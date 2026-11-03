"""HTTP upload -> shared runner -> status/result/download, including refusal paths."""
import http.client
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from urllib.parse import urlencode

from cascade.demo import make_server


class DemoHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls: list[tuple] = []

        def runner(video, output, **kwargs):
            self.calls.append((video, kwargs))
            self.started.set()
            self.release.wait(timeout=5)
            if video.read_bytes() == b'bad':
                raise ValueError('無法解碼影片')
            output.mkdir()
            report = {'decisions': [{'label': None, 'decision': 'FAKE'}]}
            (output/'report.json').write_text(json.dumps(report), encoding='utf-8')
            (output/'status.json').write_text('{"phase":"complete"}', encoding='utf-8')
            (output/'trace.csv').write_text('label,score\n,0.9\n')
            kwargs['progress']('inference', '推論中')
            return report

        self.server = make_server(self.root, port=0, device='cpu', runner=runner)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)

    def close_server(self):
        self.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, method, path, body=None, headers=None):
        client = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        client.request(method, path, body=body, headers=headers or {})
        response = client.getresponse()
        result = response.status, response.read()
        client.close()
        return result

    def upload(self, body=b'valid movie', filename='同名.mp4'):
        return self.request('POST', '/api/jobs?'+urlencode(dict(filename=filename, policy='early', exploratory='true')), body)

    def test_config_explains_twenty_minute_limit_and_sparse_sampling(self):
        code, body = self.request('GET', '/api/config')
        self.assertEqual(code, 200)
        config = json.loads(body)
        self.assertEqual(config['max_bytes'], 200 * 1024 * 1024)
        self.assertIn('1200 秒', config['limits'])
        self.assertIn('32 幀', config['limits'])
        code, page = self.request('GET', '/')
        self.assertEqual(code, 200)
        self.assertIn('短暫換臉', page.decode('utf-8'))

    def settled(self, job_id):
        for _ in range(100):
            code, content = self.request('GET', '/api/jobs/'+job_id)
            result = json.loads(content)
            if result['phase'] in ('complete', 'failed'):
                return result
            time.sleep(.01)
        self.fail('worker did not settle')

    def test_upload_busy_result_range_download_and_same_filename(self):
        code, content = self.upload()
        self.assertEqual(code, 202)
        job_id = json.loads(content)['id']
        self.assertTrue(self.started.wait(timeout=2))
        self.assertEqual(self.upload()[0], 409)
        self.release.set()
        result = self.settled(job_id)
        self.assertEqual(result['phase'], 'complete')
        self.assertIsNone(result['report']['decisions'][0]['label'])
        self.assertEqual(self.calls[0][1]['original_filename'], '同名.mp4')
        code, video = self.request('GET', '/jobs/'+job_id+'/input.mp4', headers={'Range':'bytes=0-3'})
        self.assertEqual((code, video), (206, b'vali'))
        self.assertEqual(self.request('GET', '/jobs/'+job_id+'/trace.csv')[0], 200)
        self.assertEqual(self.request('GET', '/jobs/'+job_id+'/../../secret')[0], 404)
        code, again = self.upload()
        self.assertEqual(code, 202)
        other = json.loads(again)['id']
        self.assertNotEqual(job_id, other)
        self.assertEqual(self.settled(other)['phase'], 'complete')
        self.assertEqual((self.root/job_id/'input.mp4').read_bytes(), b'valid movie')

    def test_failed_runner_does_not_become_real_and_releases_busy(self):
        self.release.set()
        code, content = self.upload(b'bad')
        self.assertEqual(code, 202)
        result = self.settled(json.loads(content)['id'])
        self.assertEqual(result['phase'], 'failed')
        self.assertNotIn('report', result)
        self.assertIn('無法解碼', result['message'])
        self.assertEqual(self.upload()[0], 202)

    def test_completion_is_not_reported_until_result_is_available(self):
        def runner(video, output, **kwargs):
            kwargs['progress']('complete', '推論完成')
            self.started.set()
            self.release.wait(timeout=5)
            return {'decisions':[{'label':None,'decision':'FAKE'}]}
        self.server.demo_state.runner = runner
        code, content = self.upload()
        self.assertEqual(code, 202)
        job_id = json.loads(content)['id']
        self.assertTrue(self.started.wait(timeout=2))
        code, content = self.request('GET', '/api/jobs/'+job_id)
        pending = json.loads(content)
        self.assertEqual(pending['phase'], 'finalizing')
        self.assertNotIn('report', pending)
        self.release.set()
        self.assertEqual(self.settled(job_id)['phase'], 'complete')

    def test_completed_jobs_are_available_after_server_restart(self):
        self.release.set()
        _, content = self.upload()
        job_id = json.loads(content)['id']
        self.assertEqual(self.settled(job_id)['phase'], 'complete')
        restarted = make_server(self.root, port=0, device='cpu')
        worker = threading.Thread(target=restarted.serve_forever, daemon=True)
        worker.start()
        try:
            client = http.client.HTTPConnection('127.0.0.1', restarted.server_port, timeout=5)
            client.request('GET', '/api/jobs/'+job_id)
            response = client.getresponse()
            self.assertEqual(response.status, 200)
            self.assertEqual(json.loads(response.read())['phase'], 'complete')
            client.close()
        finally:
            restarted.shutdown()
            restarted.server_close()
            worker.join(timeout=2)

    def test_origin_size_policy_and_acknowledgment_are_checked(self):
        self.assertEqual(self.request('POST', '/api/jobs?filename=a.mp4&exploratory=true', b'x',
                                      {'Origin':'https://example.com'})[0], 403)
        self.assertEqual(self.request('POST', '/api/jobs?filename=a.mp4', b'x')[0], 400)
        self.assertEqual(self.request('POST', '/api/jobs?filename=a.exe&exploratory=true', b'x')[0], 400)
        self.assertEqual(self.request('POST', '/api/jobs?filename=a.mp4&policy=unknown&exploratory=true', b'x')[0], 400)
        self.assertEqual(self.upload(b'')[0], 400)
        self.assertEqual(len(self.calls), 0)


if __name__ == '__main__':
    unittest.main()
