"""Public HTTP streaming contract with controlled source/model speed and errors."""
import http.client
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from cascade.demo import make_server
from cascade.streaming import StreamingSession, Peer, VideoSource


def stalled_child(connection, kind, args):
    connection.send(('ok', dict(test_only=True)))
    connection.recv()
    time.sleep(30)


class Source:
    duration = .55
    def __init__(self, *args, **kwargs):
        self.info = dict(duration_seconds=self.duration, fps=100, frame_count=55)
    def read(self, index):
        return np.zeros((12, 12, 3), dtype=np.uint8)
    def finish(self):
        pass
    def close(self):
        pass


class Worker:
    delay = 0.
    failure = False
    noface = False
    calls = []
    def __init__(self, *args, **kwargs):
        self.provenance = dict(test_only=True)
    def infer(self, window):
        self.calls.append(window['index'])
        time.sleep(self.delay)
        if self.failure:
            raise ValueError('controlled model error')
        return dict(outcome='unable_to_decide' if self.noface else 'decided',
                    decision=None if self.noface else 'FAKE', score=None if self.noface else .9,
                    exit_stage=None if self.noface else 1, trace=[], profiles={},
                    label=None, planned_samples=len(window['samples']))
    def close(self):
        pass


class StreamingHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.worker_type = type('ControlledWorker', (Worker,), dict(calls=[], delay=0., failure=False, noface=False))
        self.sessions = []
        def factory(video, output, **kwargs):
            session = StreamingSession(video, output, source_factory=Source,
                worker_factory=self.worker_type, window_seconds=.1, sample_hz=20, **kwargs)
            self.sessions.append(session)
            return session
        self.server = make_server(self.root, port=0, device='cpu', stream_factory=factory)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.shutdown)

    def shutdown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)

    def request(self, method, path, body=None):
        c = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        c.request(method, path, body=body)
        r = c.getresponse()
        result = r.status, r.read()
        c.close()
        return result

    def start(self):
        code, body = self.request('POST', '/api/jobs?filename=a.mp4&exploratory=true&analysis_mode=stream', b'fixture')
        self.assertEqual(code, 202, body)
        return json.loads(body)['id']

    def settled(self, job):
        for _ in range(300):
            code, body = self.request('GET', '/api/jobs/'+job)
            self.assertEqual(code, 200)
            state = json.loads(body)
            if state['phase'] in ('complete', 'stopped', 'failed'):
                return state
            time.sleep(.01)
        self.fail('session did not finish')

    def test_paced_results_tail_download_and_restart(self):
        job = self.start()
        self.assertEqual(self.request('POST', '/api/jobs?filename=b.mp4&exploratory=true', b'x')[0], 409)
        state = self.settled(job)
        self.assertEqual(state['phase'], 'complete')
        rows = state['report']['windows']
        self.assertEqual([r['index'] for r in rows], list(range(6)))
        self.assertAlmostEqual(rows[-1]['end_seconds'], .55)
        self.assertTrue(all(r['outcome']=='decided' for r in rows))
        self.assertTrue(all(r['result_available_elapsed_seconds'] >= r['end_seconds'] for r in rows))
        self.assertEqual(self.request('GET', f'/jobs/{job}/windows.csv')[0], 200)
        self.assertEqual(self.request('GET', f'/jobs/{job}/report.json')[0], 200)
        restarted = make_server(self.root, port=0, device='cpu')
        try:
            loaded = restarted.demo_state.snapshot(job)
            self.assertEqual(loaded['phase'], 'complete')
            self.assertEqual(loaded['report']['windows'], rows)
        finally:
            restarted.server_close()

    def test_overload_keeps_latest_window_and_accounts_for_every_range(self):
        self.worker_type.delay = .26
        state = self.settled(self.start())
        rows = state['report']['windows']
        self.assertEqual(len(rows), 6)
        self.assertTrue(any(r['outcome']=='skipped' for r in rows))
        self.assertEqual(rows[-1]['outcome'], 'decided')
        self.assertTrue(all(r.get('decision') is None for r in rows if r['outcome']=='skipped'))
        self.assertLess(len(self.worker_type.calls), 6)

    def test_stop_finishes_running_window_and_cancels_pending_and_partial(self):
        self.worker_type.delay = .35
        job = self.start()
        for _ in range(100):
            if self.worker_type.calls:
                break
            time.sleep(.005)
        time.sleep(.125)
        self.assertEqual(self.request('POST', f'/api/jobs/{job}/stop', b'')[0], 202)
        state = self.settled(job)
        self.assertEqual(state['phase'], 'stopped')
        outcomes = [r['outcome'] for r in state['report']['windows']]
        self.assertIn('decided', outcomes)
        self.assertIn('cancelled', outcomes)
        self.assertEqual(len(self.worker_type.calls), 1)

    def test_no_face_continues_and_model_failure_preserves_evidence_and_releases_busy(self):
        self.worker_type.noface = True
        state = self.settled(self.start())
        self.assertEqual(state['phase'], 'complete')
        self.assertTrue(all(r['decision'] is None for r in state['report']['windows']))
        self.worker_type.failure = True
        failed_job = self.start()
        failed = self.settled(failed_job)
        self.assertEqual(failed['phase'], 'failed')
        self.assertIn('controlled model error', failed['message'])
        self.assertEqual(self.request('GET', f'/jobs/{failed_job}/report.json')[0], 200)
        self.worker_type.failure = False
        self.assertEqual(self.settled(self.start())['phase'], 'complete')

    def test_invalid_stream_options_and_stop_unknown_job(self):
        self.assertEqual(self.request('POST','/api/jobs?filename=a.mp4&exploratory=true&analysis_mode=bad',b'x')[0],400)
        self.assertEqual(self.request('POST','/api/jobs?filename=a.mp4&exploratory=true&analysis_mode=stream&policy=all_stages',b'x')[0],400)
        self.assertEqual(self.request('POST','/api/jobs/'+'0'*32+'/stop',b'')[0],404)

    def test_native_worker_timeout_terminates_process_and_http_job_fails(self):
        class StalledWorker(Worker):
            def __init__(self, *args, **kwargs):
                self.peer = Peer('test', (), 10, target=stalled_child)
                self.peer.timeout = .1
                self.provenance = self.peer.initial
            def infer(self, window):
                return self.peer.call('stall')
            def close(self):
                self.peer.close()
        self.worker_type = StalledWorker
        state = self.settled(self.start())
        self.assertEqual(state['phase'], 'failed')
        self.assertIn('terminated', state['message'])
        self.assertEqual(state['report']['windows'][0]['outcome'], 'failed')

    def test_readiness_failure_never_starts_media_clock(self):
        class MissingAssets(Worker):
            def __init__(self, *args, **kwargs):
                raise ValueError('official asset missing')
        self.worker_type = MissingAssets
        state = self.settled(self.start())
        self.assertEqual(state['phase'], 'failed')
        self.assertNotIn('playback_started_unix_seconds', state['report'])
        self.assertEqual(state['report']['windows'], [])

    def test_cache_overflow_fails_without_classification(self):
        self.server.demo_state.cache_max_bytes = 1
        state = self.settled(self.start())
        self.assertEqual(state['phase'],'failed')
        self.assertIn('cache budget',state['message'])
        self.assertTrue(all(r['decision'] is None for r in state['report']['windows']))

    def test_persistence_failure_stops_consumer_without_false_completion(self):
        from cascade.streaming import save_report
        raised = False
        def failing(output, report):
            nonlocal raised
            if not raised and any(r['outcome']=='processing' for r in report['windows']):
                raised = True
                raise OSError('controlled persistence failure')
            return save_report(output,report)
        with patch('cascade.streaming.save_report',side_effect=failing):
            state = self.settled(self.start())
        self.assertEqual(state['phase'],'failed')
        self.assertIn('persistence failure',state['message'])
        self.assertTrue(all(r['outcome'] not in ('processing','waiting','collecting') for r in state['report']['windows']))

    def test_stop_during_startup_never_consumes_source(self):
        entered = threading.Event()
        class Loading(Worker):
            def __init__(self, *args, **kwargs):
                entered.set()
                kwargs['stop_event'].wait(2)
                super().__init__()
        self.worker_type = Loading
        job = self.start()
        self.assertTrue(entered.wait(2))
        self.assertEqual(self.request('POST', f'/api/jobs/{job}/stop', b'')[0], 202)
        state = self.settled(job)
        self.assertEqual(state['phase'], 'stopped')
        self.assertNotIn('playback_started_unix_seconds', state['report'])

    def test_interrupted_restart_marks_unfinished_rows_and_csv_consistently(self):
        from cascade.preprocessing import atomic_json
        job = 'a'*32
        folder = self.root/job
        (folder/'result').mkdir(parents=True)
        atomic_json(folder/'job.json', dict(id=job, filename='a.mp4', analysis_mode='stream'))
        atomic_json(folder/'result/report.json', dict(phase='playing',windows=[
            dict(index=0,start_seconds=0,end_seconds=4,outcome='processing',decision=None)]))
        restarted = make_server(self.root, port=0, device='cpu')
        try:
            state = restarted.demo_state.snapshot(job)
            self.assertEqual(state['phase'], 'failed')
            self.assertEqual(state['report']['summary']['cancelled'], 1)
            self.assertEqual(state['report']['windows'][0]['reason'], 'service_interrupted')
            self.assertIn('service_interrupted',(folder/'result/windows.csv').read_text())
        finally:
            restarted.server_close()


class NativeSourceTests(unittest.TestCase):
    def test_real_decoder_reads_requested_frames_and_rejects_corrupt_input(self):
        import cv2
        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary)/'tiny.mp4'
            writer = cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),10,(32,32))
            for i in range(5):
                writer.write(np.full((32,32,3),i*40,dtype=np.uint8))
            writer.release()
            source = VideoSource(video, stop_event=threading.Event())
            try:
                self.assertAlmostEqual(source.info['duration_seconds'], .5)
                self.assertLess(source.read(0).mean(), source.read(3).mean())
                source.finish()
            finally:
                source.close()
            video.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'metadata'):
                VideoSource(video, stop_event=threading.Event())


if __name__ == '__main__':
    unittest.main()
