"""Public HTTP explanation/download/restart journeys, actual Cascade routing."""
import csv
import http.client
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from cascade.tests import test_window_journey as fixtures


class ExplanationJourneyTests(unittest.TestCase):
    journey=fixtures.WindowJourneyTests.journey
    short_events=staticmethod(fixtures.WindowJourneyTests.short_events)

    def test_both_window_modes_explain_real_fake_and_provenance(self):
        for mode in ('offline_window','stream'):
            with self.subTest(mode=mode),self.journey(self.short_events()) as (request,start,settled):
                job=settled(start(mode));self.assertEqual(job['phase'],'complete')
                report=job['report'];self.assertEqual(report['trace_schema_version'],1)
                self.assertTrue(any(w['outcome']=='decided' for w in report['windows']))
                for w in report['windows']:
                    x=w['explanation']
                    self.assertEqual(x['context']['unit'],'window')
                    self.assertEqual(x['context']['window_id'],w['window_id'])
                    self.assertEqual(x['context']['policy_sha256'],report['policy_sha256'])
                    if w['outcome']=='decided':
                        self.assertEqual(x['stages'][0]['execution_status'],'scored')
                        self.assertIn(x['stages'][0]['route_reason'],('final_fake_ge_threshold','final_real_lt_threshold'))
                    else:
                        self.assertEqual(w['outcome'],'skipped')
                        self.assertEqual(x['stages'][0]['execution_status'],'not_executed')
                        self.assertIsNone(x['stages'][0]['score'])
                    self.assertFalse(x['score_is_calibrated_confidence'])
                _,full=request('GET',f"/jobs/{job['id']}/report.json")
                self.assertEqual(full['windows'][0]['explanation'],report['windows'][0]['explanation'])

    def test_unknown_failure_stage_is_not_guessed_from_prior_score(self):
        class Broken(fixtures.Worker):
            def infer(self,window):raise ValueError('fixture failure without stage evidence')
        with self.journey(self.short_events(),worker=Broken) as (_,start,settled):
            job=settled(start());self.assertEqual(job['phase'],'failed')
            row=next(w for w in job['report']['windows'] if w['outcome']=='failed')
            x=row['explanation'];self.assertIsNone(x['failure_stage'])
            self.assertIsNone(x['stages'][0]['score'])
            self.assertEqual(x['stages'][0]['execution_status'],'unreported')

    def test_known_validation_failure_and_overload_have_distinct_explanations(self):
        class Invalid(fixtures.Worker):
            def infer(self,window):
                from cascade.controller import run_cascade
                from cascade.video_scoring import VideoScore
                class Adapter:
                    def infer(self,records):return (VideoScore('window_'+str(window['index']),None,float('nan'),1),),{}
                return run_cascade([dict(video_name='window_'+str(window['index']),label=None)],self.policy,{'ucf':Adapter()})
        with self.journey(self.short_events(),worker=Invalid) as (_,start,settled):
            report=settled(start())['report']
            row=next(w for w in report['windows'] if w['outcome']=='failed')
            self.assertEqual(row['explanation']['failure_stage'],1)
            self.assertEqual(row['explanation']['failure_phase'],'score_validation')
            self.assertEqual(row['explanation']['stages'][0]['execution_status'],'failed')
        class Slow(fixtures.Worker):
            def infer(self,window):time.sleep(.3);return super().infer(window)
        with self.journey(self.short_events(),worker=Slow) as (_,start,settled):
            report=settled(start('stream'))['report']
            skipped=[w for w in report['windows'] if w['outcome']=='skipped']
            self.assertTrue(skipped)
            for row in skipped:
                self.assertEqual(row['explanation']['stages'][0]['execution_status'],'not_executed')
                self.assertIsNone(row['explanation']['stages'][0]['score'])

    def test_offline_sparse_http_route_export_and_restart(self):
        from cascade.demo import make_server
        from cascade.tests.test_preprocessing_profiles import fake_process,video
        from cascade.video_scoring import VideoScore
        class Adapter:
            def __init__(self,*a,**kw):self.provenance={}
            def infer(self,records):
                return tuple(VideoScore(r['video_name'],None,.9,len(r['face_groups'])) for r in records),{}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.mp4';video(source)
            with patch('cascade.preprocessing.preflight',return_value={'dlib81':{'id':'dlib81'},'retinaface':{'id':'retinaface'}}),patch('cascade.preprocessing.run_process',side_effect=fake_process),patch('cascade.adapters.FrameDetectorAdapter',Adapter):
                server=make_server(root/'jobs',port=0,device='cpu')
                thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
                def request(method,path,body=None,active=server):
                    conn=http.client.HTTPConnection('127.0.0.1',active.server_port,timeout=10)
                    conn.request(method,path,body);r=conn.getresponse();raw=r.read();conn.close()
                    self.assertIn(r.status,(200,202),raw);return raw
                try:
                    job=json.loads(request('POST','/api/jobs?filename=x.mp4&exploratory=true',source.read_bytes()))
                    for _ in range(300):
                        state=json.loads(request('GET','/api/jobs/'+job['id']))
                        if state['phase'] in ('complete','failed'):break
                        time.sleep(.01)
                    self.assertEqual(state['phase'],'complete',state)
                    x=state['report']['explanation']
                    self.assertEqual(x['stages'][0]['route_reason'],'fake_ge_high')
                    self.assertEqual(x['stages'][1]['execution_status'],'not_executed')
                    self.assertEqual(x['stages'][1]['reason'],'prior_exit')
                    rows=list(csv.DictReader(io.StringIO(request('GET',f"/jobs/{job['id']}/explanation.csv").decode())))
                    self.assertEqual(len(rows),3);self.assertEqual(rows[1]['score'],'')
                    restarted=make_server(root/'jobs',port=0,device='cpu')
                    other=threading.Thread(target=restarted.serve_forever,daemon=True);other.start()
                    try:
                        loaded=json.loads(request('GET','/api/jobs/'+job['id'],active=restarted))
                        self.assertEqual(loaded['report']['explanation'],x)
                    finally:restarted.shutdown();restarted.server_close();other.join(2)
                finally:server.shutdown();server.server_close();thread.join(2)

    def test_interrupted_new_trace_is_recovered_without_stale_processing_view(self):
        from cascade.demo import make_server
        from cascade.streaming import save_report
        from cascade.explanation import canonical_hash
        from cascade.tests.test_stream_policy import policy
        from cascade.temporal import TemporalAggregator
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);identity='a'*32;folder=root/identity/'result';folder.mkdir(parents=True)
            p=policy();row=dict(window_id='session:0',index=0,start_seconds=0.,end_seconds=.2,observed_end_seconds=.2,
                               outcome='processing',decision=None,source_frames=[])
            report=dict(schema_version=4,trace_schema_version=1,phase='playing',policy=p,policy_sha256=canonical_hash(p),
                        input_contract={'unit':'window'},windows=[row],observed_horizon_seconds=.2,
                        aggregate=TemporalAggregator().snapshot(.2,pending=1))
            save_report(folder,report)
            (folder.parent/'job.json').write_text(json.dumps(dict(id=identity,filename='x.mp4',analysis_mode='stream')))
            server=make_server(root,port=0,device='cpu');thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                conn=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=5)
                conn.request('GET','/api/jobs/'+identity);job=json.loads(conn.getresponse().read());conn.close()
                self.assertEqual(job['phase'],'failed')
                x=job['report']['windows'][0]['explanation']
                self.assertEqual(x['observation_outcome'],'cancelled')
                self.assertEqual(x['stages'][0]['execution_status'],'not_executed')
                self.assertEqual(x['operational_failure']['kind'],'service_interrupted')
            finally:server.shutdown();server.server_close();thread.join(2)


if __name__=='__main__':unittest.main()
