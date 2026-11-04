"""Phase 3 acceptance through the actual HTTP/session/report seam."""
import http.client
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager
from unittest.mock import patch
import numpy as np
from cascade.tests.test_stream_policy import policy


class Events:
    info = {'kind':'controlled_events', 'source_id':'fixture', 'timestamp_basis':'fixture_media_time'}

    def __init__(self, *args, **kwargs):
        self.events = iter([e for i in range(8) for e in (
            dict(type='frame', frame_id=str(i), frame_index=i, timestamp_seconds=i*.5,
                 timestamp_basis='fixture_media_time', image=np.full((2,2,3),i,dtype=np.uint8)),
            dict(type='progress', seconds=(i+1)*.5))]+[dict(type='end',seconds=4.)])

    def read_event(self, **kwargs):
        return next(self.events)

    def close(self):
        pass


class Worker:
    provenance = {'fixture':True}

    def __init__(self,*args,**kwargs):
        self.policy=kwargs.get('policy',policy())

    def infer(self,window):
        from cascade.controller import run_cascade
        from cascade.video_scoring import VideoScore
        assert all(s['timestamp_seconds'] <= window['observed_end_seconds'] for s in window['samples'])
        score=.9 if window['index']==1 else .1
        class Adapter:
            def infer(self,records):
                return tuple(VideoScore(r['video_name'],r['label'],score,len(window['samples'])) for r in records),{}
        result=run_cascade([dict(video_name='window_'+str(window['index']),label=None)],
                           self.policy,{s['detector']:Adapter() for s in self.policy['stages']})
        decision=result['decisions'][0]
        return dict(outcome='decided',decision=decision['decision'],score=decision['score'],
                    exit_stage=decision['exit_stage'],trace=result['trace'],profiles={})

    def close(self):
        pass


class WindowJourneyTests(unittest.TestCase):
    @contextmanager
    def journey(self, events, worker=Worker, **settings):
        from cascade.demo import make_server
        from cascade.streaming import StreamingSession
        class Source:
            info={'kind':'controlled_events','source_id':'source'}
            def __init__(self,*a,**kw):
                self.iterator=iter(events)
            def read_event(self,**kw):
                return next(self.iterator)
            def close(self):pass
        factory_source=settings.pop('event_source_factory',Source)
        def factory(video,output,**kw):
            return StreamingSession(video,output,event_source_factory=factory_source,worker_factory=worker,**dict(kw,**settings))
        with tempfile.TemporaryDirectory() as tmp:
            server=make_server(Path(tmp),port=0,device='cpu',stream_factory=factory,
                               stream_policies={'fixture':policy(),'bound':dict(policy(),input_contract={
                                   'unit':'window','sampling':'admitted_tick_grid_v2','window_contract_version':2,
                                   'window_seconds':.2,'stride_seconds':.1,'sample_hz':20.,
                                   'profiles':{'ucf':'dlib81'},'aggregation':{'ucf':'max_face_then_mean_frame'}})})
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            def request(method,url,body=None):
                conn=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=5)
                conn.request(method,url,body);r=conn.getresponse();raw=r.read();conn.close()
                return r.status,json.loads(raw)
            def start(mode='offline_window',extra=''):
                code,job=request('POST',f'/api/jobs?filename=x.mp4&analysis_mode={mode}&policy=fixture&exploratory=true&window_seconds=.2&stride_seconds=.1&sample_hz=20'+extra,b'x')
                self.assertEqual(code,202,job)
                return job['id']
            def settled(identity):
                for _ in range(400):
                    _,job=request('GET','/api/jobs/'+identity)
                    if job['phase'] in ('complete','stopped','failed'):return job
                    time.sleep(.01)
                self.fail('window job did not settle')
            try:yield request,start,settled
            finally:
                server.shutdown();server.server_close();thread.join(2)

    @staticmethod
    def short_events():
        return [e for i in range(8) for e in (
            dict(type='frame',frame_id=str(i),frame_index=i,timestamp_seconds=i*.05,
                 timestamp_basis='fixture_media_time',image=np.full((2,2,3),i,dtype=np.uint8)),
            dict(type='progress',seconds=(i+1)*.05))]+[dict(type='end',seconds=.4)]

    def test_offline_stream_same_evidence_and_overlap_union(self):
        reports=[]
        for mode in ('offline_window','stream'):
            with self.journey(self.short_events()) as (_,start,settled):
                job=settled(start(mode));self.assertEqual(job['phase'],'complete',job)
                reports.append(job['report'])
        for field in ('state','decided_seconds','fake_intervals','gap_intervals','dispositions'):
            self.assertEqual(reports[0]['aggregate'][field],reports[1]['aggregate'][field])
        def evidence(r):
            return [{k:w[k] for k in ('source_frames','decision','score','exit_stage','trace')} for w in r['windows']]
        self.assertEqual(evidence(reports[0]),evidence(reports[1]))
        self.assertAlmostEqual(reports[0]['aggregate']['decided_seconds'],.4)

    def test_duplicates_late_events_empty_progress_and_conflicting_identity(self):
        frame=self.short_events()[0]
        events=[frame,dict(frame),dict(type='progress',seconds=.2),
                dict(frame,frame_id='late',frame_index=100),dict(type='end',seconds=.4)]
        with self.journey(events) as (_,start,settled):
            r=settled(start())['report']
            self.assertEqual([d['reason'] for d in r['source_diagnostics']],['duplicate_frame','late_frame_rejected'])
            self.assertTrue(any(w['outcome']=='unable_to_decide' for w in r['windows']))
            self.assertEqual(r['aggregate']['state'],'INSUFFICIENT_EVIDENCE')
        events=[frame,dict(frame,image=np.ones((2,2,3),dtype=np.uint8))]
        with self.journey(events) as (_,start,settled):
            job=settled(start());self.assertEqual(job['phase'],'failed')
            self.assertIn('Conflicting frame identity',job['message'])
            self.assertEqual(job['report']['source_diagnostics'][0]['reason'],'conflicting_frame_identity')

    def test_overload_keeps_dispositions_even_with_overlap(self):
        class Slow(Worker):
            def infer(self,w):time.sleep(.23);return super().infer(w)
        with self.journey(self.short_events(),Slow) as (_,start,settled):
            job=settled(start('stream'));self.assertEqual(job['phase'],'complete',job)
            r=job['report'];self.assertTrue(any(w['outcome']=='skipped' for w in r['windows']))
            self.assertTrue(any(w['outcome']=='skipped' for w in r['aggregate']['unknown_windows']))
            self.assertLessEqual(r['aggregate']['decided_seconds'],.4)

    def test_stop_drains_running_cancels_pending_and_observed_tails(self):
        entered=threading.Event()
        class Slow(Worker):
            def infer(self,w):entered.set();time.sleep(.25);return super().infer(w)
        with self.journey(self.short_events(),Slow) as (request,start,settled):
            identity=start('stream');self.assertTrue(entered.wait(3))
            self.assertEqual(request('POST',f'/api/jobs/{identity}/stop',b'')[0],202)
            job=settled(identity);self.assertEqual(job['phase'],'stopped',job)
            r=job['report'];a=r['aggregate']
            self.assertEqual(a['finalization_reason'],'user_stop')
            self.assertTrue(any(w['outcome']=='decided' for w in r['windows']))
            self.assertTrue(any(w['outcome']=='cancelled' for w in r['windows']))
            self.assertLess(a['observed_horizon_seconds'],.4)
            self.assertTrue(all(w['observed_end_seconds']<=a['observed_horizon_seconds'] for w in r['windows']))

    def test_no_face_continues_and_model_source_failures_finalize(self):
        class NoFace(Worker):
            def infer(self,w):
                return dict(outcome='unable_to_decide',decision=None,score=None,trace=[],profiles={}) if w['index']==0 else super().infer(w)
        with self.journey(self.short_events(),NoFace) as (_,start,settled):
            r=settled(start())['report']
            self.assertEqual(r['windows'][0]['outcome'],'unable_to_decide')
            self.assertEqual(r['aggregate']['state'],'FAKE_EVIDENCE')
        class Failed(Worker):
            def infer(self,w):raise ValueError('fixture model failure')
        with self.journey(self.short_events(),Failed) as (_,start,settled):
            job=settled(start());self.assertEqual(job['phase'],'failed',job)
            self.assertEqual(job['report']['aggregate']['finalization_reason'],'failure')
        with self.journey(self.short_events()[:8]+[dict(type='error',message='disconnect')]) as (_,start,settled):
            job=settled(start());self.assertEqual(job['phase'],'failed',job)
            self.assertIn('disconnect',job['message'])
            self.assertEqual(job['report']['aggregate']['state'],'INSUFFICIENT_EVIDENCE')

    def test_policy_cadence_gate_and_invalid_resource_settings(self):
        with self.journey(self.short_events()) as (request,_,__):
            for cadence in ('window_seconds=nan','window_seconds=1&stride_seconds=2',
                            'window_seconds=4&stride_seconds=.00001','window_seconds=inf'):
                code,_=request('POST','/api/jobs?filename=x.mp4&analysis_mode=offline_window&policy=fixture&exploratory=true&'+cadence,b'x')
                self.assertEqual(code,400)
            code,response=request('POST','/api/jobs?filename=x.mp4&analysis_mode=offline_window&policy=bound&exploratory=true&window_seconds=.2&stride_seconds=.2&sample_hz=20',b'x')
            self.assertEqual(code,400);self.assertIn('input_contract',response['error'])

    def test_bounded_source_timeout_and_missing_timestamp_basis(self):
        class Waiting:
            info={'kind':'controlled_wait'}
            def __init__(self,*a,**kw):pass
            def read_event(self,**kw):time.sleep(.1);raise TimeoutError('fixture wait')
            def close(self):pass
        with self.journey([],event_source_factory=Waiting,profile_timeout=.03) as (_,start,settled):
            job=settled(start());self.assertEqual(job['phase'],'failed',job)
            self.assertIn('timed out',job['message'])
        frame=self.short_events()[0];del frame['timestamp_basis']
        with self.journey([frame]) as (_,start,settled):
            job=settled(start());self.assertEqual(job['phase'],'failed',job)
            self.assertIn('basis',job['message'])

    def test_unknown_source_publishes_before_eof_and_stop_cancels_read(self):
        waiting=threading.Event()
        class Ongoing:
            info={'kind':'controlled_ongoing','source_id':'unknown-length'}
            def __init__(self,*a,**kw):self.iterator=iter(WindowJourneyTests.short_events()[:8])
            def read_event(self,*,stop_event,timeout):
                try:return next(self.iterator)
                except StopIteration:
                    waiting.set();stop_event.wait(timeout)
                    raise TimeoutError('controlled source stopped')
            def close(self):pass
        with self.journey([],event_source_factory=Ongoing) as (request,start,settled):
            identity=start();self.assertTrue(waiting.wait(3))
            for _ in range(100):
                _,job=request('GET','/api/jobs/'+identity)
                if job.get('report',{}).get('windows',[{}])[0].get('outcome')=='decided':break
                time.sleep(.01)
            self.assertEqual(job['phase'],'playing')
            self.assertEqual(job['report']['windows'][0]['outcome'],'decided')
            self.assertTrue(job['report']['aggregate']['provisional'])
            self.assertNotIn('duration_seconds',job['report']['source'])
            request('POST',f'/api/jobs/{identity}/stop',b'')
            self.assertEqual(settled(identity)['phase'],'stopped')

    def test_mp4_adapter_low_fps_dedup_preserves_empty_tails(self):
        class MP4:
            info=dict(fps=1.,frame_count=1,duration_seconds=.4,width=2,height=2)
            def __init__(self,*a,**kw):pass
            def read(self,index):
                self_index=index
                if self_index!=0:raise AssertionError('future source frame')
                return np.zeros((2,2,3),dtype=np.uint8)
            def finish(self):pass
            def close(self):pass
        with self.journey([],event_source_factory=None,source_factory=MP4) as (_,start,settled):
            job=settled(start());self.assertEqual(job['phase'],'complete',job)
            r=job['report'];self.assertIn('input_sha256',r)
            self.assertEqual([w['planned_samples'] for w in r['windows']],[1,0,0,0])
            self.assertEqual(r['aggregate']['state'],'INSUFFICIENT_EVIDENCE')
            self.assertEqual(r['windows'][0]['source_frames'][0]['timestamp_basis'],'frame_index_over_fps')

    def test_attained_event_window_and_raw_limits_fail_explicitly(self):
        for settings,reason in ((dict(max_events=3),'event budget'),(dict(max_windows=1),'record budget'),
                                (dict(cache_max_bytes=16),'raw cache budget'),
                                (dict(max_session_seconds=.1),'duration budget')):
            with self.journey(self.short_events(),**settings) as (_,start,settled):
                job=settled(start());self.assertEqual(job['phase'],'failed',job)
                self.assertIn(reason,job['message'])
                self.assertEqual(job['report']['aggregate']['finalization_reason'],'failure')

    def test_persistence_error_keeps_immutable_completed_evidence(self):
        from cascade.streaming import save_report
        raised=False
        def fail_once(output,report):
            nonlocal raised
            if not raised and any(w['outcome']=='decided' for w in report['windows']):
                raised=True
                raise OSError('completed evidence persistence failure')
            return save_report(output,report)
        with patch('cascade.streaming.save_report',side_effect=fail_once):
            with self.journey(self.short_events()) as (_,start,settled):
                job=settled(start());self.assertEqual(job['phase'],'failed',job)
                self.assertEqual(job['report']['windows'][0]['outcome'],'decided')
                self.assertEqual(job['report']['aggregate']['finalization_reason'],'failure')

    def test_interrupted_restart_finalizes_declared_rule_without_upgrading_legacy(self):
        from cascade.demo import make_server
        from cascade.temporal import TemporalAggregator
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for version in (3,4):
                identity=str(version)*32;folder=root/identity/'result';folder.mkdir(parents=True)
                row=dict(window_id=identity+':0',index=0,start_seconds=0.,end_seconds=.2,
                         observed_end_seconds=.2,outcome='waiting',decision=None)
                report=dict(schema_version=version,phase='playing',observed_horizon_seconds=.2,windows=[row])
                if version==4:report['aggregate']=TemporalAggregator().snapshot(.2,pending=1)
                (folder/'report.json').write_text(json.dumps(report))
                (folder.parent/'job.json').write_text(json.dumps(dict(id=identity,filename='x.mp4',analysis_mode='offline_window')))
            server=make_server(root,port=0,device='cpu')
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                for version in (3,4):
                    conn=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=5)
                    conn.request('GET','/api/jobs/'+str(version)*32)
                    job=json.loads(conn.getresponse().read());conn.close()
                    self.assertEqual(job['phase'],'failed')
                    self.assertEqual(job['report']['windows'][0]['reason'],'service_interrupted')
                    if version==3:self.assertNotIn('aggregate',job['report'])
                    else:
                        a=job['report']['aggregate']
                        self.assertEqual(a['rule_version'],'temporal_evidence_v1')
                        self.assertEqual(a['finalization_reason'],'failure')
                        self.assertEqual(a['state'],'INSUFFICIENT_EVIDENCE')
            finally:
                server.shutdown();server.server_close();thread.join(2)

    def test_http_offline_overlap_download_restart(self):
        from cascade.demo import make_server
        from cascade.streaming import StreamingSession
        def factory(video,output,**kw):
            return StreamingSession(video,output,event_source_factory=Events,worker_factory=Worker,**kw)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            server=make_server(root,port=0,device='cpu',stream_factory=factory,stream_policies={'fixture':policy()})
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            def request(method,url,body=None):
                conn=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=5)
                conn.request(method,url,body);r=conn.getresponse();data=r.read();conn.close()
                return r.status,json.loads(data)
            try:
                _,config=request('GET','/api/config')
                self.assertIn('offline_window',config['analysis_modes'])
                status,job=request('POST','/api/jobs?filename=x.mp4&analysis_mode=offline_window&policy=fixture&exploratory=true&window_seconds=2&stride_seconds=1&sample_hz=2',b'fixture')
                self.assertEqual(status,202);identity=job['id']
                for _ in range(200):
                    _,job=request('GET','/api/jobs/'+identity)
                    if job['phase'] in ('complete','failed'):
                        break
                    time.sleep(.01)
                self.assertEqual(job['phase'],'complete',job)
                report=job['report']
                self.assertEqual([w['planned_samples'] for w in report['windows']],[4,4,4,2])
                self.assertEqual(report['aggregate']['state'],'FAKE_EVIDENCE')
                self.assertEqual(report['aggregate']['decided_seconds'],4.)
                self.assertEqual(report['aggregate']['finalization_reason'],'eof')
                self.assertEqual(report['windows'][0]['source_frames'][2],report['windows'][1]['source_frames'][0])
                self.assertEqual(report['windows'][0]['trace'][0]['detector'],'ucf')
                _,download=request('GET',f'/jobs/{identity}/report.json')
                self.assertEqual(download['aggregate'],report['aggregate'])
                self.assertEqual(download['windows'][0]['window_contract_version'],2)
                self.assertNotIn('input_sha256',report)
            finally:
                server.shutdown();server.server_close();thread.join(2)
            restored=make_server(root,port=0,device='cpu')
            try:
                self.assertEqual(restored.demo_state.snapshot(identity)['report']['aggregate'],report['aggregate'])
            finally:
                restored.server_close()
