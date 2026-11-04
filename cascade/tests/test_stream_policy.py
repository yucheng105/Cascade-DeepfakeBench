"""Selected streaming policy is frozen, bound and scoped to active models."""
import json
import http.client
import threading
import time
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


def policy():
    return dict(schema_version=1,kind='sequential',dataset_id='uploaded_video',policy_id='ucf-only',
        deployment_ready=False,metric_definition='fake_positive_far_is_fnr',
        comparators='intermediate_real_le_fake_ge_final_fake_ge',weights=[],
        stages=[dict(detector='ucf',low=.5,high=.5)])


class StreamPolicyTests(unittest.TestCase):
    def test_session_freezes_policy_and_rejects_video_domain(self):
        from cascade.streaming import StreamingSession
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'policy.json';p=policy();path.write_text(json.dumps(p))
            session=StreamingSession(root/'x.mp4',root/'out',policy_path=path)
            path.write_text('{}')
            self.assertEqual(session.report['policy']['policy_id'],'ucf-only')
            self.assertEqual(session.settings['policy']['stages'][0]['detector'],'ucf')
            p['input_contract']=dict(unit='video',sampling='uniform_unique_frame_indices_v1',max_frames=32,
                profiles={'ucf':'dlib81'},aggregation={'ucf':'max_face_then_mean_frame'})
            with self.assertRaisesRegex(ValueError,'input_contract'):
                StreamingSession(root/'x.mp4',root/'bad',policy=p)

    def test_engine_loads_only_policy_models_and_checks_hashes_before_open(self):
        from cascade.streaming import OfficialWindowEngine
        created=[];opened=[]
        class Adapter:
            def __init__(self,name,*a,**kw):
                created.append(name);self.name=name
                self.provenance=dict(checkpoint_sha256='a'*64,config_sha256='b'*64,preprocessing={})
            def open(self):opened.append(self.name)
            def close(self):pass
        with tempfile.TemporaryDirectory() as tmp, patch('cascade.adapters.FrameDetectorAdapter',Adapter), \
                patch('cascade.preprocessing.preflight',return_value={'dlib81':dict(id='dlib81',assets={})}), \
                patch('cascade.face_profiles.build_faces',return_value=lambda image:[]):
            engine=OfficialWindowEngine(Path(tmp),'cpu','official',None,10000,policy=policy())
            engine.close()
            self.assertEqual(created,['ucf']);self.assertEqual(opened,['ucf'])
            opened.clear();p=policy();p['checkpoint_sha256']={'ucf':'c'*64}
            with self.assertRaisesRegex(ValueError,'checkpoint_sha256'):
                OfficialWindowEngine(Path(tmp),'cpu','official',None,10000,policy=p)
            self.assertEqual(opened,[])

    def test_http_registration_freezes_custom_order_and_downloads_it_after_restart(self):
        from cascade.demo import make_server
        captured=[]
        class Session:
            def __init__(self,video,output,**kwargs):
                self.output=output;self.kwargs=kwargs;captured.append(kwargs['policy'])
            def run(self):
                self.output.mkdir()
                report=dict(phase='complete',message='fixture',policy=self.kwargs['policy'],windows=[],summary={})
                for name in ('report.json','manifest.json'):
                    (self.output/name).write_text(json.dumps(report))
                return report
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'selected.json';p=policy();path.write_text(json.dumps(p))
            server=make_server(root/'runs',port=0,device='cpu',stream_factory=Session,stream_policies={'selected':path})
            path.write_text('{}')
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            def request(method,url,body=None):
                c=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=5)
                c.request(method,url,body=body);r=c.getresponse();result=(r.status,r.read());c.close();return result
            try:
                config=json.loads(request('GET','/api/config')[1])
                self.assertIn('selected',[r['name'] for r in config['stream_policies']])
                code,raw=request('POST','/api/jobs?filename=x.mp4&analysis_mode=stream&policy=selected&exploratory=true',b'fixture')
                self.assertEqual(code,202);identity=json.loads(raw)['id']
                for _ in range(100):
                    state=json.loads(request('GET','/api/jobs/'+identity)[1])
                    if state['phase'] in ('complete','failed'):break
                    time.sleep(.01)
                self.assertEqual(state['phase'],'complete')
                self.assertEqual(captured[0],p)
                self.assertEqual(state['stream_policy'],p)
                saved=json.loads(request('GET',f'/jobs/{identity}/manifest.json')[1])
                self.assertEqual(saved['policy']['stages'],p['stages'])
                self.assertEqual(request('POST','/api/jobs?filename=x.mp4&analysis_mode=stream&policy=missing&exploratory=true',b'x')[0],400)
            finally:
                server.shutdown();server.server_close();thread.join(2)
            restored=make_server(root/'runs',port=0,device='cpu')
            try:self.assertEqual(restored.demo_state.snapshot(identity)['stream_policy'],p)
            finally:restored.server_close()


if __name__=='__main__':unittest.main()
