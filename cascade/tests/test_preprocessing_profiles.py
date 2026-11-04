"""Public profile/cascade behavior, including multi-face source-frame semantics."""
import json
import subprocess
import sys
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from cascade.video_scoring import VideoScore


def video(path):
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),5,(64,64))
    for n in range(5): writer.write(np.full((64,64,3),n*20,dtype=np.uint8))
    writer.release()


def fake_process(request, timeout):
    from cascade.preprocessing import execute_request
    def faces(profile,image):
        if profile=='retinaface':
            return [(image,dict(bbox=[0,0,20,20])),(image,dict(bbox=[0,0,16,16]))]
        return [(image,{})]
    return execute_request(request, face_factory=lambda profile:lambda image:faces(profile,image))


class ProfileRunTests(unittest.TestCase):
    def test_public_run_uses_lazy_shared_profile_and_distinct_sbi_faces(self):
        from cascade.video import run_video, POLICIES
        class Adapter:
            def __init__(self,name,*args,**kwargs):self.name=name;self.provenance={}
            def infer(self,records):
                for r in records:
                    self.assert_record(r)
                return tuple(VideoScore(r['video_name'],r['label'],.6,len(r['face_groups'])) for r in records),{}
            @staticmethod
            def assert_record(r):
                assert len(r['face_groups'])==5
                assert len(r['frames']) in (5,10)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'in.mp4';video(source)
            with patch('cascade.preprocessing.preflight',return_value={'dlib81':{'id':'dlib81'},'retinaface':{'id':'retinaface'}}), \
                 patch('cascade.preprocessing.run_process',side_effect=fake_process), \
                 patch('cascade.adapters.FrameDetectorAdapter',Adapter):
                result=run_video(source,root/'out',policy_path=POLICIES['all_stages'],device='cpu',allow_exploratory=True)
            self.assertEqual(result['schema_version'],2)
            self.assertEqual(result['preprocessing']['mode'],'official')
            self.assertEqual(set(result['preprocessing']['profiles']),{'dlib81','retinaface'})
            self.assertEqual(result['preprocessing']['profiles']['dlib81']['state'],'used')
            self.assertEqual(result['preprocessing']['profiles']['retinaface']['result']['face_count'],10)
            self.assertEqual(result['decisions'][0]['frame_count'],5)
            self.assertIsNone(result['metrics']['fnr'])

    def test_missing_readiness_fails_without_silent_legacy_fallback(self):
        from cascade.video import run_video
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'in.mp4';video(source)
            with patch('cascade.preprocessing.preflight',side_effect=ValueError('setup required')):
                with self.assertRaisesRegex(ValueError,'setup required'):
                    run_video(source,root/'out',device='cpu',allow_exploratory=True)
            self.assertFalse((root/'out/report.json').exists())

    def test_budget_failure_preserves_error_without_binary_report(self):
        from cascade.video import run_video
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'in.mp4';video(source)
            with patch('cascade.preprocessing.run_process',side_effect=fake_process):
                with self.assertRaisesRegex(ValueError,'cache'):
                    run_video(source,root/'out',preprocessing_mode='legacy',cache_max_bytes=1,
                              device='cpu',allow_exploratory=True)
            self.assertTrue((root/'out/error.json').is_file())
            self.assertFalse((root/'out/report.json').exists())

    def test_early_exit_does_not_run_unreached_profile(self):
        from cascade.video import run_video
        class Adapter:
            def __init__(self,*args,**kwargs):self.provenance={}
            def infer(self,records):return tuple(VideoScore(r['video_name'],None,.9,len(r['face_groups'])) for r in records),{}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'in.mp4';video(source)
            with patch('cascade.preprocessing.preflight',return_value={'dlib81':{'id':'dlib81'},'retinaface':{'id':'retinaface'}}), \
                 patch('cascade.preprocessing.run_process',side_effect=fake_process), \
                 patch('cascade.adapters.FrameDetectorAdapter',Adapter):
                result=run_video(source,root/'out',device='cpu',allow_exploratory=True)
            self.assertEqual(result['preprocessing']['profiles']['retinaface']['state'],'not_reached')
            self.assertFalse((root/'out/profiles/retinaface').exists())

    def test_worker_timeout_is_interruptible(self):
        from cascade.preprocessing import run_process
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            request=dict(kind='raw',output=str(root/'worker'))
            began=time.monotonic()
            real_popen=subprocess.Popen
            def stalled(*args,**kwargs):
                return real_popen([sys.executable,'-c','import time;time.sleep(30)'],**kwargs)
            with patch('cascade.preprocessing.subprocess.Popen',side_effect=stalled):
                with self.assertRaisesRegex(ValueError,'timeout'):
                    run_process(request,.3)
            self.assertLess(time.monotonic()-began,10)

    def test_reached_no_face_profile_preserves_partial_trace(self):
        from cascade.video import run_video,POLICIES
        from cascade.preprocessing import execute_request
        class Adapter:
            def __init__(self,*args,**kwargs):self.provenance={}
            def infer(self,rows):return tuple(VideoScore(r['video_name'],None,.6,len(r['face_groups'])) for r in rows),{}
        def process(request,timeout):
            return execute_request(request,face_factory=lambda profile:(
                (lambda im:[]) if profile=='retinaface' else lambda im:[(im,{})]))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'in.mp4';video(source)
            with patch('cascade.preprocessing.preflight',return_value={'dlib81':{'id':'dlib81'},'retinaface':{'id':'retinaface'}}), \
                 patch('cascade.preprocessing.run_process',side_effect=process), \
                 patch('cascade.adapters.FrameDetectorAdapter',Adapter):
                with self.assertRaisesRegex(ValueError,'no valid faces'):
                    run_video(source,root/'out',policy_path=POLICIES['all_stages'],device='cpu',allow_exploratory=True)
            self.assertEqual(len(json.loads((root/'out/partial_trace.json').read_text())),2)
            self.assertFalse((root/'out/report.json').exists())
            info=json.loads((root/'out/preprocessing.json').read_text())
            self.assertEqual(info['profiles']['retinaface']['state'],'failed')


class MultiFaceTests(unittest.TestCase):
    def test_sbi_uses_max_per_frame_then_mean(self):
        from cascade.adapters import FrameDetectorAdapter
        class Model:
            def __call__(self,data,inference=False):return {'prob':torch.tensor([.1,.9,.2])}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);checkpoint=root/'ckpt';checkpoint.write_bytes(b'x')
            config=root/'sbi.yaml';config.write_text('model_name: sbi\nresolution: 8\nmean: [0,0,0]\nstd: [1,1,1]\n')
            paths=[]
            for n in range(3):
                p=root/f'{n}.png';cv2.imwrite(str(p),np.zeros((8,8,3),dtype=np.uint8));paths.append(str(p))
            adapter=FrameDetectorAdapter('sbi',config,checkpoint,'cpu',warmup=False)
            rows=[dict(video_name='a',label=None,frames=paths,face_groups=[
                dict(frame_index=1,faces=paths[:2]),dict(frame_index=2,faces=paths[2:])])]
            with patch('cascade.adapters.load_model',return_value=Model()):scores,timing=adapter.infer(rows)
            self.assertAlmostEqual(scores[0].score,.55,places=6)
            self.assertEqual(scores[0].frame_count,2)
            self.assertEqual(timing['face_count'],3)


if __name__=='__main__':unittest.main()
