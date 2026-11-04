"""Resident worker contract against actual adapters/controller and grouped faces."""
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
import torch

from cascade.streaming import OfficialWindowEngine


class StreamingProfileTests(unittest.TestCase):
    def test_resident_all_stage_grouping_shared_profile_and_partial_trace(self):
        calls = {'dlib81':0,'retinaface':0}
        guard = threading.Lock()
        no_sbi = False
        class Detector:
            detector = object()
            def __init__(self, name):
                self.name = name
            def __call__(self, image):
                with guard:
                    calls[self.name] += 1
                if self.name=='dlib81':
                    return [(image, {})]
                if no_sbi:
                    self.last_rejection_reason = 'no_face'
                    return []
                values = (25,230) if image[0,0,0]==0 else (51,76)
                return [(np.full((8,8,3),v,dtype=np.uint8),{}) for v in values]
        class Model:
            def __init__(self, name):
                self.name = name
            def __call__(self, data, inference=False):
                count = len(data['image'])
                return dict(prob=data['image'][:,0,0,0] if self.name=='sbi' else torch.full((count,),.6))
        definitions = {n:dict(id=n,assets={}) for n in calls}
        with tempfile.TemporaryDirectory() as tmp, \
                patch('cascade.preprocessing.preflight', return_value=definitions), \
                patch('cascade.face_profiles.build_faces',side_effect=lambda p,*args:Detector(p)), \
                patch('cascade.streaming.reuse_retina_priors'), \
                patch('cascade.adapters.load_model',side_effect=lambda n,*args:Model(n)) as load:
            engine = OfficialWindowEngine(Path(tmp),'cpu','official',None,1024*1024)
            window = dict(index=0,samples=[dict(frame_index=i,timestamp_seconds=i/2,
                image=np.full((8,8,3),i,dtype=np.uint8)) for i in range(2)])
            try:
                result = engine.infer(window)
                self.assertEqual(result['exit_stage'],3)
                self.assertAlmostEqual(result['score'],153/255, places=5)
                self.assertEqual(result['profiles']['retinaface']['face_count'],4)
                self.assertEqual(result['profiles']['retinaface']['valid_face_frames'],2)
                self.assertEqual(calls,dict(dlib81=2,retinaface=2))
                self.assertEqual(load.call_count,3)
                self.assertTrue(all(s['timing']['tensor_seconds']>=0 for s in result['stages']))
                self.assertIn('scope',result['worker_rss'])
                self.assertIsNone(result['gpu_peak_reserved_bytes'])
                no_sbi = True
                missing = engine.infer(dict(window,index=1))
                self.assertEqual(missing['outcome'],'unable_to_decide')
                self.assertIsNone(missing['decision'])
                self.assertEqual(len(missing['trace']),2)
                self.assertEqual(load.call_count,3)
                self.assertEqual(list((Path(tmp)/'scratch').iterdir()),[])
                engine.policy['stages'][0].update(low=.7,high=.8)
                early = engine.infer(dict(window,index=2))
                self.assertEqual(early['exit_stage'],1)
                self.assertEqual(early['profiles']['retinaface']['state'],'not_reached')
                self.assertEqual(calls['retinaface'],4)
            finally:
                engine.close()


if __name__=='__main__':
    unittest.main()
