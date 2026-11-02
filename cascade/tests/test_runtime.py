"""Public runtime contract: policy + adapters -> complete, auditable decisions."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cascade.controller import run_cascade
from cascade.policy import validate_policy
from cascade.video_scoring import VideoScore


def policy():
    return dict(schema_version=1,kind='sequential',dataset_id='ffpp',policy_id='test-only',
                deployment_ready=False,metric_definition='fake_positive_far_is_fnr',
                comparators='intermediate_real_le_fake_ge_final_fake_ge',weights=[],stages=[
                    dict(detector='xception',low=.2,high=.8),
                    dict(detector='ucf',low=.2,high=.8),
                    dict(detector='sbi',low=.5,high=.5)])


class FakeAdapter:
    def __init__(self,name,scores):
        self.name,self.scores=name,scores
        self.calls=[]
        self.provenance={'model':name,'checkpoint_sha256':'fixture'}

    def infer(self, records):
        self.calls.append(tuple(r['video_name'] for r in records))
        return (tuple(VideoScore(r['video_name'],r['label'],self.scores[r['video_name']],1)
                      for r in records), {'forward_seconds': .01, 'stage_seconds': .02})


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.records=[dict(video_name=i,label=y) for i,y in [('a',0),('b',1),('c',0),('d',1)]]
        self.adapters=dict(xception=FakeAdapter('xception',dict(a=.2,b=.8,c=.5,d=.5)),
                           ucf=FakeAdapter('ucf',dict(a=.1,b=.9,c=.2,d=.5)),
                           sbi=FakeAdapter('sbi',dict(a=.1,b=.9,c=.1,d=.5)))

    def test_three_stages_exact_subset_inclusive_exits_and_final_tie(self):
        result=run_cascade(self.records,policy(),self.adapters)
        self.assertEqual([r['prediction'] for r in result['decisions']],[0,1,0,1])
        self.assertEqual([r['exit_stage'] for r in result['decisions']],[1,1,2,3])
        self.assertEqual([s['reached'] for s in result['stages']],[4,2,1])
        self.assertEqual(self.adapters['sbi'].calls,[('d',)])
        self.assertEqual(result['metrics']['tp'],2)
        self.assertEqual(result['metrics']['tn'],2)
        self.assertEqual(result['metrics']['far'],0.)
        self.assertEqual(sum(s['exited'] for s in result['stages']),4)
        self.assertEqual(len(result['trace']),7)

    def test_no_uncertain_does_not_load_later_models(self):
        p=policy();p['stages'][0].update(low=.6,high=.8)
        run_cascade(self.records,p,self.adapters)
        self.assertEqual(self.adapters['ucf'].calls,[])
        self.assertEqual(self.adapters['sbi'].calls,[])

    def test_single_stage_has_final_threshold_semantics(self):
        p=policy();p['stages']=[dict(detector='xception',low=.5,high=.5)]
        result=run_cascade(self.records,p,self.adapters)
        self.assertEqual([r['prediction'] for r in result['decisions']],[0,1,1,1])

    def test_policy_contract_matches_phase1_and_rejects_invalid(self):
        p=policy();validate_policy(p,'ffpp',allow_exploratory=True)
        for field,value in [('schema_version',2),('schema_version',True),('kind','parallel'),('dataset_id','dfdc'),
                            ('comparators','other'),('weights',[1]),('stages',[])]:
            bad=copy.deepcopy(p);bad[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                validate_policy(bad,'ffpp',allow_exploratory=True)
        for low,high in [(float('nan'),.8),(.8,.2),(.2,.2)]:
            bad=copy.deepcopy(p);bad['stages'][0].update(low=low,high=high)
            with self.assertRaises(ValueError):validate_policy(bad,'ffpp',allow_exploratory=True)
        bad=copy.deepcopy(p);bad['stages'][-1]['high']=.9
        with self.assertRaises(ValueError):validate_policy(bad,'ffpp',allow_exploratory=True)
        with self.assertRaises(ValueError):validate_policy(p,'ffpp')
        p['deployment_ready']=True
        with self.assertRaises(ValueError):validate_policy(p,'ffpp')  # metadata is insufficient

    def test_duplicate_or_changed_cohort_is_rejected(self):
        with self.assertRaises(ValueError):run_cascade(self.records+self.records[:1],policy(),self.adapters)
        original=self.adapters['xception'].infer
        with patch.object(self.adapters['xception'],'infer',side_effect=lambda records:(original(records)[0][:-1],{})):
            with self.assertRaisesRegex(ValueError,'cohort'):run_cascade(self.records,policy(),self.adapters)

    def test_changed_label_and_nonfinite_score_are_rejected(self):
        with patch.object(self.adapters['xception'],'infer',side_effect=lambda records:(
                tuple(VideoScore(r['video_name'],1-r['label'],.5,1) for r in records),{})):
            with self.assertRaisesRegex(ValueError,'label'):run_cascade(self.records,policy(),self.adapters)
        with patch.object(self.adapters['xception'],'infer',side_effect=lambda records:(
                tuple(VideoScore(r['video_name'],r['label'],float('nan'),1) for r in records),{})):
            with self.assertRaises(ValueError):run_cascade(self.records,policy(),self.adapters)

    def test_same_scores_as_phase1_replay(self):
        import sys
        repo_a=Path(__file__).resolve().parents[3]/'far-frr-graph'
        sys.path.insert(0,str(repo_a))
        from cascade_sim.data import Scores
        from cascade_sim.engine import Policy,replay
        import numpy as np
        names=tuple(self.adapters)
        data=Scores('ffpp',tuple(r['video_name'] for r in self.records),names,
                    np.array([r['label'] for r in self.records]),
                    np.array([[self.adapters[d].scores[r['video_name']] for d in names] for r in self.records]),
                    np.ones((4,3)),{})
        expected=replay(data,Policy.from_dict(policy()))
        actual=run_cascade(self.records,policy(),self.adapters)
        self.assertEqual([r['prediction'] for r in actual['decisions']],expected.predictions.tolist())
        self.assertEqual([r['exit_stage'] for r in actual['decisions']],expected.exit_stages.tolist())


class InputTests(unittest.TestCase):
    def test_json_preserves_subtype_and_missing_frame_fails(self):
        from cascade.offline_data import load_records
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);frame=root/'rgb/FaceForensics++/frames/000_003/000.png'
            frame.parent.mkdir(parents=True);frame.write_bytes(b'fixture')
            info=dict(label='FF-DF',frames=['FaceForensics++/frames/000_003/000.png'])
            path=root/'dataset.json';path.write_text(json.dumps({'FaceForensics++':{
                'FF-DF':{'test':{'c23':{'000_003':info}}},
                'FF-FS':{'test':{'c23':{'000_003':{**info,'label':'FF-FS'}}}}}}))
            records=load_records(path,root/'rgb','FaceForensics++')
            self.assertEqual([r['video_name'] for r in records],['FF-DF_000_003','FF-FS_000_003'])
            frame.unlink()
            with self.assertRaises(FileNotFoundError):load_records(path,root/'rgb','FaceForensics++')

    def test_path_escape_and_empty_cohort_rejected(self):
        from cascade.offline_data import load_records
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/'data.json'
            path.write_text(json.dumps({'FaceForensics++':{'FF-real':{'test':{'c23':{
                'a':dict(label='FF-real',frames=['../outside.png'])}}}}}))
            with self.assertRaises(ValueError):load_records(path,root/'rgb','FaceForensics++')
            path.write_text(json.dumps({'FaceForensics++':{}}))
            with self.assertRaises(ValueError):load_records(path,root/'rgb','FaceForensics++')


class CheckpointTests(unittest.TestCase):
    def test_sbi_backbone_matches_official_efficientnet_logits(self):
        import os
        import sys
        import torch
        from efficientnet_pytorch import EfficientNet
        from cascade.model_loader import checkpoint_state,REPO_ROOT
        sys.path.insert(0,str(REPO_ROOT/'training'))
        torch.set_num_threads(4)
        with patch.dict(os.environ,{'DEEPFAKEBENCH_RUNTIME_ONLY':'1'}):
            from networks.efficientnetb4 import EfficientNetB4
            reference=EfficientNet.from_name('efficientnet-b4',num_classes=2).eval()
            adapter=EfficientNetB4(dict(num_classes=2,inc=3,dropout=False,mode='original',
                                      pretrained=None,inference_checkpoint_only=True)).eval()
            state=checkpoint_state({'model':{'net.'+k:v for k,v in reference.state_dict().items()}},'sbi')
            adapter.load_state_dict({k.removeprefix('backbone.'):v for k,v in state.items()},strict=True)
            image=torch.rand(1,3,380,380)
            with torch.inference_mode():
                expected=reference(image)
                actual=adapter.classifier(adapter.features(image))
            torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)

    def test_sbi_mapping_preserves_all_keys_and_rejects_collision(self):
        from cascade.model_loader import checkpoint_state
        original={'model':{'module.net._fc.weight':1,'module.net._conv_stem.weight':2}}
        self.assertEqual(checkpoint_state(original,'sbi'),
                         {'backbone.last_layer.weight':1,'backbone.efficientnet._conv_stem.weight':2})
        self.assertIn('module.net._fc.weight',original['model'])
        with self.assertRaises(ValueError):checkpoint_state({'module.a':1,'a':2},'xception')
        with self.assertRaises(ValueError):checkpoint_state({'model':{}},'sbi')

    def test_complete_checkpoint_loads_without_backbone_file_and_missing_keys_fail(self):
        import os
        from types import SimpleNamespace
        from unittest.mock import patch
        import torch
        from cascade.model_loader import load_model
        captured=[]
        def build(config):
            captured.append(config)
            return torch.nn.Linear(2,1)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'model.pth'
            layer=torch.nn.Linear(2,1)
            torch.save(layer.state_dict(),path)
            with patch.dict(os.environ),patch('cascade.model_loader.importlib.import_module',
                    return_value=SimpleNamespace(DETECTOR={'xception':build})):
                loaded=load_model('xception',{'model_name':'xception','pretrained':'missing.pth'},path,'cpu')
                self.assertTrue(torch.equal(layer.weight,loaded.weight))
                self.assertIsNone(captured[0]['pretrained'])
                self.assertTrue(captured[0]['inference_checkpoint_only'])
                torch.save({'weight':layer.weight},path)
                with self.assertRaises(RuntimeError):
                    load_model('xception',{'model_name':'xception'},path,'cpu')


class AdapterTests(unittest.TestCase):
    def test_default_official_sbi_profile_maps_black_white_to_zero_one(self):
        import numpy as np
        import cv2
        from cascade.adapters import FrameDetectorAdapter
        from cascade.offline import DEFAULT_CONFIGS
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);ckpt=root/'model.pth';ckpt.write_bytes(b'fixture')
            adapter=FrameDetectorAdapter('sbi',DEFAULT_CONFIGS['sbi'],ckpt,'cpu')
            for intensity,expected in [(0,0.),(255,1.)]:
                path=root/f'{intensity}.png'
                cv2.imwrite(str(path),np.full((8,8,3),intensity,dtype=np.uint8))
                tensor=adapter._image(str(path)).numpy()
                self.assertEqual(tensor.shape,(3,380,380))
                np.testing.assert_allclose(tensor,expected,atol=1e-7)

    def test_specific_normalization_uses_same_rgb_pixels(self):
        import numpy as np
        import cv2
        from cascade.adapters import FrameDetectorAdapter
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);image=root/'black.png'
            cv2.imwrite(str(image),np.zeros((8,8,3),dtype=np.uint8))
            for name,mean,std in [('xception',[.5]*3,[.5]*3),
                                  ('sbi',[.485,.456,.406],[.229,.224,.225])]:
                # Small dummy checkpoint: constructor hashes it, but decoding doesn't load a model.
                ckpt=root/(name+'.pth');ckpt.write_bytes(b'fixture')
                config=root/(name+'.json');config.write_text(json.dumps(dict(model_name=name,
                    resolution=8,mean=mean,std=std)))
                adapter=FrameDetectorAdapter(name,config,ckpt,'cpu')
                value=adapter._image(str(image)).numpy()
                np.testing.assert_allclose(value[:,0,0],-np.array(mean)/np.array(std),rtol=1e-6)
                self.assertEqual(value.shape,(3,8,8))
                with self.assertRaises(ValueError):adapter._image(str(root/'missing.png'))


class CommandTests(unittest.TestCase):
    def test_cli_writes_auditable_artifacts_and_refuses_overwrite(self):
        from unittest.mock import patch
        from cascade.offline import main
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);rgb=root/'rgb';frame=rgb/'FaceForensics++/frames/000/000.png'
            frame.parent.mkdir(parents=True);frame.write_bytes(b'fixture')
            source=root/'data.json'
            source.write_text(json.dumps({'FaceForensics++':{
                'FF-real':{'test':{'c23':{'000':dict(label='FF-real',frames=['FaceForensics++/frames/000/000.png'])}}},
                'FF-DF':{'test':{'c23':{'000_003':dict(label='FF-DF',frames=['FaceForensics++/frames/000/000.png'])}}}}}))
            p=policy();p['stages']=[dict(detector='xception',low=.5,high=.5)]
            policy_path=root/'policy.json';policy_path.write_text(json.dumps(p))
            output=root/'out'
            command=['--policy',str(policy_path),'--dataset-json',str(source),'--rgb-root',str(rgb),
                     '--output',str(output),'--device','cpu','--allow-exploratory']
            fake=FakeAdapter('xception',{'FF-real_000':.1,'FF-DF_000_003':.9})
            with patch('cascade.adapters.FrameDetectorAdapter',return_value=fake):
                self.assertEqual(main(command),0)
                report=json.loads((output/'report.json').read_text())
                self.assertEqual(report['metrics']['acc'],1.)
                self.assertEqual(report['evaluation_status'],'functional_smoke_only')
                for file in ('manifest.json','decisions.csv','trace.csv','reached_scores.csv'):
                    self.assertTrue((output/file).is_file())
                with self.assertRaises(SystemExit):main(command)
                unapproved=[x for x in command if x!='--allow-exploratory']
                unapproved[unapproved.index('--output')+1]=str(root/'blocked')
                with self.assertRaises(SystemExit):main(unapproved)
                self.assertFalse((root/'blocked').exists())


if __name__=='__main__':unittest.main()
