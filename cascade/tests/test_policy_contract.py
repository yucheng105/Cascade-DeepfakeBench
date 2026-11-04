"""Policy loading and public offline execution compatibility, without real weights."""
import copy
import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from cascade.policy import validate_policy
from cascade.video_scoring import VideoScore


def exploratory():
    return dict(schema_version=1, kind='sequential', dataset_id='ffpp', policy_id='contract-test',
                deployment_ready=False, weights=[], metric_definition='fake_positive_far_is_fnr',
                comparators='intermediate_real_le_fake_ge_final_fake_ge',
                stages=[dict(detector='xception', low=.5, high=.5)])


def contract():
    return dict(unit='video', sampling='first_N_numeric_sorted_frames', max_frames=32,
                profiles={'xception':'preprocessed_face_crops'},
                aggregation={'xception':'mean_frame_probability'})


def provenance():
    return dict(checkpoint_sha256='a'*64, config_sha256='b'*64, preprocessing_sha256='c'*64,
                preprocessing=dict(input='preprocessed_face_crops',
                    sampling='first_N_numeric_sorted_frames', aggregation='mean_frame_probability'))


def bound_policy():
    p = exploratory()
    p.update(deployment_ready=True, target_far=.05, latency_model='sum_reached_csv_ms_serial',
             input_contract=contract())
    for key in ('checkpoint_sha256', 'config_sha256', 'preprocessing_sha256'):
        p[key] = {'xception':provenance()[key]}
    return p


class PolicyMetadataTests(unittest.TestCase):
    def test_existing_exploratory_policy_is_accepted_and_copied(self):
        p = exploratory()
        loaded = validate_policy(p, 'ffpp', allow_exploratory=True)
        loaded['stages'][0]['low'] = .1
        self.assertEqual(p['stages'][0]['low'], .5)

    def test_supplied_metadata_is_validated_even_in_exploratory_mode(self):
        invalid = [
            ('target_far', -1), ('target_far', 1.1), ('target_far', True),
            ('target_far', float('nan')), ('target_far', float('inf')), ('target_far', '0.1'),
            ('deployment_ready', 'false'), ('latency_model', 'invalid'), ('latency_model', []),
            ('objective', 'unknown'), ('preprocessing_sha256', 'invalid'),
            ('preprocessing_sha256', {'xception':'unknown'}), ('config_sha256', {}),
            ('checkpoint_sha256', {'ucf':'a'*64}), ('input_contract', None),
        ]
        for key, value in invalid:
            with self.subTest(key=key, value=value):
                p = exploratory(); p[key] = value
                with self.assertRaisesRegex(ValueError, key):
                    validate_policy(p, 'ffpp', allow_exploratory=True)

    def test_input_contract_rejects_invalid_or_ambiguous_semantics(self):
        invalid = [('unit','clip'), ('sampling','unknown'), ('max_frames',True),
                   ('max_frames',0), ('profiles',{'xception':'official'}),
                   ('profiles',{'ucf':'dlib81'}), ('aggregation',{'xception':'mean_logits'}),
                   ('sample_hz',2), ('typo_field',1)]
        for key, value in invalid:
            with self.subTest(key=key, value=value):
                p = exploratory(); p['input_contract'] = contract(); p['input_contract'][key] = value
                with self.assertRaisesRegex(ValueError, 'input_contract'):
                    validate_policy(p, 'ffpp', allow_exploratory=True)

    def test_window_cadence_is_positive_and_separate_from_video_contract(self):
        p = exploratory()
        p['input_contract'] = dict(unit='window', sampling='paced_media_time_no_future_frames',
            profiles={'xception':'dlib81'}, aggregation={'xception':'max_face_then_mean_frame'},
            window_seconds=4., stride_seconds=4., sample_hz=2.)
        validate_policy(p, 'ffpp', allow_exploratory=True)
        for key in ('window_seconds', 'stride_seconds', 'sample_hz'):
            for value in (0, -1, True, float('nan')):
                bad = copy.deepcopy(p); bad['input_contract'][key] = value
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, 'input_contract'):
                    validate_policy(bad, 'ffpp', allow_exploratory=True)

    def test_formal_requirements_cannot_be_bypassed_by_exploratory_flag(self):
        p = bound_policy()
        validate_policy(p, 'ffpp')
        for key in ('target_far','latency_model','input_contract','checkpoint_sha256',
                    'config_sha256','preprocessing_sha256'):
            for allow in (False, True):
                bad = copy.deepcopy(p); del bad[key]
                with self.subTest(key=key, allow=allow), self.assertRaisesRegex(ValueError, key):
                    validate_policy(bad, 'ffpp', allow_exploratory=allow)


class BoundOfflineEntryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rgb = self.root/'rgb'
        frame = self.rgb/'FaceForensics++/frames/000/000.png'
        frame.parent.mkdir(parents=True); frame.write_bytes(b'fixture')
        self.source = self.root/'data.json'
        self.source.write_text(json.dumps({'FaceForensics++':{
            'FF-real':{'test':{'c23':{'000':dict(label='FF-real',frames=['FaceForensics++/frames/000/000.png'])}}},
            'FF-DF':{'test':{'c23':{'001':dict(label='FF-DF',frames=['FaceForensics++/frames/000/000.png'])}}}}}))
        self.p = self.root/'policy.json'
        self.calls = []
        owner = self
        class Adapter:
            def __init__(self, *args, **kwargs):
                self.provenance = provenance()
            def infer(self, records):
                owner.calls.append(records)
                return tuple(VideoScore(r['video_name'],r['label'],.1 if r['label']==0 else .9,1)
                             for r in records), {'per_video_forward_ms':{r['video_name']:1. for r in records}}
        self.adapter = Adapter

    def execute(self, p, name='out', *, allow=False):
        from cascade.offline import main
        self.p.write_text(json.dumps(p))
        command = ['--policy',str(self.p),'--dataset-json',str(self.source),'--rgb-root',str(self.rgb),
                   '--output',str(self.root/name),'--device','cpu']
        if allow: command.append('--allow-exploratory')
        with patch('cascade.adapters.FrameDetectorAdapter',self.adapter), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return main(command)

    def test_matching_formal_contract_runs_and_records_legacy_crop_scope(self):
        self.assertEqual(self.execute(bound_policy()), 0)
        report = json.loads((self.root/'out/report.json').read_text())
        manifest = json.loads((self.root/'out/manifest.json').read_text())
        self.assertEqual(report['metrics']['acc'], 1.)
        self.assertEqual(manifest['input_contract'], contract())
        self.assertEqual(manifest['input_profile_status'], 'legacy_preprocessed_face_crops')
        self.assertEqual(len(self.calls), 1)

    def test_runtime_mismatches_reject_before_inference_or_output(self):
        mutations = [
            lambda p:p.update(input_contract=dict(unit='window',sampling='paced_media_time_no_future_frames',
                profiles={'xception':'dlib81'},aggregation={'xception':'max_face_then_mean_frame'},
                sample_hz=2.,window_seconds=4.,stride_seconds=4.)),
            lambda p:p['input_contract'].update(max_frames=16),
            lambda p:p['input_contract'].update(sampling='uniform_unique_frame_indices_v1'),
            lambda p:p['input_contract']['profiles'].update(xception='dlib81'),
            lambda p:p['input_contract']['aggregation'].update(xception='max_face_then_mean_frame'),
            lambda p:p['preprocessing_sha256'].update(xception='d'*64),
            lambda p:p['config_sha256'].update(xception='d'*64),
            lambda p:p['checkpoint_sha256'].update(xception='d'*64),
        ]
        for i, mutate in enumerate(mutations):
            p = bound_policy(); mutate(p)
            with self.subTest(i=i), self.assertRaises(SystemExit):
                self.execute(p,str(i))
            self.assertFalse((self.root/str(i)).exists())
        self.assertEqual(self.calls, [])

    def test_exploratory_flag_does_not_ignore_declared_bindings(self):
        p = bound_policy(); p['deployment_ready'] = False
        p['checkpoint_sha256']['xception'] = 'd'*64
        with self.assertRaises(SystemExit): self.execute(p,allow=True)
        self.assertEqual(self.calls, [])
        self.assertFalse((self.root/'out').exists())

    def test_malformed_metadata_rejects_before_adapter_creation(self):
        from cascade.offline import main
        p = exploratory(); p['target_far'] = -1
        self.p.write_text(json.dumps(p))
        with patch('cascade.adapters.FrameDetectorAdapter') as adapter, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main(['--policy',str(self.p),'--dataset-json',str(self.source),'--rgb-root',str(self.rgb),
                      '--output',str(self.root/'out'),'--device','cpu','--allow-exploratory'])
        adapter.assert_not_called()
        self.assertFalse((self.root/'out').exists())


class UploadBindingTests(unittest.TestCase):
    def test_upload_reloads_matching_manifest_bindings(self):
        from cascade.video import run_video
        p = exploratory(); p['dataset_id'] = 'uploaded_video'
        class Adapter:
            def __init__(self, *args, **kwargs):
                self.provenance = provenance()
            def infer(self, records):
                return tuple(VideoScore(r['video_name'],None,.5,1) for r in records), {}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root/'input.mp4'; source.write_bytes(b'fixture')
            path = root/'policy.json'; path.write_text(json.dumps(p))
            cache = SimpleNamespace(mapping={'xception':'dlib81'},
                profiles={'dlib81':{'definition':{'id':'dlib81'}}}, summary=lambda:dict(profiles={}),
                get=lambda name:dict(face_groups=[dict(frame_index=0,faces=['fixture.png'])]))
            with patch('cascade.video.ProfileCache',return_value=cache), \
                    patch('cascade.adapters.FrameDetectorAdapter',Adapter):
                first = run_video(source,root/'first',policy_path=path,device='cpu',allow_exploratory=True)
                manifest = json.loads((root/'first/manifest.json').read_text(encoding='utf-8'))
                p['input_contract'] = manifest['input_contract']
                for field in ('checkpoint_sha256','config_sha256','preprocessing_sha256'):
                    p[field] = {'xception':manifest['models']['xception'][field]}
                path.write_text(json.dumps(p))
                second = run_video(source,root/'second',policy_path=path,device='cpu',allow_exploratory=True)
            self.assertEqual(first['decisions'],second['decisions'])
            self.assertEqual(second['decisions'][0]['decision'],'FAKE')
            self.assertTrue(all(second['metrics'][k] is None for k in ('acc','far','frr','fnr','fpr')))

    def test_detector_failure_preserves_partial_trace_without_binary_result(self):
        from cascade.video import run_video
        p = exploratory(); p['dataset_id'] = 'uploaded_video'
        p['stages'] = [dict(detector='xception',low=.2,high=.8), dict(detector='ucf',low=.5,high=.5)]
        class Adapter:
            def __init__(self, name, *args, **kwargs):
                self.name = name
                self.provenance = provenance()
            def infer(self, records):
                if self.name == 'ucf':
                    raise ValueError('controlled detector failure')
                return tuple(VideoScore(r['video_name'],None,.6,1) for r in records), {}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root/'input.mp4'; source.write_bytes(b'fixture')
            path = root/'policy.json'; path.write_text(json.dumps(p))
            cache = SimpleNamespace(mapping={'xception':'dlib81','ucf':'dlib81'},
                profiles={'dlib81':{'definition':{'id':'dlib81'}}}, summary=lambda:{},
                get=lambda name:dict(face_groups=[dict(frame_index=0,faces=['fixture.png'])]))
            output = root/'out'
            with patch('cascade.video.ProfileCache',return_value=cache), \
                    patch('cascade.adapters.FrameDetectorAdapter',Adapter):
                with self.assertRaisesRegex(ValueError, 'controlled detector failure'):
                    run_video(source,output,policy_path=path,device='cpu',allow_exploratory=True)
            trace = json.loads((output/'partial_trace.json').read_text())
            self.assertEqual([(s['detector'],s['decision']) for s in trace], [('xception','UNCERTAIN')])
            self.assertEqual(json.loads((output/'status.json').read_text(encoding='utf-8'))['phase'], 'failed')
            self.assertEqual(json.loads((output/'error.json').read_text())['error_type'], 'ValueError')
            self.assertFalse((output/'report.json').exists())
            self.assertFalse((output/'decisions.csv').exists())

    def test_upload_rejects_declared_profile_before_preprocessing(self):
        from cascade.video import run_video
        p = exploratory(); p['dataset_id'] = 'uploaded_video'
        p['input_contract'] = contract()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root/'input.mp4'; source.write_bytes(b'fixture')
            path = root/'policy.json'; path.write_text(json.dumps(p))
            with patch('cascade.video.ProfileCache') as cache:
                with self.assertRaisesRegex(ValueError, 'input_contract'):
                    run_video(source,root/'out',policy_path=path,device='cpu',allow_exploratory=True)
            cache.assert_not_called()
            self.assertFalse((root/'out/report.json').exists())

    def test_upload_declared_asset_binding_rejects_before_detector_inference(self):
        from cascade.video import run_video
        p = exploratory(); p['dataset_id'] = 'uploaded_video'
        p['checkpoint_sha256'] = {'xception':'d'*64}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root/'input.mp4'; source.write_bytes(b'fixture')
            path = root/'policy.json'; path.write_text(json.dumps(p))
            cache = SimpleNamespace(mapping={'xception':'dlib81'},profiles={'dlib81':{'definition':{'id':'dlib81'}}},summary=lambda:{})
            adapter = SimpleNamespace(provenance=provenance())
            with patch('cascade.video.ProfileCache',return_value=cache), \
                    patch('cascade.adapters.FrameDetectorAdapter',return_value=adapter), \
                    patch('cascade.video.run_cascade') as controller:
                with self.assertRaisesRegex(ValueError, 'checkpoint_sha256'):
                    run_video(source,root/'out',policy_path=path,device='cpu',allow_exploratory=True)
            controller.assert_not_called()
            self.assertTrue((root/'out/error.json').is_file())
            self.assertFalse((root/'out/report.json').exists())


if __name__ == '__main__':
    unittest.main()
