"""Public single-video behavior and unlabeled-inference contract."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from cascade.controller import run_cascade
from cascade.video_scoring import VideoScoreAccumulator


def make_video(path, frames=5):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 5, (64, 64))
    if not writer.isOpened():
        raise RuntimeError('MP4 encoder unavailable')
    for index in range(frames):
        writer.write(np.full((64, 64, 3), index * 20, dtype=np.uint8))
    writer.release()


class FixtureFaces:
    def crop(self, image):
        return cv2.resize(image, (256, 256)), dict(face_count=1)


class VideoInputTests(unittest.TestCase):
    def test_twenty_minute_boundary_and_uniform_sampling(self):
        from cascade.video_input import prepare_video, VideoLimits
        class Capture:
            def __init__(self, count): self.count, self.index = count, 0
            def isOpened(self): return True
            def get(self, field):
                return {cv2.CAP_PROP_FPS: 1, cv2.CAP_PROP_FRAME_COUNT: self.count,
                        cv2.CAP_PROP_FRAME_WIDTH: 64, cv2.CAP_PROP_FRAME_HEIGHT: 64}[field]
            def read(self):
                self.index += 1
                return (True, np.zeros((64, 64, 3), dtype=np.uint8)) if self.index <= self.count else (False, None)
            def release(self): pass
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root/'source.mp4'; source.write_bytes(b'fixture')
            with patch('cascade.video_input.cv2.VideoCapture', return_value=Capture(1200)):
                frames, info = prepare_video(source, root, VideoLimits(), FixtureFaces())
            self.assertEqual(len(frames), 32)
            self.assertEqual(info['decoded_frame_count'], 1200)
            self.assertEqual(info['duration_seconds'], 1200)
            self.assertEqual([info['samples'][i]['frame_index'] for i in (0, -1)], [0, 1199])
            with patch('cascade.video_input.cv2.VideoCapture', return_value=Capture(1201)):
                with self.assertRaisesRegex(ValueError, '1200'):
                    prepare_video(source, root, VideoLimits(), FixtureFaces())

    def test_truncated_decode_is_not_silently_classified(self):
        from cascade.video_input import prepare_video, VideoLimits
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root/'truncated.mp4'; source.write_bytes(b'fixture')
            class TruncatedCapture:
                def isOpened(self): return True
                def get(self, field):
                    return {cv2.CAP_PROP_FPS:5, cv2.CAP_PROP_FRAME_COUNT:5,
                            cv2.CAP_PROP_FRAME_WIDTH:64, cv2.CAP_PROP_FRAME_HEIGHT:64}[field]
                def read(self): return False, None
                def release(self): pass
            with patch('cascade.video_input.cv2.VideoCapture', return_value=TruncatedCapture()):
                with self.assertRaisesRegex(ValueError, '提前結束'):
                    prepare_video(source, root, VideoLimits(), FixtureFaces())

    def test_real_decode_unique_sampling_and_no_face_failure(self):
        from cascade.video_input import prepare_video, VideoLimits
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/'input.mp4'
            make_video(source)
            output = root/'success'; output.mkdir()
            frames, info = prepare_video(source, output, VideoLimits(max_frames=3), FixtureFaces())
            self.assertEqual([s['frame_index'] for s in info['samples']], [0, 2, 4])
            self.assertEqual(info['decoded_frame_count'], 5)
            self.assertEqual(len(frames), 3)
            self.assertTrue(all(Path(p).is_file() for p in frames))
            class NoFaces:
                def crop(self, image):
                    return None, dict(reason='no_face', face_count=0)
            failure = root/'no_faces'; failure.mkdir()
            with self.assertRaisesRegex(ValueError, '沒有產生真假判定'):
                prepare_video(source, failure, VideoLimits(), NoFaces())

    def test_limits_and_invalid_video_fail(self):
        from cascade.video_input import prepare_video, VideoLimits
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/'input.mp4'; make_video(source)
            with self.assertRaises(ValueError):
                prepare_video(source, root, VideoLimits(max_bytes=1), FixtureFaces())
            with self.assertRaises(ValueError):
                prepare_video(source, root, VideoLimits(max_pixels=1), FixtureFaces())
            source.write_bytes(b'not a movie')
            with self.assertRaises(ValueError):
                prepare_video(source, root, VideoLimits(), FixtureFaces())

    def test_alignment_is_repeatable_and_rejects_degenerate_landmarks(self):
        from cascade.video_input import align_face
        image = np.zeros((160, 160, 3), dtype=np.uint8)
        points = [[38, 52], [74, 52], [56, 72], [42, 93], [71, 93]]
        first, matrix = align_face(image, points)
        second, again = align_face(image, points)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(matrix, again)
        self.assertEqual(first.shape, (256, 256, 3))
        with self.assertRaises(ValueError):
            align_face(image, [[0, 0]]*5)


class VideoPipelineTests(unittest.TestCase):
    def test_no_face_keeps_error_artifacts_without_a_prediction(self):
        from cascade.video import run_video
        from cascade.preprocessing import execute_request
        def process(request,timeout):
            return execute_request(request,face_factory=lambda profile:lambda image:[])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root/'input.mp4'; make_video(source)
            output = root/'failed'
            with patch('cascade.preprocessing.run_process', side_effect=process), \
                    patch('cascade.adapters.FrameDetectorAdapter',return_value=SimpleNamespace(provenance={})):
                with self.assertRaisesRegex(ValueError, 'no valid faces'):
                    run_video(source, output, device='cpu', allow_exploratory=True,preprocessing_mode='legacy')
            self.assertTrue((output/'error.json').is_file())
            self.assertTrue((output/'preprocessing.json').is_file())
            self.assertFalse((output/'report.json').exists())
            self.assertEqual(json.loads((output/'status.json').read_text(encoding='utf-8'))['phase'], 'failed')

    def test_mp4_to_unknown_label_artifacts_and_no_overwrite(self):
        from cascade.video import run_video, POLICIES
        from cascade.video_scoring import VideoScore
        from cascade.preprocessing import execute_request
        def process(request,timeout):
            def faces(image):
                crop,info=FixtureFaces().crop(image)
                return [(crop,info)]
            return execute_request(request,face_factory=lambda profile:faces)
        class Adapter:
            def __init__(self, name, *args, **kwargs):
                self.name = name
                self.provenance = dict(detector=name)
            def infer(self, records):
                return tuple(VideoScore(r['video_name'], r['label'], .6, len(r['frames']))
                             for r in records), dict(frame_count=sum(len(r['frames']) for r in records))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root/'input.mp4'; make_video(source)
            output = root/'run'
            with patch('cascade.preprocessing.run_process', side_effect=process), \
                    patch('cascade.adapters.FrameDetectorAdapter', Adapter):
                result = run_video(source, output, policy_path=POLICIES['all_stages'],
                                   device='cpu', allow_exploratory=True,preprocessing_mode='legacy')
                self.assertEqual(len(result['trace']), 3)
                self.assertIsNone(result['metrics']['acc'])
                self.assertIsNone(result['decisions'][0]['label'])
                self.assertEqual(result['decisions'][0]['exit_stage'], 3)
                self.assertEqual(json.loads((output/'status.json').read_text(encoding='utf-8'))['phase'], 'complete')
                for name in ['manifest.json', 'preprocessing.json', 'decisions.csv', 'trace.csv']:
                    self.assertTrue((output/name).is_file())
                with self.assertRaisesRegex(ValueError, '已存在'):
                    run_video(source, output, device='cpu', allow_exploratory=True)
                with self.assertRaisesRegex(ValueError, 'allow-exploratory'):
                    run_video(source, root/'blocked', device='cpu')
                self.assertFalse((root/'blocked').exists())


class UnknownLabelTests(unittest.TestCase):
    def test_ucf_runtime_does_not_accumulate_placeholder_labels_or_metrics(self):
        import os
        import sys
        import torch
        from cascade.model_loader import REPO_ROOT
        sys.path.insert(0, str(REPO_ROOT/'training'))
        with patch.dict(os.environ, {'DEEPFAKEBENCH_RUNTIME_ONLY':'1'}):
            from detectors.ucf_detector import UCFDetector
        model = UCFDetector.__new__(UCFDetector)
        torch.nn.Module.__init__(model)
        model.features = lambda data: {'forgery': torch.ones(1, 2), 'content': torch.ones(1, 2)}
        model.classifier = lambda features: (features, features)
        model.head_sha = lambda features: (features, features)
        model.head_spe = lambda features: (features, features)
        model.prob, model.label, model.correct, model.total = [], [], 0, 0
        output = model({'label':torch.tensor([0]), 'track_metrics':False}, inference=True)
        self.assertEqual(model.label, [])
        self.assertEqual(model.prob, [])
        self.assertEqual(model.total, 0)
        legacy = model({'label':torch.tensor([0])}, inference=True)
        self.assertEqual(model.total, 1)
        torch.testing.assert_close(output['prob'], legacy['prob'])

    def test_unknown_labels_survive_aggregation_and_have_no_quality_metrics(self):
        accumulator = VideoScoreAccumulator()
        accumulator.update([.8, .6], [None, None], ['uploaded', 'uploaded'])
        score = accumulator.finalize()[0]
        self.assertIsNone(score.label)
        class Adapter:
            def infer(self, records):
                return (score,), {}
        policy = dict(schema_version=1, kind='sequential', dataset_id='uploaded_video',
                      policy_id='demo', deployment_ready=False, weights=[],
                      metric_definition='fake_positive_far_is_fnr',
                      comparators='intermediate_real_le_fake_ge_final_fake_ge',
                      stages=[dict(detector='xception', low=.5, high=.5)])
        result = run_cascade([dict(video_name='uploaded', label=None)], policy, {'xception': Adapter()})
        self.assertEqual(result['decisions'][0]['decision'], 'FAKE')
        self.assertIsNone(result['decisions'][0]['label'])
        for key in ('acc', 'fnr', 'fpr', 'far', 'frr', 'tp', 'fn', 'fp', 'tn'):
            self.assertIsNone(result['metrics'][key])


if __name__ == '__main__':
    unittest.main()
