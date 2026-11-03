import unittest

from cascade.metrics import final_classification_metrics, operational_auc
from cascade.video_scoring import (
    FAKE,
    REAL,
    UNCERTAIN,
    VideoScore,
    VideoScoreAccumulator,
    merge_cascade_decisions,
    route_final_stage,
    route_video_scores,
)


class VideoScoreAccumulatorTest(unittest.TestCase):
    def test_arithmetic_mean_does_not_depend_on_frame_order(self):
        accumulator = VideoScoreAccumulator()
        accumulator.update(
            probabilities=[0.2, 0.9, 0.4, 0.7],
            labels=[0, 1, 0, 1],
            video_names=['video_a', 'video_b', 'video_a', 'video_b'],
        )

        scores = accumulator.finalize()

        self.assertEqual([score.video_name for score in scores], ['video_a', 'video_b'])
        self.assertAlmostEqual(scores[0].score, 0.3)
        self.assertAlmostEqual(scores[1].score, 0.8)
        self.assertEqual(scores[0].frame_count, 2)

    def test_conflicting_frame_labels_are_rejected(self):
        accumulator = VideoScoreAccumulator()
        accumulator.update([0.1], [0], ['video_a'])

        with self.assertRaises(ValueError):
            accumulator.update([0.2], [1], ['video_a'])


class DoubleThresholdRoutingTest(unittest.TestCase):
    def test_threshold_boundaries_are_inclusive_exits(self):
        scores = (
            VideoScore('low', 0, 0.2, 1),
            VideoScore('middle', 1, 0.5, 1),
            VideoScore('high', 1, 0.8, 1),
        )

        routed = route_video_scores(scores, 0.2, 0.8)

        self.assertEqual(
            [video.decision for video in routed],
            [REAL, UNCERTAIN, FAKE],
        )

    def test_invalid_threshold_order_is_rejected(self):
        with self.assertRaises(ValueError):
            route_video_scores([], 0.8, 0.2)


class OperationalAucTest(unittest.TestCase):
    def test_auc_uses_only_supplied_stage_cohort(self):
        scores = (
            VideoScore('real', 0, 0.1, 2),
            VideoScore('fake', 1, 0.9, 2),
        )

        self.assertEqual(operational_auc(scores), 1.0)

    def test_auc_is_na_for_single_class(self):
        scores = (VideoScore('real', 0, 0.1, 2),)

        self.assertIsNone(operational_auc(scores))


class FinalStageTest(unittest.TestCase):
    def test_explicit_rates_use_asymmetric_class_denominators(self):
        labels = [0, 0, 0, 1, 1, 1, 1]
        scores = [0.9, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9]
        videos = route_final_stage(tuple(VideoScore(str(i), y, s, 1)
            for i, (y, s) in enumerate(zip(labels, scores))), 0.5)
        metrics = final_classification_metrics(videos)
        self.assertEqual(metrics['fnr'], 2 / 4)
        self.assertEqual(metrics['fpr'], 1 / 3)
        self.assertEqual(metrics['far'], metrics['fnr'])
        self.assertEqual(metrics['frr'], metrics['fpr'])

    def test_absent_class_has_null_explicit_rate(self):
        videos = route_final_stage((VideoScore('real', 0, 0.1, 1),), 0.5)
        metrics = final_classification_metrics(videos)
        self.assertIsNone(metrics['fnr'])
        self.assertEqual(metrics['fpr'], 0.)

    def test_final_threshold_is_fake_inclusive(self):
        scores = (
            VideoScore('below', 0, 0.49, 2),
            VideoScore('equal', 1, 0.5, 2),
        )

        routed = route_final_stage(scores, 0.5)

        self.assertEqual([video.decision for video in routed], [REAL, FAKE])

    def test_merge_replaces_exact_uncertain_cohort(self):
        stage1 = route_video_scores(
            (
                VideoScore('early_real', 0, 0.1, 2),
                VideoScore('continue', 1, 0.5, 2),
                VideoScore('early_fake', 1, 0.9, 2),
            ),
            0.2,
            0.8,
        )
        stage2 = route_final_stage(
            (VideoScore('continue', 1, 0.7, 2),),
            0.5,
        )

        final_videos = merge_cascade_decisions(stage1, stage2)

        self.assertEqual(
            [video.decision for video in final_videos],
            [REAL, FAKE, FAKE],
        )

    def test_merge_rejects_incomplete_stage2_cohort(self):
        stage1 = route_video_scores(
            (VideoScore('continue', 1, 0.5, 2),),
            0.2,
            0.8,
        )

        with self.assertRaises(ValueError):
            merge_cascade_decisions(stage1, ())

    def test_final_acc_far_frr_follow_fake_positive_definition(self):
        final_videos = (
            route_final_stage((VideoScore('real_ok', 0, 0.1, 2),), 0.5)[0],
            route_final_stage((VideoScore('real_rejected', 0, 0.9, 2),), 0.5)[0],
            route_final_stage((VideoScore('fake_accepted', 1, 0.1, 2),), 0.5)[0],
            route_final_stage((VideoScore('fake_ok', 1, 0.9, 2),), 0.5)[0],
        )

        result = final_classification_metrics(final_videos)

        self.assertEqual(result['acc'], 0.5)
        self.assertEqual(result['far'], 0.5)
        self.assertEqual(result['frr'], 0.5)


if __name__ == '__main__':
    unittest.main()
