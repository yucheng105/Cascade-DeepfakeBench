"""Video-level score aggregation and double-threshold routing."""

import math
from collections import OrderedDict
from dataclasses import dataclass


REAL = 'REAL'
UNCERTAIN = 'UNCERTAIN'
FAKE = 'FAKE'

# 一支影片經過某個 detector 完整 inference 和 frame aggregation 後的結果。
@dataclass(frozen=True)
class VideoScore:
    """Arithmetic mean of all sampled frame scores for one video."""

    video_name: str
    label: int | None
    score: float
    frame_count: int

# 和 VideoScore 一樣但是多了 Decision
@dataclass(frozen=True)
class RoutedVideo:
    """A video score and its decision at one cascade stage."""

    video_name: str
    label: int | None
    score: float
    frame_count: int
    decision: str


class VideoScoreAccumulator:
    """Accumulate frame probabilities without depending on batch ordering."""

    def __init__(self):
        self._videos = OrderedDict()

    def update(self, probabilities, labels, video_names):
        """Add one batch of frame scores, binary labels, and video identities."""
        probabilities = list(probabilities)
        labels = list(labels)
        video_names = list(video_names)

        if not (
            len(probabilities) == len(labels) == len(video_names)
        ):
            raise ValueError(
                'probabilities, labels, and video_names must have equal lengths.'
            )

        for probability, label, video_name in zip(
            probabilities, labels, video_names
        ):
            probability = float(probability)
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(
                    'Every frame probability must be finite and within [0, 1].'
                )

            if label is not None:
                numeric_label = float(label)
                if numeric_label not in (0.0, 1.0):
                    raise ValueError('Every known video label must be binary: REAL=0, FAKE=1.')
                label = int(numeric_label)
            if not isinstance(video_name, str) or not video_name:
                raise ValueError('Every video_name must be a non-empty string.')

            # create for new video if have not included in _videos
            if video_name not in self._videos:
                self._videos[video_name] = {
                    'label': label,
                    'score_sum': 0.0,
                    'frame_count': 0,
                }

            state = self._videos[video_name]
            if state['label'] != label:
                raise ValueError(
                    'Video {!r} is associated with multiple labels.'.format(
                        video_name
                    )
                )

            state['score_sum'] += probability
            state['frame_count'] += 1

    def finalize(self):
        """Return immutable video scores in first-seen video order."""
        return tuple(
            VideoScore(
                video_name=video_name,
                label=state['label'],
                score=state['score_sum'] / state['frame_count'],
                frame_count=state['frame_count'],
            )
            for video_name, state in self._videos.items()
        )

# check threshold legality
def validate_double_threshold(low_threshold, high_threshold):
    """Validate one intermediate stage's real/fake exit thresholds."""
    low_threshold = float(low_threshold)
    high_threshold = float(high_threshold)
    if not 0.0 <= low_threshold < high_threshold <= 1.0:
        raise ValueError(
            'Thresholds must satisfy 0 <= low_threshold < high_threshold <= 1.'
        )
    return low_threshold, high_threshold


def route_video_scores(video_scores, low_threshold, high_threshold):
    """Route videos using REAL <= low, FAKE >= high, otherwise UNCERTAIN."""

    # check threshold legality
    low_threshold, high_threshold = validate_double_threshold(
        low_threshold, high_threshold
    )

    routed = []
    for video_score in video_scores:
        if video_score.score <= low_threshold:
            decision = REAL
        elif video_score.score >= high_threshold:
            decision = FAKE
        else:
            decision = UNCERTAIN

        routed.append(
            RoutedVideo(
                video_name=video_score.video_name,
                label=video_score.label,
                score=video_score.score,
                frame_count=video_score.frame_count,
                decision=decision,
            )
        )
    return tuple(routed)


def validate_final_threshold(threshold):
    """Validate the final stage's single real/fake threshold."""
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError('Final threshold must be within [0, 1].')
    return threshold


def route_final_stage(video_scores, threshold):
    """Resolve every final-stage video using REAL < threshold, else FAKE."""
    threshold = validate_final_threshold(threshold)
    return tuple(
        RoutedVideo(
            video_name=video_score.video_name,
            label=video_score.label,
            score=video_score.score,
            frame_count=video_score.frame_count,
            decision=REAL if video_score.score < threshold else FAKE,
        )
        for video_score in video_scores
    )


def merge_cascade_decisions(stage1_routed, stage2_routed):
    """Replace Stage 1 uncertain decisions with final Stage 2 decisions."""
    stage1_routed = tuple(stage1_routed)
    stage2_routed = tuple(stage2_routed)
    uncertain_names = {
        video.video_name
        for video in stage1_routed
        if video.decision == UNCERTAIN
    }
    stage2_by_name = {}
    for video in stage2_routed:
        if video.decision == UNCERTAIN:
            raise ValueError('The final stage cannot return UNCERTAIN.')
        if video.video_name in stage2_by_name:
            raise ValueError(
                'Duplicate Stage 2 video_name: {!r}.'.format(video.video_name)
            )
        stage2_by_name[video.video_name] = video

    stage2_names = set(stage2_by_name)
    if stage2_names != uncertain_names:
        missing = sorted(uncertain_names.difference(stage2_names))
        unexpected = sorted(stage2_names.difference(uncertain_names))
        raise ValueError(
            'Stage 2 cohort must exactly match Stage 1 uncertain videos. '
            'Missing: {}; unexpected: {}.'.format(missing, unexpected)
        )

    final_videos = []
    for stage1_video in stage1_routed:
        if stage1_video.decision == UNCERTAIN:
            stage2_video = stage2_by_name[stage1_video.video_name]
            if stage1_video.label != stage2_video.label:
                raise ValueError(
                    'Ground-truth label changed between stages for {!r}.'.format(
                        stage1_video.video_name
                    )
                )
            final_videos.append(stage2_video)
        else:
            final_videos.append(stage1_video)
    return tuple(final_videos)
