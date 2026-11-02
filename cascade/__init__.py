"""Sequential cascade evaluation components."""

from .video_scoring import (
    REAL,
    UNCERTAIN,
    FAKE,
    RoutedVideo,
    VideoScore,
    VideoScoreAccumulator,
    merge_cascade_decisions,
    route_final_stage,
    route_video_scores,
)

__all__ = [
    'REAL',
    'UNCERTAIN',
    'FAKE',
    'RoutedVideo',
    'VideoIdentityDataset',
    'VideoScore',
    'VideoScoreAccumulator',
    'merge_cascade_decisions',
    'route_final_stage',
    'route_video_scores',
]


def __getattr__(name):
    # Pure policy/routing consumers should not require the torch Dataset runtime.
    if name == 'VideoIdentityDataset':
        from .data import VideoIdentityDataset
        return VideoIdentityDataset
    raise AttributeError(name)
