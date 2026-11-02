"""Bounded MP4 decode, deterministic sampling and YuNet five-point alignment."""
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import time

import cv2
import numpy as np

from .download_face_model import MODEL_PATH, MODEL_SHA256
from .model_loader import sha256

SAMPLING = 'uniform_unique_frame_indices_v1'
FACE_PROFILE = 'yunet2023mar_largest_face_similarity256_margin1.3_v1'


@dataclass(frozen=True)
class VideoLimits:
    max_bytes: int = 200 * 1024 * 1024
    max_seconds: int = 120
    max_pixels: int = 1920 * 1080
    max_frames: int = 32
    max_fps: int = 240
    decode_timeout_seconds: int = 180


def align_face(image, landmarks):
    """Least-squares similarity fit of five landmarks to a padded 256 template."""
    source = np.asarray(landmarks, dtype=np.float64).reshape(5, 2)
    target = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                       [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float64) * (256 / 112)
    target = (target + 256 * .3 / 2) / 1.3
    if not np.isfinite(source).all() or np.linalg.norm(source[0]-source[1]) < 1:
        raise ValueError('Invalid face landmarks')
    src_center, dst_center = source.mean(axis=0), target.mean(axis=0)
    src, dst = source-src_center, target-dst_center
    variance = (src * src).sum() / 5
    u, singular, vt = np.linalg.svd(dst.T @ src / 5)
    signs = np.ones(2)
    signs[-1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag(signs) @ vt
    scale = (singular * signs).sum() / variance
    matrix = np.column_stack((scale * rotation, dst_center-scale * rotation @ src_center))
    if not np.isfinite(matrix).all() or scale <= 0:
        raise ValueError('Degenerate face alignment')
    return cv2.warpAffine(image, matrix, (256, 256)), matrix.tolist()


class YuNetFaces:
    def __init__(self, model_path: Path = MODEL_PATH):
        if not model_path.is_file():
            raise FileNotFoundError('YuNet missing; run python -m cascade.download_face_model')
        if sha256(model_path) != MODEL_SHA256:
            raise ValueError('YuNet model hash mismatch')
        self.detector = cv2.FaceDetectorYN.create(str(model_path), '', (320, 320), .8, .3, 5000)

    def crop(self, image):
        height, width = image.shape[:2]
        ratio = min(1., 640 / max(height, width))
        size = (max(1, round(width * ratio)), max(1, round(height * ratio)))
        resized = cv2.resize(image, size)
        self.detector.setInputSize(size)
        _, faces = self.detector.detect(resized)
        if faces is None or not len(faces):
            return None, dict(reason='no_face', face_count=0)
        valid = [f for f in faces if np.isfinite(f).all() and f[2] > 0 and f[3] > 0]
        if not valid:
            return None, dict(reason='invalid_detection', face_count=0)
        face = max(valid, key=lambda f: float(f[2] * f[3])).copy()
        # Resize rounding may produce slightly different x/y ratios.
        points = face[4:14].reshape(5, 2).astype(np.float64)
        points *= np.array([width / size[0], height / size[1]])
        try:
            crop, matrix = align_face(image, points)
        except (ValueError, np.linalg.LinAlgError):
            return None, dict(reason='alignment_failed', face_count=len(valid))
        bbox = (face[:4] * np.array([width/size[0], height/size[1]]*2)).tolist()
        return crop, dict(face_count=len(valid), bbox=bbox, landmarks=points.tolist(),
                          detection_score=float(face[14]), alignment_matrix=matrix)


def prepare_video(path: Path, output: Path, limits: VideoLimits, faces, progress=None):
    if path.suffix.lower() != '.mp4' or not path.is_file():
        raise ValueError('請選擇可讀取的 MP4 檔案')
    if not 0 < path.stat().st_size <= limits.max_bytes:
        raise ValueError('影片為空或超過 200 MiB 上限')
    if not 1 <= limits.max_frames <= 32:
        raise ValueError('取樣影格數需介於 1–32')
    cap = cv2.VideoCapture(str(path))
    began = time.perf_counter()
    samples = []
    decoded = 0
    try:
        if not cap.isOpened():
            raise ValueError('無法解碼影片；請使用有效的 MP4')
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        count_value = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or not 0 < fps <= limits.max_fps or \
                not math.isfinite(count_value) or not 1 <= count_value <= limits.max_seconds * limits.max_fps:
            raise ValueError('影片 FPS／影格數無效或超限')
        count = int(count_value)
        if width <= 0 or height <= 0 or width*height > limits.max_pixels or count/fps > limits.max_seconds:
            raise ValueError('影片需在 120 秒及 1920×1080 像素範圍內')
        selected = set(np.linspace(0, count-1, min(count, limits.max_frames), dtype=int).tolist())
        frame_dir = output / 'frames'
        frame_dir.mkdir(exist_ok=False)
        # Decode to EOF; do not silently accept lying metadata or truncated inputs.
        while True:
            if time.perf_counter()-began > limits.decode_timeout_seconds:
                raise ValueError('影片解碼／人臉處理超過時間限制')
            ok, image = cap.read()
            if not ok:
                break
            index = decoded
            decoded += 1
            if decoded > count or decoded/fps > limits.max_seconds or \
                    image.shape[0]*image.shape[1] > limits.max_pixels or image.shape[:2] != (height, width):
                raise ValueError('實際解碼內容與 metadata 不符或超限')
            if index not in selected:
                continue
            crop, details = faces.crop(image)
            sample = dict(frame_index=index, timestamp_seconds=index/fps, **details)
            if crop is not None:
                destination = frame_dir / f'{index:08d}.png'
                if not cv2.imwrite(str(destination), crop):
                    raise OSError('無法儲存人臉影格')
                sample.update(path=str(destination.resolve()), sha256=sha256(destination))
            samples.append(sample)
            if progress:
                progress('preprocessing', f'抽幀／人臉處理 {len(samples)}/{len(selected)}')
        if decoded != count:
            raise ValueError('影片提前結束或影格數與 metadata 不符')
    finally:
        cap.release()
    usable = [s for s in samples if 'path' in s]
    summary = dict(profile=FACE_PROFILE, sampling=SAMPLING, limits=asdict(limits),
                   source_fps=fps, source_frame_count=count, decoded_frame_count=decoded,
                   source_width=width, source_height=height, duration_seconds=count/fps,
                   sampled_frame_count=len(samples), valid_face_frames=len(usable), samples=samples,
                   face_selection='largest_face_per_sample_no_identity_tracking',
                   face_detector_threshold=.8, face_detector_nms=.3,
                   seconds=time.perf_counter()-began)
    if not usable:
        error = ValueError('抽樣影格未找到可用人臉；沒有產生真假判定')
        error.preprocessing = summary  # type: ignore[attr-defined]
        raise error
    return [s['path'] for s in usable], summary
