"""Download the small official YuNet model with a pinned content hash."""
import argparse
from pathlib import Path
from urllib.request import urlopen
import hashlib

ASSETS = Path(__file__).resolve().parent / 'assets'
MODEL_PATH = ASSETS / 'face_detection_yunet_2023mar.onnx'
MODEL_SHA256 = '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4'
MODEL_SIZE = 232589
MODEL_URL = 'https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'


def download(destination: Path = MODEL_PATH) -> Path:
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() != MODEL_SHA256:
            raise ValueError('Existing YuNet model has a different hash; it was not overwritten')
        return destination
    with urlopen(MODEL_URL, timeout=60) as response:
        content = response.read(MODEL_SIZE + 1)
    if len(content) != MODEL_SIZE or hashlib.sha256(content).hexdigest() != MODEL_SHA256:
        raise ValueError('YuNet download size/hash mismatch; no model written')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        stream.write(content)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=MODEL_PATH)
    print(download(parser.parse_args().output))
