# YuNet face detector

Download with `python -m cascade.download_face_model` from the repository root.
No download occurs implicitly during inference or server startup.

Official model: OpenCV Zoo `face_detection_yunet_2023mar.onnx` (232589 bytes).
SHA-256: `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`.
The downloader rejects LFS pointers, incomplete downloads and unexpected content.

Source: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
License: MIT, copyright 2020 Shiqi Yu. See LICENSE-YuNet.txt.

This is a new largest-face/five-landmark alignment profile, not an assertion of
equivalence to the old dlib preprocessing or the official SBI video pipeline.
