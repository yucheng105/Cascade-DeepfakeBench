# Cascade pipeline

## Offline MP4 demo — official preprocessing

Prepare the separate environment from the workspace parent:

```powershell
powershell -ExecutionPolicy Bypass -File tools/setup_preprocessing.ps1
```

Double-click `Start_Offline_Demo.cmd`, or from this repository run:

```powershell
.\.venv-preprocessing\Scripts\python.exe -B -m cascade.demo --open-browser
```

Stop any older server on 8765 before restarting. The default upload path shares
raw decoding, uses official dlib crops for Xception/UCF and original RetinaFace
bbox crops for SBI, and preserves per-profile coverage and diagnostics. Select
legacy explicitly to compare YuNet. Missing dependencies fail without fallback.

Setup, CLI commands, resource limits, compatibility packaging and failure
artifacts are documented in [PREPROCESSING.md](PREPROCESSING.md). The cropped
dataset runtime below remains separate. Upload policies remain exploratory;
unknown labels and FAR/FRR stay null. See workspace
`docs/preprocessing_repair_acceptance.md` for the current parity evidence, and
`docs/phase2_video_demo.md` for the historical YuNet demo record.

### Research demonstration interface

The light interface pairs the source video with the Xception → UCF → SBI
decision path. Reached stages show their actual scores and thresholds;
unexecuted stages remain explicitly marked. Preprocessing and policy controls
are under advanced settings, while detailed traces and provenance are expandable.
Offline results include captioned face samples and report downloads.

Streaming windows can be selected to inspect their own traces. Selection does
not interrupt active playback; after completion it also seeks to the window's
start. Skipped, cancelled and unable-to-decide windows are never presented as
REAL, and no whole-stream verdict is inferred. Fake scores are not accuracy or
the probability of a correct decision. The interface uses local font fallbacks
and requires no external asset service or frontend build.

Frontend behavior regressions (requires Node.js, no packages to install):

```powershell
node --test cascade/tests/test_demo_ui.cjs
```

## Phase 2: policy-driven offline runtime

The new entry point is `python -m cascade.offline`; the existing `cascade/test.py`
two-stage entry is retained. Phase 2 uses the original Xception, UCF and SBI model
classes with a strict full detector checkpoint load, without requiring a separate
ImageNet checkpoint. Training defaults are unchanged. The runtime explicitly
opts into three-model registration using `DEEPFAKEBENCH_RUNTIME_ONLY=1`.

Create an isolated Python 3.12 environment from this repository root. Install the
official PyTorch CUDA wheel and the bounded inference dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r cascade/requirements-runtime.txt
```

The CUDA wheel command follows the [official PyTorch 2.7.1 installation matrix](https://pytorch.org/get-started/previous-versions/#v271).
This runtime consumes already cropped RGB frames; dlib, torchvision, training
augmentations and unrelated detector packages are not required. This environment
is for the new runtime, not a claim that every legacy training entry works in it.

From the repository root, run the prepared small FF++ test cohort:

```powershell
.\.venv\Scripts\python.exe -B -m cascade.offline `
  --policy cascade/policies/ffpp_smoke_all_stages.json `
  --dataset FaceForensics++ `
  --dataset-json ../data/phase2_ffpp_smoke/dataset_json/FaceForensics++.json `
  --rgb-root ../data/phase2_ffpp_smoke/rgb `
  --device cuda --batch-size 16 --max-frames 32 --allow-exploratory `
  --output ../DOCS/phase2_runs/my_unique_all_stages_run
```

For early-exit testing, use `cascade/policies/ffpp_smoke_early_exit.json` and a
different output directory. These policies are functional fixtures, **not calibrated
security policies**. Thresholds of 0/1 encourage forwarding but exact endpoint
scores still follow normal inclusive exit rules.

Policy JSON uses Phase 1 schema v1, a `policy_id` and a matching canonical
`dataset_id` (`ffpp`, `celeb_df_v2`, `dfdc`). Only sequential policies are supported
by this entry. Intermediate real <= low / fake >= high; final real < t / fake >= t.
Each policy detector is unique, thresholds finite, terminal low=high. Unsupported
metrics/comparators, wrong datasets and fusion policies are refused. Phase 1
artifacts marked `deployment_ready=false` require `--allow-exploratory`; omitting
that flag requires a deployment marker plus matching `checkpoint_sha256` and
`config_sha256` dictionaries keyed by detector. Hash binding is compatibility
validation, not proof of statistical calibration or provenance correctness.

Default weights are `training/weights/{xception_best.pth,ucf_best.pth,FFc23.tar}`.
Use repeatable `--checkpoint DETECTOR=PATH` or `--config DETECTOR=PATH` overrides
for files actually used. Each stage hashes its source checkpoint/config before
execution, but the model is loaded only if at least one video reaches it.
SBI's `checkpoint['model']` / `net.*` format is mapped to DeepfakeBench names and
strict-loaded. Other wrappers and missing/incompatible tensors fail explicitly;
there is no silent partial weight load or random-model fallback.
The checkpoint-only SBI path also retains EfficientNet's native SAME-padding stem;
the legacy training constructor's unpadded replacement is not the official graph.
An independent logits-parity test checks the restored model against
`efficientnet_pytorch.EfficientNet` without downloading reference weights.

The JSON input reader preserves `source_label + '_' + source_video_id` as opaque
identity, rejects missing/escaping paths, and selects the first N numeric-sorted
frames to match the current legacy exporter's effective selection. It does not
claim uniform temporal sampling. All stages receive the same frame paths; each
adapter separately applies its YAML resolution, RGB conversion, configured OpenCV
resize, tensor /255 and mean/std normalization.
The new runtime defaults SBI to `cascade/configs/sbi_unit_range.yaml` (380x380,
linear resize, RGB /255, mean 0, std 1), following the
[official SBI input preprocessing](https://github.com/mapooon/SelfBlendedImages/blob/master/src/inference/preprocess.py)
and [probability inference](https://github.com/mapooon/SelfBlendedImages/blob/master/src/inference/inference_video.py).
Xception/UCF retain their 256x256 cubic-resize and mean/std 0.5 profiles.
The legacy `training/config/detector/sbi.yaml` ImageNet normalization remains
available via an explicit `--config sbi=...` override for models trained for that
normalization. This runtime's SBI graph targets the official-style checkpoint;
normalization override alone does not certify a legacy DeepfakeBench-trained SBI
checkpoint's architecture. Full equivalence to official face cropping and historical CSV provenance
is not asserted. Landmarks are not used by these models'
RGB-only inference forward; SBI synthesis/training is a different data path.

Outputs (new directory only; no overwrite):

- `manifest.json`: policy, hashes of checkpoints/configs/frame inputs, selected IDs,
  preprocessing profile, source revision/dirty state and runtime file hashes,
  device/CUDA/Python details.
- `report.json`: complete final-cohort confusion metrics, stage reach/exit and
  reached-cohort AUC, model load/warmup/forward/stage durations.
- `decisions.csv`, `trace.csv`: every video's final result, exit stage and every
  reached detector score/threshold/decision.
- `reached_scores.csv`: video scores with full opaque IDs, frame count and
  allocated forward milliseconds. **Later-stage coverage is only reached videos**;
  do not treat this as a full joint-score benchmark without checking coverage.
- `error.json` after an execution failure, if the output directory was created.
  Failed directories are preserved; retry with a new run name.

CUDA timing synchronizes before/after forward. Forward time excludes model load,
warmup, decode and H2D; video costs allocate each batch's forward ms equally to
frames, then sum by video. Stage time includes those operations and model release.
No per-video end-to-end latency or streaming throughput claim is made.

Tests: `python -B -m unittest discover -s cascade/tests -v`. Runtime parity tests
expect Repo A as sibling `far-frr-graph`; production runtime has no cross-repo
Python import. A real checkpoint/CUDA smoke run is additional to unit tests.

---

New sequential-cascade components live in this directory. The original
DeepfakeBench code should only receive small compatibility changes when the
cascade cannot be implemented cleanly as an adapter.

## Video identity contract

`video_name` is an opaque, non-empty string that uniquely identifies one video
within a test dataset. Cascade code must compare the complete value and must not
derive it by splitting a frame path.

For every dataset sample, these lists must stay index-aligned:

```text
data_dict["image"][i]
data_dict["label"][i]
data_dict["video_name"][i]
```

`VideoIdentityDataset` preserves the original DeepfakeBench sample and collate
formats while adding this batch metadata（原本 Dataset 的 Wrapper）:

```python
batch["video_name"]  # list[str], aligned with batch["image"]
```

The adapter also provides `indices_for_videos()` and `subset_for_videos()` so a
later stage can receive all and only the frames belonging to uncertain videos.

## Stage 1: Xception

Inside the Docker container, run the first stage from the repository mount:

```bash
cd /app
python cascade/test.py \
  --low_threshold 0.2 \
  --high_threshold 0.8 \
  --final_threshold 0.5
```

Replace the example thresholds with the operational values supplied to the
system. The stage groups frames by the opaque `video_name`, computes the
arithmetic mean fake probability, and applies these inclusive exit rules:

```text
score <= low_threshold   -> REAL
score >= high_threshold  -> FAKE
otherwise                -> UNCERTAIN
```

Stage 1 AUC is calculated over every FaceForensics++ video. A cohort with fewer
than two ground-truth classes is reported as `N/A`.

Stage 1 uncertain videos are selected from this same Dataset object, so UCF
receives the exact same sampled frames. Stage 2 uses a single inclusive fake
threshold:

```text
score < final_threshold   -> REAL
score >= final_threshold  -> FAKE
```

Stage 2 AUC covers only videos actually processed by UCF. Final ACC, FAR, and
FRR cover the original full FaceForensics++ test cohort, with fake as class 1:

```text
FNR = FAR = fake videos classified as REAL / all fake videos
FPR = FRR = real videos classified as FAKE / all real videos
```

FNR/FPR are the explicit error-rate names. FAR/FRR are compatibility aliases
using "accept = classify as REAL". They do not change score semantics: Fake=1,
Real=0, high fake score predicts FAKE. A missing class has a null rate; an
unlabeled upload has null evaluation metrics. See the sibling
`far-frr-graph/METRIC_DEFINITIONS.md` for historical fusion exports whose aliases
were reversed and a non-destructive prediction audit.

Long videos: at most 32 frames are sampled uniformly over the entire video; brief manipulations can be missed. Decode time increases with video length; the existing 180-second decode/face-processing timeout and 200 MiB size limit still apply. This is not segmented detection.
## 模擬即時串流

本機 Demo 新增 MP4 原速逐窗分析；操作與語意見 [STREAMING.md](STREAMING.md)。
