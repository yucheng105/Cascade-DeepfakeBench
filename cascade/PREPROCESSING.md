# Official uploaded-video preprocessing

The MP4 CLI and local demo default to `official`. Shared validation and full
decode produce up to 32 uniform, unique, lossless raw samples. Face geometry
belongs to each model profile; tensor normalization belongs to its adapter.

| Models | Face profile | Tensor / aggregation |
| --- | --- | --- |
| Xception, UCF | Full-resolution dlib, largest face, 81-point predictor, official five-point similarity alignment, post-alignment detection | RGB 256, cubic resize, mean/std .5; mean valid frames |
| SBI FFc23 | Original ternaus RetinaFace ResNet50, max_size 2048, confidence .7 / NMS .4; bbox margin 1/8 per side; retain area >= half largest | Direct linear resize to RGB 380, /255; max faces within each frame, then mean valid frames |

Xception/UCF share the same cached crop. SBI has its own cache and source-frame
groups. No profile runs until its stage is reached. Weights and policy thresholds
are unchanged. Fake score is the class-1 (fake) probability, not accuracy.

## Setup and launch

From the workspace parent, explicitly prepare the isolated Python 3.12 environment:

```powershell
powershell -ExecutionPolicy Bypass -File tools/setup_preprocessing.ps1
```

If the original `.venv` is unavailable, add `-BasePython C:/path/to/python.exe`.
This preserves `.venv` and installs the pinned lock into `.venv-preprocessing`.
Setup downloads packages and missing detector assets, validates hashes, and
executes both detectors. Classifier checkpoints must already exist in
`training/weights/`. Normal inference never downloads dependencies or weights.

From this repository:

```powershell
.\.venv-preprocessing\Scripts\python.exe -B -m cascade.prepare_preprocessing --check-only
.\.venv-preprocessing\Scripts\python.exe -B -m cascade.demo --open-browser
.\.venv-preprocessing\Scripts\python.exe -B -m cascade.video --video C:/path/video.mp4 --output ../data/new_run --allow-exploratory
```

The workspace `Start_Offline_Demo.cmd` uses this environment. Stop the old server
with Ctrl+C before restarting on 8765, or choose `--port 8766`. A running old
process does not reload Python changes.

Use the UI selector or `--preprocessing-mode legacy` for explicit YuNet comparison.
Legacy requires its existing pinned asset (`python -m cascade.download_face_model`
is the explicit preparation command). There is no automatic fallback.
Use `--policy cascade/policies/uploaded_video_all_stages.json` to exercise all
three stages. These policies remain exploratory and uncalibrated for the repaired
profiles; deployment_ready is false. Profile-bound policies reject hash mismatch.
The existing `cascade.offline` cropped-dataset path retains its behavior.

## Compatibility and reproducibility

`assets/preprocessing/assets.json` pins package versions, official asset URLs,
SHA-256 values and upstream references. Windows uses `dlib-bin` 20.0.1.post1 from
alesanfra/dlib-wheels, an upstream dlib binary distribution, not an official dlib
PyPI wheel. This is disclosed rather than described as an original official
environment. `retinaface-pytorch` 0.0.8 pins obsolete torch training dependencies.
`build_retinaface_compat` deterministically creates local version `0.0.8+torch27`,
changing only wheel version/dependency metadata and RECORD. Every upstream Python
source byte is preserved; the source wheel SHA is hardcoded and verified.
`pip check` must pass. This validates inference, not upstream training support.

The workspace verifier `tools/verify_preprocessing_reference.py` independently
loads pinned upstream crop functions and original RetinaFace get_model, using the
verified offline checkpoint. It checks exact crop pixels and tensor error <=1e-6,
logit/probability/video-score error <=1e-4. SBI also uses an independent
EfficientNet-b4 graph. Xception/UCF reuse the existing graph, so their score check
does not independently verify architecture. Reference sources can be restored
with `tools/restore_preprocessing_references.py --restore`.

## Limits and diagnostics

Defaults: 200 MiB, 1200 seconds, 2073600 pixels/frame, 240 fps, 32 samples;
raw decode 180 seconds, each reached profile 180 seconds including cold load,
512 MiB total encoded PNG cache. Override `--profile-timeout` and `--cache-mib`
on CLI or server. Workers are separate processes and are killed at timeout.
Resource exhaustion fails explicitly; faces are never silently capped.

Schema-2 manifests record profile settings, implementation/package/asset hashes,
source-frame identities, bbox/landmarks, coverage, cache keys and
used/not_reached/failed states. `preprocessing.json` and profile `partial.json`
retain progress; `partial_trace.json` retains completed classifier stages.
Zero valid frames in any reached profile fails the whole upload, without a final
REAL/FAKE report. The UI exposes root diagnostic JSON and restricted crop/profile
artifact downloads. Historical jobs remain readable and are labeled as historical
when their profile version was not recorded.

Official parity does not establish model quality. Uniform sampling can miss short
manipulations; there is no identity tracking. Small engineering controls are not
an independent accuracy or FAR/FRR evaluation.
