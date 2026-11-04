# Complete score export and streaming policy selection

Phase 2 reuses the existing adapters, crop profiles and window engine. All-detector
export is an explicit **offline** operation: normal cascade requests still early-exit.
No runtime request performs threshold search, downloads assets or trains models.

## Cohort and export

Cohort JSON contains `dataset_id` and nonempty `records`. Each record declares
unique `sample_id`, `source_id`, `group_id`, `label` (0, 1 or null),
`label_provenance`, optional `lineage`, and either `frames` or `video`.
CLI paths resolve relative to the cohort file. For example:

```json
{"dataset_id":"ffpp","records":[{
  "sample_id":"FF-real_000","source_id":"original_000","group_id":"family_000",
  "label":0,"label_provenance":"dataset annotation reference",
  "lineage":{"person_ids":["person_000"],"family_id":"family_000"},
  "frames":["rgb/FaceForensics++/frames/000/000.png"]
}]}
```

This illustrates the format, not a sufficient calibration dataset. Group/label
claims require independent review. Known related originals, people and derived
fakes must have consistent identifiers. `person_id`/`person_ids`,
`family_id`/`family_ids`, `original_id`/`original_ids` share their own namespace.

```powershell
python -B -m cascade.export_scores --cohort C:/data/cohort.json --mode crops --unit video --output C:/runs/new_scores --device cpu
```

For raw MP4, replace `frames` with `video`; use `dataset_id=uploaded_video` as the
runtime domain and retain source dataset details in `lineage`. Select
`--mode official` (dlib81/RetinaFace) or explicit `--mode legacy` (YuNet).
Video export uses uniform unique sampling up to `--max-frames` (default 32).
`--unit window` uses the current nonoverlapping 4-second/2-Hz samples and sampled
tail, without pacing playback or adding live/sliding input. Library callers can
specify the same cadence parameters supported by the session.
`--detectors xception,ucf,sbi` may select a subset. Repeatable
`--checkpoint DETECTOR=PATH` / `--config DETECTOR=PATH` bind actual files.
Only run a deliberately selected cohort/device; complete export may be expensive.

Window export calls the existing engine with `full_scores=True`; normal HTTP
workers never enable that flag. Every requested detector gets a disposition,
including failures/no-face. Missing evidence is not zero or REAL.

A video label is **not automatically a window label**. Supply per-record
`window_labels` as an object such as `{"0":{"label":1,"label_provenance":"time-local annotation"}}`,
or explicitly assert `window_label_scope=all_windows` when the annotation truly
covers the whole source. Otherwise exported window labels remain null, with the
source label retained separately, and optimization refuses them. Missing windows
in a per-window label map stay unknown unless the all-windows assertion is explicit.

## Artifact and eligibility

Only new output directories are accepted. Completed export creates:

- `manifest.json`: schema 1, kind `complete_detector_scores`, execution domain,
  input contract, all samples/models, actual hashes, exporter/runtime revision,
  hardware and complete coverage summary.
- `scores.jsonl`: one sample × detector disposition, score or null, valid-frame
  count, cost breakdown and frame/profile/face coverage evidence.
- `manifest.sha256`: exact manifest byte fingerprint; the manifest binds the score
  payload hash. These fingerprints detect changes; they are not signatures.

Window samples retain time/index/tail, source IDs, frame/pixel hashes and actual
FPS/count. Timestamps are `frame_index_over_fps`, not container PTS. Crop-source
hashes bind selected PNG contents and do not reconstruct original-video lineage.
The preprocessing fingerprint now binds adapter implementation bytes; windows
also bind cadence. One helper describes/hashes face profiles across export and
runtime. Old bound fingerprints may reject: re-export rather than guess hashes.

Repo A rejects incomplete/duplicate/unlabeled/nonfinite records, inconsistent
profile/unit/cadence/coverage, or unknown/mixed cost scope. It never silently drops
failed or no-face samples. Audit the manifest and explicitly version any new
eligible cohort, reporting original coverage and exclusion criteria. Reached-only
CSV output is not a complete joint-score artifact. Interrupted export directories
without a complete manifest cannot be optimized.

The search objective is **sum of reached batch-allocated forward milliseconds**.
Preprocessing, load, warmup and stage wall time are recorded separately. Startup,
queue and persistence are outside this objective; replay p95 is not measured
end-to-end latency. Preserve hardware/cost scope when comparing policies.

## Selected streaming policy

The session library accepts policy content or `policy_path` and optional
per-detector config/checkpoint maps. It freezes policy content, validates actual
window domain/cadence/profile and bindings before classifier initialization, loads
only selected models and retains resident reuse. Incompatible assets are rejected.

Register window JSON for the local API/UI:

```powershell
python -B -m cascade.demo --stream-policy experiment=C:/runs/search/policies/sequential_far0.05_latency.json
```

Registration freezes at server startup; each job saves its snapshot/canonical hash.
Restart/re-register to use changed JSON. The API catalog exposes IDs/order; the
selector, window traces, history and downloads preserve the chosen policy. Browser
queries select registered names rather than supplying arbitrary filesystem paths.
Default asset files are the runtime defaults; custom asset maps are available to
library sessions. Stop older demo servers before restarting; Python does not hot-reload.

Window early exit, no-face, overload, tail/stop and worker failure semantics stay
unchanged. No session risk aggregator or formal uploaded-video service is added.
All generated policies remain exploratory pending independent scientific review;
see [POLICY_CONTRACT.md](POLICY_CONTRACT.md) and the analysis workflow README.
