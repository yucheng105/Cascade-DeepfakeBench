# Sequential policy execution contract (schema v1 extension)

This contract checks execution compatibility. It does not certify population FAR,
independent evaluation, dataset lineage or the provenance of a supplied hash.
Existing bundled policies remain uncalibrated and `deployment_ready=false`.
Phase 2 supplies score export, split/evaluation evidence and optimizer integration;
this change does not manufacture a deployment-ready policy.

## Loading and exploratory compatibility

The existing schema, dataset, unique sequential stages, metric/comparator and
threshold rules remain unchanged. `deployment_ready`, when supplied, must be a
JSON boolean. Policies without new metadata remain usable through explicit
exploratory access. Any new field below is checked whenever it is supplied;
`--allow-exploratory` does not ignore malformed metadata or declared bindings.

For a policy marked `deployment_ready=true`, all these fields are required even
when the caller allows exploratory policies:

| Field | Validation |
| --- | --- |
| `target_far` | Finite JSON number in [0,1], not boolean; FAR means fake classified REAL / all fake, i.e. FNR |
| `latency_model` | `sum_reached_csv_ms_serial`, `sum_reached_forward_ms_serial`, or `measured_end_to_end_ms` |
| `checkpoint_sha256` | Object with exactly the active detector names, each a lowercase 64-character SHA-256 |
| `config_sha256` | Same mapping rule |
| `preprocessing_sha256` | Same mapping rule; exact adapter profile fingerprint, including face profile when supplied by that entry |
| `input_contract` | The complete object below |

The optional `objective` is `accuracy` or `latency`. Existing descriptive fields
such as `purpose`, `split_status` and `input_manifest_sha256` are preserved;
loading does not independently validate their scientific assertions. Latency
enums label cost scope, not a real-time SLA or a worst-case bound. CLI dataset
reports still use batch-allocated forward time and stage wall time with their
existing timing_scope; a policy cost model does not change those measurements.

## Input contract

For the cropped-dataset entry, a one-detector example is:

```json
{
  "unit": "video",
  "sampling": "first_N_numeric_sorted_frames",
  "max_frames": 32,
  "profiles": {"xception": "preprocessed_face_crops"},
  "aggregation": {"xception": "mean_frame_probability"}
}
```

Both mappings must cover exactly the policy's active detectors. Profiles are
`preprocessed_face_crops`, `dlib81`, `retinaface` or `yunet_legacy`;
aggregation is `mean_frame_probability` or `max_face_then_mean_frame`.
Unknown or extra keys inside `input_contract` are rejected to catch misspellings.

Video contracts require exactly `unit`, `sampling`, `max_frames`, `profiles`,
`aggregation`. `max_frames` is a positive integer, describing the configured
sampling limit rather than guaranteeing that many valid face frames. Video
sampling is `first_N_numeric_sorted_frames` or `uniform_unique_frame_indices_v1`.

Window contracts require exactly `unit`, `sampling`, `profiles`, `aggregation`,
`window_seconds`, `stride_seconds`, `sample_hz`. Their sampling is
`paced_media_time_no_future_frames`; all three cadence values are finite positive
numbers, not booleans. Validation defines their shape, not new sliding-window or
custom streaming-policy execution support. Streaming policy injection remains
Phase 2; neither offline entry accepts a window contract as a video contract.

## Entry-point gates

`cascade.offline` constructs its actual input contract from the selected stage
names and `--max-frames`: video, first-N numeric-sorted existing face crops,
mean-frame probability for every detector. It compares a declared input contract
before reading the cohort or creating output, then compares all declared hashes
against adapter provenance before classification or output creation. The manifest
records `input_contract` and `input_profile_status=legacy_preprocessed_face_crops`.
SBI resize/normalization cannot restore official RetinaFace geometry from these
crops; a policy declaring a RetinaFace profile is refused by this entry.

`cascade.video` remains an explicit **exploratory-only** uploaded-MP4 entry.
It requires `--allow-exploratory` and `deployment_ready=false`; formal upload
execution is outside this phase. Its actual contract is video, uniform unique
sampling at the requested limit, dlib81 for Xception/UCF and RetinaFace for SBI
(or explicitly selected legacy YuNet), and max-face then mean-frame aggregation.
Its declared input contract is compared before preprocessing/output creation;
all declared checkpoint/config/preprocessing hashes are compared before classifier
inference. Asset checks can occur after shared raw decoding and retain error
diagnostics if an output directory already exists. Existing profile fingerprints
are unchanged; callers must use the actual entry's manifest/provenance, not copy
a dataset-crop fingerprint into an uploaded-video policy.

`run_cascade` is the shared decision engine. It validates policy structure, but
cannot inspect actual source samples or assets. Direct library callers must
validate metadata and apply `validate_runtime_input` / `validate_runtime_bindings`
at their execution boundary. These comparison helpers take an already validated
policy. They do not load checkpoints, run optimization or decide deployment quality.

## Terminal and failure semantics (unchanged)

- Intermediate stages: `score <= low` REAL, `score >= high` FAKE; strictly between
  the thresholds is UNCERTAIN and advances only that sample to the next stage.
- Final stage: `low == high == t`, `score < t` REAL, `score >= t` FAKE. There is no
  terminal UNCERTAIN for a valid finite score. Changing this requires a separate
  versioned decision/metric design, not a UI relabeling.
- An uploaded video with no valid faces in a reached profile fails. A detector
  exception also fails the job. Diagnostics and completed partial trace survive;
  there is no final REAL/FAKE report or decisions export. An unreached profile is
  not needed for an already completed early exit.
- Streaming NoFaces means `unable_to_decide` for that window; later windows may
  continue. Other worker/model errors fail the session and cancel pending work.
  A window's normal early exit does not stop its session. Skipped/cancelled/failed
  windows are never implicitly REAL. There is still no whole-session verdict.
- Unknown labels remain unknown; uploaded/window inference has null accuracy,
  FAR/FRR, FNR/FPR and confusion counts. Fake probability is not the probability
  that the final decision is correct.

## Legacy boundary and verification

`cascade/test.py` is the legacy Xception-to-UCF dataset benchmark. Its shared-tensor
validator requires matching resolution/normalization; it is not the generic
three-detector runtime and must not impose its tensor contract on SBI.
Prefer `cascade.offline` for policy-driven dataset execution and `cascade.video`
for uploaded MP4. Model weights and routing thresholds are unchanged.

Focused public-contract regression:

```powershell
.\.venv-preprocessing\Scripts\python.exe -B -m unittest cascade.tests.test_policy_contract -v
```

Fixtures verify valid bound execution, malformed metadata, mode-independent
bindings, input/profile mismatch and detector-failure evidence without a real
checkpoint or GPU. Actual statistical quality and large GPU benchmarks require
separate evidence.
