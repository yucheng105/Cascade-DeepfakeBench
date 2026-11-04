# Execution explanations

New video/Window reports declare `trace_schema_version: 1`. The existing scored
`trace` remains scored rows only, now with a `route_reason`. Each result adds an
`explanation` containing `schema_version`, context and the selected Policy's full
stage view. Unreached models do not receive invented trace scores.

Context includes input unit, sample/Window/session/source identities, observed
time range/basis, source-frame evidence hash/count, selected Policy hash, input
contract, available profile coverage/model bindings and temporal rule version.
Full frame/profile evidence remains in the original report. Missing context is
null/unrecorded; the wrapper provides it, not the Controller.

| Execution status | Meaning |
| --- | --- |
| not_started | No per-input stage-entry evidence yet |
| entered | Adapter call entered; can include preparation, not proof of forward |
| scored | A valid detector score was returned and routed |
| not_executed | Explicit prior exit, upstream failure, window skip/cancel reason |
| unable_to_decide | Reached adapter reports no valid face evidence |
| failed | Known adapter/validation failure, with recorded phase |
| unreported | Current/failed execution cannot be located to a stage |

The worker's resident models/warmup do not establish forward on a particular
Window. A native RPC that has not returned stage events is not displayed as
three models all executing. `forward_evidence` only claims a valid score when
one was returned; failures without that evidence do not invent forward completion.

Route reasons preserve the original comparisons: `real_le_low`, `fake_ge_high`,
`continue_between`, `final_real_lt_threshold`, `final_fake_ge_threshold`.
The stage `decision` keeps the original REAL/FAKE/UNCERTAIN vocabulary. A terminal
score belongs to its detector; it is not calibrated confidence or session risk.

Downloads:

- `report.json`: original results plus explanation/context.
- `trace.csv` (Offline sparse): scored stages only, with route reason.
- `explanation.csv`: all planned stage states, blank unavailable scores, explicit
  reasons and `context_json`. `trace_schema_version` is in each exported row.
- `partial_trace.json` / `partial_explanation.json` (Offline sparse): last
  available execution evidence, including known preparation/model/validation
  failures. Available diagnostic downloads are listed by the job API.

Offline sparse preserves its original fail-job behavior on no faces; the Window
engine returns unable-to-decide and continues according to the existing contract.
Source/overload/cancellation/persistence failures stay operational dispositions,
never REAL. Accepted classification remains intact when later persistence fails;
`operational_failure` records the session problem separately.

JSON/CSV/API/UI use the same explanation. On interrupted new-format Window
history, cancellation and failure scope are updated with the stored observation;
completed histories are read verbatim. Legacy histories lack recorded reasons
and are shown as historical/unrecorded rather than silently upgraded. Existing
metadata budgets still apply; exports can require extra transient disk space and
are not a total process RSS/global history quota. No detector reruns or new raw
pixel/tensor retention are introduced for explanations.

Research comparisons remain offline. Their `UNCERTAIN` route means continue;
legacy analysis NEXT can be normalized to the same meaning for audits without
changing routing. Serial allocated forward cost and measured wall latency retain
different scopes. No parallel inference endpoint or policy search is added here.
