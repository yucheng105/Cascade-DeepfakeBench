# MP4 模擬即時串流

使用已準備好的官方前處理環境，從 runtime repository 啟動：

```powershell
.\.venv-preprocessing\Scripts\python.exe -B -m cascade.demo --open-browser
```

或從 workspace 雙擊既有 Start_Offline_Demo.cmd。開啟本機頁面，選擇 MP4，將
「分析方式」改為「模擬即時串流」，確認探索性說明後開始。分類模型與人臉
偵測器就緒後才從第 0 秒原速播放；播放預設靜音。這是本機 MP4 的 paced replay，
尚未接攝影機或 RTSP。

- 每 4 秒一個不重疊時間窗、每秒取樣 2 個來源影格，每窗最多 8 幀；不能使用尚未送入的未來影格。
- 官方模式沿用已驗證的 dlib81／RetinaFace。Xception/UCF 共用最大臉裁切；只有到達 SBI 的窗才執行其人臉偵測／裁切，保留官方合格多臉、幀內 max，再跨有效來源幀 mean。沒有身分追蹤。
- 每窗獨立 Early Exit。頁面列出時間範圍、結果、分數、退出階段、逐 profile 覆蓋與延遲；詳細逐關 trace 可展開及下載。沒有整場串流的 binary 最終判定。
- 一窗推論、一窗等待；新窗取代較舊的等待窗，所有略過區間標為未分析。畫面結果不是當下播放影格的判定，請看其時間範圍。
- 已到達的任一 profile 無有效臉：該窗無法判定，保留先前 trace，繼續下一窗。依賴、解碼、模型、資源及逾時錯誤則停止 session。
- 自然播完會處理已有取樣的片尾短窗並等待保留的工作完成；按停止只完成當下推論的窗，其餘標為取消。
- 原生解碼與推論位於可終止的 child processes；停止暖機可終止尚未就緒的 worker。已開始的推論可完成，超過設定預算則終止。
- 原始／人臉 scratch 為有界工作資料，逐窗處理後回收；下載保留原始像素／crop hashes、模型／profile／實作版本、分組、原因及耗時，不永久保存所有串流裁切。
- JSON 報告、時間窗 CSV、manifest 於各工作目錄保存；重啟後可查已保存紀錄。中斷的 session 顯示失敗／中斷，不自動續跑。離線與串流共享單工作限制。

輸入上限仍為 MP4 200 MiB、1200 秒、每幀 2073600 pixels、240 FPS。原離線
「全片最多 32 幀」只適用離線模式。預設 512 MiB 是 active raw/crop cache
的邏輯資料預算，另需模型、偵測器、IPC 複本及 runtime 記憶體；完整取樣率不代表
所有幀或每個人物都已分析。缺官方資產不會自動切回舊 YuNet。

效能目標是指定機器與素材的工程驗收：RTX 3090、至少 120 秒原速輸入、無過載
略窗，暖機後窗結束至可讀取結果的 nearest-rank p95 ≤4 秒。啟動另計；所有無臉、
略過與取消另列。不能將此目標當成任意來源的速度保證或準確率驗收。

測試：`python -B -m unittest cascade.tests.test_streaming cascade.tests.test_streaming_profiles -v`。
HTTP 測試使用可控來源與模型；真實 GPU、官方參考及瀏覽器驗收另存 workspace 證據。
# Phase 2 policy selection

Register a frozen window policy using `--stream-policy NAME=JSON` on the demo
server, then select its registered identity in the existing policy selector.
The job/manifest/report retains policy content, canonical hash and actual model
order. Only active detectors are initialized; declared cadence/profile/asset
mismatches fail before classifier loading. Defaults remain exploratory. See
[SCORE_ARTIFACT.md](SCORE_ARTIFACT.md) for complete-score export and asset maps.

## Phase 3: Offline Window, ongoing sources and temporal evidence

The original `offline` whole-video sparse mode remains unchanged. Choose
`offline_window` for every window without playback pacing or overload replacement.
`stream` still has one running and one waiting window; newer work replaces only
the waiting window. Both use the resident `OfficialWindowEngine` and frozen policy.
The HTTP capability is now API v3; window reports are schema v4.

In the UI choose Window duration, stride and Hz. The default stream keeps the
legacy 4/4-second, 2-Hz sampling contract. Choose **共用滑窗取樣** to use the new
domain even at those defaults. Offline Window always uses the new domain.
Changing any cadence setting also selects the new domain; a bound legacy policy
is rejected. Register a compatible policy with `--stream-policy NAME=JSON`.
Unbound built-in policies remain exploratory and uncalibrated.

For example, upload a local MP4 using the existing `/api/jobs` endpoint with:

```text
?filename=sample.mp4&analysis_mode=offline_window&policy=early&exploratory=true&window_seconds=4&stride_seconds=2&sample_hz=2
```

The new input domain is `window_contract_version=2`,
`sampling=admitted_tick_grid_v2`. Ticks are anchored at media time zero; committed
progress selects the newest admitted frame at/before each tick. A frame identity
is selected once globally, then shared across windows by reference. MP4 uses the
existing floor(tick × FPS) selection and explicitly declares
`timestamp_basis=frame_index_over_fps`; this is not original PTS. The default 4/4,
2-Hz MP4 frame selection remains reproducible. Non-integral cadence may differ
from legacy per-window anchors and must be treated as a different policy domain.

`windowing.WindowInput` / `WindowResult` carry source/session/window identities,
nominal and observed bounds, tail, sample ticks, source timestamps, pixel hashes,
profile coverage, disposition and policy binding. Pixels stay in bounded working
storage, never in reports. Equivalent v2 profile geometry is cached by exact
source/frame/pixels/profile; different profiles keep separate transforms/tensors.
Legacy/unversioned direct engine calls do not reuse across-window geometry.

For a library integration, pass `event_source_factory` to `StreamingSession`.
Its instance exposes optional `info` metadata, `read_event(timeout=..., stop_event=...)`
and `close()`. The factory and metadata setup must return promptly. No camera or
RTSP production endpoint is provided. Supported event dictionaries:

```python
dict(type='frame', frame_id='source-frame-17', frame_index=17,
     timestamp_seconds=8.5, timestamp_basis='source_pts', image=bgr_uint8_hwc)
dict(type='progress', seconds=9.0)  # commits samples strictly before this watermark
dict(type='end', seconds=9.3)       # EOF; emits observed short tails
dict(type='error', message='source disconnected')
```

Duration, total frame count and file hash are optional for this source. Source
IDs are explicit (or a session-local fallback); the system never invents a file
hash. Sources must not mutate admitted image storage. Identical identity/content
replay is deduplicated; conflicting content fails the session. Frames older than
the committed watermark are rejected and logged, never inserted into closed windows.
Frame indices, when supplied, must be unique to their source identity. A missing
timestamp basis, regressing watermark or future frame is refused.

Reads have bounded waits and cancellation. Native MP4/model calls remain in
killable processes. Injected Python reads use one bounded reader thread: a source
that ignores cancellation is quarantined, and replacement readers are refused
until it returns. It cannot leak one new thread for every timeout. Reader factory
construction itself is the integrator's cooperative responsibility.

Defaults: at most 128 overlapping active windows, 10,000 windows, 3,600 media
seconds, 200,000 source events/ticks/persisted sample memberships and 1,000 source
diagnostics. MP4's existing 1,200-second/size/pixel limits still apply. Exceeding a
limit fails explicitly. The logical raw/crop cache budget is split into timeline
and worker ceilings; model memory, native decoder storage and IPC/runtime copies
are additional. Face work is submitted in batches of eight; each v2 frame's crop
geometry is capped at 1/16 of the worker budget, and the reusable geometry cache
at 1/4 (4,096 entries). These are finite ceilings, not a total process RSS promise.

`TemporalAggregator` emits `temporal_evidence_v1` independently of Cascade routing:

- Any valid FAKE window: `FAKE_EVIDENCE`, retaining its source intervals.
- Only REAL observations, with no unknown dispositions or uncovered observed time:
  `REAL_EVIDENCE_ONLY`. An open source is explicitly provisional.
- Otherwise: `INSUFFICIENT_EVIDENCE`; failure cannot certify REAL-only evidence.

There is no session confidence, averaged exit score or calibrated session FAR.
Decided seconds use interval union. Profile/frame coverage, unknown dispositions,
conflicting REAL/FAKE intervals and uncovered time remain separate. Overlap cannot
erase a skipped/cancelled/failed window. Replayed immutable results are idempotent;
conflicting results are rejected. EOF, user stop and failure finalize separately.
Stop drains in-flight inference and cancels waiting/unfinished observations without
extending coverage into future media time. Offline latency is measured from window
admission; streaming latency remains from media-window end.

API polling, JSON download and restarted history use the stored rule/result.
Completed histories are not recomputed; histories without an aggregate show
unavailable. An interrupted schema-v4/v1-rule report is explicitly finalized as
failure at restart. Download `report.json` for complete frame/profile evidence.

For complete score export use the same domain explicitly:

```text
python -B -m cascade.export_scores --cohort cohort.json --output scores-new --mode official --unit window --window-contract-version 2 --window-seconds 4 --stride-seconds 2 --sample-hz 2
```

The analysis loader checks complete window/sample/model coverage and shared frame
hashes. Policy search still requires grouped calibration/evaluation roles and
explicit window labels or a declared `all_windows` truth scope. No source-level
label is implicitly copied to its windows, and overlap is not independent data.
Emitted policies remain pending independent review. Real connectors, v2 GPU
throughput and scientific quality are UNVERIFIED by the lightweight fixture suite.
