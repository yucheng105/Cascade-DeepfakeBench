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
