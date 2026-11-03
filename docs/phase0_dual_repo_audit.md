# CASCAde Phase 0 — Dual-Repository Audit & Integration Planning

稽核日期：2026-10-30（Asia/Taipei）。範圍：本機唯讀 Audit；唯一新增產物為本文件。未修改程式、CSV、設定或資料；未 clone/fetch、安裝套件、下載權重／資料集、執行分析入口、訓練或 GPU 推論；未 commit 或發布 Issue。

本文 `A/` 代表 `C:\Users\oplab\Desktop\DeepFake_Cascadeguard\far-frr-graph`，`B/` 代表 `C:\Users\oplab\Desktop\DeepFake_Cascadeguard\Cascade-DeepfakeBench`。程式證據以「路徑 :: 函式／類別」標示；資料證據以 CSV 路徑及本次唯讀檢查結果標示。結論只適用下列本機 commit，未宣稱與遠端最新版本一致。

狀態：**Implemented**＝Source 已有對應實作，不表示本機端到端驗收通過；**Partial**＝有基礎但缺必要契約／整合；**Missing**＝本次指定目錄及 Python Source 搜尋未找到；**Unverified**＝缺執行環境、來源或資料證據。

## 1. Executive Summary

1. **兩個 Repo 都已存在且起始 working tree 乾淨。** 外層另有第三個 Git Repo，A/B 並非外層已追蹤的 submodule；執行 Git 操作應明確指定 Repo。盤點詳見第 2、3 節。
2. **可重用分析與推論基礎，不能直接把既有結果當 Phase 1 baseline。** A 已有 sweep、EER、排列／門檻窮舉、固定 FAR 篩選、GP 與 fusion；B 已有 Xception→UCF 的兩階段離線推論、影片聚合與 early exit。證據：`A/exhaustive-search/threshold-sweep/src/calculate_permutation.py :: calculate_cascade_far_frr`；`A/bmma_gpt_lookup/build_lookup_table.py :: solve_gp_thresholds`；`B/cascade/test.py :: main`。
3. **A 尚缺真正依聯合分數逐樣本 routing 的序列 Cascade simulator。** 既有序列結果來自邊際 FAR/FRR 曲線相乘；GP 後的「重算」也只是曲線差式，沒有讀入共同樣本。不同 Detector 錯誤相關時，這不等於真實系統錯誤率。Parallel Fusion 則確實有逐樣本重算。證據：`calculate_permutation.py :: calculate_cascade_far_frr`；`build_lookup_table.py :: compute_cascade_far_frr`；`A/parallel-fusion/exhaustive/exhaustive_search.py :: evaluate_grid_point`。
4. **CSV 可用，但有身分及 lineage 缺口。** Celeb-DF-v2 每模型 518 筆、DFDC 4,704 筆，可依字串 ID 一對一 join。FF++ 每模型 700 筆卻只有 280 個 ID；140 個假片 ID 各出現四次。現行 occurrence join 的值一致性已核對，但生成方法對映未能證實。證據：第 5 節九份原始 CSV；`A/parallel-fusion/merge_scores.py :: index_by_sample_occurrence`。
5. **必須先統一指標及版本。** 序列 A 與 B 使用 FAR=FNR、FRR=FPR；A 的 parallel exhaustive/MILP 則稱 FPR 為 FAR、FNR 為 FRR。Xception/UCF 舊 sweep 與目前 raw scores 不一致；GP `main()` 尚引用搬移前的不存在路徑。證據：第 4.2、5.3、6 節。
6. **Phase 1 的 CSV 原型不需下載原始影片／權重。** 可先用 Celeb-DF、DFDC；正式固定 FAR policy 與獨立測試結論，須先補 manifest、calibration/evaluation split、指標和延遲契約。FF++ 需補 subtype/video identity 映射，優先找既有 metadata／匯出紀錄，不必先下載影片。

技能判斷：本次需求已完整，未啟動 `grill-me` 訪談、`to-spec`／`to-tickets` 發布流程或 `implement` 的程式修改／commit 流程。後續實作計畫只記於本文件。

## 2. Repo A Inventory

### 2.1 Git 與檔案

| 項目 | 實測結果 |
|---|---|
| Absolute path | `C:\Users\oplab\Desktop\DeepFake_Cascadeguard\far-frr-graph` |
| Remote（fetch/push） | `https://github.com/yucheng105/far-frr-graph.git` |
| Branch | `main` |
| Commit | `6f5e59945f33b9bf6f4ff0110d32423d2e713b9d` |
| 起始 working tree | `git -C far-frr-graph status --porcelain=v1` 無輸出 |
| 檔案規模（不含 .git） | 298 個檔案，1,110,829,417 bytes；220 CSV、35 PNG（包含分析圖） |
| Checkpoint／影片 | 本 Repo 搜尋未找到 `.pth/.pt/.ckpt/.mp4`；原始分數在第 5 節 |
| Dependency lock | 未找到全域 lock 或完整環境清單；只有子模組 requirements |

`far-frr-relation/src/` 的 `threshold_sweep.py` 使用 Python stdlib；`fit_far_frr_relation.py :: fit_relation/process_csv` 需要 NumPy、SciPy，`plot_fit_diagnostics.py :: write_fit_diagnostic` 需要 Matplotlib。`exhaustive-search` 的主要 sweep、窮舉、選擇流程使用 stdlib（csv/decimal/itertools）。`bmma_gpt_lookup/requirements.txt` 宣告 pandas>=2.0、numpy>=1.24、cvxpy>=1.4；`parallel-fusion/exhaustive/requirements.txt` 為 numpy>=1.23；`parallel-fusion/milp/requirements.txt` 另需 scipy>=1.9。這些是宣告與 import 審查，非已安裝證明。

### 2.2 共用主機環境（A/B 適用）

| 項目 | 實測／限制 |
|---|---|
| OS／Shell | Windows，PowerShell |
| PATH Python | `python.exe`、`python3.exe` 指向 WindowsApps alias；不能據此判定有可用專案 Python |
| Audit Python | 既有 Codex bundled Python **3.12.14**：`C:\Users\oplab\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`；僅以 `-B -` 執行 stdlib 唯讀檢查，不代表專案正式環境 |
| Audit runtime 套件 | metadata 查得 numpy **2.3.5**、pandas **3.0.1**；torch、cvxpy、scipy、scikit-learn、PyYAML、opencv-python、dlib、efficientnet-pytorch 均未在此 runtime 安裝 |
| 專案環境 | Repo 內未找到 Python venv/conda 環境；常見使用者 Python／conda 位置未找到。其他未列入位置或外部環境：**Unverified** |
| GPU | `nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader`：**NVIDIA GeForce RTX 3090，24,576 MiB，driver 617.42** |
| CUDA | `nvidia-smi` 顯示 **CUDA UMD 13.4**；這是 driver 能力，不能當作已安裝 CUDA toolkit 或 `torch.version.cuda`。專案 PyTorch/CUDA runtime **Unverified**；PATH 未發現可用專案 torch |
| Docker／WSL | PATH 未找到 docker；`wsl --list --verbose` 回傳 `Wsl/E_ACCESSDENIED`，無法盤點 WSL 內環境；未啟動任何服務 |
| C: 可用空間 | **1,846,258,061,312 bytes**（約 1.68 TiB；查詢瞬間值，兩 Repo 位於 C:） |
| D: 可用空間 | **993,650,978,816 bytes**（約 925.41 GiB） |
| 文件中外部資料路徑 | `B/setup_instructions.md` 的 `E:\114_IMProject\data` 與舊專案位置 `C:\Users\oplab\Desktop\Cascade-DeepfakeBench`，本次 `Test-Path` 均 False；目前 FileSystem drives 為 C:/D: |

建議後續分開分析環境與 DeepfakeBench 推論環境，先確認既有環境再規劃依賴；本次未建立環境。未發現需補 clone 的 Repo，因此沒有執行 clone，也不需要提供缺失 Repo 的下載指令。

## 3. Repo B Inventory

| 項目 | 實測結果 |
|---|---|
| Absolute path | `C:\Users\oplab\Desktop\DeepFake_Cascadeguard\Cascade-DeepfakeBench` |
| Remote（fetch/push） | `https://github.com/yucheng105/Cascade-DeepfakeBench` |
| Branch | `main` |
| Commit | `561744909c918617addbcbf1265f64a53654c978` |
| 起始 working tree | `git -C Cascade-DeepfakeBench status --porcelain=v1` 無輸出 |
| 檔案規模（新增文件前，不含 .git） | 267 檔，3,063,648 bytes |
| 環境 | 第 2.2 節；本機未完成 detector 執行驗證 |
| Dataset | `datasets/` 只有 readme；設定 `datasets/rgb`、`datasets/lmdb` 目前均無實際資料 |
| Dataset JSON | `preprocessing/dataset_json/` 只有 readme/.gitignore，無完整 FF++/Celeb-DF/DFDC JSON；`preprocessing/dataset_json_smoke/FaceForensics++.json` 存在，但只有 smoke metadata，不代表影格存在 |
| Weights | `training/weights/`、`training/pretrained/` 只有 readme；`xception_best.pth`、`ucf_best.pth`、EfficientNet-B4/Xception 預訓練權重均未找到 |
| Face predictor | `preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat` 未找到 |
| 其他 `.npy` | `training/lib/component/SRM_Kernels.npy` 是模型元件資源，不能視為 dataset landmark |

設定來源：`B/training/config/test_config.yaml`：`lmdb: False`，rgb=`./datasets/rgb`、lmdb=`./datasets/lmdb`、json=`./preprocessing/dataset_json`。`B/cascade/test.py :: project_path/load_config/build_model` 會解析路徑並載入權重；Xception/UCF 建構還先讀取 backbone 預訓練權重（`XceptionDetector.build_backbone`、`UCFDetector.build_backbone`），並非只補 final checkpoint 即可。

**依賴文件互不等價：** `B/requirements.txt` 只有 mpmath/numpy/pillow/setuptools/sympy/typing_extensions，沒有 torch/yaml/cv2 等核心依賴，不能單靠它安裝可執行環境。`B/install.sh` 宣告 torch 1.12.0+cu113、torchvision 0.13.0、numpy 1.21.5、pandas 1.4.2 等；`B/Dockerfile` 基底為 `pytorch/pytorch:1.12.0-cuda11.3-cudnn8-devel`，另安裝 Python 3.7 及一批套件。Docker base 和新設的 Python interpreter 是否使用同一套 torch 未驗證；未建置 image。`training/detectors/__init__.py` eager import 多個 detector，單模型實際需要的 import 閉包大於單一檔案。

**文件不能取代現況：** `docs/sbi/SBI_PHASE1.md` 記載過去已通過模型／資料檢查及 smoke run；目前 checkout 缺資料、權重及專案環境，故本機 readiness 仍為 Unverified。`training/check_sbi_phase1.py :: main` 是可保留的未來檢查入口，本次未執行。`setup_instructions.md` 提及的 4090／舊磁碟位置不等於本次實測的 3090。

外層 Repo：`C:\Users\oplab\Desktop\DeepFake_Cascadeguard`，origin=`https://github.com/maurice1117/DeepFake_Cascadeguard.git`，branch=`main`，SHA=`875bd515b8eea48b3cd2305e6299b270bebd6e94`；起始顯示 `?? Cascade-DeepfakeBench/`、`?? far-frr-graph/`。不要在外層誤加整個 nested Repo。

## 4. Existing Functionality Matrix

### 4.1 Repo A：Policy Optimization

以下路徑均以 `A/` 為根；每列列出主要輸入、輸出與實作限制。

| 功能 | 狀態 | Source / 函式 | 輸入 → 輸出；依賴與問題 |
|---|---|---|---|
| Score CSV 讀取／合併 | Implemented | `parallel-fusion/merge_scores.py :: read_csv/merge_dataset/index_by_sample_occurrence` | 9 raw CSV → 3 fusion CSV；stdlib。檢查數量、labels/status/分數，但用 `(sample_id, occurrence)` join，非真實唯一 ID；未驗證來源 dataset 欄位一致性 |
| FAR／FRR Curve | Implemented | `far-frr-relation/src/threshold_sweep.py :: read_scores/calculate_sweep`；`exhaustive-search/threshold-sweep/src/threshold_sweep.py :: calculate_far_frr/build_sweep_rows` | raw score → threshold/FAR/FRR CSV；後者目前寫死 SBI、20 點。前者可指定 input/output/steps；曲線圖與 fitting 可用 `fit_far_frr_relation.py :: process_csv`、`plot_fit_diagnostics.py :: write_fit_diagnostic`（NumPy/SciPy/Matplotlib）。既有 X/U curve 版本不一致 |
| EER Calculation | Implemented | `exhaustive-search/threshold-sweep/src/calculate_eer.py :: find_min_eer_entry/calculate_eer`；`bmma_gpt_lookup/build_lookup_table.py :: compute_eer` | sweep → min_eer_entries.csv 或 profile。前者選最小 abs(FAR−FRR) 格點再平均，且摘要未保留單模型 Threshold；GP 版本有交點插值／無交點 fallback；不可當相同估計法 |
| Exhaustive Search | Implemented | `exhaustive-search/threshold-sweep/src/create_all_permutations.py :: get_detector_permutations`；`calculate_permutation.py :: get_threshold_pairs/write_permutation_dataset_csv` | 3 detector 的長度 1..3 共 15 排列、門檻笛卡兒積 → cascade_threshold_sweep CSV；stdlib。不是 raw-score simulator；20 點三階段單資料集單排列約 210×210×20=882,000 候選，避免本次重跑 |
| Dual-threshold Optimization | Partial | `calculate_permutation.py :: calculate_cascade_far_frr`；`exhaustive-search/far-frr-tradeoff/src/calculate_tradeoff.py :: select_tradeoff_rows` | 邊際曲線 → FAR limit 下最小 FRR；最後一關強制 L=H；前關允許 L<=H。缺聯合樣本驗證及 expected latency |
| BMMA-GPT / GP Lookup | Partial | `bmma_gpt_lookup/build_lookup_table.py :: build_detector_profile/solve_gp_thresholds/select_best_ordering` | sweep+latency → GP 與 policy；NumPy/pandas/cvxpy。主要數學修正已加入；main 路徑失效、profile 來源混合、尚無樣本級驗證，詳 4.2 |
| Parallel Fusion | Implemented | `parallel-fusion/exhaustive/exhaustive_search.py :: exhaustive_search/evaluate_grid_point`；`parallel-fusion/milp/milp_solver.py :: solve_parallel_fusion` | fusion CSV → weights/beta、prediction/grid/summary CSV；NumPy，MILP 另需 SciPy。逐樣本 score fusion 存在；FAR/FRR 命名相反，目標是 FAR+FRR，沒有固定 FAR 比較契約 |
| Latency-aware Selection | Partial | `exhaustive-search/latency-oriented-selection/src/select_lowest_latency.py :: permutation_latency/select_lowest_latency_rows`；GP `select_best_ordering` | 平均 detector time 表+候選 → selection CSV；只加總所有 detector 平均成本，不乘 reach/exit rate；排序與固定 FAR 機制可重用 |
| Policy Table 輸出 | Partial | `exhaustive-search/accuracy-oriented-selection/src/select_global_best.py :: select_global_rows/write_dataset_result`；GP `build_lookup_table/main` | tradeoff → accuracy selection；GP → real_data_vip_settings.csv。輸出機制已存在，但 schema 未含版本／切分／模型雜湊，B 無 importer；現有表不是已驗收部署 artifact |
| Score-based sequential simulation | Missing | `calculate_permutation.py :: calculate_cascade_far_frr`、GP `compute_cascade_far_frr` 只接收 sweep/profile | 全 Repo Python 搜尋未找到依 joint raw scores 對每筆執行所有候選序列 routing 的 simulator；這是 Phase 1 最小新增功能 |

`far-frr-relation/src/analyze_k_product.py :: analyze_source` 已有 shifted `(FAR−alpha)*(FRR−beta)` 分析；`fit_far_frr_relation.py :: exp_func/fit_relation` 已有指數擬合。兩者是診斷工具，不必重新開發，也不是 GP 的 `conb=max(EER²,max FAR*FRR)` 之替代定義。

### 4.2 BMMA-GPT：Source 優先於舊 Review

`A/bmma_gpt_lookup/GP_LOOKUP_TABLE_REVIEW.md` 描述舊缺陷；目前 `build_lookup_table.py` 已有以下實作，不能原樣複製舊 Review 的未修正清單：

| 核對項目 | 目前 Source |
|---|---|
| 最後一關 | `solve_gp_thresholds` 加入 `ab_lb[n-1] == ab_ub[n-1]`；`_reverse_map_thresholds` 將最後門檻取平均為同一值；`_thresholds_valid` 要求最末近似相等 |
| RB<=1 | `solve_gp_thresholds` 限制 AB_lb/AB_ub >= conb，並限制 <=1−EPS |
| 門檻反查 | `_reverse_map_thresholds` 兩側皆用 `DetectorProfile.inv_far`；超出觀測範圍回傳 None，不靜默 clip。`inv_frr` 仍存在且會 clip，但目前此路徑未使用 |
| Solver | 檢查 `problem.is_dgp()`；ECOS 可用時優先使用；`optimal_inaccurate` 只檢查 total_far 容差後接受，尚未完整記錄／驗證每條約束殘差 |
| GP 後驗算 | `select_best_ordering` 呼叫 `compute_cascade_far_frr`，以插值邊際曲線差式值再篩 alpha/beta；不是 joint-score／樣本級驗算 |

GP 目前式子（`solve_gp_thresholds`）：令 AB=FAR，RB=conb/AB，最小化 `Σ_s (Π_{j<s} RB_L[j]) RB_H[s]`，限制 `Σ_s (Π_{j<s} AB_H[j]) AB_L[s] <= alpha`。它以正項乘積替代中間帶機率差；最後單門檻與機率約束已修，但仍是 surrogate。`combine_across_datasets(max)` 對 FAR、FRR 分別取跨資料集最大值，`compute_conb` 再取最大乘積；此 profile 不一定屬於任何真實 cohort。

**數學一致性結論：Partial。** `compute_cascade_far_frr` 的差式乘積，只有在相關條件分布可由邊際曲線代表等假設下才等於實際 routing。Detector 對同一影片的錯誤具有相關性的可能性未建模；GP 上界不能據此宣稱是任意 joint-score cohort 的可靠 FAR 保證。另外 A sweep 的 real exit 使用 `< L` 的機率，B 中間關實際使用 `<= L`，在邊界質量不為零時也會不同（第 6 節）。

**延遲不是 expected latency，也不是真正最壞單片 latency。** `build_detector_profile` 的 Tb 為各資料集「平均」延遲之最大值；`select_best_ordering` 使用 `sum(Tb)` 篩預算／排序。同一順序的門檻改變不會改變此成本。`_reverse_map_thresholds` 輸出的 `expected_latency_ms` 實際只是 Tb，欄名容易誤導。窮舉 latency selector 亦只用平均 time 加總。

**無法原樣重跑的入口：** `build_lookup_table.main()` 中 repo_root 指向 A 根目錄，但 8 份來源仍指 `A/results/20/sbi/...`、`A/models/xception/...`、`A/models/ucf/...`（2+3+3），實際都在 `A/exhaustive-search/threshold-sweep/` 下。第九份 `bmma_gpt_lookup/threshold_sweep_sbi_faceforensics_v1.csv` 存在。這不是本次要修的程式，列為 Phase 1 修正。

現有 `output/real_data_vip_settings.csv` 只有 forensic 的三列：UCF→Xception→SBI，system FAR≈0.190287、FRR≈0.479463，非逐樣本實測；`output/all_orderings.csv` 僅 2 bytes，沒有可讀 header/資料，不能視為完整診斷表。`ALL_ORDERINGS_ALPHA=0.05`；SECURITY_LEVELS 的 forensic alpha=0.3，故兩輸出的涵義也不同。`validate.py :: validate_lookup_table` 只檢查格式及門檻關係，不會證明校準、效能或部署一致性。

### 4.3 Repo B：Detector / Cascade Pipeline

以下路徑以 `B/` 為根。

| 功能 | 狀態 | Source / 函式 | 核對結果、輸入／輸出 |
|---|---|---|---|
| Video Identity Preservation | Partial | `training/dataset/abstract_dataset.py :: collect_img_and_label_for_one_dataset`；`cascade/data/video_identity_dataset.py :: VideoIdentityDataset` | 基底以 subtype label+原名組 video_name，與 frames/labels 一起 shuffle；wrapper 保留 opaque ID、檢查 labels、提供 indices/subset；但匯出 CSV 又丟失 subtype；讀圖失敗 fallback 有風險，詳 9 節 |
| Face Detection / Crop / Alignment | Implemented | `preprocessing/preprocess.py :: video_manipulate/extract_aligned_face_dlib/get_keypts` | 原片 → dlib 最大臉、5 點 alignment、256 crop、81 landmark、可選 mask；檔案輸出。不是 streaming face tracker；本機 predictor/data 缺失，未執行 |
| Xception Inference | Implemented | `training/detectors/xception_detector.py :: XceptionDetector.forward/build_backbone` | batch image → cls/prob/feat；prob=softmax[:,1]；cascade adapter 可用，缺本機權重／環境 |
| UCF Inference | Implemented | `training/detectors/ucf_detector.py :: UCFDetector.forward/build_backbone` | `inference=True` 使用 shared head 的 `prob_sha`，回 cls/prob/feat；不是把完整訓練重建分支拿來作 runtime 決策 |
| SBI Inference | Partial | `training/detectors/sbi_detector.py :: SBIDetector.forward/build_backbone`；`training/networks/efficientnetb4.py :: EfficientNetB4` | 本地 binary EfficientNet-B4，回 cls/prob/feat，可作單模型 inference；未接 cascade，缺 trained checkpoint；官方權重相容性未驗證 |
| Current Cascade Controller | Partial | `cascade/test.py :: main` | **固定兩階段 Xception→UCF**，明確拒絕其他 model_name；沒有通用 N-stage 控制器 |
| Early Exit Decision | Implemented | `cascade/video_scoring.py :: route_video_scores/route_final_stage/merge_cascade_decisions` | Stage1 real<=L / fake>=H，其他 uncertain；Stage2 單門檻；沒有 uncertain 時不建立 UCF 模型；subset 精準按 ID 選取 |
| Video-level Aggregation | Implemented | `cascade/video_scoring.py :: VideoScoreAccumulator.update/finalize`；`training/test.py :: aggregate_video_predictions` | 算術平均 sampled frame fake probability；前者驗證有限數／labels、保留 frame_count。單模型 CSV 未保留 frame_count |
| Detector-specific Preprocessing | Partial | `cascade/config.py :: validate_shared_input_contract`；`training/dataset/abstract_dataset.py :: load_rgb/to_tensor/normalize` | X/U 共用同 Dataset，檢查 resolution/mean/std/mask/landmark/video_mode/clip_size/frame_num.test。不是每個 stage 自動套專用 transform；SBI 設定不同 |
| Output / Metrics / Logging | Partial | `cascade/metrics.py :: operational_auc/final_classification_metrics`；`cascade/test.py :: print_stage_summary/print_final_summary`；`training/test.py :: test_epoch/write_prediction_csv` | cascade 有 stdout Stage AUC、REAL/UNCERTAIN/FAKE 數及全 cohort ACC/FAR/FRR；單模型有 CSV/log。cascade 缺持久化逐片 decision trace、runtime latency、policy ID 等 |
| Threshold Configuration | Implemented | `cascade/test.py :: parse_args`；`video_scoring.py :: validate_double_threshold/validate_final_threshold` | CLI 必填 low/high/final_threshold，前關 0<=L<H<=1，末關 0<=t<=1 |
| 外部 Policy Table | Missing | `cascade/test.py :: parse_args/main` | 無 policy file/security_level/version 載入；A 表無法直接送入 B |
| Streaming Window | Missing | `cascade/inference.py :: run_frame_detector_stage`；`video_scoring.py :: VideoScoreAccumulator.finalize` | 目前遍歷完整離線 loader 後聚合。可重用 accumulator/routing 語意，但缺 window/session ID、時間戳、跨窗狀態、queue/backpressure、tracker、deadline |

**UCF 是否沿用 Xception 前處理：是（目前這條 cascade）。** `cascade/test.py :: build_data_loader/build_subset_data_loader/main` 只建立一次 Stage1 Dataset，UCF 取同一物件的 uncertain subset。兩者 YAML 都為 256、mean/std=[0.5,0.5,0.5]、test frame_num=32、mask/landmark=False；config guard 檢查相容性。這保留同一批 sampled frames，但不是通用 detector adapter 能力。

**SBI Dataset 與推論須區分：** `training/dataset/sbi_dataset.py :: SBIDataset.__getitem__/collate_fn` 由真實影格與 landmark 生成一對 real/fake 訓練樣本；這不應直接替代離線測試 dataset。`SBIDetector.forward` 本身只吃已處理 image；`training/config/detector/sbi.yaml` 設 256、ImageNet mean/std、with_landmark=True、pretrained=null。不能讓 SBI 沿用 X/U 已正規化 tensor 而期待結果等價。

### 4.4 SBI 官方實作與本地 DeepfakeBench 的差異

僅唯讀查閱官方 `mapooon/SelfBlendedImages` Source，2026-10-30 讀取的 master（未 pin 遠端 commit；不宣稱完整逐行 diff）。未下載 Repo／模型。

| 面向 | 官方 Source | 本地 Source／整合含意 |
|---|---|---|
| Model / weights | 官方 `src/model.py :: Detector.__init__/forward` 直接包 EfficientNet-B4、advprop=True、兩類；回 logits | `B/training/networks/efficientnetb4.py :: EfficientNetB4.__init__` 替換 conv stem、_fc、另設 last_layer，`SBIDetector.forward` 回 dict；state_dict key/架構不同，不能直接 strict-load 官方 detector weights。[官方 model](https://github.com/mapooon/SelfBlendedImages/blob/master/src/model.py) |
| 單片推論／aggregation | 官方 `src/inference/inference_video.py :: main` 使用 RetinaFace，載 checkpoint['model']，影像 /255；同 frame 多臉取最大 fake score，再對 frames 取平均 | 本地 `preprocess.py :: extract_aligned_face_dlib` 取最大臉 dlib crop；dataset normalize 用 SBI YAML 的 ImageNet mean/std；`aggregate_video_predictions`／`VideoScoreAccumulator` 對 sampled crop 分數平均。預處理與 aggregation 不等價，CSV 來源須註明。[官方 inference](https://github.com/mapooon/SelfBlendedImages/blob/master/src/inference/inference_video.py) |
| 訓練資料 | 官方 `src/utils/sbi.py :: SBI_Dataset.__getitem__` 使用 landmark 與 retina bbox、crop/self blending | 本地 `SBIDataset.__getitem__` 用 DeepfakeBench frame/landmark 與 `sbi_api.py :: SBI_API`；不能僅憑同名 SBI 認定 scores 可互換。[官方 dataset](https://github.com/mapooon/SelfBlendedImages/blob/master/src/utils/sbi.py) |

這些差異支持「需獨立 SBI adapter／preprocessing profile」；**不能證明 A 的 SBI CSV 一定來自官方模型或本地重訓模型**，目前 raw schema 無 checkpoint、exporter commit、preprocess hash，來源仍 Unverified。

## 5. CSV Data Readiness

### 5.1 原始分數實測盤點

根目錄 `A/exhaustive-search/threshold-sweep/input-csvs/`，以下九檔均完整讀取（共 17,766 列）；ID 以字串處理、保留前導零。每檔 label 僅 0/1、status 全為 ok，fake_score 全部有限且落在 [0,1]，ID/label/score 無空值。`error_message` 全空為正常成功記錄。SHA-256 見附錄。

| Detector / 相對檔名 | Dataset 欄位 | 列數 | Real / Fake | Unique ID | 重複多出的列數 |
|---|---|---:|---:|---:|---:|
| `xception/xception_Celeb-DF-v2_predictions.csv` | Celeb-DF-v2 | 518 | 178 / 340 | 518 | 0 |
| `ucf/ucf_Celeb-DF-v2_predictions.csv` | Celeb-DF-v2 | 518 | 178 / 340 | 518 | 0 |
| `sbi/scores_visual_sbi_celebdf_v0.csv` | Celeb-DF-v2 | 518 | 178 / 340 | 518 | 0 |
| `xception/xception_DFDC_predictions.csv` | DFDC | 4,704 | 2,315 / 2,389 | 4,704 | 0 |
| `ucf/ucf_DFDC_predictions.csv` | DFDC | 4,704 | 2,315 / 2,389 | 4,704 | 0 |
| `sbi/scores_visual_sbi_dfdc_v0.csv` | DFDC | 4,704 | 2,315 / 2,389 | 4,704 | 0 |
| `xception/xception_FaceForensics++_predictions.csv` | FaceForensics++ | 700 | 140 / 560 | 280 | 420 |
| `ucf/ucf_FaceForensics++_predictions.csv` | FaceForensics++ | 700 | 140 / 560 | 280 | 420 |
| `sbi/scores_visual_sbi_v0.csv` | FF++ | 700 | 140 / 560 | 280 | 420 |

共用 12 欄 schema：`sample_id, dataset, label, detector_name, modality, fake_score, score_type, inference_time_ms, window_start_sec, window_end_sec, status, error_message`。detector_name 的 X/U 為小寫，SBI 為大寫；modality=visual、score_type=probability。score 高代表 fake；本地三模型皆 `softmax[:,1]`（第 4.3 節），但不等於已驗證每份歷史 CSV 的模型來源。

**Granularity：** X/U 的 schema 與 `B/training/test.py :: aggregate_video_predictions` 相符，該函式每 video_name 平均 frames，輸出 video-level。SBI 列數／ID 結構亦是 video-style export，但來源 exporter 未找到，聚合細節 Unverified。所有 CSV 均無 frame IDs / frame_count / sampling hash；X/U window 起訖為空，SBI 為字串 `N/A`，無真實 window/frame 時序，不能直接做 streaming/window-level 評估。沒有 split 欄；B 雖有 `mode='test'` 的出口，仍不足以證實歷史 CSV 的 split 或是否與校準重疊。

### 5.2 Join 檢查與可用程度

| Cohort | 本次唯讀核對 | 判斷 |
|---|---|---|
| Celeb-DF-v2 | 三份 ID 集合完全一致，518 個，labels 無衝突；SBI 原始列順序與 X/U 不同 | **可直接以 dataset+字串 ID 做探索性 join**，禁止用 row index zip |
| DFDC | 三份 4,704 個 ID 集合一致、labels 無衝突，現有列順序亦相同 | **可直接探索性 join**，仍以 key 驗證，不能假設順序永久相同 |
| FF++ | 700 occurrence keys 可對齊、labels 一致；140 個 fake IDs 各四次，real IDs 各一次 | **Partial**；需 subtype／原始相對路徑／opaque video_name 對照，不能以 occurrence 作已驗證 canonical ID |

FF++ 例：Xception `000_003` 有四個不同分數（0.9991116654、0.9947026893、0.9930417649、0.9771589302），不是可以任意刪掉的重複輸出。`B/training/dataset/abstract_dataset.py :: collect_img_and_label_for_one_dataset` 原本保有 subtype；`B/training/test.py :: sample_id_from_image_path` 只取父目錄 video stem，`aggregate_video_predictions` 排序後刪除 `_video_name`，使四種 manipulation 的 stem 在 CSV 中碰撞。這解釋 X/U 的設計風險；SBI 的四組順序仍需來源佐證。

`A/parallel-fusion/{Celeb-DF-v2,DFDC,FaceForensics++}_fusion.csv` 三檔皆已核對：按照 `merge_scores.py :: index_by_sample_occurrence` 的 key，三個分數／label 與九份目前 raw CSV **全部一致（0 mismatch）**。這只證明 fusion export 按現有 occurrence 規則產生；不能證明 FF++ occurrence 的真實 subtype 相同。Fusion schema 六欄，label 名為 `label(ground_truth)`，且移除了 latency/status/window/版本 metadata，須由 manifest 保留 lineage。

另外，以既有 summary 的 weights/beta 在記憶體重播目前 fusion CSV（未搜尋新參數）：`parallel-fusion/exhaustive/results/exhaustive_best_summary.csv` 的 Celeb-DF、DFDC、FF++ 分別得到 **(FP,FN)=(29,69)、(538,872)、(2,22)**，全部與表中計數一致；`parallel-fusion/milp/milp-results/milp_summary.csv` 的 DFDC 亦為 (538,872)。這證實這四列現有 policy 的樣本計數可重現；不證明泛化能力、獨立 test split 或 FF++ 真實身分 join，也不改變 parallel FAR/FRR 命名反轉的問題。

### 5.3 Sweep／版本差異（未修改或產生 CSV）

本次用 stdlib 在記憶體依 `score<t` 計 FAR、`score>=t` 計 FRR，逐點比較目前 raw 與既有單模型 sweep；一般容差 5.1e−7（六位小數），legacy 四位小數則 5.1e−5。沒有呼叫既有分析腳本。

| Detector | `models/` 及 `results/20/` 各 dataset 相符點數 | `results/1000/` 相符點數：Celeb / DFDC / FF++ | 結論 |
|---|---|---|---|
| Xception | 三 dataset 均只有 **1/20** | **38/1000、3/1000、8/1000** | 舊曲線與目前 raw 不同；不能沿用為同一版本的校準依據 |
| UCF | 三 dataset 均只有 **1/20** | **4/1000、2/1000、3/1000** | 同上；原因尚未證明，不能直接歸因模型／split／程式錯誤 |
| SBI | `results/20/` 三 dataset 均 **20/20** | 三 dataset 均 **1000/1000** | 這些單模型 sweep 與目前 raw 在所查門檻一致；不代表校準/test 獨立 |

上述路徑皆相對 `A/exhaustive-search/threshold-sweep/`。FF++ t=0.5 實例：目前 Xception raw FAR=8/560≈0.014286、FRR=26/140≈0.185714，而 `results/20/xception/threshold_sweep_xception_faceforensics.csv` 為 0.016071／0.235714；UCF raw=12/560≈0.021429、14/140=0.1，舊表=0.0125／0.178571。差異大於小數精度，不可視為單純 rounding。

SBI 特別注意：

- `models/sbi/threshold_sweep_sbi_v1.csv` 的 20 點，在交換 FAR/FRR 後與目前 `input-csvs/sbi/scores_visual_sbi_v0.csv` 全部符合四位小數容差；原樣 0/20。方向確實相反。
- `bmma_gpt_lookup/threshold_sweep_sbi_faceforensics_v1.csv` 20 點方向正確，與上述目前名為 v0 的 700 筆 raw 全部符合，內嵌 latency=308.7、n_samples=700。但程式註解宣稱的 `scores_visual_sbi_ffpp_v1.csv` 不存在；**數值對得上不代表 v1 來源鏈已證實**。
- `models/sbi/threshold_sweep_sbi_v0.csv` 僅 9 點、n_samples=280、latency=1035.1，FAR 隨 t 下降／FRR 上升；即使交換也無法與現有 700 筆 raw 對齊。應保留為不同 cohort 的歷史 artifact，不能就地交換後混用。
- combined 曲線、cascade 候選、tradeoff、selection、fit 皆是下游衍生物；本次未對每個大型結果表逐列重新驗證。上游 X/U 不一致時，其下游不得視為已驗證，檔案存在不等於 lineage 成立。

### 5.4 Latency 資料

原始 CSV 的 latency 皆為有限正值；單位欄名 ms。下表為本次直接讀 raw 的算術平均，與 A 根目錄的 time 表分開列示。

| Dataset | Raw Xception ms | Raw UCF ms | Raw SBI ms | `*_average_inference_time.csv` X / U / SBI ms |
|---|---:|---:|---:|---|
| Celeb-DF-v2 | 151.84003 | 310.93869 | 2676.15058 | 158.79 / 166.85 / 2676.15 |
| DFDC | 125.73988 | 267.95466 | 365.50336 | 106.22 / 170.81 / 365.5 |
| FF++ | 151.00071 | 323.35794 | 308.70357 | 114.70 / 224.81 / **1035.13** |

證據：第 5.1 節九檔、`A/{xception,ucf,sbi}_average_inference_time.csv`、`A/bmma_gpt_lookup/detector_latency.csv`。GP latency 表只有 SBI Celeb/DFDC，FF++ 正確來源檔內嵌 308.7；目前 GP 的 SBI Tb 仍由 Celeb 的 2676.15 主導。

`B/training/test.py :: test_one_dataset` 在 model forward 前後同步 CUDA，以 batch forward ms / batch_size 分攤到 item；`aggregate_video_predictions` 對 video 累加，**不含 decode、face detect、data loading、H2D、模型載入及 queue**。歷史 raw 是否由同版計時器匯出尚未證實。SBI Celeb 最大單列達 125,412.8ms，需記錄 warmup/outlier、硬體、batch 等資訊後再比較；本次未刪除異常值。

`A/exhaustive-search/threshold-sweep/src/calculate_avg_time.py :: main` 仍引用不存在的舊 Xception scores_visual_* 檔名，且輸出在 threshold-sweep 根目錄，不是 A 根目錄。其註解把「各資料集均值再平均」叫 Micro、「全列平均」叫 Macro；與常用名稱相反，需明列 averaging_method，不能單看欄名。

### 5.5 Data Lineage

```text
外部推論／匯出（模型、checkpoint、preprocess、split 版本尚缺 manifest）
  └─ A/exhaustive-search/threshold-sweep/input-csvs/{xception,ucf,sbi}/*.csv
      ├─ far-frr-relation/src/threshold_sweep.py :: calculate_sweep
      │    └─ results/1000/<detector>/threshold_sweep_*.csv
      ├─ threshold-sweep/src/threshold_sweep.py :: build_sweep_rows（目前限 SBI）
      │    └─ 20-point sweep（入口目前寫在 threshold-sweep 根目錄）
      │         └─ [須核對版本／位置] results/20/<detector>/...
      │              └─ calculate_permutation :: calculate_cascade_far_frr
      │                   └─ far-frr-tradeoff :: select_tradeoff_rows
      │                        ├─ accuracy-oriented-selection :: select_global_rows
      │                        └─ latency-oriented-selection :: select_lowest_latency_rows
      ├─ [sweep/profile 對應需修復] bmma_gpt_lookup :: build_detector_profile
      │    └─ solve_gp_thresholds → 曲線差式驗算 → build_lookup_table → output/*.csv
      └─ parallel-fusion/merge_scores.py :: merge_dataset
           └─ *_fusion.csv → exhaustive_search / solve_parallel_fusion
                └─ per-sample predictions + grid / summary
```

此圖表示程式依賴與預期資料流，**不是每個已存檔結果都成功溯源的證明**。斷點：X/U raw→舊 sweep、SBI v0/v1 命名、GP 舊路徑、time 表版本、split／checkpoint metadata。下階段以 raw+manifest 作唯一來源，新增版本化衍生物，不覆蓋舊資料。

## 6. Metrics & Contract Inconsistencies

### 6.1 現況比較

| 面向 | A 序列／GP | A Parallel | B |
|---|---|---|---|
| GT／方向 | real=0、fake=1；分數越高越 fake | 同左 | 三 detector softmax[:,1]；推論把非零原始 subtype label 二值化 |
| FAR | fake→real / N_fake = **FNR** | real→fake / N_real = **FPR** | `cascade/metrics.py :: final_classification_metrics` 同 A 序列 |
| FRR | real→fake / N_real = **FPR** | fake→real / N_fake = **FNR** | 同 A 序列 |
| 門檻 | sweep 假判 `s>=t`；差式的 real exit 是 `s<L` | fusion fake `q>=beta` | cascade 中間 real `s<=L`、fake `s>=H`；末關 real `s<t`、fake `s>=t` |
| Legacy metric 邊界 | 各模組需辨識 | 同上 | `training/metrics/utils.py :: get_test_metrics` ACC 用 **s>0.5**，與 cascade equality 不同 |
| Metrics cohort | 邊際 profile／公式估計，可能跨 dataset max | 每 dataset 全部 fusion rows | Stage1 AUC 全 cohort；Stage2 AUC 僅 uncertain subset；最終 ACC/FAR/FRR 原始全 cohort |
| Video ID | raw stem；FF++ 重複 | 同 stem/occurrence，讀取器不保證唯一 | cascade opaque video_name 保 subtype；CSV 丟 subtype；legacy metric 用 path 父目錄分組會把 FF++ subtype 混在一起 |
| Latency | stage 平均時間相加 | solver wall time 不是 detector pipeline latency；未建模 concurrent latency | training CSV 為 allocated forward ms；cascade 沒 latency instrument |
| Model output | CSV probability | 三個分數加權 q | detector dict `cls/prob/feat`；UCF inference 取 shared prob |

Source：`A/exhaustive-search/threshold-sweep/src/threshold_sweep.py :: calculate_far_frr`；`A/parallel-fusion/exhaustive/exhaustive_search.py :: evaluate_grid_point`；`A/parallel-fusion/milp/milp_solver.py :: solve_parallel_fusion`；`B/cascade/video_scoring.py :: route_video_scores/route_final_stage`；`B/training/metrics/utils.py :: get_test_metrics`；`B/training/test.py :: aggregate_video_predictions/test_one_dataset`。

FAR+FRR 的和在交換名稱後不變，所以既有 parallel 的無約束「錯誤率和」目標可參考；**固定 FAR 約束的可行集合會改變**，不能只將圖例改名就認為比較公平。`parallel-fusion/milp/milp-results/milp_summary.csv` 目前只記錄 DFDC、optimal=False、time_limit、solution_source=grid_seed，不能宣稱所有資料集取得 MILP 最優解；其他 prediction 檔存在也不補足該證據。

### 6.2 Proposed Data Contract v1（提案，未改既有定義）

| 欄位／規則 | 契約 |
|---|---|
| 身分 | `dataset_id` canonical enum（ffpp / celeb_df_v2 / dfdc）；`video_id` 為包含來源 subtype／相對來源身分的 opaque string，保留前導零；跨資料集 key=(dataset_id,video_id)，不可只取 filename stem |
| 記錄粒度 | `granularity=video/frame/window`；video-level 用明確 sentinel 的 window_id；frame/window 需 frame_id 或 window_id 及有效時間戳。record key 另加 detector_id/model_version，確保每個粒度每模型唯一 |
| Label | `label=0 real, 1 fake`；source_manipulation 另外存，不把 subtype 數字直接當 binary label。join 必須逐筆核對 label |
| 分數 | `fake_score` 為有限 [0,1] probability；higher_is_fake=true；保留原始字串精度；不可默認重複 ID 的平均或刪除 |
| 聚合 | `aggregation=mean_sampled_frame_probability` 等明確值；保留 frame_count、sampling_profile/hash、preprocessing_profile/hash，模型間可比較的 cohort 必須相同 |
| Split | `split_id`、calibration/evaluation role、split manifest hash；與 policy tuning cohort 不重疊。以 source video／identity／manipulation family 分組避免相關樣本跨 split |
| 模型 provenance | detector_id、model_version、checkpoint SHA-256、exporter Repo+commit、run_id、source CSV SHA-256；未知欄位明示 unknown，不推測填值 |
| Latency | `latency_ms`、`latency_scope`（例如 model_forward_allocated_per_video）、hardware_id、batch_size、warmup 等；null 表未知，不能當 0；不要把 forward cost 當端到端 latency |
| Status | status=ok/error/skipped 及 error_message；缺失模型／分數不得靜默 inner join 丟樣本；預設 fail validation，或用事先宣告的缺失處理政策並回報 coverage |
| Join | dataset aliases 顯式映射；依 canonical key full coverage 檢查。FF++ occurrence mapping 可作帶 hash 的暫時 legacy manifest，需來源證實，不能當永久 identity |

現有九檔可透過未來 adapter+sidecar manifest 轉讀，不需更動原始檔。入口可重用 `merge_scores.py :: read_csv/validate_score` 的驗證概念，不能原樣沿用其 occurrence identity 作正式契約。

### 6.3 Proposed Policy Contract v1（提案）

- Header：schema_version、policy_id/version、optimizer/version、source Repo commits、input/split manifest hashes、calibration dataset/cohort、score/aggregation/preprocessing versions、metric_definition_id、target_far、可選 target_frr、objective、latency_scope/model、evaluation_status。
- Stages：連續 stage_index、detector_id/model_version/checkpoint hash、threshold_low/high；末關改用明確 `final_threshold`，可兼容表中 L=H；非末關必須 0<=L<H<=1。政策至少一關、模型不得未宣告重複；數值須有限；非法、不相容或缺模型時拒載。
- **建議沿用 B runtime 邊界：** 非末關 `s<=L → REAL`、`s>=H → FAKE`、`L<s<H → NEXT`；末關 `s<t → REAL`、`s>=t → FAKE`，不允許末關 uncertain。以明文 comparator/version 避免 A sweep 的 `<L` 被誤當 `<=L`。
- 統一 fake positive：TP=fake→fake、FN=fake→real、FP=real→fake、TN=real→real；FAR=FN/(TP+FN)=FNR，FRR=FP/(TN+FP)=FPR；輸出一併保留 TP/FN/FP/TN、分母與 explicit FPR/FNR。缺某類時 rate=null，並註明；不能當 0。
- 候選評估必須用 **joint score cohort 逐樣本**：每筆只由實際走到的 stage 決定 final label；system metrics 在原始 cohort 分母計算。stage metrics 清楚標註 reached cohort，不平均各 stage 的 FAR/FRR。
- 預測成本：固定 stage cost 時 `E[C]=Σ_s reach_rate_s*cost_s`，其中 reach_rate 從共同樣本 routing 取得；若有逐片成本，先對每片加總實際經過 stages 的成本再取平均。另列 full-path cost／p95 等，不冒稱 full-path mean cost 為嚴格 worst-case。
- Parallel 比較需同一 cohort、split、target FAR、score version；單 GPU 依序執行 cost 可作 sum baseline，真平行的 max(cost) 只能作理想假設，正式 latency 必須實測資源競爭與 overhead。
- Artifact 同時輸出估計法（gp_surrogate／marginal_curve／sample_replay）、empirical FAR/FRR、exit/reach counts、coverage、latency provenance；不能用同一 system_total_far 欄位不加註混裝不同估計法。
- B 載入前驗證 contract，並保存 per-video trace：policy_id、video/window ID、各 reached score、boundary decision、exit_stage、final_label、成本／error。A 與 B 以相同 fixture 驗證邊界、terminal 決策及 cohort conservation。

## 7. Proposed File Organization

### 7.1 現有結構摘要

```text
A/
  far-frr-relation/                 # fitting、curve diagnostics、K product
  exhaustive-search/
    threshold-sweep/{src,input-csvs,models,results}
    far-frr-tradeoff/{src,tests,results}
    accuracy-oriented-selection/{src,tests,results}
    latency-oriented-selection/{src,tests,results}
  bmma_gpt_lookup/{build_lookup_table.py,validate.py,output,...}
  parallel-fusion/{merge_scores.py,*_fusion.csv,exhaustive,milp}
  *_average_inference_time.csv
B/
  cascade/{data,tests,config.py,inference.py,video_scoring.py,metrics.py,test.py}
  training/{config,dataset,detectors,networks,metrics,trainer,weights,pretrained,...}
  preprocessing/{preprocess.py,rearrange.py,config.yaml,dataset_json,...}
  datasets/                        # 本機只有說明
  analysis/                        # DeepfakeBench 歷史分析工具
  docs/sbi/                        # SBI 實驗說明／歷史驗收
```

### 7.2 建議增量結構（未建立）

```text
A/
  contracts/                       # schema、metric/comparator 文件、small fixtures
  data/manifests/                  # 指向 immutable raw export，hash/provenance/splits
  simulation/                      # CSV score reader、sample routing、cost/trace
  policy/                          # 共用候選結果介面、固定 FAR 選擇、GP adapter
  benchmarks/                      # sequential/parallel/single-model 公平比較
  artifacts/<run_id>/               # versioned sweep/policy/metrics，不覆寫歷史
  [保留現有研究目錄]
B/
  cascade/
    adapters/                      # X/U/SBI input/output profile
    policy/                        # contract loader/validator
    controller.py                  # 未來 N-stage controller
    offline/                       # runner，重用既有 Dataset/Subset
    streaming/                     # 後續 phase；window/queue/state
    tracing/                       # decisions/errors/runtime timing
    [保留現有 data/inference/video_scoring/metrics/tests]
  [保留 training/preprocessing/datasets/docs]
```

### 7.3 保留／可能合併／搬移建議及風險

| 建議 | 保留／候選範圍及理由 | 風險 |
|---|---|---|
| 保留 raw、歷史曲線及結果 | A 九份 raw、models/results、GP output、fusion output 一律保留；新 artifact 用 manifest 區隔，便於溯源 | 低；儲存成本可控，歷史數值需標註未驗證 |
| 保留 sweep/fitting/EER | `far-frr-relation/src/*` 與 `calculate_eer.py` 可重用；未來收斂共同 loader、EER method | 中；入口路徑／精度／EER 定義不同，不能先機械搬移 |
| 合併重複 score/sweep loader | stdlib readers、GP pandas loader、fusion readers 有重複驗證；以 adapter 接共同 contract，先不刪舊入口 | 中；schema、容差、空值及 error 行為可能改變 |
| 保留窮舉與選擇器，替換上游結果來源 | `get_detector_permutations/get_threshold_pairs/select_tradeoff_rows/select_global_rows` 已有核心功能，不重寫搜尋排名 | 中；須增加 sample metrics/成本欄位，舊 schema 不可靜默混讀 |
| GP 保留研究路徑 | 保留 `solve_gp_thresholds` 與 diagnostics，作 candidate proposer，最後樣本重算才可輸出正式 policy | 中；surrogate 可能排除可行解，需記錄求解狀態及估計偏差 |
| Fusion 保留在 A | exhaustive/MILP 已有可比較 baseline；修復指標 adapter、split，再擴固定 FAR | 中；MILP epsilon／timeout／最佳性不可混淆；FF++ join 仍需證據 |
| 保留 B detector 與 preprocessing | 不搬 `training/detectors/*`、networks、dataset、preprocess；新 adapter 重用 registry/forward | 高（若搬移）；現有 import/sys.path/config/checkpoint key 耦合強 |
| 保留 B identity／routing／tests | `VideoIdentityDataset`、`VideoScoreAccumulator`、routing/metrics、cascade tests 是整合基礎 | 低至中；generic controller 仍需處理 fallback identity 與 detector preprocessing |
| B/analysis 可考慮未來轉 A | 只有確認通用、無訓練依賴的圖表分析才搬，模型 feature/heatmap 保留 B；本次不列為必要工作 | 中至高；目前相對 import/path 與 DeepfakeBench metric 語意可能不同 |
| 不立刻建立跨 Repo Python package | 先交換版本化檔案+契約 fixtures；共用邏輯有穩定需求後再抽小套件 | 低；避免 A 被 torch 綁定、B 被 cvxpy 綁定 |

## 8. Integration Architecture

**雙 Repo 分工適合現有 codebase。** A 負責 score processing、simulation、optimization、benchmark；B 負責資料前處理、detector、runtime、trace。現有分工證據分別為 A 的 `calculate_permutation/build_lookup_table/parallel-fusion` 與 B 的 `training/test.py/cascade/test.py/preprocess.py`。

```text
B：Offline detector export
   identity + detector-specific preprocessing + probabilities + forward timings
                      │ immutable scores + run manifest
                      ▼
A：contract validation / split validation / canonical join
   ├─ per-sample sequential replay + actual exit/reach statistics
   ├─ existing exhaustive / fixed-FAR selection / GP candidate proposal
   └─ single detector / parallel fusion comparison on same cohort
                      │ versioned policy + metrics + calibration manifest
                      ▼
B：policy loader → compatible adapters → N-stage controller → trace/metrics
                      │ independent evaluation export
                      └────────► A benchmark/report
```

Cross-repo dependencies：共同 schema/comparators/label/aggregation，detector/model/preprocess ID、可用 checkpoint、policy 所用 sampling profile，以及 evaluation manifest；**不需要跨 Repo 直接 import training 或 solver**。以 policy version、source SHA、輸入 hashes 保持可重現。先用固定小型 score fixtures 檢驗 A replay 與 B routing 相同，再整合第三模型；不要先做 streaming 或搬移目錄。

Streaming 適配判斷：現有純 routing 函式與累計分數概念可重用；`run_frame_detector_stage` 與全影片 `finalize()` 不具視窗 flush／狀態生命週期，`cascade/test.py` 每次在 stage 之間載入／釋放模型也不等於常駐即時服務。Phase 3 需另外設計 shared decoded crops、model-specific transforms、window aggregation、模型常駐及排程，不能只在現有 loop 加攝影機讀取就稱支援 streaming。

## 9. Known Risks

| 優先度 | 風險／證據 | 最小處理建議 |
|---|---|---|
| P0 | 指標名稱反轉：parallel `evaluate_grid_point/solve_parallel_fusion` vs sequential `calculate_far_frr` | 先使用 explicit confusion counts/FPR/FNR，再映射 CASCAde FAR/FRR；固定 FAR 比較前驗收 |
| P0 | 邊際乘積當真實系統值：`calculate_cascade_far_frr/compute_cascade_far_frr` | 逐樣本 joint-score replay；將 GP／曲線估計另列，不作實測保證 |
| P0 | FF++ stem collisions 與 occurrence join：`sample_id_from_image_path/index_by_sample_occurrence` | 取回 subtype/video_name/export mapping；不得去重、平均或任意順序配對 |
| P0 | X/U sweep 版本不符、split 未證實：第 5.3 節；所有 raw 缺 split/model hash | 以 hash manifest 鎖來源、補 split，未來才產生新版本衍生結果 |
| P1 | Runtime identity 在缺圖時可能失真：`B/training/dataset/abstract_dataset.py :: __getitem__` 讀圖失敗會 `return self.__getitem__(0)`，wrapper 仍附原 index 的 video_name | 後續改為明確失敗／帶真實 sample identity 的 error record；目前 wrapper 並不能保證缺圖情境的身分正確 |
| P1 | Frame sampling 與註解不符：`collect_img_and_label_for_one_dataset` 先把 total_frames 設為 frame_num，再算 step，超長 frame list 在 image mode 實際取前 frame_num 筆 | 紀錄 sampling profile；確認既有 scores 是否採同邏輯；修正前須評估結果版本差異，本次不改 |
| P1 | GP 入口失效與舊輸出：`build_lookup_table.main` 舊路徑、all_orderings 2 bytes | 修復配置入口／manifest 後另 run_id 輸出；不要覆寫舊表或盲目放寬 alpha |
| P1 | Latency 混用舊硬體、不同 scope／batch；`expected_latency_ms` 名稱誤導 | 先用同來源逐片成本作 predicted cost，實測 runtime 另報；保留 outlier |
| P1 | B 缺資料、權重、依賴；文件歷史成功不等於本機 ready | Phase 1 純 CSV 不依賴此阻礙；Phase 2 先補最小合法 inference 環境／必要測試 cohort |
| P1 | SBI 官方與本地架構/前處理/aggregation 不同 | adapter 與 checkpoint provenance 必須顯式區別；不要直接移用官方權重／threshold |
| P1 | Legacy video AUC 以 path parent 分組：`B/training/metrics/utils.py :: get_test_metrics` | benchmark 用 opaque ID 的 `VideoScoreAccumulator`；不要與混組 legacy AUC 混比 |
| P2 | 一次重跑三階段 exhaustive 會大量產生／覆寫結果 | Phase 1 候選格點／小 cohort smoke 起步，串流評估候選並使用新 artifact 路徑 |
| P2 | GP validation 格式通過不代表效能通過；NaN 等格式需更嚴格驗證 | policy schema 要求 finite、連續 stage、已知模型、final termination，另做獨立 sample replay |
| P2 | MILP summary 只 DFDC 且 time limit、grid seed | 只報可驗證的 incumbent，不宣稱全域 optimum；保存 bound/gap/status/source |
| P2 | Nested Repo 邊界／跨環境 Python 版本不同 | Git 操作指定 Repo；A 和 B 依賴分開，不因 docs 工作安裝／重構所有環境 |

本次未執行套件測試或 detector inference；不能據靜態 Source 評估實際 GPU throughput、checkpoint accuracy 或所有 error paths。既有 tests 保留：`A/exhaustive-search/{far-frr-tradeoff,accuracy-oriented-selection,latency-oriented-selection}/tests/`、`A/parallel-fusion/{exhaustive,milp}/tests/`、`B/cascade/tests/{test_config,test_video_identity_dataset,test_video_scoring}.py`。本次驗證是完整 raw CSV/joins/sweep/latency 的唯讀核對，不是這些 suite 已通過的宣稱。

## 10. Phase 1 Implementation Plan

### 10.1 Readiness gate

| 項目 | 能否直接開始 | 最小補充項目 |
|---|---|---|
| CSV-based Cascade Simulation | **可以開始 Celeb-DF / DFDC 探索原型；現有序列程式不能直接充當 simulator** | 實作 thin raw adapter + joint-score replay，採明確 B 邊界；補 split/provenance 才能做正式評估。FF++ 暫緩正式結果直到 identity mapping 可證實 |
| Fixed FAR Policy Optimization | **可重用 selector，但不能直接沿用舊候選結果** | 以 replay confusion counts 建候選；先統一 FAR=FNR，清楚記錄 target/cohort／可行與不可行；獨立 calibration/evaluation manifest |
| Policy Table Generation | **輸出工具可重用，部署表尚未就緒** | Contract v1、模型／preprocess／split hashes、候選 empirical metrics、schema 驗證與 B importer；GP 保留 candidate proposal 角色 |
| Sequential vs Parallel Comparison | **可開始同一 CSV cohort 的 accuracy/cost 原型；不能直接比較舊 summary** | 指標換名、同 cohort/split/target、parallel 固定 FAR 擴充；一致 latency scope。真實端到端 latency 待 B 實測 |

### 10.2 最優先三項工作（依依賴順序）

1. **鎖定資料與契約。** 對九份 raw 建 immutable hash/provenance/split manifest，先交付 Celeb-DF／DFDC validated join，定義 FAR/FPR/FNR、score direction、精度、boundary；請資料來源提供 FF++ subtype 對照與 SBI exporter/checkpoint 版本。驗收：join 不丟列、labels 一致、重複／缺失會報錯；calibration/evaluation 無 group overlap，無證據者標 unknown。
2. **完成逐樣本序列 replay 與成本驗算。** 重用候選排列／門檻枚舉、B routing 語意；輸出每筆 exit_stage/final decision、stage reach、system confusion counts、predicted mean cost。驗收：一關退化成單門檻 baseline；無 uncertain／全 uncertain／邊界相等／缺分數／最後必決的 fixture；所有樣本恰好一個 final decision；同 fixture A/B 相同。保留曲線及 GP 作對照，量化偏差。
3. **串接 fixed-FAR selection → versioned policy → 同 cohort benchmark。** 重用 tradeoff/global/latency selector 及 parallel 評估；修 GP 路徑僅作獨立 adapter 工作，不能阻擋 CSV baseline。驗收：calibration 上挑 policy 後凍結，再在獨立 evaluation 上報 empirical FAR/FRR、exit/cost、coverage；無可行解明示 infeasible；A policy 與 B loader fixture 相容。Parallel 在同一 target FAR 下比較，另列 unconstrained FAR+FRR baseline。

依賴：工作 1 → 2 → 3。最小實作 seam 為「validated score table + policy → per-sample decisions/metrics/cost」，可在 CPU、小 fixture 完成，不需 detector/GPU；不要為此重寫已有 fitting、solver、detector 或 preprocessing。B generic N-stage/SBI integration 屬後續 Phase 2；streaming、real-time optimization、XAI 屬 Phase 3–5。

**重用／不需重做：** raw readers 的基礎驗證、threshold sweep／plot/fitting、EER、15 排列與 threshold enumeration、固定 FAR selection、parallel weighted fusion／MILP 求解、Xception/UCF/SBI detector forward、B identity wrapper／subset／mean aggregation／routing／metrics。要補的是契約、identity lineage、逐樣本 simulator、reach-aware cost、policy importer／trace；不是全面重構。

**資料下載決策：** Phase 1 目前的 Celeb-DF、DFDC CSV 已足夠驗證 score-based 行為與探索候選，無需原始影片或 checkpoint。正式研究數值需要獨立 split／來源證據；FF++ 先補 metadata，若 metadata 已遺失且需重新匯出分數，才評估取得所需測試樣本與對應權重。Phase 2 實際 detector 驗收目前缺本機影格／權重，需另行準備最小必要 cohort；SBI 訓練則另需真實影格／landmark，不能把其資料需求誤當 Phase 1 必要條件。

## Appendix A — Audit Method & Reproduction Boundaries

- Git：各 Repo 的 `remote -v`、`branch --show-current`、`rev-parse HEAD`、`status --porcelain=v1`；未 fetch，不改 branch/index。
- 檔案：先讀 README、既有 `docs/sbi/*`、cascade milestone、GP review，再核對上述 Source；以 `rg --files`／targeted reads 盤點。workspace 與父層未找到適用 AGENTS.md。
- 環境：`Get-Command`、`Get-PSDrive`、`Test-Path`、`nvidia-smi`、WSL inventory 嘗試；Python 以既有 audit runtime 的 `importlib.metadata` 查版本，不 import torch 或載模型。
- CSV：Python `-B -`、stdlib `csv/collections/math/statistics/hashlib`，完整讀取九份 raw，檢查 schema、空值、label/status、finite score、ID duplicates、join sets/labels、fusion occurrence values、latency stats；讀既有單模型 sweep 在記憶體比較，不寫衍生 CSV、不執行任何 Repo 分析入口。大型 permutation tables 未全面重算。
- 外部資訊：只查 SBI 官方 Source 以比較實作，連結見 4.4；未對 A/B 遠端最新狀態作推測。
- 完成後檢查 A working tree 無變更，B 只有本文件 untracked；未執行測試而產生 cache／logs。
- 文件檢查：十個要求章節齊全、九份原始 CSV SHA-256 與附錄逐一相符、code fences 成對、無行尾空白；上述四列 fusion policy 亦以現有參數完成唯讀計數核對。

## Appendix B — Raw Score SHA-256 Manifest

以下相對路徑根為 `A/exhaustive-search/threshold-sweep/input-csvs/`；hash 只識別本次檔案內容，不代表模型／split 來源已證實。

| 檔案 | SHA-256 |
|---|---|
| `xception/xception_Celeb-DF-v2_predictions.csv` | `84da68a713944af89982d493ed01a63c10aeeff3b217a7c9ccbec8d0c3ecbd8d` |
| `xception/xception_DFDC_predictions.csv` | `da9c318601d58643e2316f16cbcc8250de3abd647ee4438ce3fd641ae3292f72` |
| `xception/xception_FaceForensics++_predictions.csv` | `d095c4b44780379a275a034913445eaa3683694d7d109a5a434391810b3c3712` |
| `ucf/ucf_Celeb-DF-v2_predictions.csv` | `21e66f43bdceba818d3b9993d8daee381b0b02f547e5a3275ae3896483fdfdd7` |
| `ucf/ucf_DFDC_predictions.csv` | `b8327265e0349f7962de7bc5cb60d525497124e14cf59a8729e8d0285a8837d4` |
| `ucf/ucf_FaceForensics++_predictions.csv` | `e68b06fc73471de15846b3956334ee5e9f3968eb88de05a7f04e6bbba1ff47d2` |
| `sbi/scores_visual_sbi_celebdf_v0.csv` | `b27845fc951fe2982fc904650369404ae7d921a3e5b45f4085bd5938bf9da267` |
| `sbi/scores_visual_sbi_dfdc_v0.csv` | `5bfa62a0f7065a4f7377b2f86b54e7735a32c481561de7a8b383268d6eaa7d08` |
| `sbi/scores_visual_sbi_v0.csv` | `9caa573dcd7afab9252eb4cb9d7b69e78c0cdbd0a386122e16da2877e0c30898` |
