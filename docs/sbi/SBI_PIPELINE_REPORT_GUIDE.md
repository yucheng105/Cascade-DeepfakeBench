# SBI Pipeline 組內報告準備稿

整理日期：2026-09-16。以下以目前 `codex/sbi-phase1` 分支、工作目錄與現有 log 為準；「已提交」與「本機尚未提交」分開列出。

## 總結

SBI（Self-Blended Images）用真實臉部影格及 landmark 即時產生合成的假臉影格，訓練真假二分類器。這次已修正 SBI 模型設定與 EfficientNet-B4 權重載入、準備 FF++ RGB 資料路徑與 JSON 清單，並加入訓練前的模型／資料檢查。現在已有 smoke run，以及 FF++ 訓練和 Celeb-DF-v2 測試的本機紀錄。**目前 log 完整記錄 Epoch 0～37，並在 Epoch 38 中途停止，不能說 50 個 epoch 已完成。**

## Pipeline 怎麼走

```text
FF++ c23 真實影片的裁切影格 + 同影格 81 點 landmark + train JSON
    → SBIDataset 載入真實影格
    → SBI_API 隨機變換臉部區域並融合，產生 real / synthetic fake
    → 每筆資料組成 2 張影像，標籤 real=0、fake=1
    → EfficientNet-B4 輸出 2 類 logits，以 cross-entropy 訓練
    → FF++ 測試、儲存最佳 checkpoint
    → 用 checkpoint 在 Celeb-DF-v2 做跨資料集測試
```

注意：landmark 用於**製作訓練樣本**；`collate_fn` 傳給模型的 `landmark` 欄位是 `None`。SBI 的訓練假影格是即時合成；一般測試資料集則依 JSON 與實際影格載入真／假樣本。

## 階段

| 階段 | 證據 | 報告時的說法 |
| --- | --- | --- |
| Phase 1 | `SBI_FFPP_DATA_TODO.md` 已勾選檢查；檢查程式預期印出 `Model check passed` 與 `Dataset check passed` | 模型初始化、影格／landmark、SBI 樣本生成已完成基本檢查 |
| Smoke run | `logs/training-smoke/sbi_2026-09-16-10-02-12/training.log` 顯示 epoch 0 的訓練、測試和停止訊息 | 短版端到端流程已跑通；測試 AUC 0.375 是極小資料的流程檢查，不能當模型表現 |
| FF++ 訓練 | `logs/training/sbi_2026-09-16-10-03-00/training.log` 顯示設定 `nEpochs: 50`，完整完成 Epoch 0～37，並在 Epoch 38 中途停止；現有最佳 FF++ 影格 AUC 為 0.79802，且已產生 `test/avg/ckpt_best.pth` | 正式設定訓練已有 checkpoint，但未完成 50 epochs。這是 FF++ 內部測試的影格 AUC，不是 video AUC 或跨資料集結果 |
| Celeb-DF-v2 測試 | `logs/testing/celeb_df_v2_run1`、`run2` 各有 16,420 筆逐影格預測與相同的 JSON 指標：影格 AUC 0.67202、影片 AUC 0.72087、ACC 0.56334、AP 0.79226、EER 0.37616 | 已有兩份跨資料集評估輸出；現有輸出檔沒有記錄測試命令和 checkpoint 路徑，報告前應補上 provenance 才能宣稱可重現或比較兩次 run |

## 建議閱讀順序

1. `SBI 啟動方式.md`：先看環境、資料檢查、smoke run、正式訓練與測試命令。
2. `SBI_PHASE1.md`：了解本次修正的範圍、前置依賴與驗收條件。
3. `SBI_FFPP_DATA_TODO.md`：了解 FF++ c23 影格、landmark、JSON 如何放置；留意路徑是助教／本機環境相關。
4. `training/config/detector/sbi.yaml`、`training/config/train_config.yaml`、`training/config/test_config.yaml`：確認模型、資料來源、batch、epoch、測試設定；`train_smoke.yaml` 只用於短版驗證。
5. `training/dataset/sbi_dataset.py`、`training/dataset/sbi_api.py`：看資料如何由真實影格產生合成假影格，以及兩類標籤怎麼組 batch。
6. `training/detectors/sbi_detector.py`、`training/check_sbi_phase1.py`：看 EfficientNet-B4、分類 loss 和模型／資料檢查。
7. `training/train.py`、`training/test.py`：看設定合併、訓練與評估入口；接著看上述 log、metrics JSON、predictions CSV。若需建立環境，再看專案根目錄的 `README.md` 和 `setup_instructions.md`。

## 報告前要懂的背景

- **SBI 的目的**：從同一張真實臉產生融合痕跡，讓分類器學到造假線索；它和直接以 FF++ 假影片影格作訓練資料不同。
- **FF++ `c23`**：是影格資料的壓縮級別；JSON 中的 split、類別與影格相對路徑必須和磁碟上的 `frames` 對得上。每個真實 `.png` 應有同路徑的 `.npy` landmark。
- **權重的兩種角色**：`efficientnet-b4-6ed6700e.pth` 是 backbone 初始化權重；`ckpt_best.pth` 是 SBI 訓練後的完整 detector checkpoint。兩者不能互換，Xception 權重也不能給 EfficientNet-B4 使用。
- **評估指標**：AUC 衡量不同閾值下的排序能力；ACC 依固定閾值算正確率；AP 是 precision-recall 曲線摘要；EER 是假陽性率與假陰性率相等附近的錯誤率。影格 AUC 與影片 AUC 的統計單位不同，不應互換。
- **跨資料集測試**：FF++ 訓練、Celeb-DF-v2 測試可檢查泛化能力；先確認使用哪個 checkpoint、測試影格數及 JSON split，再解讀數字。

正式 FF++ 訓練完整完成 Epoch 0～37，在 Epoch 38 中途停止；目前最佳 FF++ 影格 AUC 為 0.798。Celeb-DF-v2 的跨資料集輸出顯示影格 AUC 0.672、影片 AUC 0.721
