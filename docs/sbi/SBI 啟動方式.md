# SBI 文件與啟動方式

本資料夾集中管理 SBI（Self-Blended Images）Pipeline 的說明文件：

- [SBI_PIPELINE_REPORT_GUIDE.md](SBI_PIPELINE_REPORT_GUIDE.md)：改動摘要、Pipeline、實驗紀錄與組內報告稿。
- [SBI_PHASE1.md](SBI_PHASE1.md)：模型及資料載入的 Phase 1 驗收方式。
- [SBI_FFPP_DATA_TODO.md](SBI_FFPP_DATA_TODO.md)：FaceForensics++ 訓練資料準備清單。

以下命令都要在專案根目錄執行。

## 1. 啟動前準備

確認 Python 環境已安裝專案依賴，且電腦有可用的 NVIDIA GPU。先檢查核心套件與 CUDA：

```powershell
python -c "import torch, yaml, efficientnet_pytorch; print('CUDA:', torch.cuda.is_available())"
```

SBI 訓練至少需要：

```text
E:/114_IMProject/data/FaceForensics++/original_sequences/youtube/c23/
  frames/<video>/<frame>.png
  landmarks/<video>/<frame>.npy

preprocessing/dataset_json/FaceForensics++.json
training/pretrained/efficientnet-b4-6ed6700e.pth
```

目前 `training/config/train_config.yaml` 與 `training/config/test_config.yaml` 的 `rgb_dir` 都是 `E:/114_IMProject/data`。資料放在別處時，先修改這兩個設定檔。影格和 landmark 必須有相同的相對路徑與檔名，副檔名分別為 `.png` 和 `.npy`。

## 2. 執行 Phase 1 檢查

不載入預訓練權重，只檢查模型結構、資料、landmark 與 SBI 樣本生成：

```powershell
python training/check_sbi_phase1.py --data-mode rgb
```

連同 EfficientNet-B4 初始化權重一起檢查：

```powershell
python training/check_sbi_phase1.py --data-mode rgb --weights ./training/pretrained/efficientnet-b4-6ed6700e.pth
```

成功時應看到：

```text
Model check passed: binary output shape (1, 2)
Dataset check passed: real/SBI pair and landmark available
```

## 3. 執行 smoke run

smoke run 使用少量 JSON、batch size 2、workers 0，只確認完整訓練與測試流程可以運行：

```powershell
python training/train.py `
  --detector_path ./training/config/detector/sbi.yaml `
  --train_config_path ./training/config/train_smoke.yaml `
  --pretrained ./training/pretrained/efficientnet-b4-6ed6700e.pth `
  --train_dataset "FaceForensics++" `
  --test_dataset "FaceForensics++"
```

結果會寫入 `logs/training-smoke/`。smoke run 的指標只用於確認流程，不能代表正式模型表現。

## 4. 啟動正式訓練

目前 SBI 設定為 EfficientNet-B4、真假二分類、FF++ c23、50 epochs：

```powershell
python training/train.py `
  --detector_path ./training/config/detector/sbi.yaml `
  --pretrained ./training/pretrained/efficientnet-b4-6ed6700e.pth `
  --train_dataset "FaceForensics++" `
  --test_dataset "FaceForensics++"
```

訓練 log 與最佳 checkpoint 會放在新的 `logs/training/sbi_<時間>/` 資料夾。最佳模型通常位於：

```text
logs/training/sbi_<時間>/test/avg/ckpt_best.pth
```

現有 `train.py` 沒有載入完整訓練 checkpoint、optimizer 與 scheduler 狀態的 resume 參數。`start_epoch` 只改迴圈起點，不能還原已中斷的訓練；因此 Epoch 38 中斷的 run 不能直接無損續訓。

## 5. 測試訓練完成的 SBI checkpoint

在 Celeb-DF-v2 上做跨資料集測試：

```powershell
python training/test.py `
  --detector_path ./training/config/detector/sbi.yaml `
  --test_dataset "Celeb-DF-v2" `
  --weights_path ./logs/training/sbi_<時間>/test/avg/ckpt_best.pth `
  --output_dir ./logs/testing/celeb_df_v2_run
```

請把 `<時間>` 換成實際資料夾名稱。輸出包含：

- `Celeb-DF-v2_predictions.csv`：逐影格路徑、標籤與 fake probability。
- `Celeb-DF-v2_metrics.json`：ACC、AUC、AP、EER 與 video AUC。

每次實驗應另外記下執行命令、checkpoint 路徑、Git commit 與資料 split，否則只靠 metrics JSON 無法完整重現結果。

## 6. 觀察執行狀態

查看 log 最後 30 行：

```powershell
Get-Content ./logs/training/sbi_<時間>/training.log -Tail 30
```

列出 epoch 開始與完成紀錄：

```powershell
Select-String -Path ./logs/training/sbi_<時間>/training.log -Pattern 'Epoch\[[0-9]+\] (start|end)'
```

只有出現 `Epoch[n] end` 才代表該 epoch 完整完成。現有 `sbi_2026-09-16-10-03-00` log 完整完成 Epoch 0～37，並在 Epoch 38 中途停止。
