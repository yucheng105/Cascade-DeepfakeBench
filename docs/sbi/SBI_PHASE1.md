# SBI Phase 1：模型與資料載入檢查

請在專案根目錄、使用 DeepfakeBench 的 Python 環境執行下列命令。本階段只檢查模型初始化與一組訓練樣本，不會開始訓練。

## 已完成的修改

- SBI 設定改為真假二分類，預設不載入預訓練權重；原本指定的 Xception 權重與 EfficientNet-B4 不相容。
- SBI 偵測器現在只透過 EfficientNet-B4 backbone 載入可選的預訓練權重；若設定了不存在的檔案路徑，會直接報錯。
- Phase 1 檢查程式驗證模型輸出、真實影格與 landmark 的對應，以及一組真實／SBI 樣本。

## 執行前準備

1. 依 README.md 的安裝說明建立 Python 環境。檢查程式至少需要 PyTorch、PyYAML、efficientnet_pytorch，以及資料集和偵測器引用的其他套件。
2. 將預處理好的 FaceForensics++ 訓練資料放進 datasets/rgb 或 datasets/lmdb。每張真實影格都應有對應的 81 點 landmark：frames 下的 .png 應對應 landmarks 下的 .npy，或 LMDB 中相應的 key。RGB 模式的資料集 JSON 放在 preprocessing/dataset_json；LMDB 模式放在 preprocessing/dataset_json_v3。
3. 若資料放在其他位置，修改 training/config/train_config.yaml 中的 rgb_dir、lmdb_dir 或 dataset_json_folder。檢查程式會像 training/train.py 一樣合併 SBI 與訓練設定。
4. **權重需自行取得，但基本初始化檢查可以先不下載。** 專案 README.md 提供[預訓練權重壓縮檔](https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.0/pretrained.zip)。下載、解壓後，將 EfficientNet-B4 權重放在 training/pretrained，確認檔名為 efficientnet-b4-6ed6700e.pth；若檔名不同，執行時指定實際路徑。不要將 Xception 權重或已訓練好的 SBI 偵測器 checkpoint 當作 backbone 預訓練權重。

## 執行命令

~~~powershell
python -c "import torch, yaml, efficientnet_pytorch; print('Dependencies OK')"
python training/check_sbi_phase1.py --data-mode rgb
~~~

若使用 LMDB 資料，把第二行改成 --data-mode lmdb。若要驗證下載的預訓練權重，加入：

~~~powershell
python training/check_sbi_phase1.py --data-mode rgb --weights ./training/pretrained/efficientnet-b4-6ed6700e.pth
~~~

預期成功訊息如下：

~~~text
Model check passed: binary output shape (1, 2)
Dataset check passed: real/SBI pair and landmark available
~~~

如果模型檢查通過而資料檢查失敗，先確認 train_dataset、資料集 JSON 目錄，以及錯誤訊息指出的影格或 landmark 路徑。RGB 模式會明確拒絕缺少 landmark 的檔案，因為基礎資料載入器原本會以全零 landmark 代替。

## 目前驗收狀態

Phase 1 已使用 RGB 資料與 EfficientNet-B4 預訓練權重通過模型及資料檢查，後續也已完成 smoke run。正式訓練前若更換資料路徑、JSON、環境或權重，應重新執行檢查並確認兩項成功訊息。
