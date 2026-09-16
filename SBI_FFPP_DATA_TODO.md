# SBI 訓練資料準備清單（助教電腦）

此清單使用 DeepfakeBench 已預處理的 FaceForensics++ RGB 資料。請在專案根目錄操作；本階段只準備與檢查資料，不會開始訓練。

## 待辦事項

- [ ] 切換到 SBI 分支：`git fetch origin`、`git switch codex/sbi-phase1`。
- [ ] 依專案 `README.md` 建立 Python 環境，確認 PyTorch、PyYAML、NumPy 與資料集引用的其他套件可用。
- [ ] 從 [DeepfakeBench 提供的 RGB 預處理資料](https://drive.google.com/drive/folders/1N4X3rvx9IhmkEZK-KIk4OxBrQb9BRUcs?usp=drive_link) 下載 FaceForensics++ 的 `c23` 資料。SBI 訓練需要真實影片的裁切影格與 landmark；只有原始 `.mp4` 不夠。
- [ ] 將資料解壓至 `datasets/rgb/FaceForensics++/`，確認至少有以下對應檔案：

  ~~~text
  datasets/rgb/FaceForensics++/original_sequences/youtube/c23/
    frames/000/000.png
    landmarks/000/000.npy
  ~~~

  實際影片與影格編號可以不同，但同一影格的 `frames`、`landmarks` 子路徑和檔名必須一致（副檔名分別為 `.png`、`.npy`）。landmark 應為 81 點資料；SBI 資料集會用真實影格即時產生假影格。若後續要在同一資料集測試，也須準備對應的假影格資料。
- [x] 從 [DeepfakeBench 提供的 JSON 設定資料](https://drive.google.com/drive/folders/1ZV3fz5MZZU5BTB5neziN6i8Yv0Z21_LO?usp=drive_link) 取得 `FaceForensics++.json`，放至 `preprocessing/dataset_json/FaceForensics++.json`。確認其中包含 `FaceForensics++`、`FF-real`、`train`、`c23` 的資料，且記錄的影格路徑在助教電腦上確實存在。
- [x] 使用 RGB 資料時，將 `training/config/train_config.yaml` 的 `lmdb` 從 `True` 改為 `False`；`rgb_dir` 應指向 `./datasets/rgb`，`dataset_json_folder` 應指向 `./preprocessing/dataset_json`。若資料放在其他磁碟，改成實際路徑。
- [ ] 在專案根目錄執行 Phase 1 資料與模型檢查：

  ~~~powershell
  python training/check_sbi_phase1.py --data-mode rgb
  ~~~

  若已取得相容的 EfficientNet-B4 預訓練權重，可加上 `--weights ./training/pretrained/efficientnet-b4-6ed6700e.pth`。看到 `Model check passed` 和 `Dataset check passed` 兩行，才算資料與模型的 Phase 1 檢查通過。

## 下載資料不適用時

若取得的是[原始 FaceForensics++ 影片](https://github.com/ondyari/FaceForensics/tree/master/dataset)，須依 `README.md` 執行 `preprocessing/preprocess.py`，產生裁切影格與 landmark，並準備 train/val/test split 與資料集 JSON。這比使用預處理資料費時。

目前此專案的 `preprocessing/config.yaml` 將 rearrange 輸出設為 `dataset_json_v6`，與訓練設定預設的 `dataset_json` 不同；`rearrange.py` 對 FaceForensics++ 也假設存在額外資料項目，且不直接產生 `FaceForensics++.json`。若下載的 JSON 路徑不合用，先確認這些設定與腳本再重新產生 JSON，不要直接執行後就視為資料已備妥。

資料、權重與模型的完整 Phase 1 驗收方式另見 `SBI_PHASE1.md`。
