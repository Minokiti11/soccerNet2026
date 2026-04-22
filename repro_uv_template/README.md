# SoccerNet2026 RF-DETR＋ViTPose 再現リポジトリ（uv）

このディレクトリは，`Spiideo SoccerNet SynLoc 2026` の実験で使った主要スクリプトを，`uv` で再現しやすい形にまとめたテンプレートである．  
既存 organization 上で新規 GitHub リポジトリを作り，**このディレクトリ配下をそのままアップロード**すればよい．

## 1. アップロード対象ファイル

```text
repro_uv_template/
├── README.md
├── pyproject.toml
├── .gitignore
├── scripts/
│   ├── train_rfdetr_synloc.py
│   ├── prepare_rfdetr_synloc_tile_dataset.py
│   ├── convert_synloc_keypoints_to_coco.py
│   ├── eval_vitpose_synloc_split.py
│   ├── run_rfdetr_vitpose_locsym_eval.py
│   ├── run_rfdetr_vitpose_challenge_submit.py
│   ├── render_rfdetr_predictions.py
│   └── render_vitpose_test_samples.py
└── slurm/
    ├── train_rfdetr_synloc_4gpu.sbatch
    ├── train_vitpose_synloc_2gpu.sbatch
    ├── eval_rfdetr_vitpose_locsym_2gpu.sbatch
    └── rfdetr_vitpose_challenge_submit_1gpu.sbatch
```

## 2. `uv` 環境の構築

前提として，GPU ノード上に CUDA 対応 PyTorch を入れる．`uv` 本体は公式手順で導入する．

```bash
cd <this-repo>
uv venv --python 3.11
source .venv/bin/activate

# 基本依存関係
uv sync --extra all

# GPU 用 PyTorch（CUDA バージョンは環境に合わせて変更）
uv pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
```

補足：
- `mmcv` はまず `mmcv-lite` で動かす想定．必要なら `mmcv`（CUDA 拡張入り）へ差し替える．
- `sskit` は `pyproject.toml` で GitHub から取得する設定にしてある．

## 3. データ準備の流れ

### 3.1 RF-DETR 用タイル（tile）データ作成

```bash
export SOCCERNET_SOURCE_ROOT=/path/to/SpiideoSynLoc/4k
export RFDETR_TILE_ROOT=/path/to/synloc_4k_tile960_overlap240_vr03_coco
uv run python scripts/prepare_rfdetr_synloc_tile_dataset.py
```

### 3.2 ViTPose 用 COCO keypoint 形式へ変換

```bash
export SOCCERNET_SOURCE_ROOT=/path/to/SpiideoSynLoc/4k
export SOCCERNET_COCO_ROOT=/path/to/SpiideoSynLoc/4k_coco_kpt
uv run python scripts/convert_synloc_keypoints_to_coco.py
```

## 4. 学習（fine-tuning）

## 4.1 RF-DETR

```bash
export RFDETR_DATASET_DIR=/path/to/synloc_4k_tile960_overlap240_vr03_coco
export RFDETR_OUTPUT_DIR=/path/to/outputs/rfdetr_run_001
uv run python scripts/train_rfdetr_synloc.py
```

Slurm で流す場合は，次のテンプレートを使う．

```bash
sbatch slurm/train_rfdetr_synloc_4gpu.sbatch
```

## 4.2 ViTPose

このテンプレートでは，`mmpose` の `tools/train.py` を使う前提で `slurm/train_vitpose_synloc_2gpu.sbatch` を用意している．  
`CONFIG_PATH`，`DATA_ROOT`，`WORK_DIR` を自環境へ合わせて修正してから実行する．
このスクリプトは `sbatch ... <GPU台数>` の引数で GPU 台数を受け取り，学習開始前に `nvidia-smi` で空き GPU 数を確認する．
要求台数が空き台数を超える場合は，利用可能台数と空き GPU ID を表示して終了する．

```bash
# 2 GPU を使う場合
sbatch --gres=gpu:2 slurm/train_vitpose_synloc_2gpu.sbatch 2

# 4 GPU を使う場合
sbatch --gres=gpu:4 slurm/train_vitpose_synloc_2gpu.sbatch 4
```

## 5. 評価（val／test／challenge）

## 5.1 ViTPose 単体評価（GT bbox）

```bash
export VITPOSE_CONFIG=/path/to/mmpose_config.py
export VITPOSE_CKPT=/path/to/vitpose_checkpoint.pth
export SOCCERNET_COCO_ROOT=/path/to/SpiideoSynLoc/4k_coco_kpt
uv run python scripts/eval_vitpose_synloc_split.py
```

## 5.2 RF-DETR＋ViTPose 統合評価（val／test，LocSim）

```bash
export TILE_ROOT=/path/to/synloc_4k_tile960_overlap240_vr03_coco
export SOURCE_ROOT=/path/to/SpiideoSynLoc/4k
export RFDETR_CKPT=/path/to/rfdetr.ckpt_or_pth
export VITPOSE_CONFIG=/path/to/mmpose_config.py
export VITPOSE_CKPT=/path/to/vitpose_checkpoint.pth
export OUT_DIR=/path/to/artifacts/rfdetr_vitpose_eval_001
uv run python scripts/run_rfdetr_vitpose_locsym_eval.py
```

主な成果物：
- `val_pipeline_val_stats.json`
- `test_pipeline_test_stats.json`
- `results.json`
- `metadata.json`
- `test_submission.zip`

## 5.3 challenge 提出物生成

```bash
export TILE_ROOT=/path/to/synloc_4k_tile960_overlap240_vr03_coco
export SOURCE_ROOT=/path/to/SpiideoSynLoc/4k
export RFDETR_CKPT=/path/to/rfdetr.ckpt_or_pth
export VITPOSE_CONFIG=/path/to/mmpose_config.py
export VITPOSE_CKPT=/path/to/vitpose_checkpoint.pth
export OUT_DIR=/path/to/artifacts/challenge_submit_001
export SCORE_THRESHOLD=<valで決めた閾値>
uv run python scripts/run_rfdetr_vitpose_challenge_submit.py
```

主な成果物：
- `results.json`
- `metadata.json`
- `challenge_submission.zip`

## 6. GT / prediction の可視化

## 6.1 検出 bbox 可視化（GT と予測）

```bash
export RFDETR_DATASET_SPLIT_DIR=/path/to/.../test
export RFDETR_ANN_JSON=/path/to/.../test/_annotations.coco.json
export RFDETR_PRED_JSON=/path/to/.../test_predictions.coco.json
export RFDETR_OVERLAY_OUT_DIR=/path/to/vis/rfdetr
uv run python scripts/render_rfdetr_predictions.py
```

## 6.2 ViTPose 可視化（GT keypoint と予測 keypoint）

```bash
export VITPOSE_ANN_JSON=/path/to/.../annotations/test.json
export VITPOSE_PRED_JSON=/path/to/.../test_eval.keypoints.json
export VITPOSE_IMG_ROOT=/path/to/.../test
export VITPOSE_OVERLAY_OUT_DIR=/path/to/vis/vitpose
uv run python scripts/render_vitpose_test_samples.py
```

**challenge 提出は通常 `results.json` と `metadata.json` を zip 化して提出する．**
