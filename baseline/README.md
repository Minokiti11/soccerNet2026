# baseline/README.md

## Purpose

このディレクトリは、`Spiideo SoccerNet SynLoc 2026` の公式 baseline を再現・実験するための作業場所である。

## Current State

- 公式 baseline repo を `baseline/mmpose` に clone 済み
- branch は `spiideo_scenes` に checkout 済み
- `baseline/.python-version` は `3.10.13`
- `baseline/.venv` は作成済み
- `torch` / `torchvision` は `baseline/.venv` に導入済み
- `mmcv` は macOS arm64 のローカル環境では prebuilt wheel が取れず、source build に落ちて失敗した
- このため、baseline 実行環境は Docker を第一候補にする

## Official Upstream

- Repo: <https://github.com/Spiideo/mmpose>
- Branch: `spiideo_scenes`
- Paper / README: [baseline/mmpose/README.md](/Users/minorisugimura/GitHub/soccerNet2026/baseline/mmpose/README.md)

## Expected Layout

```text
baseline/
├── README.md
├── checkpoints/
├── outputs/
├── run_eval.sh
├── run_challenge.sh
├── run_mmpose_docker.sh
└── mmpose/
```

## First Things To Try

### 1. Install dependencies

公式 README では Docker イメージ `hakanardo/mmpose` の利用例が書かれている。ローカル実行するなら、まず `baseline/mmpose` 側の依存関係を満たす必要がある。

少なくとも確認すべきもの:

- `torch`
- `mmcv`
- `mmengine`
- `mmdet`
- `mmpose`

現時点の確認結果:

- 現在のシステム Python は `3.13.11` で、baseline には不向き
- `baseline/.venv` は `3.10.13`
- `torch==2.11.0`, `torchvision==0.26.0` までは導入済み
- `mmcv==2.1.0` は macOS arm64 で build に失敗

そのため、実務上はローカルネイティブ環境より Docker の方が堅い。

### 2. Prepare data

データ配置の基準はこのリポジトリの `data/SoccerNet/SpiideoSynLoc`。

公式 baseline はこのデータを前提に動くため、dataset download が完了していることが必要。

### 3. Get pretrained checkpoint

公式 README にある pretrained model は `research.spiideo.com` から取得する形式になっている。まだこのリポジトリには checkpoint は置いていない。

候補:

- `yoloxpose_tiny_4xb64-300e_640`
- `yoloxpose_s_4xb64-300e_640`
- `yoloxpose_m_4xb64-300e_960`

最初は paper / leaderboard 的に `YOLOX-m 960` が基準候補。

checkpoint は例えば以下に置く。

```text
baseline/checkpoints/yoloxpose_m_4xb64-300e_960.pth
```

### 4. Run evaluation

このリポジトリでは wrapper script を使う。

```bash
./baseline/run_eval.sh baseline/checkpoints/yoloxpose_m_4xb64-300e_960.pth
```

challenge 用提出ファイルを出す場合:

```bash
./baseline/run_challenge.sh baseline/checkpoints/yoloxpose_m_4xb64-300e_960.pth
```

## Next Work

- baseline 実行環境を整える
- checkpoint を置く場所を決める
- validation / test / challenge の実行コマンドをこのリポジトリ側に固定する

## Docker

Docker daemon が起動していれば、以下で baseline 用コンテナを立ち上げられる。

```bash
./baseline/run_mmpose_docker.sh
```
