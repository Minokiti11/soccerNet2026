# baseline/README.md

## 概要

このディレクトリは，`Spiideo SoccerNet SynLoc 2026` の公式 baseline をこのリポジトリから実行するための入口である．上流の実装本体は [baseline/mmpose](/Users/minorisugimura/GitHub/soccerNet2026/baseline/mmpose) にあり，このディレクトリには checkpoint の置き場所と実行用 script を置く．

## 上流

- repository: <https://github.com/Spiideo/mmpose>
- branch: `spiideo_scenes`
- upstream README: [baseline/mmpose/README.md](/Users/minorisugimura/GitHub/soccerNet2026/baseline/mmpose/README.md)

## 前提

- データセットが [data/SoccerNet/SpiideoSynLoc](/Users/minorisugimura/GitHub/soccerNet2026/data/SoccerNet/SpiideoSynLoc) にあること
- checkpoint を [baseline/checkpoints](/Users/minorisugimura/GitHub/soccerNet2026/baseline/checkpoints) に置くこと
- `Docker` が使えること

## ディレクトリ構成

```text
baseline/
├── README.md
├── Colab_Jupyter.md
├── checkpoints/
├── outputs/
├── run_eval.sh
├── run_challenge.sh
├── run_eval_local.sh
├── run_challenge_local.sh
└── mmpose/
```

## checkpoint の置き場所

例えば次のように置く．

```text
baseline/checkpoints/yoloxpose_tiny_4xb64-300e_640_epoch_300.pth
baseline/checkpoints/yoloxpose_m_4xb64-300e_960_epoch_300.pth
```

## Docker で validation / test を実行する

`YOLOX-tiny 640` の例:

```bash
zsh baseline/run_eval.sh \
  baseline/checkpoints/yoloxpose_tiny_4xb64-300e_640_epoch_300.pth \
  configs/body_bev_position/spiideo_soccernet/docker_yoloxpose_tiny_4xb64-300e_640.py
```

`YOLOX-m 960` の例:

```bash
zsh baseline/run_eval.sh \
  baseline/checkpoints/yoloxpose_m_4xb64-300e_960_epoch_300.pth
```

第 2 引数を省略した場合は，`YOLOX-m 960` 用の Docker config を使う．

## Docker で challenge 用出力を作る

`YOLOX-tiny 640` の例:

```bash
zsh baseline/run_challenge.sh \
  baseline/checkpoints/yoloxpose_tiny_4xb64-300e_640_epoch_300.pth \
  configs/body_bev_position/spiideo_soccernet/docker_yoloxpose_tiny_4xb64-300e_640.py
```

`YOLOX-m 960` の例:

```bash
zsh baseline/run_challenge.sh \
  baseline/checkpoints/yoloxpose_m_4xb64-300e_960_epoch_300.pth
```

## notebook で実行する

`Google Colab` など notebook 環境で実行する場合は [baseline/Colab_Jupyter.md](/Users/minorisugimura/GitHub/soccerNet2026/baseline/Colab_Jupyter.md) を参照すること．
