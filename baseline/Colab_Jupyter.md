# baseline/Colab_Jupyter.md

## 結論

一旦，baseline は `Google Colab` で動かす．`baseline/mmpose` は Linux 前提の依存関係を持っており，手元 macOS の `JupyterLab` より `Colab` の方が素直に動かしやすいからである．

`Colab` で実行するには，このリポジトリを `Google Drive` に置く必要がある．

## 前提

- `Google Drive` にこのリポジトリを置くこと
- 少なくとも次の 3 つを `Colab` から見える場所に置くこと
  - `soccerNet2026/`
  - `soccerNet2026/data/SoccerNet/SpiideoSynLoc/`
  - `soccerNet2026/baseline/checkpoints/yoloxpose_m_4xb64-300e_960_epoch_300.pth`

## 推奨配置

`Google Drive` 上で，このような配置にしておくと扱いやすい．

```text
MyDrive/
└── soccerNet2026/
    ├── About_Competition.md
    ├── AGENTS.md
    ├── baseline/
    │   ├── checkpoints/
    │   │   └── yoloxpose_m_4xb64-300e_960_epoch_300.pth
    │   ├── mmpose/
    │   ├── run_eval_local.sh
    │   └── run_challenge_local.sh
    └── data/
        └── SoccerNet/
            └── SpiideoSynLoc/
```

## 1. Drive を mount して作業ディレクトリへ移動する


```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
%cd /content/drive/MyDrive/soccerNet2026
```


## 2. データと checkpoint が見えることを確認する

```bash
!ls data/SoccerNet/SpiideoSynLoc/annotations
!ls baseline/checkpoints
```

ここで `train.json`, `val.json`, `test.json`, `challenge_public.json` や checkpoint 名が見えなければ，配置がずれている．

## 3. 依存関係を入れる

公式 baseline は OpenMMLab 系の依存関係があるため，Linux 上で入れる前提になる．`Colab` では次を実行する．

```bash
!python -m pip install -U pip
!python -m pip install -U openmim
!mim install "mmengine>=0.4.0,<1.0.0"
!mim install "mmcv>=2.0.0,<3.0.0"
!mim install "mmdet>=3.0.0,<3.3.0"
!python -m pip install -r baseline/mmpose/requirements/runtime.txt
!python -m pip install -e baseline/mmpose
```

`Colab` の GPU ランタイムでは `torch` が最初から入っていることが多い．まずはそのまま進めてよい．もし互換性エラーが出たら，その時点で `torch` の組み合わせを調整する．

## 4. repository root を環境変数で渡す

このリポジトリでは，baseline 用 config が `SOCCERNET2026_ROOT` を見てデータパスを解決する．ノートブックでは，実行する前に環境変数を設定する．

```python
import os
os.environ["SOCCERNET2026_ROOT"] = "/content/drive/MyDrive/soccerNet2026"
```

## 5. validation / test を実行する

次で baseline を回せる．

```bash
!zsh baseline/run_eval_local.sh \
  baseline/checkpoints/yoloxpose_m_4xb64-300e_960_epoch_300.pth
```

`YOLOX-m 960` は重いので，まず流れだけ確認したいなら `tiny 640` や `s 640` の checkpoint を使う方がよい．

## 6. challenge 提出ファイルを生成する

challenge 用の出力は次で作る．

```bash
!zsh baseline/run_challenge_local.sh \
  baseline/checkpoints/yoloxpose_m_4xb64-300e_960_epoch_300.pth
```

## よくある詰まりどころ

- `FileNotFoundError`
  `SOCCERNET2026_ROOT` が通っていないか，`Drive` 上の配置が `data/SoccerNet/SpiideoSynLoc` になっていない可能性が高い．
- `mmcv` install failure
  `Colab` でも version の相性で起こることがある．その場合は `torch`, `mmcv`, `mmdet` の組み合わせを見直す．
- 実行が遅い
  `YOLOX-m 960` は重い．まず `tiny 640` か `s 640` で流れを確認する．
- `Drive` が遅い
  `Drive` 上の大きいデータは I/O が遅いことがある．必要なら作業中だけ `/content` にコピーする．

