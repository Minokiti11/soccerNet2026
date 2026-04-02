# About\_Competition.md

## Competition

* 名称：`Spiideo SoccerNet SynLoc 2026`

* 主催：`SoccerNet Challenges 2026`

* タスク：単一フレームから全選手を検出し，各選手の pitch 上のワールド座標をメートル単位で推定する．

## Official Links

* SoccerNet 2026 challenge page：<https://www.soccer-net.org/challenges/2026>

* Competition devkit `sskit`：<https://github.com/Spiideo/sskit>

* Official baseline：<https://github.com/Spiideo/mmpose/tree/spiideo_scenes>

* Codabench test server：<https://www.codabench.org/competitions/10128/>

* Codabench challenge server：<https://www.codabench.org/competitions/10155/>

* Rules：<https://github.com/Spiideo/mmpose/blob/spiideo_scenes/ChallengeRules.md>

* Paper：<https://www.scitepress.org/publishedPapers/2025/131082/pdf/index.html>

## タスクの概要

SoccerNet 2026 の公式ページと `sskit` README では，この課題を `Single-Frame World-Coordinate Athlete Detection & Localization` と説明している．

1 枚の画像を入力として，各選手について以下を出力する．

* その選手が存在することを示す detection

* その選手がピッチ上のどこにいるかを表す `position_on_pitch`

この課題では，通常の物体検出と違って「bbox が合っているか」だけではなく，選手ごとの ピッチ上の位置である．そのため，画像上の bbox や pose は中間表現であり，提出物として必要なのは `position_on_pitch` である．画像上の座標→ピッチ座標へ変換してくれる関数が用意されていて，（`image_to_ground()`）ベースラインもこれを使っている．

## データセット

### フォーマット

公式 baseline が直接読むのは，`COCO` 風に拡張した JSON annotation と split ごとの画像ディレクトリである．

* 画像本体：`train/`, `val/`, `test/`, `challenge/` に `.jpg`

* annotation：`annotations/*.json`

* category：実質 1 クラスで `person`

`annotations/train.json` や `annotations/val.json` は，大まかには次の 3 つの top-level key を持つ．

* `images`

* `annotations`

* `categories`

### `images` の形式

`images` 要素には，通常の COCO の `id`, `file_name`, `width`, `height` に加えて，pitch への射影に必要な camera 情報が入る．`sskit/make_coco.py` の実装では，少なくとも次が保存される．

```json
{
  "id": 12,
  "file_name": "000012.jpg",
  "width": 3840,
  "height": 2160,
  "camera_matrix": [[...], [...], [...]],
  "dist_poly": [...],
  "undist_poly": [...]
}
```

ここで重要なのは，このタスクでは camera calibration を自力で推定しなくてよい点である．画像ごとに camera model が annotation 側へ付属しており，baseline もそれをそのまま使う．

### `annotations` の形式

`annotations` 要素は，COCO 風 bbox に加えて，pitch 上位置と 2 個の keypoint を持つ．`sskit/make_coco.py` が生成している形式は次の通りである．

```json
{
  "id": 345,
  "image_id": 12,
  "category_id": 1,
  "bbox": [529.0, 521.0, 69.0, 113.0],
  "area": 4287,
  "keypoints": [
    [572.03, 570.68, 1],
    [572.39, 633.88, 1]
  ],
  "keypoints_3d": [
    [-8.60, -1.61, 0.93, 1],
    [-8.60, -1.61, 0.00, 1]
  ],
  "position_on_pitch": [-8.60, -1.61]
}
```

各項目の意味は以下である．

* `bbox`：画像上の人 bbox．形式は COCO と同じ `[x, y, width, height]`

* `keypoints[0]`：`pelvis` の画像座標

* `keypoints[1]`：`pelvis_ground` の画像座標．`pelvis` を地面平面 `z=0` に落とした点

* `keypoints_3d[0]`：3D の `pelvis = (x, y, z)`

* `keypoints_3d[1]`：3D の `pelvis_ground = (x, y, 0)`

* `position_on_pitch`：pitch 上の真値位置．実装上は最初の 2 次元 `x, y` が評価に使われる

### アノテーションの元データ

competition 用に配布・消費されるのは上の COCO 風形式だが，`sskit/make_coco.py` を見ると，元データはさらに低レベルな scene 情報から作られている．少なくとも次のようなファイル群を参照している．

* `rgb.jpg`

* `objects.json`

* `camera_matrix.npy`

* `lens.json`

* `segmentations.npy.gz`

この段階では，各人物に対して `objects.json` の中に 3D `pelvis` や詳細な SMPL 系 keypoint があり，そこから `pelvis` と `pelvis_ground` を画像へ再投影して competition 用 annotation が構築される．

**重要なのは，この task の正解** **`position_on_pitch`** **は，人手で 2D 画像上に打った点ではないことだ．データセット構築時の 3D レンダリング空間で持っている人物の world 座標，具体的には** **`pelvis`** **の** **`(x, y, z)`** **と，それを ground plane** **`z=0`** **に落とした** **`pelvis_ground`** **から作られている．したがって，正解 pitch 座標は「3D レンダリング時に既知な world 座標を，dataset annotation として書き出したもの」と理解するのが正しい．**

## タスクの入出力

### Input

モデルが 1 サンプルとして受け取る情報は，実用上は次の組である．

* RGB 画像 1 枚

* その画像の `camera_matrix`, `dist_poly`, `undist_poly`

### Output の 2 形式

`sskit` と challenge server は，提出 `results.json` として 2 種類の表現を受けられる．

#### 1. 直接 `position_on_pitch` を書く形式

もっとも素直な形式は，各 detection が直接 pitch 座標を持つ JSON list である．

```json
[
  {
    "id": 1,
    "image_id": 1,
    "category_id": 1,
    "area": 0,
    "position_on_pitch": [1.2, 2.0, 0.0],
    "score": 0.91
  },
  {
    "id": 2,
    "image_id": 1,
    "category_id": 1,
    "area": 0,
    "position_on_pitch": [5.0, 10.0, 0.0],
    "score": 0.85
  }
]
```

#### 2. keypoint を書き，server 側で pitch へ射影してもらう convenience format

公式 baseline はこの形式を使う．`results.json` には画像上 keypoint を書き，`metadata.json` で「どの keypoint を pitch 上位置として使うか」を指定する．

`CocoMetric.results2json()` と `tools/test.py` から読むと，baseline の `results.json` 各要素は概ね次の形になる．

```json
[
  {
    "image_id": 1,
    "category_id": 1,
    "bbox": [529.0, 521.0, 69.0, 113.0],
    "keypoints": [
      572.03, 570.68, 0.98,
      572.39, 633.88, 0.99
    ],
    "score": 0.91
  }
]
```

この flattened `keypoints` は，

* `[x_pelvis, y_pelvis, score_pelvis, x_pelvis_ground, y_pelvis_ground, score_pelvis_ground]`

を表す．

対応する `metadata.json` は，baseline 実装では次の内容になる．

```json
{
  "score_threshold": 0.6848765313625336,
  "position_from_keypoint_index": 1
}
```

`position_from_keypoint_index = 1` は，2 個目の keypoint，つまり `pelvis_ground` を「地面へ射影すべき点」として使う指定である．

### Challenge submission zip

提出物は zip で，通常は以下を含む．

* `results.json`

* `metadata.json`

`Spiideo/mmpose` の `tools/test.py --challenge` は，validation を一度回して最適 `score_threshold` を決めた上で，`results.json` と `metadata.json` を生成し，さらに `challenge_submission.zip` を作る．

## Coordinate Systems

`sskit` の camera model は，`standard projective pinhole camera model with radial distortion` と説明されている．使われる変換は概ね次の通りである．

```text
Camera pixel
  -> normalize()
Normalized image
  -> undistort()
Undistorted image
  -> undistorted_to_ground()
World / ground plane
```

逆方向には `world_to_image()` がある．

## Evaluation

評価指標は `mAP-LocSim` である．これは COCO 形式の `mAP` をベースにしつつ，通常の `IoU` の代わりに `LocSim` を使って，検出とピッチ座標推定を同時に評価する指標である．

**mAP を復習する：**

まず `AP`（`Average Precision`，平均適合率）は，ある対応付け条件を固定したときの `precision-recall` 曲線の面積である．SynLoc では，この対応付け条件が `LocSim >= t` で与えられる．高 confidence の prediction から順に並べ，

* その prediction が GT とマッチすれば `TP`

* マッチしなければ `FP`

として `precision` と `recall` を累積的に計算し，その曲線の面積を `AP` とする．

`mAP` はその `AP` を複数の閾値で平均したものだと考えればよい．この課題ではクラスは `person` の 1 つだけである．

参考：<https://qiita.com/cv_carnavi/items/08e11426e2fac8433fed#5-ap-map>

SynLoc では，通常の COCO の `IoU threshold = 0.50, 0.55, ..., 0.95` の代わりに `LocSim threshold = 0.50, 0.55, ..., 0.95` を使う．したがって概念的には次である．

```text
mAP-LocSim = mean_t AP_at_LocSim_threshold(t)
where t in {0.50, 0.55, ..., 0.95}
```

### What is evaluated

`LocSim` の実装は `sskit.coco.LocSimCOCOeval` にある．予測 pitch 座標を `p_dt = (x_dt, y_dt)`，正解を `p_gt = (x_gt, y_gt)` とすると，まず二乗距離

```text
d^2 = (x_dt - x_gt)^2 + (y_dt - y_gt)^2
```

を計算し，その後

```text
LocSim = exp(log(0.05) * d^2 / tau^2)
```

で類似度を定義している．この challenge では許容距離パラメータは `tau = 1` で固定である．同値な書き方をすると

```text
LocSim = 0.05^(d^2)
```

となる．性質は次の通りである．

* 距離 `d = 0` なら `LocSim = 1`

* 距離 `d = 1m` なら `LocSim = 0.05`

* 距離が増えると指数関数的に急減する

frame accuracy では，各画像について「偽陽性も偽陰性もなく，全選手を正しく検出できたか」を見る．ここでの境界は `LocSim = 0.5` であり，およそ `0.48m` の位置誤差に対応する．

baseline config は evaluation を 3 系統に分けている．

* `bbox`：通常の bbox 検出評価

* `locsim_bbox`：bbox の下辺中央を pitch へ射影した比較用指標

* `locsim`：予測 `position_on_pitch`，または `position_from_keypoint_index` で指定した keypoint を地面へ射影した指標

challenge で本当に効くのは `locsim` である．

## Submission / Validation Practice

* validation で `score_threshold` を決める

* test / challenge ではその threshold を固定して使う

* 提出前に `results.json` と `metadata.json` の整合性を確認する

* baseline convenience format を使う場合は，`metadata.json` に `position_from_keypoint_index` を必ず入れる

## Data Usage Rules

公式ガイドラインから重要な制約を抜き出すと以下である．

* private dataset の使用は禁止

* challenge 期間中の手動追加 annotation は禁止

* すべての参加者が同条件で利用できる公開データのみ使用可能

* 既存公開データに対する自動的なラベル補完・擬似ラベル生成・特徴抽出は，再現可能で説明可能なら許可される

* challenge 中に自チームだけが使う手動修正版ラベルは不可

* 新しい共有データは，締切の少なくとも 1 か月前までに全参加者へ公開される必要がある

## Important Dates

`ChallengeRules.md` の記載は次の通りである．

* `2025-09-09`：test evaluation server open

* `2025-09-09`：challenge evaluation server open

* `2026-04-25`：evaluation server close

* `2026-05-01`：report submission deadline

* `TBD`：CVSports Workshop at CVPR 2026 で表彰

## Practical Implications For This Repo

* まず baseline の validation / test / challenge 出力を再現する

* `score_threshold` を validation で固定し，どの checkpoint と設定で決めたかを記録する

* `position_on_pitch` の直書き形式と，baseline convenience format の両方を理解しておく

* 改良案を試す場合も，camera calibration 推定ではなく，「より良い player ground point をどれだけ安定に出せるか」を中心に考える

## Sources

* SoccerNet 2026 challenge page：<https://www.soccer-net.org/challenges/2026>

* `sskit` README：<https://github.com/Spiideo/sskit/blob/master/README.md>

* `sskit` camera utilities：<https://github.com/Spiideo/sskit/blob/master/sskit/camera.py>

* `sskit` LocSim evaluator：<https://github.com/Spiideo/sskit/blob/master/sskit/coco.py>

* `sskit` annotation builder：<https://github.com/Spiideo/sskit/blob/master/make_coco.py>

* baseline README：<https://github.com/Spiideo/mmpose/blob/spiideo_scenes/README.md>

* baseline config `synloc.py`：<https://github.com/Spiideo/mmpose/blob/spiideo_scenes/configs/body_bev_position/spiideo_soccernet/synloc.py>

* baseline dataset metainfo：<https://github.com/Spiideo/mmpose/blob/spiideo_scenes/configs/_base_/datasets/spiideo_soccernet_synloc.py>

* challenge rules：<https://github.com/Spiideo/mmpose/blob/spiideo_scenes/ChallengeRules.md>

* official paper：<https://www.scitepress.org/publishedPapers/2025/131082/pdf/index.html>
