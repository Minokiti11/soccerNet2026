# Research\_Notes.md

## Scope

このメモは，`Spiideo SoccerNet SynLoc 2026` に近い過去タスク・関連研究をもとに，実際に有望そうなパイプラインとアルゴリズムを整理したものである．

## Executive Summary

* SynLoc 系では `bbox -> bottom center -> camera projection` より `pose/keypoint -> pelvis_ground -> camera projection` が明確に有利．

* 公式 baseline も `YOLOX-pose` 系で，`pelvis` と `pelvis_ground` の 2 keypoint を直接回帰している．

* 去年や近い年の broadcast 系タスクで強い `tracking + ReID + field calibration` は，SynLoc 2026 では主役ではない．

* SynLoc 2026 は camera calibration が与えられる単一フレーム課題なので，勝負どころは「どの image-space 表現から world-space の位置を最も正確に復元できるか」にある．

* 実装優先度としては，まず baseline 再現，その後 `higher resolution`, `better point representation`, `synthetic-to-real adaptation` を入れるのが妥当．

## 1. Official SynLoc Pipeline

### 1.1 Devkit baseline

`sskit` の簡易 baseline は，一般物体検出器で人を検出し，bbox 下辺中央を選手位置とみなして地面へ射影する．

概略：

1. image 上で `person` detection
2. bbox の `bottom center` を取る
3. データ付属の camera model で ground plane に射影
4. world coordinate 上で評価

これは最小構成としては分かりやすいが，真の player location の proxy としては粗い．

実装上の具体像も単純で，`sskit/baseline.py` は `ultralytics/yolov5` の `yolov5x6` を読み込み，`person` detection の各 bbox から `(x_center, y_bottom)` を作り，それを

```python
image_to_ground(camera_matrix, undist_poly, normalized_image_point)
```

で pitch 座標へ変換している．

つまり，この baseline は次のことをしていない．

* pitch registration の推定

* camera calibration の推定

* tracking

* ReID

* direct world-coordinate regression

やっているのは「既知の camera model を使って，bbox 下辺中央を地面へ落とす」だけである．

Source：

* <https://github.com/Spiideo/sskit/blob/master/baseline.py>

* <https://github.com/Spiideo/sskit/blob/master/sskit/camera.py>

### 1.2 Official mmpose baseline

公式 baseline は `Spiideo/mmpose` の `spiideo_scenes` ブランチで，`YOLOX-pose` を SynLoc 用に縮約した one-stage / bottom-up 系を使う．

重要な点：

* 検出対象は人

* 1 個の head が bbox と keypoint を同時に出す

* 回帰する keypoint は 2 個だけ

* `pelvis` = 骨盤

* `pelvis_ground` = その骨盤を地面平面 `z=0` へ落とした点

`pelvis_ground` は player pelvis を ground plane に正射影した点に対応する．評価もこの点を world-space に戻した位置で行う．

ここで重要なのは，これは「人検出器の後ろに別の pose 推定器を載せる top-down 2 段構成」ではないことだ．`YOLOXPoseHead` の実装を見ると，3 つの feature map stride 上で同時に次を予測している．

* class score

* objectness

* bbox

* keypoint offset

* keypoint visibility

つまり，公式 baseline は「物体検出だけ」でも「姿勢推定だけ」でもなく，`bbox + 2 keypoints` を 1 段で同時予測する detector-pose hybrid である．

Source：

* <https://github.com/Spiideo/mmpose/tree/spiideo_scenes>

* <https://github.com/Spiideo/mmpose/blob/spiideo_scenes/configs/body_bev_position/spiideo_soccernet/yoloxpose_m_4xb64-300e_640.py>

* <https://github.com/Spiideo/mmpose/blob/spiideo_scenes/configs/body_2d_keypoint/yoloxpose/coco/yoloxpose_s_8xb32-300e_coco-640.py>

* <https://github.com/Spiideo/mmpose/blob/spiideo_scenes/mmpose/models/heads/hybrid_heads/yoloxpose_head.py>

* <https://github.com/Spiideo/mmpose/blob/spiideo_scenes/configs/_base_/datasets/spiideo_soccernet_synloc.py>

### 1.3 How the official baseline is trained

config を追うと，SynLoc baseline は COCO 用 `YOLOX-pose` config を土台にして，SynLoc 用に次を差し替えている．

* dataset を `SpiideoSoccerNetSynLocDataset` に変更

* evaluator を `bbox`, `locsim_bbox`, `locsim` に変更

* keypoint 数を `17` から `2` へ変更

* OKS の metainfo を COCO 17 点ではなく `pelvis`, `pelvis_ground` 用に変更

* learning rate などを SynLoc 用に調整

学習 label 側も，`sskit/make_coco.py` が明示的に

* `keypoints[0] = pelvis`

* `keypoints[1] = pelvis_ground`

* `position_on_pitch = [x_world, y_world]`

を作っている．ここでの `x_world, y_world` は，データセット構築時の 3D レンダリング空間で既知な人物 world 座標に由来する．つまり正解 annotation 自体が，3D レンダリング時の world 座標から自動生成されている．したがって，baseline は画像から直接 pitch 座標を回帰しているのではなく，「まず画像上の 2 点を当てる」設計だと言える．

Source：

* <https://github.com/Spiideo/mmpose/blob/spiideo_scenes/configs/body_bev_position/spiideo_soccernet/synloc.py>

* <https://github.com/Spiideo/mmpose/blob/spiideo_scenes/configs/_base_/datasets/spiideo_soccernet_synloc.py>

* <https://github.com/Spiideo/sskit/blob/master/make_coco.py>

### 1.4 What the baseline outputs at inference time

`tools/test.py` と `CocoMetric.results2json()` を読むと，baseline 推論の最終出力は `position_on_pitch` 直書きではない．出力される `results.json` は convenience format であり，各 detection は概ね次を持つ．

* `image_id`

* `category_id`

* `bbox`

* `keypoints`

* `score`

つまり，出力値は「pitch 上の `(x, y)` そのもの」ではなく，「画像上の `pelvis` と `pelvis_ground` の 2 点とその score」である．そこから evaluator / server 側が `position_from_keypoint_index=1` を見て `pelvis_ground` を pitch 上位置へ変換する．

`metadata.json` は baseline 実装では

```json
{
  "score_threshold": ...,
  "position_from_keypoint_index": 1
}
```

となる．ここで `1` は 2 個目の keypoint，すなわち `pelvis_ground` を指す．

Source：

* <https://github.com/Spiideo/mmpose/blob/spiideo_scenes/mmpose/evaluation/metrics/coco_metric.py>

* <https://github.com/Spiideo/mmpose/blob/spiideo_scenes/tools/test.py>

### 1.5 How `pelvis_ground` is converted to pitch coordinates

この点は implementation 上かなり明確で，`sskit.coco.LocSimCOCOeval` の流れは次である．

1. detection の `keypoints` から `position_from_keypoint_index` 番目の image-space point を取り出す
2. 画像サイズ `w, h` を使って
   `((x, y) - ((w - 1) / 2, (h - 1) / 2)) / w`
   に正規化する
3. それを `image_to_ground(camera_matrix, undist_poly, normalized_point)` へ入れる
4. 返ってきた ground 座標の先頭 2 次元を `bev_dt[:, :2]` として評価に使う

式の形だけ書くと，

```text
pixel keypoint
  -> normalized image point
  -> undistort with undist_poly
  -> project to ground with inverse camera geometry
  -> (x_pitch, y_pitch)
```

である．

ここでもポイントは，同じく baseline は pitch registration を推定していないことだ．camera geometry は image ごとに annotation に入っており，それを使って幾何変換しているだけである．

比較用の `locsim_bbox` では，この `position_from_keypoint_index` の代わりに bbox 下辺中央を使う．だからこそ，baseline config が `locsim_bbox` を別 metric として残しているのは，「bbox 下辺中央だけに頼るとどこまで落ちるか」を可視化するためと理解できる．

Source：

* <https://github.com/Spiideo/sskit/blob/master/sskit/coco.py>

* <https://github.com/Spiideo/sskit/blob/master/sskit/camera.py>

### 1.6 Public pre-trained models are what kind of models?

公式 README に載っている `YOLOX-tiny`, `YOLOX-s`, `YOLOX-m` は，いずれも SynLoc 専用にゼロから設計された新規アーキテクチャではない．

* backbone / neck / head の骨格は OpenMMLab 系 `YOLOX-pose`

* tiny と s は COCO 学習済み YOLOX checkpoint を初期値に使う

* m も COCO 系 pretrained checkpoint を初期値に使う

* その上で SynLoc の 2-keypoint task に fine-tune した checkpoint が `research.spiideo.com` から配布される

したがって，「公開されている事前学習モデル」は，

1. OpenMMLab / COCO 側の一般人体検出・姿勢推定向け pretrained model
2. それを SynLoc task に fine-tune した task-specific checkpoint

の 2 層に分けて理解するとよい．

Source：

* <https://github.com/Spiideo/mmpose/blob/spiideo_scenes/README.md>

* <https://github.com/Spiideo/mmpose/blob/spiideo_scenes/configs/body_2d_keypoint/yoloxpose/coco/yoloxpose_tiny_4xb64-300e_coco-640.py>

* <https://github.com/Spiideo/mmpose/blob/spiideo_scenes/configs/body_2d_keypoint/yoloxpose/coco/yoloxpose_m_8xb32-300e_coco-640.py>

* <https://github.com/Spiideo/mmpose/blob/spiideo_scenes/configs/body_bev_position/spiideo_soccernet/yoloxpose_m_4xb64-300e_960.py>

## 2. What The SynLoc Paper Shows

SynLoc 論文では，bbox ベースと pose ベースを直接比較している．結果として，`world-space localization` では pose ベースが大きく上回った．

論文中の主要な示唆:

* bbox の下辺中央は，真の player ground position の近似として弱い

* `pelvis` と `pelvis_ground` のような幾何的に意味のある点を学習した方がよい

* 入力解像度を上げると成績がかなり改善する

論文の Table 2 では，`YOLOX-m` の比較で以下の傾向が見える．

* `bbox 960x960`: `mAP-LocSim 52.4`

* `pose 960x960`: `mAP-LocSim 79.3`

Source：

* <https://www.scitepress.org/Papers/2025/131082/131082.pdf>

## 3. Related Pipeline Families From Similar Tasks

### 3.1 Broadcast soccer: calibration-first pipelines

類似する過去のタスクとしては `SoccerNet Game State Reconstruction (GSR)` がある．ここでは通常，以下のような pipeline が使われる．

1. field registration / camera calibration
2. player detection
3. tracking
4. ReID
5. pitch projection
6. minimap / world-state reconstruction

この系統では `TVCalib`, `PnLCalib`, `No Bells Just Whistles` のような calibration の改善や，`StrongSORT`, `DeepSORT`, `BoT-SORT` 系の tracking が重要になる．

ただし，これは moving camera の broadcast 動画だから成立する話で，SynLoc 2026 では calibration が与えられ，しかも単一フレームなので，tracking/ReID/calibration については考えなくてよい．

Source：

* <https://raw.githubusercontent.com/SoccerNet/sn-gamestate/main/README.md>

* <https://www.sciencedirect.com/science/article/pii/S1077314226000792>

### 3.2 Player localization from a single moving view

過去研究 `Individual Locating of Soccer Players from a Single Moving View` でも，

1. sports field registration
2. 2D player tracking
3. homography による pitch projection

の 3 段構成が中心である．ここでも本質は，camera が未知または時間変化することへの対処にある．

SynLoc 2026 では calibration が既知なので，この pipeline 全体を持ち込む必要はない．参考にすべきなのは「image-space から world-space への幾何の扱い方」であり，tracking 系そのものではない．

Source：

* <https://www.mdpi.com/1424-8220/23/18/7938>

## 4. Practical Implications For SynLoc 2026

### 4.1 What likely matters（大事そうなもの）

* `pelvis_ground` のような world location に直結した keypoint 表現

* 遠方小物体に効く高解像度入力

* synthetic-to-real gap を詰める augmentation / adaptation

* score threshold の適切な最適化

* 提出 JSON と evaluator の完全整合

### 4.2 What likely matters less（重要ではないもの）

* tracking

* jersey number recognition

* ReID

* camera calibration

これらは broadcast video や minimap reconstruction では重要だが，SynLoc 2026 の単一フレーム challenge には関係ない．

## 5. Candidate Improvement Directions

### 5.1 Better point representation

最も自然な改善案は，`2-keypoint` 表現を拡張すること．

候補：

* `pelvis`, `pelvis_ground`, `left_foot`, `right_foot`

* `pelvis`, `head`, `pelvis_ground`

* `bbox` と `pose point` のハイブリッド

狙い：

* 遮蔽時に単一 keypoint より頑健にする

* player scale / stance に応じて ground position をより安定に推定する

### 5.2 Direct world-coordinate regression

image keypoint を経由せず，検出ごとに `(x_world, y_world)` を直接回帰する案もある．

利点：

* task に対して表現が直接的

* image-space keypoint の誤差を world-space へ変換する段階を減らせる

懸念：

* 公式 baseline や devkit と表現がずれる

* 学習と後処理の実装コストが上がる

* multi-person assignment の扱いを自前で設計する必要が出やすい

したがって，最初の改良としてはやや重い．

### 5.3 Higher resolution and crop strategy

論文結果から，`640 -> 960` でかなり改善する．さらに遠方選手向けに以下も検討余地がある．

* 4K 原画像を活かす multi-scale inference

* 左右や縦長のタイル分割

* half-pitch 構造を意識した crop

このタスクでは pitch 全体が写り，遠距離選手が極小になるため，単純 resize には限界がある可能性が高い．

### 5.4 Synthetic-to-real adaptation

SynLoc は synthetic player を real background に合成したデータであり，ドメインギャップは依然としてある．

有望な方向：

* color jitter の強化

* blur, noise, jpeg compression

* illumination / weather の擬似変化

* background texture 変動への頑健化

* stronger occlusion augmentation

さらに可能なら：

* 公開 real soccer imagery を使った self-training

* 公開データでの person / pose pretraining

ただし，利用データは challenge rules の範囲内である必要がある．

### 5.5 Confidence calibration

`mAP-LocSim` は「見つけること」と「位置が近いこと」の両方を見るため，confidence の設計が重要である．

やるべきこと：

* validation で `score_threshold` を決める

* precision / recall / F1 のトレードオフを見る

* 遮蔽や遠距離 player に対する過信を抑える

## 6. Recommended Order Of Work

このリポジトリでの優先順位は以下が妥当．

1. 公式 baseline を再現する
2. `960x960` 系を安定実行する
3. 提出生成 `results.json` / `metadata.json` を固める
4. `pelvis_ground` まわりの表現改善を試す
5. synthetic-to-real gap 対策を入れる
6. 高解像度 / タイル推論を検証する
7. 必要なら direct world-coordinate head を試す

## 7. Working Hypotheses

現時点での作業仮説は以下．

* 最初の実用的な上積みは `bbox detector` ではなく `pose/keypoint localization` から出る

* 小さく遠い選手に対する recall 改善が大きな差になる

* `pelvis_ground` を直接当てる設計は task と metric に素直に一致している

* calibration 既知という条件を活かし，tracking や registration に寄り道しない方がよい

## Sources

* SynLoc devkit: <https://github.com/Spiideo/sskit>

* SynLoc official baseline: <https://github.com/Spiideo/mmpose/tree/spiideo_scenes>

* SynLoc paper: <https://www.scitepress.org/Papers/2025/131082/131082.pdf>

* SoccerNet GSR baseline: <https://raw.githubusercontent.com/SoccerNet/sn-gamestate/main/README.md>

* PnLCalib article page: <https://www.sciencedirect.com/science/article/pii/S1077314226000792>

* Single moving view player localization: <https://www.mdpi.com/1424-8220/23/18/7938>
