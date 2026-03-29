# Research\_Notes.md

## Scope

このメモは，`Spiideo SoccerNet SynLoc 2026` に近い過去タスク・関連研究をもとに，実際に有望そうなパイプラインとアルゴリズムを整理したものである．主眼は「このコンペで何を優先して試すべきか」を決めることにある．

## Executive Summary

* SynLoc 系では `bbox -> bottom center -> camera projection` より `pose/keypoint -> pelvis_ground -> camera projection` が明確に有利．

* 公式 baseline も `YOLOX-pose` 系で，`pelvis` と `pelvis_ground` の 2 keypoint を直接回帰する．

* 去年や近い年の broadcast 系タスクで強い `tracking + ReID + field calibration` は，SynLoc 2026 では主役ではない．

* SynLoc 2026 は camera calibration が与えられる単一フレーム課題なので，勝負どころは「どの image-space 表現から world-space の位置を最も正確に復元できるか」にある．

* 実装優先度としては，まず baseline 再現，その後 `higher resolution`, `better point representation`, `synthetic-to-real adaptation` を入れるのが妥当．

## 1. Official SynLoc Pipeline

### 1.1 Devkit baseline

`sskit` の簡易 baseline は，一般物体検出器で人を検出し，bbox 下辺中央を選手位置とみなして地面へ射影する．

概略:

1. image 上で `person` detection
2. bbox の `bottom center` を取る
3. camera calibration で ground plane に射影
4. world coordinate 上で評価

これは最小構成としては分かりやすいが，真の player location の proxy としては粗い．

Source:

* <https://github.com/Spiideo/sskit>

### 1.2 Official mmpose baseline

公式 baseline は `Spiideo/mmpose` の `spiideo_scenes` ブランチで，`YOLOX-pose` ベースの top-down pose 系を使う．

重要な点:

* 検出対象は人

* 回帰する keypoint は 2 個

* `pelvis`

* `pelvis_ground`

`pelvis_ground` は player pelvis を ground plane に正射影した点に対応する．評価もこの点を world-space に戻した位置で行う．

Source:

* <https://github.com/Spiideo/mmpose/tree/spiideo_scenes>

* <https://raw.githubusercontent.com/Spiideo/mmpose/spiideo_scenes/configs/body_bev_position/spiideo_soccernet/yoloxpose_m_4xb64-300e_640.py>

* <https://raw.githubusercontent.com/Spiideo/mmpose/spiideo_scenes/configs/_base_/datasets/spiideo_soccernet_synloc.py>

## 2. What The SynLoc Paper Shows

SynLoc 論文では，bbox ベースと pose ベースを直接比較している．結果として，`world-space localization` では pose ベースが大きく上回った．

論文中の主要な示唆:

* bbox の下辺中央は，真の player ground position の近似として弱い

* `pelvis` （'骨盤'という意味）と `ground projection of pelvis` のような幾何的に意味のある点を学習した方がよい

* 入力解像度を上げると成績がかなり改善する

論文の Table 2 では，`YOLOX-m` の比較で以下の傾向が見える．

* `bbox 960x960`: `mAP-LocSim 52.4`

* `pose 960x960`: `mAP-LocSim 79.3`

差が大きいため，このタスクで最優先なのは bbox を磨くことではなく，`position-defining point` をどれだけ正確に出せるかだと考えてよい．

Source:

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

Source:

* <https://raw.githubusercontent.com/SoccerNet/sn-gamestate/main/README.md>

* <https://www.sciencedirect.com/science/article/pii/S1077314226000792>

### 3.2 Player localization from a single moving view

過去研究 `Individual Locating of Soccer Players from a Single Moving View` でも，

1. sports field registration
2. 2D player tracking
3. homography による pitch projection

の 3 段構成が中心である．ここでも本質は，camera が未知または時間変化することへの対処にある．

SynLoc 2026 では calibration が既知なので，この pipeline 全体を持ち込む必要はない．参考にすべきなのは「image-space から world-space への幾何の扱い方」であり，tracking 系そのものではない．

Source:

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

候補:

* `pelvis`, `pelvis_ground`, `left_foot`, `right_foot`

* `pelvis`, `head`, `pelvis_ground`

* `bbox` と `pose point` のハイブリッド

狙い:

* 遮蔽時に単一 keypoint より頑健にする

* player scale / stance に応じて ground position をより安定に推定する

### 5.2 Direct world-coordinate regression

image keypoint を経由せず，検出ごとに `(x_world, y_world)` を直接回帰する案もある．

利点:

* task に対して表現が直接的

* image-space keypoint の誤差を world-space へ変換する段階を減らせる

懸念:

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

有望な方向:

* color jitter の強化

* blur, noise, jpeg compression

* illumination / weather の擬似変化

* background texture 変動への頑健化

* stronger occlusion augmentation

さらに可能なら:

* 公開 real soccer imagery を使った self-training

* 公開データでの person / pose pretraining

ただし，利用データは challenge rules の範囲内である必要がある．

### 5.5 Confidence calibration

`mAP-LocSim` は「見つけること」と「位置が近いこと」の両方を見るため，confidence の設計が重要である．

やるべきこと:

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
