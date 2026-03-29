# About\_Competition.md

## Competition

* 名称: `Spiideo SoccerNet SynLoc 2026`

* 主催: `SoccerNet Challenges 2026`

* タスク: 単一フレームから全選手を検出し，各選手の pitch 上のワールド座標をメートル単位で推定する．

* データの特徴: synthetic data と real broadcast data の混在．課題の本質は，スタジアム・カメラ・ドメイン差をまたいだ選手検出と位置推定．

## Official Links

* SoccerNet 2026: <https://www.soccer-net.org/challenges/2026>

* Competition devkit: <https://github.com/Spiideo/sskit>

* Baseline: <https://github.com/Spiideo/mmpose/tree/spiideo_scenes>

* Codabench challenge: <https://www.codabench.org/competitions/10155/>

* Rules: <https://github.com/Spiideo/mmpose/blob/spiideo_scenes/ChallengeRules.md>

## Task Summary

SoccerNet 2026 の公式ページでは，この課題を「Single-Frame World-Coordinate Athlete Detection & Localization」と説明している．要求されるのは，各フレーム中の全選手を検出し，その選手がピッチ上のどこにいるかをワールド座標で予測すること．評価は検出と位置推定の両方を反映する `mAP-LocSim` で行われる．

開発キット `sskit` では，結果は COCO 風フォーマットを拡張した JSON で表現され，各 detection に `position_on_pitch` を含める．便宜上，画像上の keypoint を与えてカメラモデルで ground plane に射影する形式もサポートされている．

## Evaluation

評価指標は `mAP-LocSim` である．これは COCO 形式のmAP(`mean Average Precision`, 平均適合率) をベースにしつつ，通常の IoU の代わりに `LocSim` を使って，検出とピッチ座標推定を同時に評価する指標である．

**mAPってなんだっけ：**

まず適合率曲線の面積 (`Average Precision`, `AP`) は，ある対応付け条件 (`matching criterion`) を固定したときの `precision-recall curve` の面積である．ここで対応付け条件とは，「ある予測を正解とみなしてよい条件」のことである．SynLoc では，この条件は後述の `LocSim >= t` で与えられる．高confidenceの prediction から順に並べ，

* その prediction が GT とマッチすれば `TP` (`True Positive`)

* match しなければ `FP` (`False Positive`)

として `precision` (適合率) と `recall` (再現率) を累積的に計算し，その曲線の面積を `AP` とする．

平均適合率 (`mAP`) はその `AP` を複数の閾値で平均したものだと考えればよい．

<br />

このmAPは，COCO 形式では通常 `IoU threshold` `0.50, 0.55, ..., 0.95` で AP を計算して平均するが，SynLoc ではこの IoU の代わりに `LocSim threshold` `t` を使う．ここで使う `t` は `0.50, 0.55, ..., 0.95` の 10 個である．

したがって `mAP-LocSim` は概念的には

```text
mAP-LocSim = mean_t AP_at_LocSim_threshold(t)
where t in {0.50, 0.55, ..., 0.95}
```

である．このタスクでは画像上 (`image space`) の bbox や keypoint ではなく，最終的に `position_on_pitch` として出るワールド座標 (`world coordinate`) が正しいかが評価される．

### What is evaluated

`LocSim` の実装は公式 devkit `sskit.coco.LocSimCOCOeval` にある．予測 (`prediction`) の pitch 座標を `p_dt = (x_dt, y_dt)`，正解 (`ground truth`) を `p_gt = (x_gt, y_gt)` とすると，まず二乗距離

```text
d^2 = (x_dt - x_gt)^2 + (y_dt - y_gt)^2
```

を計算し，その後

```text
LocSim = exp(log(0.05) * d^2 / tau^2)
```

で類似度 (`similarity`) を定義している．このコンペでは許容距離パラメータは `tau = 1` で固定されている．したがって同値な書き方をすると

```text
LocSim = 0.05^(d^2)
```

となる．つまり 1m を基準スケールにして〜いることになる．性質は以下：

* 距離 `d = 0` なら `LocSim = 1`

* 距離 `d = 1m` なら `LocSim = 0.05`

* 距離が増えると指数関数的に急減する

フレーム正解率 (`frame accuracy`) では，各画像について「偽陽性 (`false positive`) も偽陰性 (`false negative`) もなく，全選手が正しく検出されたか」を見る．このとき正しい検出 (`detection`) とみなす境界は `LocSim = 0.5` であり，これは約 `0.48m` の位置誤差に対応する．なお，ここでの `LocSim threshold` は評価指標そのものの定義に使う閾値であり，提出時に使う `score_threshold` とは別物である．`LocSim threshold` は prediction と GT を対応付けてよいかを決めるための閾値であり，`score_threshold` は confidence の低い prediction を捨てるかを決めるための閾値である．

公式 baseline config では，evaluation を 3 系統に分けている．

* `bbox`: 通常の bbox 検出評価

* `locsim_bbox`: bbox の下辺中央 (`bottom-center`) を pitch へ射影して測る比較用指標

* `locsim`: 予測した `position_on_pitch`，または `position_from_keypoint_index` で指定した keypoint を地面平面 (`ground plane`) へ射影した位置で測る指標

`locsim_bbox` は「bbox 下辺中央を位置とみなしたらどこまで行けるか」を見るためのもので，`locsim` がこのチャレンジの評価指標である．baseline では `position_from_keypoint_index=1` が使われ，これは `pelvis_ground` keypoint をワールド上の位置 (`world location`) として扱うことを意味する．（補足：pelvisは'骨盤'という意味らしい）

## Submission Format

提出物は zip で，以下の 2 ファイルを含む．

* `results.json`

* `metadata.json`

`metadata.json` の最小例:

```json
{"score_threshold": 0.6848765313625336}
```

baseline README では，`Spiideo/mmpose` の `spiideo_scenes` ブランチで `python tools/test.py ... --challenge` を実行すると `results.json` と `metadata.json` が生成され，それを zip 化して提出すると説明されている．

## Data Usage Rules

公式ガイドラインから重要な制約を抜き出すと以下：

* private dataset の使用は禁止

* challenge 期間中の手動追加アノテーションは禁止

* すべての参加者が同条件で利用できる公開データのみ使用可能

* 既存公開データに対する自動的なラベル補完・擬似ラベル生成・特徴抽出は，再現可能で説明可能なら許可される．

* 既存ラベルの修正提案は challenge 外で共有・反映するのはよいが，challenge 中に手動修正した版を自チームだけで使うのは不可．

* 新しい共有データは，締切の少なくとも 1 か月前までに Discord 等で全参加者へ公開される必要がある．

## Important Dates

`ChallengeRules.md` の記載:

* `2025-09-09`: test/challenge evaluation server open

* `2026-04-25`: evaluation server close

* `2026-05-01`: report submission deadline

* `TBD`: CVSports Workshop at CVPR 2026 で表彰

## Baseline / Toolkit Notes

* baseline 実装は `Spiideo/mmpose` の `spiideo_scenes` ブランチ

* devkit は `Spiideo/sskit`

* baseline README には，paper 掲載値と leaderboard 値に差があり，これは `sskit` 側の bug fix 後に評価値が少し変わったためと記載されている．

* データダウンロードは `research.spiideo.com` 経由，または `SoccerNet` Python package を用いた取得手順が案内されている．

* データ配置： `data/SoccerNet/SpiideoSynLoc`．

## Practical Implications For This Repo

* まずは baseline 追試と提出 zip 生成までを再現する

* validation で決めた `score_threshold` を記録する

* 追加改善を入れる場合も，使用データが公開・再現可能であることを都度確認する

* 提出直前は締切日・提出形式・Codabench の受付状態を再確認する

## Sources

* SoccerNet 2026 challenge page: <https://www.soccer-net.org/challenges/2026>

* Spiideo SynLoc devkit README: <https://github.com/Spiideo/sskit>

* Spiideo SynLoc baseline README: <https://github.com/Spiideo/mmpose/tree/spiideo_scenes>

* Official rules: <https://github.com/Spiideo/mmpose/blob/spiideo_scenes/ChallengeRules.md>

* Codabench challenge page: <https://www.codabench.org/competitions/10155/>
