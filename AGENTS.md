# AGENTS.md

このリポジトリは、`Spiideo SoccerNet SynLoc 2026` への参加・提出作業を進めるための作業ディレクトリである。エージェントは以下を前提に行動すること。

## 目的

- 目標は `Spiideo SoccerNet SynLoc 2026` で有効かつ優秀な成績を収める提出物を作ること。
- コンペ概要・制約・提出仕様の要約は [About_Competition.md](/Users/minorisugimura/GitHub/soccerNet2026/About_Competition.md) を参照すること。
- 競技上の最終的な正本は公式ページ・公式リポジトリ・公式ガイドラインとし、このファイルは運用上の作業ルールを定める。

## このリポジトリでの基本方針

- まず現状の構成・既存ファイル・未コミット変更を確認してから作業する。
- ユーザーが明示していない既存変更は巻き戻さない。
- 生データや配布物を破壊する操作はしない。特に `data/` 配下の削除・上書きは必要性を確認してから行う。
- 変更は再現可能性を優先し、手順・依存関係・実行コマンドをファイルに残す。
- コンペ用の重要な仮定は README 系ドキュメントに明記し、暗黙知にしない。

## 競技ルールに関する必須制約

- 追加の private dataset は使わない。
- challenge 期間中に手動で追加した独自アノテーションは使わない。
- 使うデータは、全参加者が同条件でアクセスできる公開データに限る。
- 既存公開データからの自動的なラベル補完・特徴抽出・擬似ラベル生成は、再現可能で説明可能なら許容される。
- 新しい公開データや共有資産を前提にする場合は、公式ガイドライン上「締切の少なくとも 1 か月前までに全参加者へ共有」が必要である点を前提に判断する。

## 実装・実験の優先順位

- 最優先は「提出可能な推論パイプライン」を早く固めること。
- 次に、validation/test 相当での再現評価、閾値の扱い、提出 zip 生成を安定化する。
- ベースライン追試だけでなく、提出形式 `results.json` と `metadata.json` の整合性確認を必ず行う。
- metric は `mAP-LocSim` が中心であるため、単純な検出精度だけでなく pitch 上の位置推定誤差も意識して改善する。

## データ・パスの前提

- データ配置の基準パスは `data/SoccerNet/SpiideoSynLoc` とする。
- 公式 devkit は `sskit`、公式 baseline は `Spiideo/mmpose` の `spiideo_scenes` ブランチである。
- 外部リポジトリ由来のコードや設定を取り込む場合は、参照元 URL または commit を記録する。

## 提出物に関する前提

- challenge 提出は通常 `results.json` と `metadata.json` を zip 化したものになる。
- `metadata.json` には少なくとも `score_threshold` が必要で、必要に応じて `position_from_keypoint_index` を含める。
- 提出前に、どの split で閾値を決めたかを明確に記録する。validation で決めた閾値を test/challenge に適用する前提を崩さない。

## 情報源の優先順位

優先順位は以下。

1. 公式ガイドライン `ChallengeRules.md`
2. 公式 challenge ページと SoccerNet 2026 challenge ページ
3. 公式 devkit / baseline README
4. このリポジトリ内のメモ

情報が矛盾する場合は、矛盾点を明示してから作業を進めること。
