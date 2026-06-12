# 20260612 Data Loader Audit

## 目的

10年分のスクレイピング生データ `race_results.csv` と `race_meta.csv` を監査し、未来データのリークを避けるデータローダーを実装・検証する。

## 変更内容

- `src/data/dataloader.py` を追加。
- `race_id` による `race_results -> race_meta` の many-to-one 結合を検証。
- 特徴量行列 `X`、正解・利益計算用 `y`、行メタデータ `metadata` を分離。
- post-race列が `X` に混入しない禁止列チェックを追加。
- 馬名ベースの過去走特徴量を、日付順にソートして過去行だけから生成。
- 開催日ベースの train/test split と time-series CV index generator を追加。

## 使用データ期間

- 開始日: 2015-01-04
- 終了日: 2026-05-24
- 開催日数: 1,234日

## 入力データ監査結果

### race_results.csv

- 行数: 526,813
- 列数: 18
- `race_id` ユニーク数: 37,627
- `(race_id, 馬番)` 重複: 0
- 着順の数値化率: 99.187%
- 非数値着順: 4,284行
  - `中`: 2,369
  - `除`: 1,066
  - `取`: 829
  - その他降着/失格表記: 少数
- 単勝オッズ coverage: 99.640%
- 人気 coverage: 99.640%
- 馬体重 parse coverage: 99.826%

### race_meta.csv

- 行数: 39,400
- 列数: 4
- `race_id` ユニーク数: 39,400
- `race_id` 重複: 0
- `kaisai_date` 不正値: 0
- `course_type`: `dirt`, `turf`, `obstacle`
- `distance_m`: 1000mから4260m

## 結合監査

- `race_results` 側で `race_meta` に結合できない行: 0
- `race_meta` 側だけに存在するレース: 1,773
- `results -> meta` の many-to-one left merge: OK

## リーク分類

### レース前情報

`race_id`, `kaisai_date`, `course_type`, `distance_m`, `枠番`, `馬番`, `馬名`, `性齢`, `斤量`, `騎手`, `単勝`, `人気`, `馬体重`, `調教師`, `馬主`

### 特徴量に直接入れてはいけないレース後情報

`着順`, `タイム`, `着差`, `通過`, `上り`, `賞金(万円)`

### ターゲット・バックテスト情報

`finish_position`, `is_win`, `is_top3`, `win_payout_per_100yen`, `win_profit_per_100yen`

## モデル設定・ハイパーパラメータ

モデル学習は未実施。今回の対象はデータ監査とデータローダー実装のみ。

## 評価指標

モデル評価は未実施。データローダー検証では以下を確認。

- `X` shape: `(522548, 37)`
- `y` shape: `(522548, 5)`
- `metadata` shape: `(522548, 5)`
- `X` 内の禁止列混入: 0
- 日付範囲: 2015-01-04 から 2026-05-24

## 検証コマンド

```bash
py -m py_compile src\data\dataloader.py
py src\data\dataloader.py
```

## 検証結果

成功。

```text
X shape: (522548, 37)
y shape: (522548, 5)
date range: 2015-01-04 00:00:00 -> 2026-05-24 00:00:00
```

追加確認として、`load_dataset()`, `split_dataset_by_date()`, `iter_time_series_splits()` の実データ実行も成功。

## 得られた知見

- 現CSVだけでも、単勝オッズ・人気・距離・芝ダート・枠・斤量・馬体重・騎手・調教師を使った基礎的な期待値探索は可能。
- 複勝オッズ、公式払戻金、競馬場、馬場状態、天候、各種IDは存在しない。
- 過去走ロールアップは `馬名 + kaisai_date` で実装可能だが、将来的には `horse_id` が望ましい。
- 明示的な払戻金がないため、単勝回収は `着順==1` と `単勝` から近似生成している。

## Next Action

- `.gitignore` に raw CSV、キャッシュ、中間ファイルを追加する。
- 不足データの取得方針を決める。
- 単勝のみの最小バックテストを実装し、ROI、投資額、払戻額、最大ドローダウンを算出する。
- `複勝`, `馬場状態`, `競馬場`, `天候`, `horse_id` を追加できる取得経路を調査する。
