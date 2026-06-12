# SPEC-3 時系列オッズロガー実装・検証レポート

- 日付: 2026-06-12
- 担当: Claude（設計・実装・検証）
- 関連設計書: `results/20260612_value_loss_design.md` §7 SPEC-3

## 目的

確定オッズと購入時点オッズの乖離（バックテスト上方バイアスの定量化）および直前オッズ変動特徴量のため、JRA各レースの単勝・複勝オッズを発走前の複数時点で前向きに収集する。過去への遡及取得は不可能であり、収集開始の遅れは検証データの恒久的損失となる。

## 変更内容

- 新規: `scraper/odds_timeseries_logger.py`（ロガー本体）
- 新規: `scraper/probe_netkeiba.py`（エンドポイント検証プローブ。SPEC-2のカバレッジプローブにも転用可）
- `.gitignore` に `scraper/data/` を追加（収集データのコミット混入防止）

## 確定したエンドポイント仕様（2026-06-12 実測）

1. レース一覧+発走時刻: `https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={YYYYMMDD}`
   - HTML断片。`<li>`内の `race_id=(\d{12})` と時刻 `HH:MM` をペアリング
2. オッズJSON: `https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={id}&type=1&action=init`
   - `data.odds["1"]` = 単勝 `{馬番: [オッズ, "0.0", 人気]}`
   - `data.odds["2"]` = 複勝 `{馬番: [下限, 上限, 人気]}` ← **複勝レンジが取得可能（SPEC-2の主要懸念を解消）**
   - `data.official_datetime` = 公式オッズ時刻

## 最重要の発見: api_status の弁別

- `status == "result"`: 実市場オッズ（過去レースで確認。公式オッズ時刻つき）
- `status == "yoso"`: **netkeiba 予想オッズ。市場オッズではない**（発売前に返る。複勝レンジは空）
- 金曜13:32時点の翌日レースは全件 `yoso` だった。**ドリフト分析・特徴量生成では `api_status == "result"` のみを使用すること（厳守）**。レコードには全件 `api_status` を保存済み

## 主要設定

- スナップショット時点: 発走 T−60 / −30 / −15 / −5 / −1 分（`SNAPSHOT_OFFSETS_MIN`）
- レートリミット: 全リクエスト共通 1.5秒以上間隔、リトライ 2/5/10 秒バックオフ
- 期限超過スキップ: 予定時刻+120秒を過ぎたスナップショットは取得しない（T-ラベルの意味を汚染しないため）
- 再起動安全: 取得済みラベルを JSONL から読み戻して二重取得を防止
- タイムゾーン: JST 固定オフセット（DSTなし、tzdata 非依存）
- 出力: `scraper/data/odds_timeseries/{YYYYMMDD}/{race_id}.jsonl`（1行=1スナップショット）

## 実行した検証コマンド

```
py -m py_compile scraper/odds_timeseries_logger.py
py scraper/odds_timeseries_logger.py --list-only --date 20260613   # 36レース・発走時刻ペアリング確認
py scraper/odds_timeseries_logger.py --once --date 20260613        # 36/36 取得成功・JSONL書込確認
py scraper/odds_timeseries_logger.py --dry-run --date 20260613     # 180イベント、08:45〜16:29 整列確認
```

## 成功/失敗

成功。一覧パース（3場36R）、API取得（36/36）、JSONL書込、スケジュール生成（180件）まで実機検証済み。
未検証: 当日デーモンの長時間実走（明日 2026-06-13 が初回実走）。`result` ステータスのオッズが当日朝に取得できることは過去レースの応答形式から間接確認のみ。

## 運用手順

開催日の朝（初レースT−60より前、目安 08:20 まで）に起動:

```
py C:\Users\morim\dev\Pegasus-Edge\scraper\odds_timeseries_logger.py
```

自動化する場合（毎日08:25起動。非開催日は即終了するので安全）:

```
schtasks /Create /TN "PegasusEdge_OddsLogger" /SC DAILY /ST 08:25 /TR "cmd /c py C:\Users\morim\dev\Pegasus-Edge\scraper\odds_timeseries_logger.py >> C:\Users\morim\dev\Pegasus-Edge\scraper\data\odds_logger.log 2>&1"
```

注意: 実行中はPCのスリープを無効にすること（スリープすると該当時間帯のスナップショットは期限超過で欠測になる）。

## 得られた知見

- netkeiba は発売前に予想オッズを返すため、`api_status` を保存しない収集設計は将来のデータ汚染事故につながる（本実装は保存済み）
- 複勝レンジ（下限・上限）が同一APIで取れるため、SPEC-2 で計画していた複勝オッズ取得は同エンドポイントの過去レースカバレッジプローブに帰着する

## Next Action

1. 2026-06-13（土）にデーモン初実走、T−60〜T−1 の `result` オッズ取得を確認
2. 数週間分蓄積後、確定オッズ（race_results.単勝）との乖離分布を分析（バックテスト補正項の推定）
3. SPEC-2: 同APIで過去レースの複勝確定オッズがどこまで遡れるか、各年5レースのカバレッジプローブ（`scraper/probe_netkeiba.py` を拡張）

## Codex にレビューしてほしい点

- `fetch_race_list` の `<li>` ベースの時刻ペアリングの頑健性（netkeiba の DOM 変更耐性）
- 期限超過スキップ（120秒）とリトライ（最大17秒+リクエスト時間）の整合性: T−1 スナップショットがリトライで T+0 を跨いだ場合の扱い
- `--once` モードは done-label チェックを意図的にしない（毎回追記）仕様の妥当性
