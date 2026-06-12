# 20260612 payout cache audit

## 目的

`_payout_cache.jsonl` のスキーマを無仮定で確定し、レシピ群C実装のゲートを通す。

## 基本統計

- ファイル: `_payout_cache.jsonl` (6,845,762 bytes)
- 行数: 14,054 / JSONパース失敗: 0
- race_id抽出キー: {'race_id': 14054}
- ユニーク race_id: 14,054

## トップレベルキー出現数

- `race_id`: 14,054
- `tansho`: 14,047
- `fukusho`: 14,047
- `umaren`: 14,047
- `wide`: 14,047
- `umatan`: 14,047
- `fuku3`: 14,047
- `tan3`: 14,047
- `wakuren`: 13,247
- `_empty`: 7

## キー別の値型

- `race_id:str`: 14,054
- `tansho:list`: 14,047
- `fukusho:list`: 14,047
- `umaren:list`: 14,047
- `wide:list`: 14,047
- `umatan:list`: 14,047
- `fuku3:list`: 14,047
- `tan3:list`: 14,047
- `wakuren:list`: 13,247
- `_empty:bool`: 7

## 券種の存在数

- 8券種のキーは検出できず。サンプル構造から命名を確認すること。

## サンプル構造（切り詰め表示）

### record 1

```json
{
  "race_id": "202604010701",
  "tansho": [
    {
      "nums": [
        "..."
      ],
      "yen": 990
    }
  ],
  "fukusho": [
    {
      "nums": [
        "..."
      ],
      "yen": 180
    },
    {
      "nums": [
        "..."
      ],
      "yen": 110
    },
    {
      "nums": [
        "..."
      ],
      "yen": 130
    }
  ],
  "...": "(+6 keys)"
}
```

### record 2

```json
{
  "race_id": "202604010702",
  "tansho": [
    {
      "nums": [
        "..."
      ],
      "yen": 510
    }
  ],
  "fukusho": [
    {
      "nums": [
        "..."
      ],
      "yen": 160
    },
    {
      "nums": [
        "..."
      ],
      "yen": 200
    },
    {
      "nums": [
        "..."
      ],
      "yen": 150
    }
  ],
  "...": "(+6 keys)"
}
```

### record 3

```json
{
  "race_id": "202604010703",
  "tansho": [
    {
      "nums": [
        "..."
      ],
      "yen": 1370
    }
  ],
  "fukusho": [
    {
      "nums": [
        "..."
      ],
      "yen": 250
    },
    {
      "nums": [
        "..."
      ],
      "yen": 1080
    },
    {
      "nums": [
        "..."
      ],
      "yen": 130
    }
  ],
  "...": "(+6 keys)"
}
```

## raw CSV との突合

- cache ∩ race_meta: 14,054 / cache 14,054 / meta 39,400
- cache ∩ race_results: 14,054 / results 37,627
- meta のみ（resultsなし）レースの cache 充足: 0 / 1,773

## 合格条件（report 20260612 §6.0）

- [x] 全行JSONパース可（失敗0 / 14,054行）
- [x] race_id join率 ≥ 95%（meta・resultsとも100%）
- [x] 8券種の存在率を確定（ローマ字キー: tansho/fukusho/wakuren/umaren/wide/umatan/fuku3/tan3、形式は `{nums: [馬番...], yen: int}`。wakuren は13,247件=少頭数レースで非発売、`_empty`フラグ7件）

## 追加所見（2026-06-12 23:00 JST 手動確認）

- 年次カバレッジ: 2022=3,433 / 2023=3,453 / 2024=2,310 / 2025=3,449 / 2026=1,409。**2022年以降のみ**。
- 【2026-06-13 訂正】2024年の見かけの欠落（約1,100件）は cache 側の穴ではなく **`race_results.csv` 側が2024年を2,313レースしか持っていない**ことが原因。cache は results の2024年レースを 2,310/2,313 カバーしており欠落は3件のみ。root CSV の2024年欠落分は SPEC-1 `scraper/data/race_db/2024.jsonl`（3,442レース、payouts/weather/track_condition/class_text保持）から正規化取り込みで補完可能。
- 含意:
  - レシピ群C（ゾーン×文脈EVマップ）は約4年分の履歴で構築可能。時間減衰半減期は2年のままでよいが、縮約kは強めに。
  - 2015-2021年の単勝払戻は引き続き `単勝×100` 近似に依存（14,047レースでの厳密一致検証済み、`results/20260612_value_loss_design.md` 参照）。
- 結論: **レシピ群Cの実装ゲートは通過**。ただし学習履歴の制約上、EVマップ検証は2023年以降をテスト期間とするwalk-forwardに限定される。
