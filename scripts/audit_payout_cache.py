"""Schema-agnostic audit of the local ``_payout_cache.jsonl`` (report §6.0).

The cache is gitignored and its schema is undocumented, so this script makes
no assumption beyond "JSON Lines".  It reports line/parse counts, the key
inventory with value types, presence of the eight JRA bet types, a truncated
structural dump of sample records, and race_id join coverage against the raw
CSVs.  Output goes to stdout and to a Markdown report under ``results/``.

Usage (local machine, where the data lives):
    py scripts/audit_payout_cache.py --cache _payout_cache.jsonl \
        --results race_results.csv --meta race_meta.csv
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
from pathlib import Path

JRA_BET_TYPES = ("単勝", "複勝", "枠連", "馬連", "ワイド", "馬単", "三連複", "三連単")
RACE_ID_CANDIDATE_KEYS = ("race_id", "raceId", "id", "race")


def truncate(obj: object, *, max_items: int = 3, max_str: int = 60, depth: int = 0) -> object:
    if depth >= 4:
        return "..."
    if isinstance(obj, dict):
        items = list(obj.items())[:max_items]
        shortened = {k: truncate(v, max_items=max_items, max_str=max_str, depth=depth + 1) for k, v in items}
        if len(obj) > max_items:
            shortened["..."] = f"(+{len(obj) - max_items} keys)"
        return shortened
    if isinstance(obj, list):
        shortened_list = [truncate(v, max_items=max_items, max_str=max_str, depth=depth + 1) for v in obj[:max_items]]
        if len(obj) > max_items:
            shortened_list.append(f"... (+{len(obj) - max_items} items)")
        return shortened_list
    if isinstance(obj, str) and len(obj) > max_str:
        return obj[:max_str] + "..."
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="_payout_cache.jsonl")
    parser.add_argument("--results", default="race_results.csv")
    parser.add_argument("--meta", default="race_meta.csv")
    parser.add_argument("--out", default=None, help="Markdown report path (default: results/YYYYMMDD_payout_cache_audit.md)")
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()

    cache_path = Path(args.cache)
    n_lines = 0
    n_parse_errors = 0
    key_counter: collections.Counter[str] = collections.Counter()
    type_counter: collections.Counter[str] = collections.Counter()
    bet_type_counter: collections.Counter[str] = collections.Counter()
    race_ids: set[str] = set()
    race_id_key_counter: collections.Counter[str] = collections.Counter()
    samples: list[object] = []

    with cache_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                n_parse_errors += 1
                continue
            if len(samples) < args.samples:
                samples.append(truncate(record))
            if not isinstance(record, dict):
                type_counter[f"<top-level>:{type(record).__name__}"] += 1
                continue
            for key, value in record.items():
                key_counter[key] += 1
                type_counter[f"{key}:{type(value).__name__}"] += 1
            for bet_type in JRA_BET_TYPES:
                if bet_type in record:
                    bet_type_counter[bet_type] += 1
                else:
                    for value in record.values():
                        if isinstance(value, dict) and bet_type in value:
                            bet_type_counter[f"nested:{bet_type}"] += 1
                            break
            for candidate in RACE_ID_CANDIDATE_KEYS:
                if candidate in record:
                    race_ids.add(str(record[candidate]))
                    race_id_key_counter[candidate] += 1
                    break

    lines: list[str] = []
    lines.append(f"# {dt.date.today():%Y%m%d} payout cache audit")
    lines.append("")
    lines.append("## 目的")
    lines.append("")
    lines.append("`_payout_cache.jsonl` のスキーマを無仮定で確定し、レシピ群C実装のゲートを通す。")
    lines.append("")
    lines.append("## 基本統計")
    lines.append("")
    lines.append(f"- ファイル: `{cache_path}` ({cache_path.stat().st_size:,} bytes)")
    lines.append(f"- 行数: {n_lines:,} / JSONパース失敗: {n_parse_errors:,}")
    lines.append(f"- race_id抽出キー: {dict(race_id_key_counter) or '検出できず（要手動確認）'}")
    lines.append(f"- ユニーク race_id: {len(race_ids):,}")
    lines.append("")
    lines.append("## トップレベルキー出現数")
    lines.append("")
    for key, count in key_counter.most_common():
        lines.append(f"- `{key}`: {count:,}")
    lines.append("")
    lines.append("## キー別の値型")
    lines.append("")
    for key_type, count in type_counter.most_common(40):
        lines.append(f"- `{key_type}`: {count:,}")
    lines.append("")
    lines.append("## 券種の存在数")
    lines.append("")
    if bet_type_counter:
        for bet_type, count in bet_type_counter.most_common():
            lines.append(f"- {bet_type}: {count:,}")
    else:
        lines.append("- 8券種のキーは検出できず。サンプル構造から命名を確認すること。")
    lines.append("")
    lines.append("## サンプル構造（切り詰め表示）")
    lines.append("")
    for index, sample in enumerate(samples, start=1):
        lines.append(f"### record {index}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(sample, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")

    lines.append("## raw CSV との突合")
    lines.append("")
    try:
        import pandas as pd

        meta_ids = set(pd.read_csv(args.meta, usecols=["race_id"], dtype=str)["race_id"])
        result_ids = set(pd.read_csv(args.results, usecols=["race_id"], dtype=str)["race_id"])
        lines.append(f"- cache ∩ race_meta: {len(race_ids & meta_ids):,} / cache {len(race_ids):,} / meta {len(meta_ids):,}")
        lines.append(f"- cache ∩ race_results: {len(race_ids & result_ids):,} / results {len(result_ids):,}")
        lines.append(f"- meta のみ（resultsなし）レースの cache 充足: {len((meta_ids - result_ids) & race_ids):,} / {len(meta_ids - result_ids):,}")
    except FileNotFoundError as error:
        lines.append(f"- CSVが見つからないため突合をスキップ: {error}")
    lines.append("")
    lines.append("## 合格条件（report 20260612 §6.0）")
    lines.append("")
    lines.append("- [ ] 全行JSONパース可（失敗0）")
    lines.append("- [ ] race_id join率 ≥ 95%")
    lines.append("- [ ] 8券種の存在率と払戻金の数値化率を確定")

    report = "\n".join(lines)
    out_path = Path(args.out) if args.out else Path("results") / f"{dt.date.today():%Y%m%d}_payout_cache_audit.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
