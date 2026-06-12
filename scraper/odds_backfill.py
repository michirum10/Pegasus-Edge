"""SPEC-2: 全過去レースの確定オッズ (単勝 + 複勝レンジ) バックフィル。

カバレッジプローブ (2026-06-12) で、オッズAPIが2015年まで全年
status=result + 全馬の複勝レンジを返すことを確認済み。
これにより設計書の判定C (複勝EV検証不可) が覆る。

- 出力: scraper/data/odds_final/{year}.jsonl (1行 = 1レース)
- 再開安全: 既存JSONLに記録済みの race_id はスキップする
- レートリミットは odds_timeseries_logger と共通 (1.5秒/リクエスト)

使い方:
  py scraper/odds_backfill.py --limit 5            # 動作確認
  py scraper/odds_backfill.py                      # 全年 (新しい年から)
  py scraper/odds_backfill.py --years 2024 2023    # 年を指定
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import requests

from odds_timeseries_logger import fetch_odds_snapshot

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent / "data" / "odds_final"
RETRY_WAITS_SEC = (2, 5, 10)


def load_race_ids_by_year() -> dict[str, list[str]]:
    by_year: dict[str, list[str]] = {}
    with (REPO_ROOT / "race_meta.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_year.setdefault(row["race_id"][:4], []).append(row["race_id"])
    return by_year


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["race_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def fetch_with_retry(race_id: str) -> dict | None:
    last_error: Exception | None = None
    for backoff in (0,) + RETRY_WAITS_SEC:
        if backoff:
            time.sleep(backoff)
        try:
            return fetch_odds_snapshot(race_id)
        except (requests.RequestException, ValueError) as e:
            last_error = e
    print(f"  FAILED {race_id}: {last_error}", flush=True)
    return None


def backfill_year(year: str, race_ids: list[str], limit: int | None) -> tuple[int, int]:
    path = DATA_DIR / f"{year}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_ids(path)
    todo = [r for r in race_ids if r not in done]
    if limit is not None:
        todo = todo[:limit]
    print(f"[{year}] 全{len(race_ids)} / 取得済{len(done)} / 今回{len(todo)}", flush=True)

    n_ok = n_fail = 0
    with path.open("a", encoding="utf-8") as f:
        for i, race_id in enumerate(todo, 1):
            snap = fetch_with_retry(race_id)
            if snap is None:
                n_fail += 1
                continue
            f.write(json.dumps({"race_id": race_id, **snap}, ensure_ascii=False) + "\n")
            n_ok += 1
            if i % 100 == 0:
                f.flush()
                print(f"[{year}] {i}/{len(todo)} (status={snap['api_status']})", flush=True)
    print(f"[{year}] 完了: 成功{n_ok} 失敗{n_fail}", flush=True)
    return n_ok, n_fail


def main() -> None:
    parser = argparse.ArgumentParser(description="SPEC-2 確定オッズバックフィル")
    parser.add_argument("--years", nargs="*", default=None,
                        help="対象年 (既定: 全年を新しい順)")
    parser.add_argument("--limit", type=int, default=None,
                        help="各年の取得数上限 (動作確認用)")
    args = parser.parse_args()

    by_year = load_race_ids_by_year()
    years = args.years or sorted(by_year, reverse=True)
    total_ok = total_fail = 0
    for year in years:
        if year not in by_year:
            print(f"[{year}] race_meta.csv に存在しない年のためスキップ")
            continue
        ok, fail = backfill_year(year, by_year[year], args.limit)
        total_ok += ok
        total_fail += fail
    print(f"全体完了: 成功{total_ok} 失敗{total_fail}")


if __name__ == "__main__":
    main()
