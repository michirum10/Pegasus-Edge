"""SPEC-1: db.netkeiba.com レースページのスクレイパー。

取得物 (IG>0 達成のための外部情報):
- 真の horse_id / jockey_id / trainer_id (馬番との対応)
- 天候、馬場状態、発走時刻、コース文字列
- レース名とクラス文字列 (新馬/未勝利/1勝/OP/G1等の判別材料)
- 払戻テーブル (2015-2021年の払戻キャッシュ拡張用)

- 出力: scraper/data/race_db/{year}.jsonl (1行 = 1レース)
- HTMLキャッシュ: scraper/data/race_db_html/{race_id}.html.gz
  (パーサ改修時に再クロール不要にするため生HTMLを保全する)
- 再開安全: 既存JSONLに記録済みの race_id はスキップ

使い方:
  py scraper/race_db_scraper.py --limit 2 --years 2015 2026   # 動作確認
  py scraper/race_db_scraper.py                                # 全年 (新しい順)
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)
HEADERS = {"User-Agent": UA, "Referer": "https://db.netkeiba.com/"}
RACE_URL = "https://db.netkeiba.com/race/{race_id}/"

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent / "data" / "race_db"
HTML_CACHE_DIR = Path(__file__).resolve().parent / "data" / "race_db_html"

MIN_REQUEST_INTERVAL_SEC = 1.5
RETRY_WAITS_SEC = (2, 5, 10)
PAYOUT_KINDS = ("単勝", "複勝", "枠連", "馬連", "ワイド", "馬単", "三連複", "三連単")

_last_request_at = 0.0


def fetch_html(race_id: str) -> str:
    """HTMLを取得する。gzipキャッシュがあればネットワークを使わない。"""
    cache = HTML_CACHE_DIR / f"{race_id}.html.gz"
    if cache.exists():
        return gzip.decompress(cache.read_bytes()).decode("EUC-JP", errors="replace")

    global _last_request_at
    wait = MIN_REQUEST_INTERVAL_SEC - (time.monotonic() - _last_request_at)
    if wait > 0:
        time.sleep(wait)
    resp = requests.get(RACE_URL.format(race_id=race_id), headers=HEADERS, timeout=20)
    _last_request_at = time.monotonic()
    resp.raise_for_status()
    resp.encoding = "EUC-JP"
    text = resp.text

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(gzip.compress(text.encode("EUC-JP", errors="replace")))
    return text


def parse_race_page(race_id: str, html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    record: dict = {"race_id": race_id}

    intro = soup.find("div", class_="data_intro")
    if intro is not None:
        h1 = intro.find("h1")
        record["race_name"] = h1.get_text(strip=True) if h1 else None
        intro_text = intro.get_text(" ", strip=True)
        record["weather"] = _search(r"天候\s*:\s*(\S+)", intro_text)
        record["track_condition"] = _search(r"(?:芝|ダート|障害)\s*:\s*(\S+)", intro_text)
        record["post_time"] = _search(r"発走\s*:\s*(\d{1,2}:\d{2})", intro_text)
        record["course_text"] = _search(r"(芝|ダ|障)[^/]*?(\d{3,4})m", intro_text, group=0)
        small = intro.find("p", class_="smalltxt")
        record["class_text"] = small.get_text(" ", strip=True) if small else None

    horses = []
    table = soup.find("table", class_=re.compile("race_table"))
    if table is not None:
        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) < 5:
                continue
            row_html = str(tr)
            horse_id = _search(r"/horse/(\w+)/", row_html)
            if horse_id is None:
                continue
            horses.append({
                "umaban": tds[2].get_text(strip=True),
                "horse_id": horse_id,
                "jockey_id": _search(r"/jockey/(?:result/recent/)?(\w+)/", row_html),
                "trainer_id": _search(r"/trainer/(?:result/recent/)?(\w+)/", row_html),
            })
    record["horses"] = horses

    payouts: dict[str, list[dict]] = {}
    for table in soup.find_all("table", class_=re.compile("pay_table")):
        for tr in table.find_all("tr"):
            th = tr.find("th")
            tds = tr.find_all("td")
            if th is None or len(tds) < 2:
                continue
            kind = th.get_text(strip=True)
            if kind not in PAYOUT_KINDS:
                continue
            nums = tds[0].get_text("\n", strip=True).splitlines()
            yens = tds[1].get_text("\n", strip=True).splitlines()
            payouts[kind] = [
                {"nums": n, "yen": _parse_yen(y)}
                for n, y in zip(nums, yens)
            ]
    record["payouts"] = payouts
    return record


def _search(pattern: str, text: str, group: int = 1) -> str | None:
    m = re.search(pattern, text)
    return m.group(group) if m else None


def _parse_yen(text: str) -> int | None:
    digits = text.replace(",", "").strip()
    return int(digits) if digits.isdigit() else None


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


def scrape_year(year: str, race_ids: list[str], limit: int | None) -> tuple[int, int]:
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
            record = None
            for backoff in (0,) + RETRY_WAITS_SEC:
                if backoff:
                    time.sleep(backoff)
                try:
                    record = parse_race_page(race_id, fetch_html(race_id))
                    break
                except requests.RequestException as e:
                    print(f"  retry {race_id}: {e}", flush=True)
            if record is None or not record["horses"]:
                n_fail += 1
                print(f"  FAILED {race_id} (horses={0 if record is None else len(record['horses'])})",
                      flush=True)
                continue
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_ok += 1
            if i % 100 == 0:
                f.flush()
                print(f"[{year}] {i}/{len(todo)}", flush=True)
    print(f"[{year}] 完了: 成功{n_ok} 失敗{n_fail}", flush=True)
    return n_ok, n_fail


def main() -> None:
    parser = argparse.ArgumentParser(description="SPEC-1 db.netkeiba レースページ取得")
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
        ok, fail = scrape_year(year, by_year[year], args.limit)
        total_ok += ok
        total_fail += fail
    print(f"全体完了: 成功{total_ok} 失敗{total_fail}")


if __name__ == "__main__":
    main()
