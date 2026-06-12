"""netkeiba エンドポイント検証プローブ (SPEC-3 事前調査)。

時系列オッズロガーが依存する2つのエンドポイントの生死と応答形式を確認する:
  1. レース一覧 + 発走時刻: race_list_sub.html
  2. 単勝・複勝オッズ JSON API: api_get_jra_odds.html

使い方:
  py scraper/probe_netkeiba.py 20260613
"""

import json
import sys
import time

import requests

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)
HEADERS = {"User-Agent": UA, "Referer": "https://race.netkeiba.com/"}
SLEEP_SEC = 1.5


def fetch(url: str) -> requests.Response:
    time.sleep(SLEEP_SEC)
    resp = requests.get(url, headers=HEADERS, timeout=20)
    print(f"GET {url}\n  -> status={resp.status_code} bytes={len(resp.content)} "
          f"content-type={resp.headers.get('content-type')}")
    return resp


def probe_race_list(kaisai_date: str) -> None:
    print(f"\n=== [1] レース一覧プローブ kaisai_date={kaisai_date} ===")
    url = f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={kaisai_date}"
    resp = fetch(url)
    resp.encoding = "EUC-JP"
    text = resp.text
    import re
    race_ids = sorted(set(re.findall(r"race_id=(\d{12})", text)))
    times = re.findall(r"(\d{1,2}:\d{2})", text)
    print(f"  race_ids found: {len(race_ids)}")
    print(f"  sample: {race_ids[:6]}")
    print(f"  time-like strings: {times[:10]}")
    snippet = text[:1200].replace("\n", " ")
    print(f"  head: {snippet[:600]}")


def probe_odds_api(race_id: str, label: str) -> None:
    print(f"\n=== [2] オッズAPIプローブ race_id={race_id} ({label}) ===")
    candidates = [
        f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=1&action=init",
        f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=1",
        f"https://race.netkeiba.com/odds/odds_get_form.html?type=b1&race_id={race_id}",
    ]
    for url in candidates:
        try:
            resp = fetch(url)
        except requests.RequestException as e:
            print(f"  !! request failed: {e}")
            continue
        body = resp.text.strip()
        print(f"  head(400): {body[:400]!r}")
        try:
            data = json.loads(body)
            print(f"  JSON OK. top-level keys: {list(data)[:10]}")
            if isinstance(data, dict) and "data" in data:
                inner = data["data"]
                if isinstance(inner, dict):
                    print(f"  data keys: {list(inner)[:10]}")
                    odds = inner.get("odds")
                    if isinstance(odds, dict):
                        for k, v in list(odds.items())[:3]:
                            sample = list(v.items())[:3] if isinstance(v, dict) else v
                            print(f"    odds[{k}] sample: {sample}")
        except (json.JSONDecodeError, TypeError):
            print("  (JSONではない)")


def main() -> None:
    kaisai_date = sys.argv[1] if len(sys.argv) > 1 else "20260613"
    probe_race_list(kaisai_date)
    # 過去レース (払戻キャッシュに存在することが確認済みのID) で形式確認
    probe_odds_api("202604010701", "past, 2026年・キャッシュ整合済み")


if __name__ == "__main__":
    main()
