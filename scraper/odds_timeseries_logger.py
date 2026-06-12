"""SPEC-3: 時系列オッズロガー（前向き収集）。

netkeiba から JRA 各レースの単勝・複勝オッズを発走前の複数時点
(T-60/-30/-15/-5/-1 分) でスナップショットし、JSONL に追記する。

確定オッズと購入時点オッズの乖離分布の推定、および直前オッズ変動
特徴量の材料となる。過去に遡って取得する手段は存在しないため、
このロガーは開催日に常時稼働させること。

使い方:
  py scraper/odds_timeseries_logger.py --list-only            # レース一覧の確認のみ
  py scraper/odds_timeseries_logger.py --once                 # 今すぐ全レースを1回取得
  py scraper/odds_timeseries_logger.py --date 20260613        # 当日デーモン（推奨）

出力: scraper/data/odds_timeseries/{YYYYMMDD}/{race_id}.jsonl
  1行 = 1スナップショット:
  {"race_id", "kaisai_date", "label", "fetched_at", "post_time",
   "api_status", "official_datetime", "tan": {馬番: オッズ},
   "fuku": {馬番: [下限, 上限]}, "ninki": {馬番: 人気}}

注意: api_status の弁別が重要（実測で確認済み）。
  "result" = 実市場オッズ（公式オッズ時刻つき）
  "yoso"   = netkeiba 予想オッズ（市場オッズではない。複勝レンジも空）
発売前は "yoso" が返る。記録自体は残すが（発売状況も情報のため）、
ドリフト分析・特徴量生成では api_status == "result" のみを使うこと。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# JST は DST を持たないため固定オフセットで十分（tzdata 依存を避ける）
JST = timezone(timedelta(hours=9), name="JST")

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)
HEADERS = {"User-Agent": UA, "Referer": "https://race.netkeiba.com/"}

RACE_LIST_URL = "https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date}"
ODDS_API_URL = (
    "https://race.netkeiba.com/api/api_get_jra_odds.html"
    "?race_id={race_id}&type=1&action=init"
)

DATA_DIR = Path(__file__).resolve().parent / "data" / "odds_timeseries"

SNAPSHOT_OFFSETS_MIN = (60, 30, 15, 5, 1)
MIN_REQUEST_INTERVAL_SEC = 1.5
RETRY_WAITS_SEC = (2, 5, 10)
LATE_TOLERANCE_SEC = 120

_last_request_at = 0.0


def _now() -> datetime:
    return datetime.now(JST)


def _log(msg: str) -> None:
    print(f"[{_now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _polite_get(url: str) -> requests.Response:
    """全リクエスト共通のレートリミット付き GET。"""
    global _last_request_at
    wait = MIN_REQUEST_INTERVAL_SEC - (time.monotonic() - _last_request_at)
    if wait > 0:
        time.sleep(wait)
    resp = requests.get(url, headers=HEADERS, timeout=20)
    _last_request_at = time.monotonic()
    resp.raise_for_status()
    return resp


def fetch_race_list(kaisai_date: str) -> list[dict]:
    """開催日のレース一覧を (race_id, 発走時刻) で返す。"""
    resp = _polite_get(RACE_LIST_URL.format(date=kaisai_date))
    resp.encoding = "UTF-8"
    soup = BeautifulSoup(resp.text, "html.parser")

    races: dict[str, dict] = {}
    for li in soup.find_all("li"):
        anchor = li.find("a", href=re.compile(r"race_id=\d{12}"))
        if anchor is None:
            continue
        race_id = re.search(r"race_id=(\d{12})", anchor["href"]).group(1)
        time_match = re.search(r"(\d{1,2}:\d{2})", li.get_text(" ", strip=True))
        if race_id in races:
            continue
        races[race_id] = {
            "race_id": race_id,
            "post_time": time_match.group(1) if time_match else None,
        }
    return sorted(races.values(), key=lambda r: r["race_id"])


def fetch_odds_snapshot(race_id: str) -> dict:
    """単勝・複勝オッズの生 JSON を取得して整形する。"""
    resp = _polite_get(ODDS_API_URL.format(race_id=race_id))
    payload = resp.json()
    data = payload.get("data") or {}
    odds = data.get("odds") or {}

    def _f(value: str) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    tan: dict[str, float | None] = {}
    ninki: dict[str, int | None] = {}
    for umaban, row in (odds.get("1") or {}).items():
        tan[umaban] = _f(row[0])
        try:
            ninki[umaban] = int(row[2])
        except (TypeError, ValueError, IndexError):
            ninki[umaban] = None

    fuku: dict[str, list[float | None]] = {}
    for umaban, row in (odds.get("2") or {}).items():
        fuku[umaban] = [_f(row[0]), _f(row[1])]

    return {
        "api_status": payload.get("status"),
        "official_datetime": data.get("official_datetime"),
        "tan": tan,
        "fuku": fuku,
        "ninki": ninki,
    }


def snapshot_path(kaisai_date: str, race_id: str) -> Path:
    return DATA_DIR / kaisai_date / f"{race_id}.jsonl"


def load_done_labels(path: Path) -> set[str]:
    """再起動時の二重取得防止: 取得済みスナップショットのラベル集合。"""
    if not path.exists():
        return set()
    labels = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                labels.add(json.loads(line)["label"])
            except (json.JSONDecodeError, KeyError):
                continue
    return labels


def take_snapshot(kaisai_date: str, race: dict, label: str,
                  deadline: datetime | None = None) -> str:
    """1レース1スナップショットを取得して JSONL に追記する。

    deadline (=発走時刻) を渡された場合、取得開始前と保存直前の両方で
    跨ぎを検査する。リトライ待ちやタイムアウトで発走を越えたデータは
    「T-x分前のオッズ」ではないため保存しない (戻り値 "late")。
    戻り値: "saved" | "failed" | "late"
    """
    race_id = race["race_id"]
    if deadline is not None and _now() >= deadline:
        _log(f"  LATE {race_id} {label}: 発走時刻を跨いだため取得しない")
        return "late"
    last_error: Exception | None = None
    for attempt, backoff in enumerate((0,) + RETRY_WAITS_SEC):
        if backoff:
            time.sleep(backoff)
        if deadline is not None and _now() >= deadline:
            _log(f"  LATE {race_id} {label}: リトライ中に発走時刻を跨いだ")
            return "late"
        try:
            snap = fetch_odds_snapshot(race_id)
            break
        except (requests.RequestException, ValueError) as e:
            last_error = e
            _log(f"  retry {attempt + 1} {race_id} {label}: {e}")
    else:
        _log(f"  FAILED {race_id} {label}: {last_error}")
        return "failed"
    if deadline is not None and _now() >= deadline:
        _log(f"  LATE {race_id} {label}: 取得中に発走時刻を跨いだため保存しない")
        return "late"

    record = {
        "race_id": race_id,
        "kaisai_date": kaisai_date,
        "label": label,
        "fetched_at": _now().isoformat(timespec="seconds"),
        "post_time": race["post_time"],
        **snap,
    }
    path = snapshot_path(kaisai_date, race_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    n_tan = sum(1 for v in record["tan"].values() if v)
    status = record["api_status"]
    warn = "" if status == "result" else f" ※市場オッズではない (status={status})"
    _log(f"  saved {race_id} {label} status={status} "
         f"(単勝 {n_tan}/{len(record['tan'])}頭){warn}")
    return "saved"


def build_schedule(kaisai_date: str, races: list[dict]) -> list[tuple[datetime, dict, str]]:
    """全レース×全オフセットのスナップショット予定表を作る。"""
    base = datetime.strptime(kaisai_date, "%Y%m%d").replace(tzinfo=JST)
    events = []
    for race in races:
        if not race["post_time"]:
            _log(f"発走時刻が取れないためスキップ: {race['race_id']}")
            continue
        hh, mm = map(int, race["post_time"].split(":"))
        post_dt = base.replace(hour=hh, minute=mm)
        for offset in SNAPSHOT_OFFSETS_MIN:
            events.append((post_dt - timedelta(minutes=offset), post_dt, race, f"T-{offset}"))
    events.sort(key=lambda e: e[0])
    return events


def run_daemon(kaisai_date: str, races: list[dict]) -> None:
    events = build_schedule(kaisai_date, races)
    counts = {"saved": 0, "failed": 0, "late": 0, "skipped": 0}

    for due, post_dt, race, label in events:
        if label in load_done_labels(snapshot_path(kaisai_date, race["race_id"])):
            continue
        now = _now()
        if (now - due).total_seconds() > LATE_TOLERANCE_SEC:
            counts["skipped"] += 1
            continue
        while (wait := (due - _now()).total_seconds()) > 0:
            time.sleep(min(wait, 30))
        counts[take_snapshot(kaisai_date, race, label, deadline=post_dt)] += 1

    _log(f"完了: 取得 {counts['saved']} / 期限超過スキップ {counts['skipped']} "
         f"/ 発走跨ぎ {counts['late']} / 失敗 {counts['failed']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="SPEC-3 時系列オッズロガー")
    parser.add_argument("--date", default=_now().strftime("%Y%m%d"),
                        help="開催日 YYYYMMDD（既定: 今日 JST）")
    parser.add_argument("--once", action="store_true",
                        help="スケジュールせず全レースを今すぐ1回取得")
    parser.add_argument("--list-only", action="store_true",
                        help="レース一覧の取得・表示のみ")
    parser.add_argument("--dry-run", action="store_true",
                        help="デーモンのスナップショット予定表を表示して終了")
    args = parser.parse_args()

    races = fetch_race_list(args.date)
    if not races:
        _log(f"{args.date} の JRA レースが見つからない（非開催日の可能性）")
        return 1
    _log(f"{args.date}: {len(races)} レース検出")

    if args.list_only:
        for race in races:
            print(f"  {race['race_id']}  発走 {race['post_time']}")
        return 0

    if args.dry_run:
        events = build_schedule(args.date, races)
        now = _now()
        runnable = [e for e in events
                    if (now - e[0]).total_seconds() <= LATE_TOLERANCE_SEC]
        _log(f"予定 {len(events)} 件 / 現時点で実行対象 {len(runnable)} 件")
        for due, _post_dt, race, label in events[:3] + events[-3:]:
            print(f"  {due.strftime('%m-%d %H:%M')}  {race['race_id']}  {label}")
        return 0

    if args.once:
        ok = sum(take_snapshot(args.date, race, "manual") == "saved" for race in races)
        _log(f"manual snapshot: {ok}/{len(races)} 成功")
        return 0 if ok == len(races) else 1

    run_daemon(args.date, races)
    return 0


if __name__ == "__main__":
    sys.exit(main())
