#!/usr/bin/env python3
"""Export one recorded run from the nvgx fleet database into a replayable JSON recording.

Usage:
  export_recording.py --node <node> --t0-ns <start> --duration 1800 \
      --load-log timelapse-load.log --out poc/fleet/static/recordings/run.json

- Profiles: every profile of <node> whose window ends in [t0 - pre, t0 + duration + post], oldest first.
- Regimes: lines "HH:MM:SS === regime #N <name> for <s> s: <description> ===" from nvgx_load.py's console log
  (times are UTC on the GPU node), turned into absolute timestamps on t0's date.
The dashboard replays the file with ?replay=/recordings/run.json&frame=N.
"""
import argparse
import datetime as dt
import json
import os
import re
import sqlite3

REGIME = re.compile(r"^(\d\d):(\d\d):(\d\d) === regime #(\d+) (\S+) for (\d+) s: (.*?) ===\s*$")


def parse_regimes(path, t0_ns):
    t0 = dt.datetime.fromtimestamp(t0_ns / 1e9, tz=dt.timezone.utc)
    out = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = REGIME.match(line)
            if not m:
                continue
            hh, mm, ss, idx, name, secs, desc = m.groups()
            t = t0.replace(hour=int(hh), minute=int(mm), second=int(ss), microsecond=0)
            if t < t0 - dt.timedelta(hours=12):  # crossed midnight
                t += dt.timedelta(days=1)
            out.append({"t_utc_ns": int(t.timestamp() * 1e9), "index": int(idx), "name": name,
                        "duration_s": int(secs), "desc": desc})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=os.path.expanduser("~/.nvgx-fleet/fleet.db"))
    ap.add_argument("--node", required=True)
    ap.add_argument("--t0-ns", type=int, required=True, help="run start, ns since epoch (UTC)")
    ap.add_argument("--duration", type=int, default=1800, help="run length in seconds")
    ap.add_argument("--pre", type=int, default=30, help="seconds of profiles before t0")
    ap.add_argument("--post", type=int, default=20, help="seconds of profiles after the run")
    ap.add_argument("--load-log", help="nvgx_load.py console log (for regime captions)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lo = args.t0_ns - args.pre * 10**9
    hi = args.t0_ns + (args.duration + args.post) * 10**9
    db = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    rows = db.execute("select body from profiles where node = ? and window_end_ns between ? and ? "
                      "order by window_end_ns", (args.node, lo, hi)).fetchall()
    profiles = [json.loads(r[0]) for r in rows]
    regimes = parse_regimes(args.load_log, args.t0_ns) if args.load_log else []
    # Keep the regime already active at the window start, plus those starting inside the window.
    before = [r for r in regimes if r["t_utc_ns"] <= lo]
    regimes = before[-1:] + [r for r in regimes if lo < r["t_utc_ns"] <= hi]
    rec = {"schema": "nvgx.recording/v0", "node": args.node, "t0_utc_ns": args.t0_ns,
           "duration_s": args.duration, "regimes": regimes, "profiles": profiles}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rec, f, separators=(",", ":"))
    print(f"wrote {args.out}: {len(profiles)} profiles, {len(regimes)} regimes, "
          f"{os.path.getsize(args.out) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
