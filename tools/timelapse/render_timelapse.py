#!/usr/bin/env python3
"""Render a dashboard recording into an animated WebP time-lapse (plus optional key frames).

Usage:
  python3 poc/fleet/nvgx_fleet.py --port 9101 --db /tmp/render.db &        # serves the dashboard + recordings
  render_timelapse.py --base http://127.0.0.1:9101 --recording /recordings/run.json \
      --frames-dir /tmp/frames --webp showcases/<dir>/img/timelapse.webp --step 2 --fps 10

Each frame is the dashboard in replay mode (?replay=...&frame=N) screenshotted by headless Chrome.
Requires Google Chrome and Pillow with WebP support.
"""
import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time
import urllib.request

from PIL import Image

CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")


def shoot(base, recording, frame, window, size, path, profile_dir):
    url = f"{base}/?replay={recording}&frame={frame}&window={window}"
    cmd = [CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1",
           "--no-first-run", "--no-default-browser-check", f"--user-data-dir={profile_dir}",
           f"--window-size={size}", "--virtual-time-budget=10000", f"--screenshot={path}", url]
    for attempt in range(3):
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if r.returncode == 0 and os.path.exists(path) and os.path.getsize(path) > 10_000:
            return path
        time.sleep(1)
    raise RuntimeError(f"frame {frame} failed: {r.stderr[-300:]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:9101")
    ap.add_argument("--recording", required=True, help="URL path of the recording, e.g. /recordings/run.json")
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--webp", required=True)
    ap.add_argument("--step", type=int, default=2, help="render every Nth snapshot")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--window", type=int, default=120, help="snapshots of history visible per frame")
    ap.add_argument("--size", default="1440,1000", help="browser window WxH")
    ap.add_argument("--fps", type=float, default=10)
    ap.add_argument("--max-width", type=int, default=1000, help="downscale frames to this width")
    ap.add_argument("--quality", type=int, default=75)
    ap.add_argument("--keyframes", default="", help="comma-separated snapshot indices to save as PNG")
    ap.add_argument("--keyframes-dir", default="")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    with urllib.request.urlopen(args.base + args.recording) as r:
        n = len(json.load(r)["profiles"])
    frames = list(range(args.start, n, args.step))
    if frames[-1] != n - 1:
        frames.append(n - 1)
    os.makedirs(args.frames_dir, exist_ok=True)
    print(f"{n} profiles -> {len(frames)} frames", flush=True)

    t0 = time.time()
    paths = {}
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = {}
        for k, i in enumerate(frames):
            path = os.path.join(args.frames_dir, f"f{i:04d}.png")
            prof = os.path.join(args.frames_dir, f".chrome{k % args.workers}")
            futs[ex.submit(shoot, args.base, args.recording, i, args.window, args.size, path, prof)] = i
        for done, fut in enumerate(cf.as_completed(futs), 1):
            paths[futs[fut]] = fut.result()
            if done % 20 == 0:
                print(f"  {done}/{len(frames)} frames ({time.time() - t0:.0f} s)", flush=True)

    imgs = []
    for i in frames:
        im = Image.open(paths[i]).convert("RGB")
        if im.width > args.max_width:
            im = im.resize((args.max_width, round(im.height * args.max_width / im.width)), Image.LANCZOS)
        imgs.append(im)
    os.makedirs(os.path.dirname(os.path.abspath(args.webp)), exist_ok=True)
    imgs[0].save(args.webp, save_all=True, append_images=imgs[1:], duration=round(1000 / args.fps),
                 loop=0, quality=args.quality, method=6,
                 minimize_size=True, allow_mixed=True)  # inter-frame compression: 186 frames ≈ 4.9 MB
    print(f"wrote {args.webp}: {len(imgs)} frames, {len(imgs) / args.fps:.1f} s loop, "
          f"{os.path.getsize(args.webp) / 1e6:.2f} MB", flush=True)

    if args.keyframes:
        os.makedirs(args.keyframes_dir or os.path.dirname(args.webp), exist_ok=True)
        for k in [int(x) for x in args.keyframes.split(",") if x.strip()]:
            out = os.path.join(args.keyframes_dir or os.path.dirname(args.webp), f"frame-{k:04d}.png")
            shoot(args.base, args.recording, k, args.window, args.size, out, os.path.join(args.frames_dir, ".chromek"))
            print(f"key frame {k} -> {out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
