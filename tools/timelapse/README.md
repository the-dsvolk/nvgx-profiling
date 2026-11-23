# Dashboard time-lapse

Turn a recorded run of the [fleet dashboard](../../poc/README.md) into an **animated WebP** for the repo: no
video, no live screen capture. Frames are rendered from recorded data, so the output is repeatable and can be
re-rendered with a different crop, speed or window.

1. **Record.** Run a workload (e.g. `poc/workload/nvgx_load.py --scenario mixed --seed 2026 --duration 30m`)
   while the fleet service stores profiles in its SQLite database. Keep the generator's console log: its
   `=== regime #N ... ===` lines become the captions.
2. **Export** the run into one JSON recording (profiles of one node + regime timeline):

   ```shell
   python3 tools/timelapse/export_recording.py --node <node> --t0-ns <start-ns> --duration 1800 \
     --load-log <generator-console.log> --out poc/fleet/static/recordings/<run>.json
   ```

3. **Render.** Serve the dashboard (a second fleet instance is fine) and render frames with headless Chrome:

   ```shell
   python3 poc/fleet/nvgx_fleet.py --port 9101 --db /tmp/render.db &
   python3 tools/timelapse/render_timelapse.py --base http://127.0.0.1:9101 \
     --recording /recordings/<run>.json --frames-dir /tmp/frames \
     --webp showcases/<dir>/img/timelapse.webp --step 2 --fps 10 --size 1440,1060 \
     --keyframes 40,120,200 --keyframes-dir showcases/<dir>/img
   ```

**Replay mode** (`?replay=<recording>&frame=<N>[&window=<snapshots>][&hist=<metric>]`) makes the dashboard draw
the window ending at snapshot N of a recording instead of the live feed, with a caption banner for the active
load regime and elapsed time. It hides the tables, the lag chart and the lag tile so a frame crops to tiles,
four charts and the latency-distribution card, and sets `<html data-ready="1">` when drawn. `hist` picks the
latency metric (`nccl`, `interval`, `step`, `launch`, `sync`); `render_timelapse.py --query hist=nccl` sets it
for every frame, and `--keyframes 110:interval,236:launch` per key frame. Use `--size 1440,1410` to include the
latency card.

4. **Summarize** a run per regime type (load-generator numbers, profile averages, latency p50/p99 from the
   summed histograms) for the showcase text:

   ```shell
   python3 tools/timelapse/regime_stats.py --recording poc/fleet/static/recordings/<run>.json --load-log <generator>.jsonl
   ```

Pacing: 30 minutes of 5 s snapshots ≈ 360 frames; `--step 2 --fps 10` gives an ~18 s loop.
Requirements: Google Chrome, Python 3 with Pillow built with WebP support.
