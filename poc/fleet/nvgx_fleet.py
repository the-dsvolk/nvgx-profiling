#!/usr/bin/env python3
"""nvgx-fleet (POC): receive workload profiles from GPU nodes and serve a live dashboard.

Standard library only. Listens on 127.0.0.1 (reach it from a GPU node through an SSH tunnel:
`ssh -N -R 9100:127.0.0.1:9100 <gpu-node>` run on this machine).

Endpoints:
  POST /v1/profiles          one profile or a JSON list of profiles (schema nvgx.workload_profile/v0)
  GET  /v1/profiles?limit=N  most recent N profiles, oldest first
  GET  /events               server-sent events: one "profile" event per received profile
  GET  /healthz              liveness
  GET  /                     dashboard (static/)

Production would put a message bus (Kafka, Pub/Sub, NATS) between nodes and the fleet; the POC
receives directly. See design/fleet-profiling.md.
"""
import argparse
import json
import os
import queue
import sqlite3
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

SCHEMA = "nvgx.workload_profile/v0"
MAX_BODY = 4 * 1024 * 1024
STATIC = Path(__file__).resolve().parent / "static"
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                 ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml"}


class Store:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        self.db.execute("""create table if not exists profiles (
            id integer primary key, node text, window_end_ns integer, received_ns integer, body text)""")
        self.db.commit()

    def add(self, profile):
        with self.lock:
            self.db.execute("insert into profiles (node, window_end_ns, received_ns, body) values (?, ?, ?, ?)",
                            (profile.get("node"), profile["window"]["end_utc_ns"],
                             profile["fleet"]["received_utc_ns"], json.dumps(profile)))
            self.db.commit()

    def recent(self, limit):
        with self.lock:
            cur = self.db.execute("select body from profiles order by id desc limit ?", (limit,))
            return [json.loads(r[0]) for r in cur.fetchall()][::-1]


class Hub:
    """Fan-out of new profiles to connected dashboards."""

    def __init__(self):
        self.clients = set()
        self.lock = threading.Lock()

    def subscribe(self):
        q = queue.Queue(maxsize=200)
        with self.lock:
            self.clients.add(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            self.clients.discard(q)

    def publish(self, profile):
        data = json.dumps(profile)
        with self.lock:
            for q in list(self.clients):
                try:
                    q.put_nowait(data)
                except queue.Full:  # a stuck browser doesn't hold the others back
                    pass


def validate(p):
    if not isinstance(p, dict) or p.get("schema") != SCHEMA:
        return "schema must be " + SCHEMA
    w = p.get("window") or {}
    if not isinstance(w.get("end_utc_ns"), int) or not isinstance(w.get("start_utc_ns"), int):
        return "window.start_utc_ns / end_utc_ns must be integers"
    return None


def make_handler(store, hub, verbose):
    class Handler(BaseHTTPRequestHandler):
        server_version = "nvgx-fleet/0"

        def log_message(self, fmt, *args):
            if verbose:
                super().log_message(fmt, *args)

        def send_json(self, status, obj):
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if urlparse(self.path).path != "/v1/profiles":
                return self.send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY:
                return self.send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "bad length"})
            try:
                payload = json.loads(self.rfile.read(length))
            except json.JSONDecodeError as e:
                return self.send_json(HTTPStatus.BAD_REQUEST, {"error": f"invalid JSON: {e}"})
            items = payload if isinstance(payload, list) else [payload]
            accepted, errors = 0, []
            now = time.time_ns()
            for p in items:
                err = validate(p)
                if err:
                    errors.append(err)
                    continue
                p["fleet"] = {"received_utc_ns": now,
                              "lag_s": round((now - p["window"]["end_utc_ns"]) / 1e9, 2)}
                store.add(p)
                hub.publish(p)
                accepted += 1
                if not verbose:
                    print(f"[fleet] {p.get('node')} snapshot {p.get('snapshot_number')} "
                          f"verdict={p.get('verdict')} lag={p['fleet']['lag_s']}s", flush=True)
            status = HTTPStatus.OK if accepted else HTTPStatus.BAD_REQUEST
            self.send_json(status, {"accepted": accepted, "errors": errors[:5]})

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/healthz":
                return self.send_json(HTTPStatus.OK, {"ok": True})
            if url.path == "/v1/profiles":
                limit = int((parse_qs(url.query).get("limit") or ["180"])[0])
                return self.send_json(HTTPStatus.OK, store.recent(max(1, min(limit, 5000))))
            if url.path == "/events":
                return self.stream_events()
            return self.serve_static(url.path)

        def stream_events(self):
            q = hub.subscribe()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            try:
                self.wfile.write(b": connected\n\n")
                self.wfile.flush()
                while True:
                    try:
                        data = q.get(timeout=15)
                        self.wfile.write(f"event: profile\ndata: {data}\n\n".encode())
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                hub.unsubscribe(q)

        def serve_static(self, path):
            rel = "index.html" if path in ("", "/") else path.lstrip("/")
            target = (STATIC / rel).resolve()
            if STATIC not in target.parents or not target.is_file():
                return self.send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            body = target.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", CONTENT_TYPES.get(target.suffix, "application/octet-stream"))
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9100)
    ap.add_argument("--db", default=os.path.expanduser("~/.nvgx-fleet/fleet.db"))
    ap.add_argument("--verbose", action="store_true", help="log every HTTP request")
    args = ap.parse_args()

    store, hub = Store(Path(args.db)), Hub()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(store, hub, args.verbose))
    server.daemon_threads = True
    print(f"[fleet] listening on http://{args.host}:{args.port}  (db {args.db})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
