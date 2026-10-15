#!/usr/bin/env python3
"""Nest box watch: motion -> bird check -> message to Home Assistant."""
import os
import threading
import time
from datetime import datetime
from pathlib import Path

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2
import requests
from ultralytics import YOLO, settings

settings.update({"sync": False})  # never send usage data

# ---------- Settings ----------
RTSP = "rtsp://127.0.0.1:8554/nestbox"
HA_WEBHOOK = os.environ.get("HA_WEBHOOK", "")  # e.g. http://<home-assistant>:8123/api/webhook/<secret-id>
MODE = "both"             # what counts as a hit for "occupied": "model", "motion" or "both"
MOTION_FRAC = 0.01        # share of changed pixels that counts as motion
PIXEL_DIFF = 25           # brightness difference per pixel
BIRD_CONF = 0.15          # minimum confidence for "bird"
MIN_CONF = 0.05           # below this nothing is shown at all
CHECK_IMGSZ = 640         # image size for the bird check
HITS_NEEDED = 3           # this many hits ...
HIT_WINDOW = 20           # ... within this many seconds = occupied
EMPTY_AFTER = 90          # seconds without a hit = empty
CHECK_EVERY = 1.0         # bird check while there is motion (seconds)
CHECK_EVERY_OCC = 5.0     # bird check while occupied (a bird may sit still)
MOTION_NOTIFY_EVERY = 60  # seconds between two motion messages
SNAP_EVERY = 3            # seconds between two saved pictures
SNAP_DIR = Path.home() / "nestbox" / "snaps"
MAX_SNAPS = 1500
BIRD_CLASS = 14           # COCO: "bird"
# ------------------------------

latest = {"frame": None, "ok": False}


def log(msg):
    print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)


def reader():
    """Reads the stream continuously and keeps only the newest frame."""
    while True:
        cap = cv2.VideoCapture(RTSP, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            log("Stream not reachable, retrying in 5 s")
            time.sleep(5)
            continue
        log("Stream connected")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            latest["frame"] = frame
            latest["ok"] = True
        cap.release()
        latest["ok"] = False
        log("Stream lost, reconnecting")
        time.sleep(2)


def notify(state):
    log(f"=== Nest box: {state.upper()} ===")
    if HA_WEBHOOK:
        try:
            requests.post(HA_WEBHOOK, json={"state": state, "time": datetime.now().isoformat()}, timeout=5)
        except Exception as e:
            log(f"Home Assistant not reachable: {e}")


def save_snap(frame, label, boxes=None):
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    img = frame.copy()
    for (x1, y1, x2, y2, c) in boxes or []:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, f"{c:.2f}", (int(x1), max(int(y1) - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(str(SNAP_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{label}.jpg"), img)
    files = sorted(SNAP_DIR.glob("*.jpg"))
    for old in files[:-MAX_SNAPS]:
        old.unlink()


def main():
    threading.Thread(target=reader, daemon=True).start()
    log("Loading bird model ...")
    model = YOLO("yolov8n.pt")
    log(f"Mode: {MODE}")

    def find_birds(frame):
        res = model(frame, imgsz=CHECK_IMGSZ, conf=MIN_CONF, classes=[BIRD_CLASS], verbose=False)[0]
        return [(*b.xyxy[0].tolist(), float(b.conf[0])) for b in res.boxes]

    bg = None
    warmup = 30
    occupied = False
    hits = []
    last_hit = 0.0
    last_check = 0.0
    last_snap = 0.0
    last_motion_msg = 0.0
    last_seen = None

    while True:
        time.sleep(0.07)
        frame = latest["frame"]
        if frame is None or frame is last_seen:
            continue
        last_seen = frame

        small = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY), (9, 9), 0)
        if bg is None:
            bg = small.astype("float")
            continue
        diff = cv2.absdiff(small, cv2.convertScaleAbs(bg))
        motion = (diff > PIXEL_DIFF).mean() > MOTION_FRAC
        cv2.accumulateWeighted(small, bg, 0.05)
        if warmup > 0:
            warmup -= 1
            continue

        now = time.time()
        if motion and now - last_motion_msg > MOTION_NOTIFY_EVERY:
            last_motion_msg = now
            notify("motion")

        interval = CHECK_EVERY_OCC if occupied else CHECK_EVERY
        if (motion or occupied) and now - last_check >= interval:
            last_check = now
            boxes = find_birds(frame)
            birds = [b for b in boxes if b[4] >= BIRD_CONF]
            best = max((b[4] for b in boxes), default=0.0)
            if birds:
                log(f"Bird detected (confidence {best:.2f})")
                if now - last_snap >= SNAP_EVERY:
                    save_snap(frame, "bird", birds)
                    last_snap = now
            elif motion:
                log(f"Motion, but no bird detected (best score {best:.2f})")
                if now - last_snap >= SNAP_EVERY:
                    save_snap(frame, "nobird", boxes)
                    last_snap = now
            hit = (bool(birds) and MODE in ("model", "both")) or (motion and MODE in ("motion", "both"))
            if hit:
                hits.append(now)
                last_hit = now
            hits = [t for t in hits if now - t <= HIT_WINDOW]
            if not occupied and len(hits) >= HITS_NEEDED:
                occupied = True
                notify("occupied")

        if occupied and now - last_hit > EMPTY_AFTER:
            occupied = False
            hits = []
            notify("empty")


if __name__ == "__main__":
    main()
