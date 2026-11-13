#!/usr/bin/env python3
"""Nest box watch: motion -> bird check -> message to Home Assistant -> post."""
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 snoth0x53
import os
import random
import sys
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
BIRD_CONF = 0.30          # minimum confidence for "bird"
MIN_CONF = 0.05           # below this nothing is shown at all
CHECK_IMGSZ = 1280        # image size for the bird check
MIN_BOX = 40              # ignore frames whose shorter side is below this many pixels (dust speck)
MODEL_REQUIRED = True     # "occupied" only after at least one model hit
HITS_NEEDED = 3           # this many hits ...
HIT_WINDOW = 20           # ... within this many seconds = occupied
EMPTY_AFTER = 90          # seconds without a hit = empty
CHECK_EVERY = 1.0         # bird check while there is motion (seconds)
CHECK_EVERY_OCC = 5.0     # bird check while occupied (a bird may sit still)
MOTION_NOTIFY_EVERY = 60  # seconds between two motion messages
SNAP_EVERY = 3            # seconds between two saved pictures
SNAP_DIR = Path.home() / "nestbox" / "snaps"
MAX_SNAPS = 1500
# Second copy without the green frame, for the next round of training. None = off.
SNAP_RAW_DIR = Path.home() / "nestbox" / "snaps_raw"
MAX_SNAPS_RAW = 1500
# Own model, trained on pictures from this box. Falling back to the general model
# means setting MODEL_PATH to "yolov8n.pt" and BIRD_CLASS to 14.
MODEL_PATH = str(Path.home() / "nestbox" / "nestbox-model.pt")
BIRD_CLASS = 0            # own model: class 0 = "bird" (general COCO model: 14)

# Mastodon: one post per visit, triggered when the box turns empty.
# The token comes from the environment, never from this file.
MASTODON = True
MASTODON_SERVER = "https://mastodon.social"
MASTODON_TOKEN = os.environ.get("MASTODON_TOKEN", "")
MASTODON_VISIBILITY = "public"
MASTODON_LANGUAGE = "en"
MASTODON_TAGS = "#birds #nestbox"
MASTODON_GAP = 600        # seconds between two posts
MASTODON_ALT = ("Black-and-white infrared photograph taken inside a wooden nest box, "
                "looking down from the roof. A bird is visible in the box.")
# {date}, {time} and {duration} are filled in.
MASTODON_TEXTS = [
    "Incident report, {date}. A bird was observed inside the box at {time}. It stayed {duration} and left without comment.",
    "Incident report, {date}. Entry recorded at {time}. The visitor remained {duration} and did not state the purpose of the visit.",
    "Incident report, {date}. At {time} an unidentified bird entered the premises. Duration of stay: {duration}. No damage reported.",
    "Incident report, {date}. Unannounced inspection at {time}, {duration}. The inspector left no report.",
    "Incident report, {date}. A bird occupied the box at {time} for {duration}. It declined to register.",
    "Incident report, {date}. Movement at {time}, confirmed as a bird. Time on site: {duration}. The matter is closed.",
]
MASTODON_TEXTS_SHORT = [  # visits under 10 seconds
    "Incident report, {date}. Brief intrusion at {time}, {duration}. The visitor reconsidered.",
    "Incident report, {date}. At {time} a bird looked in for {duration} and thought better of it.",
]
MASTODON_TEXTS_LONG = [   # visits over 2 minutes
    "Incident report, {date}. Extended stay at {time}, {duration}. No lease has been signed.",
    "Incident report, {date}. The box was occupied from {time} for {duration}. The occupant made itself at home.",
]
MASTODON_TEXTS_RARE = [   # turn up now and then, see MASTODON_RARE_SHARE
    "Incident report, {date}. Entry recorded at {time}. The device remained {duration} and did not identify its operator.",
    "Incident report, {date}. At {time} the subject entered and was photographed for {duration}. No serial number was visible.",
    "Incident report, {date}. The visitor remained {duration} from {time}. It was observed observing the premises.",
    "Incident report, {date}. At {time} the unit powered down inside the box for {duration} and then departed.",
    "Incident report, {date}. Activity recorded from {time}, duration {duration}. Whether the subject is a bird remains unverified.",
    "Incident report, {date}. At {time} the subject inspected the camera for {duration}. The camera inspected the subject. Neither filed a report.",
]
MASTODON_RARE_SHARE = 0.2  # 0.0 = never, 1.0 = always
# ------------------------------

latest = {"frame": None, "ok": False}

MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]


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


def notify(state, extra=""):
    log(f"=== Nest box: {state.upper()}{extra.rstrip()} ===")
    if not HA_WEBHOOK:
        return
    try:
        r = requests.post(HA_WEBHOOK, json={"state": state, "time": datetime.now().isoformat()}, timeout=5)
        if not r.ok:
            log(f"Home Assistant answered {r.status_code}")
    except Exception as e:
        log(f"Home Assistant not reachable: {e}")


def duration_text(sec):
    """45 seconds, 2 minutes 14 seconds, 11 hours 3 minutes (roosting in winter)."""
    whole = int(round(sec))
    h, rest = divmod(whole, 3600)
    m, s = divmod(rest, 60)
    parts = []
    if h:
        parts.append(f"{h} hour" + ("s" if h != 1 else ""))
    if m:
        parts.append(f"{m} minute" + ("s" if m != 1 else ""))
    if s and not h:
        parts.append(f"{s} second" + ("s" if s != 1 else ""))
    return " ".join(parts) or "0 seconds"


def post_text(start, duration_sec):
    """Builds the caption from a random template."""
    if MASTODON_TEXTS_RARE and random.random() < MASTODON_RARE_SHARE:
        templates = MASTODON_TEXTS_RARE
    elif duration_sec < 10:
        templates = MASTODON_TEXTS_SHORT or MASTODON_TEXTS
    elif duration_sec > 120:
        templates = MASTODON_TEXTS_LONG or MASTODON_TEXTS
    else:
        templates = MASTODON_TEXTS
    text = random.choice(templates).format(
        date=f"{start.day} {MONTHS[start.month - 1]} {start.year}",
        time=f"{start:%H:%M}",
        duration=duration_text(duration_sec),
    )
    return f"{text}\n\n{MASTODON_TAGS}".strip()


last_post = 0.0
post_lock = threading.Lock()


def mastodon_send(picture, text, visibility=None):
    """Uploads the picture and publishes the post. Returns (ok, message)."""
    head = {"Authorization": f"Bearer {MASTODON_TOKEN}", "Accept": "application/json"}
    try:
        with open(picture, "rb") as f:
            answer = requests.post(f"{MASTODON_SERVER}/api/v2/media", headers=head,
                                   files={"file": (os.path.basename(picture), f, "image/jpeg")},
                                   data={"description": MASTODON_ALT}, timeout=60)
        if answer.status_code not in (200, 202):
            return False, f"picture rejected ({answer.status_code})"
        media = answer.json().get("id")
        if not media:
            return False, "no media id returned"
        # 202 means: still being processed, wait a little
        for _ in range(20):
            if answer.status_code == 200:
                break
            time.sleep(3)
            answer = requests.get(f"{MASTODON_SERVER}/api/v1/media/{media}", headers=head, timeout=30)
        data = {"status": text, "media_ids": [media],
                "visibility": visibility or MASTODON_VISIBILITY, "language": MASTODON_LANGUAGE}
        head2 = dict(head)
        head2["Idempotency-Key"] = f"nestbox-{os.path.basename(picture)}"
        answer = requests.post(f"{MASTODON_SERVER}/api/v1/statuses", headers=head2, json=data, timeout=60)
        if answer.status_code != 200:
            return False, f"post rejected ({answer.status_code})"
        return True, answer.json().get("url", "published")
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def mastodon_post(picture, start, duration_sec):
    """Checks the conditions and sends in a thread of its own."""
    global last_post
    if not MASTODON:
        return
    if not MASTODON_TOKEN:
        log("Mastodon: no token set, nothing posted")
        return
    if not picture or not Path(picture).exists():
        log("Mastodon: no picture for this visit, nothing posted")
        return
    with post_lock:
        if time.time() - last_post < MASTODON_GAP:
            log("Mastodon: too soon after the last post, nothing posted")
            return
        last_post = time.time()
    text = post_text(start, duration_sec)

    def work():
        ok, message = mastodon_send(picture, text)
        log(("Mastodon: published " if ok else "Mastodon: ") + str(message))

    threading.Thread(target=work, daemon=True).start()


class Visit:
    """Keeps track of hits, decides occupied/empty and collects values."""

    def __init__(self):
        self.occupied = False
        self.hits = []          # times of all hits (model or motion)
        self.bird_hits = []     # times of the model hits
        self.last_hit = 0.0
        self.best = 0.0         # best model value since the visit started
        self.birds = 0          # number of model hits since it started
        self.start = 0.0
        self.held = False       # "occupied" was held back and already logged
        self.picture = None     # best saved picture of this visit (no frame drawn)
        self.picture_value = 0.0
        self.last_visit = None  # details for the Mastodon post

    def values(self):
        plural = "" if self.birds == 1 else "s"
        return f"best value {self.best:.2f}, {self.birds} hit{plural}"

    def check(self, now, motion, best, n_birds):
        """Evaluates one bird check. Returns a list of (kind, text)."""
        out = []
        self.hits = [t for t in self.hits if now - t <= HIT_WINDOW]
        self.bird_hits = [t for t in self.bird_hits if now - t <= HIT_WINDOW]
        if not self.occupied and not self.hits:
            self.best, self.birds, self.start, self.held = 0.0, 0, now, False
            self.picture, self.picture_value = None, 0.0
        self.best = max(self.best, best)
        if n_birds:
            self.birds += 1
            self.bird_hits.append(now)
        hit = (n_birds > 0 and MODE in ("model", "both")) or (motion and MODE in ("motion", "both"))
        if hit:
            self.hits.append(now)
            self.last_hit = now
        if not self.occupied and len(self.hits) >= HITS_NEEDED:
            if MODEL_REQUIRED and not self.bird_hits:
                if not self.held:
                    self.held = True
                    out.append(("held", f"occupied held back, no model hit, best value {self.best:.2f}"))
            else:
                self.occupied = True
                out.append(("occupied", f"  {self.values()}"))
        return out

    def keep_picture(self, path, value):
        """Keeps the picture with the highest model value of this visit."""
        if path and value > self.picture_value:
            self.picture, self.picture_value = path, value

    def empty(self, now):
        """Returns the extra text for "empty" once it is time, otherwise None."""
        if self.occupied and now - self.last_hit > EMPTY_AFTER:
            duration = max(0.0, self.last_hit - self.start)
            text = f"  {self.values()}, visit about {duration_text(duration)}"
            self.last_visit = {
                "start": datetime.fromtimestamp(self.start),
                "duration": duration,
                "picture": self.picture,
            }
            self.occupied = False
            self.hits = []
            self.bird_hits = []
            return text
        return None


def tidy(folder, keep):
    """Keeps only the newest pictures."""
    for old in sorted(folder.glob("*.jpg"))[:-keep]:
        try:
            old.unlink()
        except OSError:
            pass


def save_snap(frame, label, boxes=None):
    """Saves the picture with the green frame and, if switched on, a copy without."""
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    img = frame.copy()
    for (x1, y1, x2, y2, c) in boxes or []:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, f"{c:.2f}", (int(x1), max(int(y1) - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(str(SNAP_DIR / f"{stamp}-{label}.jpg"), img)
    tidy(SNAP_DIR, MAX_SNAPS)
    if SNAP_RAW_DIR is None:
        return None
    try:
        SNAP_RAW_DIR.mkdir(parents=True, exist_ok=True)
        raw = SNAP_RAW_DIR / f"{stamp}-{label}.jpg"
        cv2.imwrite(str(raw), frame)
        tidy(SNAP_RAW_DIR, MAX_SNAPS_RAW)
        return str(raw)
    except Exception as e:
        log(f"picture without frame not saved: {e}")
        return None


def main():
    threading.Thread(target=reader, daemon=True).start()
    log(f"Loading bird model {MODEL_PATH} ...")
    if "/" in MODEL_PATH and not Path(MODEL_PATH).exists():
        log(f"Model not found: {MODEL_PATH}")
        sys.exit(1)
    model = YOLO(MODEL_PATH)
    log(f"Mode: {MODE}, model required: {MODEL_REQUIRED}, threshold {BIRD_CONF:.2f}")
    if MASTODON:
        log("Mastodon: " + ("token set" if MASTODON_TOKEN else "no token, nothing will be posted"))

    def detect(img, imgsz):
        res = model(img, imgsz=imgsz, conf=MIN_CONF, classes=[BIRD_CLASS], verbose=False)[0]
        boxes = [(*b.xyxy[0].tolist(), float(b.conf[0])) for b in res.boxes]
        return [b for b in boxes if min(b[2] - b[0], b[3] - b[1]) >= MIN_BOX]

    def find_birds(frame):
        # The own model also finds birds cut off at the top edge, so the extra
        # strip the earlier versions checked separately is no longer needed.
        return detect(frame, CHECK_IMGSZ)

    bg = None
    warmup = 30
    visit = Visit()
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

        interval = CHECK_EVERY_OCC if visit.occupied else CHECK_EVERY
        if (motion or visit.occupied) and now - last_check >= interval:
            last_check = now
            boxes = find_birds(frame)
            birds = [b for b in boxes if b[4] >= BIRD_CONF]
            best = max((b[4] for b in boxes), default=0.0)
            if birds:
                log(f"Bird detected (confidence {best:.2f})")
                if now - last_snap >= SNAP_EVERY:
                    visit.keep_picture(save_snap(frame, "bird", birds), best)
                    last_snap = now
            elif motion:
                log(f"Motion, but no bird detected (best value {best:.2f})")
                if now - last_snap >= SNAP_EVERY:
                    save_snap(frame, "nobird", boxes)
                    last_snap = now
            for kind, text in visit.check(now, motion, best, len(birds)):
                if kind == "occupied":
                    notify("occupied", text)
                else:
                    log(text)

        empty_text = visit.empty(now)
        if empty_text is not None:
            notify("empty", empty_text)
            done = visit.last_visit
            if done:
                mastodon_post(done["picture"], done["start"], done["duration"])


if __name__ == "__main__":
    main()
