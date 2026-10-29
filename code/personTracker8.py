#!/usr/bin/env python3
"""
Face tracker -> two TMC2208 STEP/DIR steppers (pigpio), POSITION control

  python3 personTracker.py                          # track + drive motors
  python3 personTracker.py --stream                 # also serve video at http://<pi>:8080
  python3 personTracker.py --stream --no-motor      # camera/detection test only
  python3 personTracker.py --preview                # local window (needs a display)

Motor behavior:
  * Both motors are driven in sync: same direction, same steps, same instant.
  * Wherever the motors point when you START the script is home = 0 deg (down).
  * Face horizontal position maps linearly to motor angle:
        left edge -> +90 deg (clockwise) | center -> 0 | right edge -> -90 deg
        (flip with LEFT_IS_CLOCKWISE). Never exceeds +/-MAX_DEG.
  * Face still -> motors still (deadband). Face lost -> motors hold position.
  * Motors run in their own thread so they move smoothly.

Smoothness changes (vs. the previous version):
  * Step bursts are now QUEUED back to back (pigpio WAVE_MODE_ONE_SHOT_SYNC), so the
    next burst starts the instant the previous one ends. Before, every burst ended
    with a pause while Python built the next one, which showed up as jitter and
    made the motors stall under load.
  * Deceleration is limited as well as acceleration.
  * The tracking target only updates when the face moves more than a few degrees,
    so detection-box wobble no longer makes the motors chase noise.
  * OpenCV is capped to a few threads so the detector can't starve the motor thread.

Speed / reliability design:
  * Camera + video stream run in the main loop at full camera speed.
  * Face detection runs in a separate worker thread at its own (slower) rate.
    The stream stays smooth; the box just updates when a detection finishes.
  * Detector cascade (first hit wins):
      1. YuNet            (if your OpenCV supports it)
      2. SSD res10 face net  - handles turned heads much better than Haar.
         While a face is being tracked it only searches a crop around the
         last position (fast + sees small faces); otherwise it scans the
         whole frame in overlapping squares.
      3. Haar frontal, 4. Haar profile (both sides)
  * Box colors: green=yunet, orange=ssd (full scan), yellow=ssd (tracking crop),
    blue=haar frontal, magenta=haar profile.

Requires: pigpiod running (sudo systemctl start pigpiod)
Open-loop motors (no encoder): if steps get skipped, restart with motors pointing down to re-home.
"""

import argparse
import os
import threading
import time
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import pigpio
from picamera2 import Picamera2

# ---- Motor pins (index 0 = motor 1, index 1 = motor 2) ----
STEP_PINS = [18, 21]
DIR_PINS = [14, 16]
EN_PINS = [15, 20]
# If the two motors are mounted mirrored and one spins the wrong way, set that
# motor's entry to True to invert its direction (they still step in sync).
DIR_INVERT = [False, False]
DIR_SETUP_S = 0.001

STEP_MASK = 0
for _p in STEP_PINS:
    STEP_MASK |= 1 << _p

# ---- Motor geometry ----
STEPS_PER_REV = 200
MICROSTEPS = 8
GEAR_RATIO = 1.0       # >1 if the eyes are geared down from the motor shaft
MAX_DEG = 90.0         # max rotation each way from home
STEPS_PER_DEG = STEPS_PER_REV * MICROSTEPS * GEAR_RATIO / 360.0

# ---- Motion tuning ----
MIN_FREQ_HZ = 200
MAX_FREQ_HZ = 1000     # lower for slower, gentler movement
ACCEL_HZ_PER_CHUNK = 100   # max speed-up per burst (Hz)
DECEL_HZ_PER_CHUNK = 200   # max slow-down per burst (Hz)
SLOW_ZONE_DEG = 25.0
CHUNK_STEPS = 20       # steps per queued burst
QUEUE_DEPTH = 2        # bursts kept queued ahead (keeps pulses gap-free)
START_DEG = 2.0        # target must move this far before the motors start moving
STOP_STEPS = 1
RETURN_HOME_ON_EXIT = True
RETURN_HOME_AFTER_S = None   # e.g. 10 = swing to center if no face for 10 s; None = hold forever

# ---- Mapping ----
LEFT_IS_CLOCKWISE = True
TARGET_GAIN = 1.0      # >1 reaches +/-90 deg before the face hits the screen edge (e.g. 1.3)
SMOOTHING = 0.6        # applied per detection update; 0 = raw, closer to 1 = smoother/laggier
TARGET_DEADBAND_DEG = 3.0   # ignore target changes smaller than this (stops detection-noise jitter)

CW = 1
CCW = 0

# ---- Camera ----
FRAME_SIZE = (1024, 768)    # frame size given to the detectors
SENSOR_SIZE = (1640, 1232)  # full field-of-view sensor mode
STREAM_WIDTH = 640
CV_THREADS = 2              # cap OpenCV threads so the motor thread keeps getting CPU time

# ---- Detection ----
YUNET_SCORE_MIN = 0.5
SSD_CONF_MIN = 0.45    # lower = finds more faces (and more false hits)
ROI_SCALE = 3.5        # tracking crop = face size * this
ROI_MIN_SIDE = 200
TRACK_HOLD_S = 1.0     # keep using the tracking crop this long after the last face
HAAR_WIDTH = 640

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
YUNET_BASE = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
YUNET_MODELS = ["face_detection_yunet_2022mar.onnx", "face_detection_yunet_2023mar.onnx"]

SSD_PROTO = "deploy.prototxt"
SSD_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
SSD_URLS = {
    SSD_PROTO: [
        "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt",
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    ],
    SSD_MODEL: [
        "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
        "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
    ],
}
SSD_MIN_BYTES = {SSD_PROTO: 1000, SSD_MODEL: 1_000_000}


# ======================= Motor (position control) =======================
class PositionMotor:
    """Step counter (clockwise positive, 0 = home) + thread sending exact step bursts toward the target.

    Drives all motors in STEP_PINS together: one wave toggles every STEP pin at the
    same instant, and all DIR / EN pins are set together, so the motors stay in sync.

    Bursts are queued with WAVE_MODE_ONE_SHOT_SYNC so each one starts exactly when the
    previous one ends (no gaps between bursts).
    """

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.lock = threading.Lock()
        self.pos_steps = 0
        self.target_steps = 0
        self.freq = 0
        self.moving = False
        self.direction = None
        self.queue = deque()      # (wave_id, signed_steps) for bursts sent but not yet counted
        self.running = True

        if enabled:
            self.pi = pigpio.pi()
            if not self.pi.connected:
                raise SystemExit("Could not connect to pigpiod - is it running?")
            for pin in STEP_PINS + DIR_PINS + EN_PINS:
                self.pi.set_mode(pin, pigpio.OUTPUT)
            for pin in STEP_PINS:
                self.pi.write(pin, 0)
            for pin in EN_PINS:
                self.pi.write(pin, 0)   # enable drivers (holds position)
            self.pi.wave_clear()
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def set_target_deg(self, deg):
        deg = max(-MAX_DEG, min(MAX_DEG, deg))
        with self.lock:
            self.target_steps = int(round(deg * STEPS_PER_DEG))
            if not self.enabled:
                self.pos_steps = self.target_steps   # simulate instantly

    def angle_deg(self):
        with self.lock:
            return self.pos_steps / STEPS_PER_DEG

    def target_deg(self):
        with self.lock:
            return self.target_steps / STEPS_PER_DEG

    def shutdown(self, go_home=True):
        if not self.enabled:
            return
        if go_home:
            self.set_target_deg(0)
            t0 = time.time()
            while time.time() - t0 < 10:
                with self.lock:
                    done = abs(self.pos_steps) <= STOP_STEPS and not self.queue
                if done:
                    break
                time.sleep(0.02)
        self.running = False
        self.thread.join(timeout=1.0)
        self.pi.wave_tx_stop()
        self.pi.wave_clear()
        self.queue.clear()
        for pin in STEP_PINS:
            self.pi.write(pin, 0)
        for pin in EN_PINS:
            self.pi.write(pin, 1)   # release drivers
        self.pi.stop()

    def _reap(self):
        """Count finished bursts into pos_steps and free their waves."""
        if not self.queue:
            return
        if self.pi.wave_tx_busy():
            cur = self.pi.wave_tx_at()
            while self.queue and self.queue[0][0] != cur:
                wid, steps = self.queue.popleft()
                self.pi.wave_delete(wid)
                with self.lock:
                    self.pos_steps += steps
        else:
            # nothing transmitting: every queued burst has finished
            while self.queue:
                wid, steps = self.queue.popleft()
                self.pi.wave_delete(wid)
                with self.lock:
                    self.pos_steps += steps

    def _set_direction(self, direction):
        for pin, invert in zip(DIR_PINS, DIR_INVERT):
            level = CW if direction > 0 else CCW
            if invert:
                level = 1 - level
            self.pi.write(pin, level)

    def _queue_burst(self, n, direction, freq):
        half_us = max(2, int(500000 / freq))
        pulses = []
        for _ in range(n):
            # all STEP pins go high / low together -> motors stay in sync
            pulses.append(pigpio.pulse(STEP_MASK, 0, half_us))
            pulses.append(pigpio.pulse(0, STEP_MASK, half_us))
        self.pi.wave_add_generic(pulses)
        wid = self.pi.wave_create()
        if wid < 0:
            return False
        # SYNC mode: starts when the current wave ends (or immediately if idle)
        self.pi.wave_send_using_mode(wid, pigpio.WAVE_MODE_ONE_SHOT_SYNC)
        self.queue.append((wid, n * direction))
        return True

    def _run(self):
        while self.running:
            self._reap()

            with self.lock:
                pos = self.pos_steps
                target = self.target_steps
            pending = sum(s for _, s in self.queue)
            remaining = target - (pos + pending)   # distance left after queued bursts finish
            dist = abs(remaining)

            if not self.moving:
                if dist >= START_DEG * STEPS_PER_DEG:
                    self.moving = True
                    self.freq = 0
            elif dist <= STOP_STEPS:
                self.moving = False
                if not self.queue:
                    self.freq = 0

            if not self.moving:
                time.sleep(0.005)
                continue

            direction = 1 if remaining > 0 else -1

            # Direction can only change when nothing is queued: let the queue drain first
            if self.queue and direction != self.direction:
                time.sleep(0.002)
                continue

            if len(self.queue) >= QUEUE_DEPTH:
                time.sleep(0.001)
                continue

            if direction != self.direction:
                self._set_direction(direction)
                time.sleep(DIR_SETUP_S)
                self.direction = direction
                self.freq = 0   # restart slowly after a reversal

            slow_zone = SLOW_ZONE_DEG * STEPS_PER_DEG
            want = MIN_FREQ_HZ + (MAX_FREQ_HZ - MIN_FREQ_HZ) * min(1.0, dist / slow_zone)
            if self.freq == 0:
                self.freq = MIN_FREQ_HZ
            elif want > self.freq:
                self.freq = min(want, self.freq + ACCEL_HZ_PER_CHUNK)
            else:
                self.freq = max(want, self.freq - DECEL_HZ_PER_CHUNK)
            self.freq = max(MIN_FREQ_HZ, min(self.freq, MAX_FREQ_HZ))

            if not self._queue_burst(min(dist, CHUNK_STEPS), direction, self.freq):
                time.sleep(0.002)


# ======================= Face detection =======================
def _download(name, urls, dest, min_bytes):
    if os.path.exists(dest) and os.path.getsize(dest) >= min_bytes:
        return True
    for url in urls:
        try:
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(url, dest)
            if os.path.getsize(dest) >= min_bytes:
                return True
            os.remove(dest)
        except Exception as e:
            print(f"  failed ({str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__})")
            if os.path.exists(dest):
                os.remove(dest)
    return False


class FaceDetector:
    """detect(frame, hint_box) -> (x, y, w, h, source) in frame pixels, or None."""

    def __init__(self, frame_w, frame_h):
        self.fw, self.fh = frame_w, frame_h
        self.yunet = None
        self.ssd = None
        self.haar_front = None
        self.haar_profile = None
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        os.makedirs(MODEL_DIR, exist_ok=True)

        print(f"OpenCV version: {cv2.__version__}")
        self._init_yunet()
        self._init_ssd()
        self._init_haar()
        names = [n for n, d in (("YuNet", self.yunet), ("SSD", self.ssd),
                                ("Haar frontal", self.haar_front), ("Haar profile", self.haar_profile)) if d]
        if not names:
            raise SystemExit("No face detector could be loaded - see messages above.")
        print("Detectors active:", ", ".join(names))
        if self.ssd is None:
            print("TIP: the SSD face net handles turned heads best. To install it by hand:\n"
                  f"  cd {MODEL_DIR}\n"
                  f"  wget {SSD_URLS[SSD_PROTO][0]}\n"
                  f"  wget {SSD_URLS[SSD_MODEL][0]}")

    # ---- loaders ----
    def _init_yunet(self):
        if not hasattr(cv2, "FaceDetectorYN"):
            return
        for name in YUNET_MODELS:
            path = os.path.join(MODEL_DIR, name)
            try:
                if not _download(name, [YUNET_BASE + name], path, 10000):
                    continue
                det = cv2.FaceDetectorYN.create(path, "", (self.fw, self.fh), YUNET_SCORE_MIN, 0.3, 5000)
                det.setInputSize((self.fw, self.fh))
                det.detect(np.zeros((self.fh, self.fw, 3), dtype=np.uint8))  # self-test
                self.yunet = det
                print(f"YuNet loaded ({name})")
                return
            except Exception as e:
                msg = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
                print(f"YuNet {name} not usable: {msg}")
        print("YuNet unavailable (normal on OpenCV 4.6)")

    def _init_ssd(self):
        proto = os.path.join(MODEL_DIR, SSD_PROTO)
        model = os.path.join(MODEL_DIR, SSD_MODEL)
        ok = (_download(SSD_PROTO, SSD_URLS[SSD_PROTO], proto, SSD_MIN_BYTES[SSD_PROTO]) and
              _download(SSD_MODEL, SSD_URLS[SSD_MODEL], model, SSD_MIN_BYTES[SSD_MODEL]))
        if not ok:
            print("SSD face model could not be downloaded")
            return
        try:
            net = cv2.dnn.readNetFromCaffe(proto, model)
            self.ssd = net
            self._ssd_run(np.zeros((300, 300, 3), dtype=np.uint8))  # self-test
            print("SSD face net loaded")
        except Exception as e:
            self.ssd = None
            print(f"SSD face net not usable: {str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__}")

    def _find_cascade(self, filename):
        dirs = []
        if hasattr(cv2, "data"):
            dirs.append(cv2.data.haarcascades)
        dirs += ["/usr/share/opencv4/haarcascades/", "/usr/share/opencv/haarcascades/"]
        for d in dirs:
            path = os.path.join(d, filename)
            if os.path.exists(path):
                c = cv2.CascadeClassifier(path)
                if not c.empty():
                    return c
        return None

    def _init_haar(self):
        self.haar_front = self._find_cascade("haarcascade_frontalface_default.xml")
        self.haar_profile = self._find_cascade("haarcascade_profileface.xml")

    # ---- SSD helpers ----
    def _ssd_run(self, img):
        """Run the SSD net on an image; returns [(x, y, w, h, conf)] in that image's pixels."""
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300),
                                     (104.0, 177.0, 123.0), swapRB=False)
        self.ssd.setInput(blob)
        out = self.ssd.forward()
        res = []
        for i in range(out.shape[2]):
            conf = float(out[0, 0, i, 2])
            if conf < SSD_CONF_MIN:
                continue
            x1, y1, x2, y2 = out[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(w), x2), min(float(h), y2)
            bw, bh = x2 - x1, y2 - y1
            if bw < 12 or bh < 12:
                continue
            res.append((x1, y1, bw, bh, conf))
        return res

    def _ssd_crop(self, frame, x0, y0, x1, y1):
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(self.fw, int(x1)), min(self.fh, int(y1))
        if x1 - x0 < 40 or y1 - y0 < 40:
            return []
        res = self._ssd_run(frame[y0:y1, x0:x1])
        return [(x + x0, y + y0, w, h, c) for x, y, w, h, c in res]

    def _ssd_detect(self, frame, hint):
        # Tracking: search only around the last face (fast, better for small/turned faces)
        if hint is not None:
            hx, hy, hw, hh = hint
            cx, cy = hx + hw / 2, hy + hh / 2
            side = max(ROI_MIN_SIDE, max(hw, hh) * ROI_SCALE)
            res = self._ssd_crop(frame, cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
            if res:
                b = max(res, key=lambda r: r[2] * r[3])
                return (int(b[0]), int(b[1]), int(b[2]), int(b[3]), "ssd-crop")
        # Full scan: overlapping squares (a squashed full frame makes faces too small)
        side = min(self.fw, self.fh)
        offsets = [0] if self.fw <= side * 1.1 else [0, (self.fw - side) // 2, self.fw - side]
        res = []
        for ox in offsets:
            res += self._ssd_crop(frame, ox, 0, ox + side, self.fh)
        if res:
            b = max(res, key=lambda r: r[2] * r[3])
            return (int(b[0]), int(b[1]), int(b[2]), int(b[3]), "ssd")
        return None

    # ---- Haar helpers ----
    def _haar_gray(self, frame):
        scale = self.fw / HAAR_WIDTH
        small = cv2.resize(frame, (HAAR_WIDTH, int(self.fh / scale)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return self.clahe.apply(gray), scale

    @staticmethod
    def _largest(rects):
        return max(rects, key=lambda r: r[2] * r[3])

    def detect(self, frame, hint=None):
        if self.yunet is not None:
            self.yunet.setInputSize((self.fw, self.fh))
            _, faces = self.yunet.detect(frame)
            if faces is not None and len(faces) > 0:
                f = self._largest(faces)
                return (int(f[0]), int(f[1]), int(f[2]), int(f[3]), "yunet")

        if self.ssd is not None:
            r = self._ssd_detect(frame, hint)
            if r is not None:
                return r

        if self.haar_front is None and self.haar_profile is None:
            return None
        gray, scale = self._haar_gray(frame)
        gw = gray.shape[1]

        if self.haar_front is not None:
            r = self.haar_front.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(24, 24))
            if len(r) > 0:
                x, y, w, h = [int(v * scale) for v in self._largest(r)]
                return (x, y, w, h, "haar")

        if self.haar_profile is not None:
            r = self.haar_profile.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(24, 24))
            if len(r) > 0:
                x, y, w, h = [int(v * scale) for v in self._largest(r)]
                return (x, y, w, h, "profile")
            r = self.haar_profile.detectMultiScale(cv2.flip(gray, 1), scaleFactor=1.1,
                                                   minNeighbors=4, minSize=(24, 24))
            if len(r) > 0:
                x, y, w, h = self._largest(r)
                x = gw - x - w  # un-flip
                x, y, w, h = [int(v * scale) for v in (x, y, w, h)]
                return (x, y, w, h, "profile")
        return None


class DetectionWorker(threading.Thread):
    """Runs the detector on the newest frame in the background."""

    def __init__(self, detector):
        super().__init__(daemon=True)
        self.detector = detector
        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = 0
        self.result = None
        self.result_id = 0
        self.result_time = 0.0
        self.det_fps = 0.0
        self.running = True
        self.last_box = None
        self.last_box_time = 0.0

    def submit(self, frame):
        with self.lock:
            self.frame = frame
            self.frame_id += 1

    def get(self):
        with self.lock:
            return self.result, self.result_id, self.result_time

    def run(self):
        seen = 0
        t_prev = time.time()
        while self.running:
            with self.lock:
                if self.frame_id == seen or self.frame is None:
                    frame = None
                else:
                    frame = self.frame
                    seen = self.frame_id
            if frame is None:
                time.sleep(0.002)
                continue

            now = time.time()
            hint = self.last_box if (self.last_box is not None and now - self.last_box_time < TRACK_HOLD_S) else None
            try:
                det = self.detector.detect(frame, hint)
            except Exception as e:
                print(f"detector error: {e}")
                det = None

            t = time.time()
            self.det_fps = 0.8 * self.det_fps + 0.2 * (1.0 / max(t - t_prev, 1e-3))
            t_prev = t
            if det is not None:
                self.last_box = det[:4]
                self.last_box_time = t
            with self.lock:
                self.result = det
                self.result_id += 1
                self.result_time = t


# ======================= Browser streaming =======================
class Streamer:
    """Serves annotated frames as an MJPEG stream you can open in a browser."""

    def __init__(self, port):
        self.jpeg = None
        self.cond = threading.Condition()
        streamer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    try:
                        while True:
                            with streamer.cond:
                                streamer.cond.wait()
                                data = streamer.jpeg
                            self.wfile.write(
                                b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                else:
                    page = (b'<html><body style="margin:0;background:#000">'
                            b'<img src="/stream" style="width:100%;max-width:960px"></body></html>')
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(page)))
                    self.end_headers()
                    self.wfile.write(page)

        self.server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def update(self, frame):
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with self.cond:
                self.jpeg = buf.tobytes()
                self.cond.notify_all()


# ======================= Main =======================
BOX_COLORS = {
    "yunet": (0, 255, 0),
    "ssd": (0, 165, 255),
    "ssd-crop": (0, 255, 255),
    "haar": (255, 128, 0),
    "profile": (255, 0, 255),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true", help="show local camera window (needs a display)")
    ap.add_argument("--stream", action="store_true", help="serve preview in a browser")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--no-motor", action="store_true", help="detect only, never touch the motors")
    args = ap.parse_args()

    cv2.setNumThreads(CV_THREADS)   # leave CPU headroom for the motor thread

    motor = PositionMotor(enabled=not args.no_motor)
    streamer = Streamer(args.port) if args.stream else None
    if streamer:
        print(f"Streaming at http://<pi-address>:{args.port}  (find address with: hostname -I)")

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": FRAME_SIZE, "format": "RGB888"},  # RGB888 = BGR byte order, OpenCV-ready
        raw={"size": SENSOR_SIZE})
    picam2.configure(config)
    picam2.start()
    print("Sensor mode:", picam2.camera_configuration()["sensor"])

    w, h = FRAME_SIZE
    detector = FaceDetector(w, h)
    worker = DetectionWorker(detector)
    worker.start()
    print(f"Motor home = current position (0 deg). Range +/-{MAX_DEG:.0f} deg = "
          f"+/-{int(MAX_DEG * STEPS_PER_DEG)} steps. Driving {len(STEP_PINS)} motors in sync.")

    sw = STREAM_WIDTH
    sh = int(h * sw / w)
    sx = sw / w

    last_rid = 0
    last_seen = time.time()
    smooth_err = 0.0
    have_face = False
    status = "starting"
    fps = 0.0
    t_prev = time.time()

    try:
        while True:
            frame = picam2.capture_array()
            worker.submit(frame)
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-3))
            t_prev = now

            det, rid, rtime = worker.get()
            if rid != last_rid:   # a new detection result arrived
                last_rid = rid
                if det is not None:
                    x, y, bw, bh, src = det
                    error = ((x + bw / 2) - w / 2) / (w / 2)   # -1 = far left ... +1 = far right
                    smooth_err = SMOOTHING * smooth_err + (1 - SMOOTHING) * error if have_face else error
                    have_face = True
                    last_seen = now
                    angle = -smooth_err if LEFT_IS_CLOCKWISE else smooth_err
                    new_target = angle * TARGET_GAIN * MAX_DEG
                    # Deadband: only retarget when the face has really moved (ignores box wobble)
                    if abs(new_target - motor.target_deg()) >= TARGET_DEADBAND_DEG:
                        motor.set_target_deg(new_target)
                    status = f"face [{src}] target {motor.target_deg():+.0f} deg"
                else:
                    have_face = False
                    if RETURN_HOME_AFTER_S is not None and now - last_seen > RETURN_HOME_AFTER_S:
                        motor.set_target_deg(0)
                    status = "no face - holding"
                print(f"{status} | motor {motor.angle_deg():+.0f} deg | video {fps:.0f} fps, detect {worker.det_fps:.1f}/s")

            if args.preview or streamer:
                view = cv2.resize(frame, (sw, sh))
                cv2.line(view, (sw // 2, 0), (sw // 2, sh), (255, 255, 0), 1)
                if det is not None and now - rtime < 1.0:
                    x, y, bw, bh, src = det
                    cv2.rectangle(view, (int(x * sx), int(y * sx)), (int((x + bw) * sx), int((y + bh) * sx)),
                                  BOX_COLORS.get(src, (0, 255, 0)), 2)
                cv2.putText(view, f"{status} | motor {motor.angle_deg():+.0f}", (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                cv2.putText(view, f"video {fps:.0f} fps  detect {worker.det_fps:.1f}/s", (8, sh - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                if streamer:
                    streamer.update(view)
                if args.preview:
                    cv2.imshow("googley eyes", view)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    except KeyboardInterrupt:
        pass
    finally:
        worker.running = False
        motor.shutdown(go_home=RETURN_HOME_ON_EXIT)
        picam2.stop()
        if args.preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
