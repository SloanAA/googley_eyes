#!/usr/bin/env python3
"""
Face/person tracker -> TMC2208 STEP/DIR motor (pigpio)

  python3 personTracker.py                          # track + drive motor
  python3 personTracker.py --stream                 # also serve video at http://<pi>:8080
  python3 personTracker.py --stream --no-motor      # camera/detection test only
  python3 personTracker.py --preview                # local window (needs a display)

Detection order each frame:
  1. YuNet face detector (auto-downloads a small model on first run; tries a
     model version that works with your OpenCV, and self-tests it at startup)
  2. Haar cascade face detector (if YuNet is unavailable)
  3. HOG full-body person detector (if no face is found)

Person on the LEFT  -> clockwise; on the RIGHT -> counterclockwise.
Speed scales with distance from center.
Requires: pigpiod running (sudo systemctl start pigpiod)
"""

import argparse
import os
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import pigpio
from picamera2 import Picamera2

# ---- Motor pins / settings ----
STEP_PIN = 18          # hardware PWM pin
DIR_PIN = 14
EN_PIN = 15
DIR_SETUP_S = 0.001
SETTLE_S = 0.01

MIN_FREQ_HZ = 150      # slowest step rate while tracking
MAX_FREQ_HZ = 1000     # step rate at full deflection (raise once stable)
FREQ_SLEW_HZ = 250     # max change in step rate per frame (avoids stalling)

# ---- Camera ----
# Raw sensor mode. 1640x1232 is the binned FULL field-of-view mode of the
# IMX219 (Camera V2). The default 640x480 mode is a heavy crop = "zoomed in".
FRAME_SIZE = (640, 480)
SENSOR_SIZE = (1640, 1232)

# ---- Tracking settings ----
DEADZONE = 0.08
LEFT_IS_CLOCKWISE = True
LOST_TIMEOUT = 0.7     # seconds without a detection before stopping
SMOOTHING = 0.5        # 0 = no smoothing, closer to 1 = smoother but laggier

# ---- Detection settings ----
FACE_SCORE_MIN = 0.6   # YuNet confidence threshold (lower = more detections, more false hits)
HOG_WIDTH = 320        # HOG fallback runs on a downscaled frame for speed
USE_BODY_FALLBACK = True

# OpenCV 4.6 needs the 2022mar model; 4.8+ needs 2023mar. We try both, in this order.
YUNET_BASE = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
YUNET_MODELS = [
    "face_detection_yunet_2022mar.onnx",
    "face_detection_yunet_2023mar.onnx",
]
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

CW = 1                 # DIR level for clockwise
CCW = 0


# ======================= Motor =======================
class Motor:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.direction = None
        self.freq = 0
        if not enabled:
            return
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise SystemExit("Could not connect to pigpiod - is it running?")
        for pin in (STEP_PIN, DIR_PIN, EN_PIN):
            self.pi.set_mode(pin, pigpio.OUTPUT)
        self.pi.write(EN_PIN, 0)  # enable driver

    def stop(self):
        if self.enabled and self.freq:
            self.pi.hardware_PWM(STEP_PIN, 0, 0)
            self.pi.write(STEP_PIN, 0)
            time.sleep(SETTLE_S)
        self.freq = 0

    def drive(self, signed_freq):
        """signed_freq > 0 = clockwise, < 0 = counterclockwise, 0 = stop."""
        target = abs(signed_freq)
        if target < 1:
            self.stop()
            return

        new_dir = CW if signed_freq > 0 else CCW
        if new_dir != self.direction:
            self.stop()  # ramp back up from zero after a reversal
            if self.enabled:
                self.pi.write(DIR_PIN, new_dir)
                time.sleep(DIR_SETUP_S)
            self.direction = new_dir

        # slew toward the target step rate
        if target > self.freq:
            self.freq = min(target, max(self.freq, MIN_FREQ_HZ) + FREQ_SLEW_HZ)
        else:
            self.freq = max(target, self.freq - FREQ_SLEW_HZ)
        self.freq = max(self.freq, MIN_FREQ_HZ)

        if self.enabled:
            self.pi.hardware_PWM(STEP_PIN, int(self.freq), 500000)

    def shutdown(self):
        self.stop()
        if self.enabled:
            self.pi.write(EN_PIN, 1)
            self.pi.stop()


# ======================= Detection =======================
class Detector:
    """Returns (x, y, w, h, label) in full-frame pixel coordinates, or None."""

    def __init__(self, frame_w, frame_h):
        self.frame_w, self.frame_h = frame_w, frame_h
        self.yunet = None
        self.haar = None
        self.hog = None

        print(f"OpenCV version: {cv2.__version__}")
        self._init_yunet()
        if self.yunet is None:
            self._init_haar()
        if USE_BODY_FALLBACK:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def _init_yunet(self):
        if not hasattr(cv2, "FaceDetectorYN"):
            print("This OpenCV has no FaceDetectorYN; skipping YuNet")
            return
        os.makedirs(MODEL_DIR, exist_ok=True)
        for name in YUNET_MODELS:
            path = os.path.join(MODEL_DIR, name)
            try:
                if not os.path.exists(path) or os.path.getsize(path) < 10000:
                    print(f"Downloading {name} ...")
                    urllib.request.urlretrieve(YUNET_BASE + name, path)
                det = cv2.FaceDetectorYN.create(
                    path, "", (self.frame_w, self.frame_h), FACE_SCORE_MIN, 0.3, 5000)
                # Self-test: some model/OpenCV combos load fine but crash on first detect.
                det.setInputSize((self.frame_w, self.frame_h))
                det.detect(np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8))
                self.yunet = det
                print(f"Face detector: YuNet ({name})")
                return
            except Exception as e:
                msg = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
                print(f"YuNet {name} not usable: {msg}")
                if os.path.exists(path) and os.path.getsize(path) < 10000:
                    os.remove(path)  # remove a bad/partial download
        print("YuNet unavailable; falling back to Haar cascade")

    def _init_haar(self):
        candidates = []
        if hasattr(cv2, "data"):
            candidates.append(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
        candidates += [
            "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
            "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
        ]
        for path in candidates:
            if os.path.exists(path):
                self.haar = cv2.CascadeClassifier(path)
                print("Face detector: Haar cascade")
                return
        print("No Haar cascade file found (try: sudo apt install opencv-data); using body detector only")

    def detect(self, frame_bgr):
        # 1) faces
        box = None
        if self.yunet is not None:
            self.yunet.setInputSize((self.frame_w, self.frame_h))
            _, faces = self.yunet.detect(frame_bgr)
            if faces is not None and len(faces) > 0:
                f = max(faces, key=lambda r: r[2] * r[3])
                box = (int(f[0]), int(f[1]), int(f[2]), int(f[3]))
        elif self.haar is not None:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = self.haar.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(faces) > 0:
                f = max(faces, key=lambda r: r[2] * r[3])
                box = tuple(int(v) for v in f)
        if box is not None:
            return (*box, "face")

        # 2) full body fallback
        if self.hog is not None:
            scale = self.frame_w / HOG_WIDTH
            small = cv2.resize(frame_bgr, (HOG_WIDTH, int(self.frame_h / scale)))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            boxes, _ = self.hog.detectMultiScale(
                gray, winStride=(8, 8), padding=(8, 8), scale=1.05)
            if len(boxes) > 0:
                b = max(boxes, key=lambda r: r[2] * r[3])
                x, y, w, h = [int(v * scale) for v in b]
                return (x, y, w, h, "body")
        return None


# ======================= Browser streaming =======================
class Streamer:
    """Serves the annotated frames as an MJPEG stream you can open in a browser."""

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
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true", help="show local camera window (needs a display)")
    ap.add_argument("--stream", action="store_true", help="serve preview in a browser")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--no-motor", action="store_true", help="detect only, never touch the motor")
    args = ap.parse_args()

    motor = Motor(enabled=not args.no_motor)
    streamer = Streamer(args.port) if args.stream else None
    if streamer:
        print(f"Streaming at http://<pi-address>:{args.port}  (find address with: hostname -I)")

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": FRAME_SIZE, "format": "RGB888"},  # RGB888 = BGR byte order, OpenCV-ready
        raw={"size": SENSOR_SIZE})                      # forces the full-FOV sensor mode
    picam2.configure(config)
    picam2.start()
    print("Sensor mode:", picam2.camera_configuration()["sensor"])

    w, h = FRAME_SIZE
    detector = Detector(w, h)

    last_seen = 0.0
    smooth_err = 0.0

    try:
        while True:
            frame = picam2.capture_array()
            det = detector.detect(frame)

            status = "no person"
            if det is not None:
                x, y, bw, bh, label = det
                cx = x + bw / 2
                half = w / 2
                error = (cx - half) / half  # -1 = far left ... +1 = far right
                smooth_err = SMOOTHING * smooth_err + (1 - SMOOTHING) * error
                last_seen = time.time()

                if abs(smooth_err) < DEADZONE:
                    motor.drive(0)
                    status = f"{label} centered ({smooth_err:+.2f})"
                else:
                    mag = (abs(smooth_err) - DEADZONE) / (1 - DEADZONE)  # 0..1 past deadzone
                    freq = MIN_FREQ_HZ + mag * (MAX_FREQ_HZ - MIN_FREQ_HZ)
                    left = smooth_err < 0
                    clockwise = left if LEFT_IS_CLOCKWISE else not left
                    motor.drive(freq if clockwise else -freq)
                    status = f"{label} {'CW' if clockwise else 'CCW'} {motor.freq:.0f}Hz ({smooth_err:+.2f})"
            elif time.time() - last_seen > LOST_TIMEOUT:
                motor.drive(0)
                smooth_err = 0.0

            print(status)

            if args.preview or streamer:
                view = frame.copy()
                cv2.line(view, (w // 2, 0), (w // 2, h), (255, 255, 0), 1)
                dz = int(DEADZONE * (w / 2))
                cv2.rectangle(view, (w // 2 - dz, 0), (w // 2 + dz, h), (80, 80, 80), 1)
                if det is not None:
                    x, y, bw, bh, label = det
                    color = (0, 255, 0) if label == "face" else (0, 165, 255)
                    cv2.rectangle(view, (x, y), (x + bw, y + bh), color, 2)
                cv2.putText(view, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2)
                if streamer:
                    streamer.update(view)
                if args.preview:
                    cv2.imshow("googley eyes", view)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    except KeyboardInterrupt:
        pass
    finally:
        motor.shutdown()
        picam2.stop()
        if args.preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
