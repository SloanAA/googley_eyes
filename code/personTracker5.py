#!/usr/bin/env python3
"""
Face tracker -> TMC2208 STEP/DIR stepper (pigpio), POSITION control

  python3 personTracker.py                          # track + drive motor
  python3 personTracker.py --stream                 # also serve video at http://<pi>:8080
  python3 personTracker.py --stream --no-motor      # camera/detection test only
  python3 personTracker.py --preview                # local window (needs a display)

How the motor behaves:
  * Wherever the motor is pointing when you START the script is "home" = 0 degrees (down).
  * Your face's horizontal position on screen maps linearly to a motor angle:
        left edge of screen  -> +90 deg (clockwise)      [flip with LEFT_IS_CLOCKWISE]
        center of screen     ->   0 deg (home / down)
        right edge of screen -> -90 deg (counterclockwise)
  * The motor never goes past +/-MAX_DEG.
  * If your face stays put, the motor stays put (deadband, no jitter).
  * If your face is lost, the motor holds its last position.
  * The motor runs in its own thread, so it moves smoothly even though the
    camera/detector only produces a few frames per second.

Face detection only. Tries YuNet, then Haar frontal, then Haar profile.
Requires: pigpiod running (sudo systemctl start pigpiod)

NOTE: this is open-loop (no encoder). If the motor skips steps (too fast, too
much load) its real position can drift from the angle the script thinks it has.
Restart the script with the motor pointing down to re-home.
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

# ---- Motor pins ----
STEP_PIN = 18
DIR_PIN = 14
EN_PIN = 15
DIR_SETUP_S = 0.001

# ---- Motor geometry ----
STEPS_PER_REV = 200
MICROSTEPS = 8
GEAR_RATIO = 1.0       # output turns = motor turns / GEAR_RATIO. Set >1 if you gear it down
MAX_DEG = 90.0         # max rotation each way from home
STEPS_PER_DEG = STEPS_PER_REV * MICROSTEPS * GEAR_RATIO / 360.0

# ---- Motion tuning ----
MIN_FREQ_HZ = 200      # slowest step rate (start speed)
MAX_FREQ_HZ = 1000     # fastest step rate; lower this for slower, gentler movement
ACCEL_HZ_PER_CHUNK = 150   # how quickly the step rate may rise between chunks
SLOW_ZONE_DEG = 25.0   # start slowing down when this close to the target
CHUNK_STEPS = 40       # steps sent per burst (smaller = more responsive)
START_DEG = 2.0        # face must move the target at least this far to start moving
STOP_STEPS = 1         # stop when within this many steps of the target
RETURN_HOME_ON_EXIT = True
RETURN_HOME_AFTER_S = None   # e.g. 10 = swing back to center if no face for 10 s; None = hold forever

# ---- Mapping ----
LEFT_IS_CLOCKWISE = True    # True: face on LEFT -> clockwise. Flip if backwards
TARGET_GAIN = 1.0           # >1 reaches +/-90 deg before the face hits the screen edge (e.g. 1.3)
SMOOTHING = 0.5             # 0 = raw face position, closer to 1 = smoother but laggier

CW = 1                 # DIR pin level for clockwise
CCW = 0

# ---- Camera ----
FRAME_SIZE = (1024, 768)    # detection resolution (try 800x600 if too slow)
SENSOR_SIZE = (1640, 1232)  # full field-of-view sensor mode
STREAM_WIDTH = 640

# ---- Detection ----
FACE_SCORE_MIN = 0.5
HAAR_WIDTH = 640
YUNET_BASE = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
YUNET_MODELS = [
    "face_detection_yunet_2022mar.onnx",   # for OpenCV 4.6
    "face_detection_yunet_2023mar.onnx",   # for OpenCV 4.8+
]
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


# ======================= Motor (position control) =======================
class PositionMotor:
    """
    Keeps a step counter (clockwise positive, 0 = home/down) and a background
    thread that sends short bursts of exact step pulses toward the target.
    """

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.lock = threading.Lock()
        self.pos_steps = 0          # where we believe the motor is
        self.target_steps = 0
        self.freq = 0
        self.moving = False
        self.direction = None
        self.in_flight = 0          # signed steps of the burst currently being sent
        self.wave_id = None
        self.running = True

        if enabled:
            self.pi = pigpio.pi()
            if not self.pi.connected:
                raise SystemExit("Could not connect to pigpiod - is it running?")
            for pin in (STEP_PIN, DIR_PIN, EN_PIN):
                self.pi.set_mode(pin, pigpio.OUTPUT)
            self.pi.write(STEP_PIN, 0)
            self.pi.write(EN_PIN, 0)   # enable driver (holds position)
            self.pi.wave_clear()
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    # ---- public API ----
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
        if self.enabled:
            if go_home:
                self.set_target_deg(0)
                t0 = time.time()
                while time.time() - t0 < 10:
                    with self.lock:
                        done = abs(self.pos_steps) <= STOP_STEPS and not self.in_flight
                    if done:
                        break
                    time.sleep(0.02)
            self.running = False
            self.thread.join(timeout=1.0)
            self.pi.wave_tx_stop()
            self.pi.wave_clear()
            self.pi.write(STEP_PIN, 0)
            self.pi.write(EN_PIN, 1)   # release driver
            self.pi.stop()

    # ---- background thread ----
    def _finish_burst(self):
        if self.wave_id is not None:
            self.pi.wave_delete(self.wave_id)
            self.wave_id = None
        with self.lock:
            self.pos_steps += self.in_flight
        self.in_flight = 0

    def _send_burst(self, n, direction, freq):
        if direction != self.direction:
            self.pi.write(DIR_PIN, CW if direction > 0 else CCW)
            time.sleep(DIR_SETUP_S)
            self.direction = direction
        half_us = max(2, int(500000 / freq))
        pulses = []
        for _ in range(n):
            pulses.append(pigpio.pulse(1 << STEP_PIN, 0, half_us))
            pulses.append(pigpio.pulse(0, 1 << STEP_PIN, half_us))
        self.pi.wave_clear()
        self.pi.wave_add_generic(pulses)
        wid = self.pi.wave_create()
        if wid < 0:
            return
        self.wave_id = wid
        self.in_flight = n * direction
        self.pi.wave_send_once(wid)

    def _run(self):
        while self.running:
            if self.in_flight and self.pi.wave_tx_busy():
                time.sleep(0.001)
                continue
            if self.in_flight:
                self._finish_burst()

            with self.lock:
                remaining = self.target_steps - self.pos_steps
            dist = abs(remaining)

            if not self.moving:
                if dist >= START_DEG * STEPS_PER_DEG:
                    self.moving = True
                    self.freq = 0
            elif dist <= STOP_STEPS:
                self.moving = False
                self.freq = 0

            if not self.moving:
                time.sleep(0.005)
                continue

            direction = 1 if remaining > 0 else -1
            if direction != self.direction:
                self.freq = 0   # restart slowly after a reversal

            # speed: slow near the target, accelerate gradually from a stop
            slow_zone = SLOW_ZONE_DEG * STEPS_PER_DEG
            want = MIN_FREQ_HZ + (MAX_FREQ_HZ - MIN_FREQ_HZ) * min(1.0, dist / slow_zone)
            if self.freq == 0:
                self.freq = MIN_FREQ_HZ
            else:
                self.freq = min(want, self.freq + ACCEL_HZ_PER_CHUNK)
            self.freq = max(MIN_FREQ_HZ, min(self.freq, max(want, MIN_FREQ_HZ)))

            n = min(dist, CHUNK_STEPS)
            self._send_burst(n, direction, self.freq)


# ======================= Face detection =======================
class FaceDetector:
    """detect() returns (x, y, w, h, source) in frame pixel coordinates, or None."""

    def __init__(self, frame_w, frame_h):
        self.frame_w, self.frame_h = frame_w, frame_h
        self.yunet = None
        self.haar_front = None
        self.haar_profile = None
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        print(f"OpenCV version: {cv2.__version__}")
        self._init_yunet()
        self._init_haar()
        if self.yunet is None and self.haar_front is None and self.haar_profile is None:
            raise SystemExit("No face detector could be loaded. "
                             "Check internet for the YuNet download and try: sudo apt install opencv-data")

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
                det.setInputSize((self.frame_w, self.frame_h))
                det.detect(np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8))  # self-test
                self.yunet = det
                print(f"Face detector 1: YuNet ({name})")
                return
            except Exception as e:
                msg = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
                print(f"YuNet {name} not usable: {msg}")
                if os.path.exists(path) and os.path.getsize(path) < 10000:
                    os.remove(path)
        print("YuNet unavailable")

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
        print(f"Haar frontal: {'loaded' if self.haar_front else 'missing'}, "
              f"Haar profile: {'loaded' if self.haar_profile else 'missing'}")

    def _haar_gray(self, frame_bgr):
        scale = self.frame_w / HAAR_WIDTH
        small = cv2.resize(frame_bgr, (HAAR_WIDTH, int(self.frame_h / scale)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return self.clahe.apply(gray), scale

    @staticmethod
    def _largest(rects):
        return max(rects, key=lambda r: r[2] * r[3])

    def detect(self, frame_bgr):
        if self.yunet is not None:
            self.yunet.setInputSize((self.frame_w, self.frame_h))
            _, faces = self.yunet.detect(frame_bgr)
            if faces is not None and len(faces) > 0:
                f = self._largest(faces)
                return (int(f[0]), int(f[1]), int(f[2]), int(f[3]), "yunet")

        if self.haar_front is None and self.haar_profile is None:
            return None
        gray, scale = self._haar_gray(frame_bgr)
        gw = gray.shape[1]

        if self.haar_front is not None:
            r = self.haar_front.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(24, 24))
            if len(r) > 0:
                x, y, w, h = [int(v * scale) for v in self._largest(r)]
                return (x, y, w, h, "haar")

        if self.haar_profile is not None:
            r = self.haar_profile.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(24, 24))
            if len(r) > 0:
                x, y, w, h = [int(v * scale) for v in self._largest(r)]
                return (x, y, w, h, "profile")
            r = self.haar_profile.detectMultiScale(
                cv2.flip(gray, 1), scaleFactor=1.1, minNeighbors=4, minSize=(24, 24))
            if len(r) > 0:
                x, y, w, h = self._largest(r)
                x = gw - x - w  # un-flip
                x, y, w, h = [int(v * scale) for v in (x, y, w, h)]
                return (x, y, w, h, "profile")
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
BOX_COLORS = {"yunet": (0, 255, 0), "haar": (255, 200, 0), "profile": (255, 0, 255)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true", help="show local camera window (needs a display)")
    ap.add_argument("--stream", action="store_true", help="serve preview in a browser")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--no-motor", action="store_true", help="detect only, never touch the motor")
    args = ap.parse_args()

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
    print(f"Motor home = current position (0 deg). Range +/-{MAX_DEG:.0f} deg = "
          f"+/-{int(MAX_DEG * STEPS_PER_DEG)} steps.")

    last_seen = time.time()
    smooth_err = 0.0
    have_face = False
    fps = 0.0
    t_prev = time.time()

    try:
        while True:
            frame = picam2.capture_array()
            det = detector.detect(frame)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-3))
            t_prev = now

            if det is not None:
                x, y, bw, bh, src = det
                error = ((x + bw / 2) - w / 2) / (w / 2)   # -1 = far left ... +1 = far right
                if have_face:
                    smooth_err = SMOOTHING * smooth_err + (1 - SMOOTHING) * error
                else:
                    smooth_err = error                       # snap on first sighting
                    have_face = True
                last_seen = now

                angle = -smooth_err if LEFT_IS_CLOCKWISE else smooth_err
                motor.set_target_deg(angle * TARGET_GAIN * MAX_DEG)
                status = f"face [{src}] target {motor.target_deg():+.0f} deg"
            else:
                have_face = False
                if RETURN_HOME_AFTER_S is not None and now - last_seen > RETURN_HOME_AFTER_S:
                    motor.set_target_deg(0)
                status = "no face - holding"

            status += f" | motor {motor.angle_deg():+.0f} deg"
            print(f"{status}  {fps:.1f} fps")

            if args.preview or streamer:
                view = frame.copy()
                cv2.line(view, (w // 2, 0), (w // 2, h), (255, 255, 0), 1)
                if det is not None:
                    x, y, bw, bh, src = det
                    cv2.rectangle(view, (x, y), (x + bw, y + bh), BOX_COLORS.get(src, (0, 255, 0)), 3)
                cv2.putText(view, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 255), 2)
                cv2.putText(view, f"{fps:.1f} fps", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 255), 2)
                if streamer:
                    sh = int(h * STREAM_WIDTH / w)
                    streamer.update(cv2.resize(view, (STREAM_WIDTH, sh)))
                if args.preview:
                    cv2.imshow("googley eyes", view)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    except KeyboardInterrupt:
        pass
    finally:
        motor.shutdown(go_home=RETURN_HOME_ON_EXIT)
        picam2.stop()
        if args.preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
