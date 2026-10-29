#!/usr/bin/env python3
"""
Motion tracker -> two TMC2208 STEP/DIR steppers (pigpio), POSITION control

  python3 personTracker8.py                          # track motion + drive motors
  python3 personTracker8.py --stream                 # also serve video at http://<pi>:8080
  python3 personTracker8.py --stream --no-motor      # camera/detection test only
  python3 personTracker8.py --stream --mask          # stream shows the motion mask in the corner
  python3 personTracker8.py --preview                # local window (needs a display)

Motor behavior:
  * Both motors are driven in sync: same direction, same steps, same instant.
  * Wherever the motors point when you START the script is home = 0 deg (down).
  * Horizontal position of the motion maps linearly to motor angle:
        left edge -> +90 deg (clockwise) | center -> 0 | right edge -> -90 deg
        (flip with LEFT_IS_CLOCKWISE). Never exceeds +/-MAX_DEG.
  * Target still -> motors still (deadband). Motion lost -> motors hold position,
    then swing home after RETURN_HOME_AFTER_S (set to None to hold forever).
  * Motors run in their own thread so they move smoothly.

Motion detection design:
  * Works on a small grayscale, blurred copy of every camera frame (fast, noise-resistant).
  * Adaptive background model (MOG2) learns the static scene, so lighting drift,
    flicker and furniture are ignored. Anything that moves shows up as "foreground".
  * Speckle is removed, then nearby motion pixels are merged so a whole person
    (arms, legs, costume pieces) becomes ONE blob instead of many fragments.
  * Blobs smaller than MIN_MOTION_FRAC of the frame are ignored (noise, bugs, dust).
  * The target must be seen for CONFIRM_FRAMES in a row before the motors react,
    so a one-frame flicker never makes the eyes twitch.
  * Target is the largest blob, but once locked on it sticks to the blob nearest
    the previous position (doesn't jump between two people unless one is much bigger).
  * If most of the frame changes at once (lights switch, camera bumped), that frame
    is ignored instead of being treated as motion.
  * A person who freezes slowly fades into the background (BG_LEARNING_RATE):
    the eyes hold their last position, then go home after RETURN_HOME_AFTER_S.

Stream colors: green box = locked target, yellow = candidate not yet confirmed,
gray = other motion. Cyan line = center of view. --mask shows what the detector sees.

Requires: pigpiod running (sudo systemctl start pigpiod)
Open-loop motors (no encoder): if steps get skipped, restart with motors pointing down to re-home.
The camera must stay fixed and should not be able to see the eyes moving (see IGNORE_ZONES).
"""

import argparse
import threading
import time
from collections import namedtuple
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

# ---- Motion tuning (motors) ----
MIN_FREQ_HZ = 200
MAX_FREQ_HZ = 1000     # lower for slower, gentler movement
ACCEL_HZ_PER_CHUNK = 150
SLOW_ZONE_DEG = 25.0
CHUNK_STEPS = 40
START_DEG = 2.0        # target must move this far before the motors start moving
STOP_STEPS = 1
RETURN_HOME_ON_EXIT = True
RETURN_HOME_AFTER_S = 10.0   # swing to center if no motion for this long; None = hold forever

# ---- Mapping ----
LEFT_IS_CLOCKWISE = True
TARGET_GAIN = 1.0      # >1 reaches +/-90 deg before the target hits the screen edge (e.g. 1.3)
SMOOTHING = 0.6        # applied per frame; 0 = raw, closer to 1 = smoother/laggier

CW = 1
CCW = 0

# ---- Camera ----
FRAME_SIZE = (640, 480)     # frame size captured (also the stream size)
SENSOR_SIZE = (1640, 1232)  # full field-of-view sensor mode

# ---- Motion detection ----
DET_W, DET_H = 320, 240     # detection resolution (smaller = faster)
BLUR_K = 7                  # blur before differencing (odd number); bigger = less noise
BG_VAR_THRESHOLD = 25       # lower = more sensitive (and noisier); raise if it jitters at rest
BG_LEARNING_RATE = 0.003    # how fast still things fade into the background (~1/rate frames)
WARMUP_LEARNING_RATE = 0.05
WARMUP_FRAMES = 40          # frames spent learning the empty scene at startup (keep the view still)
OPEN_K = 3                  # speckle removal kernel
MERGE_K = 25                # merge nearby motion into one blob (bigger = merges more)
MIN_MOTION_FRAC = 0.003     # blob needs this fraction of the frame in moving pixels (~230 px)
GLOBAL_MOTION_FRAC = 0.5    # if more than this much of the frame "moves", ignore the frame
CONFIRM_FRAMES = 3          # frames in a row before the motors react
LOSE_FRAMES = 15            # frames without motion before the lock is dropped
STICKY_MIN_FRAC = 0.5       # a blob must be at least this fraction of the biggest to be preferred
                            # for being near the previous position
# Parts of the view to ignore, as fractions (x0, y0, x1, y1) of the frame, e.g. if the camera
# can see the eyes themselves moving:  IGNORE_ZONES = [(0.0, 0.6, 0.3, 1.0)]
IGNORE_ZONES = []

Blob = namedtuple("Blob", "px cx cy x y w h")   # all in detection-resolution pixels
Result = namedtuple("Result", "target blobs status mask")


# ======================= Motor (position control) =======================
class PositionMotor:
    """Step counter (clockwise positive, 0 = home) + thread sending exact step bursts toward the target.

    Drives all motors in STEP_PINS together: one wave toggles every STEP pin at the
    same instant, and all DIR / EN pins are set together, so the motors stay in sync.
    """

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.lock = threading.Lock()
        self.pos_steps = 0
        self.target_steps = 0
        self.freq = 0
        self.moving = False
        self.direction = None
        self.in_flight = 0
        self.wave_id = None
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
                    done = abs(self.pos_steps) <= STOP_STEPS and not self.in_flight
                if done:
                    break
                time.sleep(0.02)
        self.running = False
        self.thread.join(timeout=1.0)
        self.pi.wave_tx_stop()
        self.pi.wave_clear()
        for pin in STEP_PINS:
            self.pi.write(pin, 0)
        for pin in EN_PINS:
            self.pi.write(pin, 1)   # release drivers
        self.pi.stop()

    def _finish_burst(self):
        if self.wave_id is not None:
            self.pi.wave_delete(self.wave_id)
            self.wave_id = None
        with self.lock:
            self.pos_steps += self.in_flight
        self.in_flight = 0

    def _set_direction(self, direction):
        for pin, invert in zip(DIR_PINS, DIR_INVERT):
            level = CW if direction > 0 else CCW
            if invert:
                level = 1 - level
            self.pi.write(pin, level)

    def _send_burst(self, n, direction, freq):
        if direction != self.direction:
            self._set_direction(direction)
            time.sleep(DIR_SETUP_S)
            self.direction = direction
        half_us = max(2, int(500000 / freq))
        pulses = []
        for _ in range(n):
            # all STEP pins go high / low together -> motors stay in sync
            pulses.append(pigpio.pulse(STEP_MASK, 0, half_us))
            pulses.append(pigpio.pulse(0, STEP_MASK, half_us))
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

            slow_zone = SLOW_ZONE_DEG * STEPS_PER_DEG
            want = MIN_FREQ_HZ + (MAX_FREQ_HZ - MIN_FREQ_HZ) * min(1.0, dist / slow_zone)
            if self.freq == 0:
                self.freq = MIN_FREQ_HZ
            else:
                self.freq = min(want, self.freq + ACCEL_HZ_PER_CHUNK)
            self.freq = max(MIN_FREQ_HZ, min(self.freq, max(want, MIN_FREQ_HZ)))

            self._send_burst(min(dist, CHUNK_STEPS), direction, self.freq)


# ======================= Motion detection =======================
class MotionDetector:
    """process(frame, last_cx) -> Result(target Blob or None, all Blobs, status text, motion mask)."""

    def __init__(self):
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=BG_VAR_THRESHOLD, detectShadows=False)
        self.open_k = np.ones((OPEN_K, OPEN_K), np.uint8)
        self.merge_k = cv2.getStructuringElement(cv2.MORPH_RECT, (MERGE_K, MERGE_K))
        self.frames = 0
        self.min_px = MIN_MOTION_FRAC * DET_W * DET_H

    def process(self, frame, last_cx=None):
        small = cv2.resize(frame, (DET_W, DET_H), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (BLUR_K, BLUR_K), 0)

        lr = WARMUP_LEARNING_RATE if self.frames < WARMUP_FRAMES else BG_LEARNING_RATE
        mask = self.bg.apply(gray, learningRate=lr)
        self.frames += 1
        if self.frames <= WARMUP_FRAMES:
            return Result(None, [], f"learning background {self.frames}/{WARMUP_FRAMES}", mask)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.open_k)   # drop single-pixel speckle
        for x0, y0, x1, y1 in IGNORE_ZONES:
            mask[int(y0 * DET_H):int(y1 * DET_H), int(x0 * DET_W):int(x1 * DET_W)] = 0

        if cv2.countNonZero(mask) > GLOBAL_MOTION_FRAC * DET_W * DET_H:
            return Result(None, [], "global change - ignoring frame", mask)

        # Merge nearby motion pixels so one person = one blob, then measure each blob
        # using the real (un-dilated) motion pixels inside it.
        merged = cv2.dilate(mask, self.merge_k)
        contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            m = cv2.moments(mask[y:y + h, x:x + w], binaryImage=True)
            px = m["m00"]
            if px < self.min_px:
                continue
            blobs.append(Blob(px, x + m["m10"] / px, y + m["m01"] / px, x, y, w, h))

        if not blobs:
            return Result(None, [], "no motion", mask)

        biggest = max(blobs, key=lambda b: b.px)
        chosen = biggest
        if last_cx is not None:
            near = [b for b in blobs if b.px >= STICKY_MIN_FRAC * biggest.px]
            chosen = min(near, key=lambda b: abs(b.cx - last_cx))
        return Result(chosen, blobs, "motion", mask)


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
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true", help="show local camera window (needs a display)")
    ap.add_argument("--stream", action="store_true", help="serve preview in a browser")
    ap.add_argument("--mask", action="store_true", help="show the motion mask in the corner of the preview/stream")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--no-motor", action="store_true", help="detect only, never touch the motors")
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
    sx = w / DET_W   # detection coords -> frame coords
    detector = MotionDetector()
    print(f"Learning the background for {WARMUP_FRAMES} frames - keep the view clear of movement.")
    print(f"Motor home = current position (0 deg). Range +/-{MAX_DEG:.0f} deg = "
          f"+/-{int(MAX_DEG * STEPS_PER_DEG)} steps. Driving {len(STEP_PINS)} motors in sync.")

    hits = 0
    misses = 0
    last_cx = None
    have_lock = False
    smooth_err = 0.0
    last_seen = time.time()
    status = "starting"
    fps = 0.0
    t_prev = time.time()
    t_print = 0.0

    try:
        while True:
            frame = picam2.capture_array()
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-3))
            t_prev = now

            res = detector.process(frame, last_cx)
            tgt = res.target

            if tgt is not None:
                hits += 1
                misses = 0
                last_cx = tgt.cx
            else:
                misses += 1
                if misses > LOSE_FRAMES:
                    hits = 0
                    last_cx = None
                    have_lock = False

            locked = tgt is not None and hits >= CONFIRM_FRAMES
            if locked:
                error = (tgt.cx / DET_W - 0.5) * 2   # -1 = far left ... +1 = far right
                smooth_err = SMOOTHING * smooth_err + (1 - SMOOTHING) * error if have_lock else error
                have_lock = True
                last_seen = now
                angle = -smooth_err if LEFT_IS_CLOCKWISE else smooth_err
                motor.set_target_deg(angle * TARGET_GAIN * MAX_DEG)
                status = f"tracking, target {motor.target_deg():+.0f} deg"
            else:
                if tgt is not None:
                    status = f"confirming {hits}/{CONFIRM_FRAMES}"
                elif res.status != "no motion":
                    status = res.status
                else:
                    status = "no motion - holding"
                if RETURN_HOME_AFTER_S is not None and now - last_seen > RETURN_HOME_AFTER_S:
                    motor.set_target_deg(0)
                    if not have_lock:
                        status = "no motion - home"

            if now - t_print > 1.0:
                t_print = now
                print(f"{status} | motor {motor.angle_deg():+.0f} deg | {fps:.0f} fps | "
                      f"{len(res.blobs)} blob(s)")

            if args.preview or streamer:
                view = frame.copy()
                cv2.line(view, (w // 2, 0), (w // 2, h), (255, 255, 0), 1)
                for b in res.blobs:
                    color = (160, 160, 160)
                    thick = 1
                    if tgt is not None and b is tgt:
                        color = (0, 255, 0) if locked else (0, 255, 255)
                        thick = 2
                    cv2.rectangle(view, (int(b.x * sx), int(b.y * sx)),
                                  (int((b.x + b.w) * sx), int((b.y + b.h) * sx)), color, thick)
                    if thick == 2:
                        cv2.circle(view, (int(b.cx * sx), int(b.cy * sx)), 5, color, -1)
                cv2.putText(view, f"{status} | motor {motor.angle_deg():+.0f}", (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                cv2.putText(view, f"{fps:.0f} fps", (8, h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                if args.mask:
                    mw, mh = 160, 120
                    inset = cv2.cvtColor(cv2.resize(res.mask, (mw, mh), interpolation=cv2.INTER_NEAREST),
                                         cv2.COLOR_GRAY2BGR)
                    view[0:mh, w - mw:w] = inset
                    cv2.rectangle(view, (w - mw, 0), (w - 1, mh - 1), (255, 255, 255), 1)
                if streamer:
                    streamer.update(view)
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
