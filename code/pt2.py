#!/usr/bin/env python3
"""
Motion tracker -> two TMC2208 STEP/DIR steppers (pigpio), POSITION control

  python3 personTracker.py                          # track + drive motors + serve video
  python3 personTracker.py --no-motor               # camera/motion test only (motors untouched)
  python3 personTracker.py --no-stream              # no browser video
  python3 personTracker.py --preview                # also open a local window (needs a display)

Video: the browser stream is ON by default. At startup the script prints the exact
address to open, e.g. http://192.168.1.50:8080/ (also /snapshot for a single still).
The motion mask (what the tracker thinks is moving) is shown in the bottom-right
corner of the video. The terminal prints a line each time a browser connects.

What it tracks:
  * ANYTHING that moves (costumes, props, people, whatever) - no face needed.
  * The camera must be STATIONARY (not mounted on the rotating eyes), otherwise the
    whole image moves whenever the motors turn.
  * Method: compare each frame to a slowly-updating background image. Pixels that
    changed a lot become a motion mask; the largest moving blob is the target.
  * Someone who stands completely still slowly fades into the background (about
    3 seconds) and stops being tracked - the eyes then hold, and swing home after
    RETURN_HOME_AFTER_S.

Motor behavior:
  * Both motors are driven in sync: same direction, same steps, same instant.
  * Wherever the motors point when you START the script is home = 0 deg (down).
  * Target horizontal position maps linearly to motor angle:
        left edge -> +90 deg (clockwise) | center -> 0 | right edge -> -90 deg
        (flip with LEFT_IS_CLOCKWISE). Never exceeds +/-MAX_DEG.
  * Nothing moving -> motors hold position, then swing back to 0 deg after
    RETURN_HOME_AFTER_S seconds.
  * Motors run in their own thread so they move smoothly.

Feedback-loop protection (the motors must never chase their own movement):
  * Motion detection is paused while the motors are moving and for SETTLE_S seconds
    afterwards, and the background is re-learned during that time. This stops the
    camera seeing the eyes (or its own shake) and treating that as motion.
  * IGNORE_REGIONS can mask off an area of the image permanently (e.g. where the
    eyes appear in the camera view).
  * Turn the pause off with BLANK_WHILE_MOVING = False only if the camera is rigidly
    fixed AND cannot see the eyes.

Noise rejection:
  * Image is blurred and downscaled, tiny blobs are ignored (MIN_AREA_FRAC).
  * Motion must be seen CONFIRM_FRAMES frames in a row before the motors react.
  * If a huge part of the image changes at once (lights switching, camera bumped,
    auto-exposure), that frame is ignored and the background is re-learned.

Smoothness / safety design (motors):
  * All step waves are created ONCE at startup (one tiny wave per speed level) and
    never deleted while running. Each move is sent as a pigpio wave CHAIN: an exact,
    gap-free list of those waves that runs once and stops. Nothing can repeat, and
    the step count is exact (counted only when the chain has finished).
  * Acceleration and deceleration are both limited.
  * The target only updates when it moves more than a few degrees.
  * `--test-motor` sweeps the motors +30 / -30 / 0 deg with no camera, to test the
    motors on their own.

Requires: pigpiod running (sudo systemctl start pigpiod)
Open-loop motors (no encoder): if steps get skipped, restart with motors pointing down to re-home.
"""

import argparse
import socket
import threading
import time
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

# ---- Motion tuning (motors) ----
MIN_FREQ_HZ = 200
MAX_FREQ_HZ = 1000     # lower for slower, gentler movement
ACCEL_HZ_PER_STEP = 5.0    # max speed-up: Hz gained per step taken
DECEL_HZ_PER_STEP = 10.0   # max slow-down: Hz lost per step taken
SLOW_ZONE_DEG = 25.0
UNIT_STEPS = 4         # steps in each pre-built wave (move resolution)
FREQ_LEVELS = 33       # number of pre-built speed levels between MIN and MAX freq
CHAIN_TIME_S = 0.25    # each chain runs about this long (max delay before a new target is used)
MAX_CHAIN_UNITS = 48   # safety cap on waves per chain
START_DEG = 2.0        # target must move this far before the motors start moving
STOP_STEPS = UNIT_STEPS - 1   # close enough: within this many steps of the target
RETURN_HOME_ON_EXIT = True
RETURN_HOME_AFTER_S = 10.0   # swing back to 0 deg after this long with no motion; None = hold forever

# ---- Mapping ----
LEFT_IS_CLOCKWISE = True
TARGET_GAIN = 1.0      # >1 reaches +/-90 deg before the target hits the screen edge (e.g. 1.3)
SMOOTHING = 0.8        # applied per frame; 0 = raw, closer to 1 = smoother/laggier
TARGET_DEADBAND_DEG = 3.0   # ignore target changes smaller than this (stops jitter)

CW = 1
CCW = 0

# ---- Camera ----
FRAME_SIZE = (640, 480)     # frame size used for tracking and the stream
SENSOR_SIZE = (1640, 1232)  # full field-of-view sensor mode
STREAM_WIDTH = 640
CV_THREADS = 2              # cap OpenCV threads so the motor thread keeps getting CPU time

# ---- Motion detection ----
MOTION_WIDTH = 320          # frames are downscaled to this width for motion analysis
BLUR_PX = 21                # Gaussian blur size (odd); bigger = ignores sensor noise/fine detail
BG_ALPHA = 0.02             # background update speed per frame (higher = still things fade faster)
MOTION_THRESH = 25          # brightness change (0-255) that counts as motion; lower = more sensitive
DILATE_PX = 15              # grows/merges motion pixels so one body = one blob
MIN_AREA_FRAC = 0.004       # ignore blobs smaller than this fraction of the image (0.4%)
MAX_MOTION_FRAC = 0.5       # if more than this much of the image changes, treat as lighting/camera change
WARMUP_FRAMES = 20          # frames used to learn the background at startup
IGNORE_REGIONS = []         # areas to never track: (x0, y0, x1, y1) as 0-1 fractions of the image,
                            # e.g. [(0.3, 0.0, 0.7, 0.5)] masks the top-middle if the eyes are in view

# ---- Feedback-loop protection ----
BLANK_WHILE_MOVING = True   # ignore motion while the motors move (stops self-tracking loops)
SETTLE_S = 0.5              # keep ignoring motion this long after the motors stop (camera wobble)

# ---- Motion confirmation ----
CONFIRM_FRAMES = 3          # motion frames in a row before the motors react
CONFIRM_MAX_JUMP = 0.35     # consecutive motion must be within this fraction of frame width
LOST_S = 2.0                # motion is considered gone after this long with nothing moving


# ======================= Motor (position control) =======================
class PositionMotor:
    """Step counter (clockwise positive, 0 = home) + thread sending exact step chains toward the target.

    Drives all motors in STEP_PINS together: every pulse toggles every STEP pin at the
    same instant, and all DIR / EN pins are set together, so the motors stay in sync.

    At startup one small wave (UNIT_STEPS steps) is created for each speed level and
    kept for the whole run. A move is a wave CHAIN built from those waves: it runs
    exactly once, with no gaps, then stops. pos_steps is updated only after the chain
    has finished, so the count is exact.
    """

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.lock = threading.Lock()
        self.pos_steps = 0
        self.target_steps = 0
        self.freq = 0
        self.moving = False
        self.direction = None
        self.inflight = 0         # signed steps in the chain currently running
        self.running = True
        self.level_freqs = [MIN_FREQ_HZ + (MAX_FREQ_HZ - MIN_FREQ_HZ) * i / (FREQ_LEVELS - 1)
                            for i in range(FREQ_LEVELS)]
        self.waves = []

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
            self.pi.wave_tx_stop()
            self.pi.wave_clear()
            self._build_waves()
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def _build_waves(self):
        """One wave per speed level, each UNIT_STEPS steps; created once, never deleted while running."""
        for f in self.level_freqs:
            half_us = max(2, int(500000 / f))
            pulses = []
            for _ in range(UNIT_STEPS):
                # all STEP pins go high / low together -> motors stay in sync
                pulses.append(pigpio.pulse(STEP_MASK, 0, half_us))
                pulses.append(pigpio.pulse(0, STEP_MASK, half_us))
            self.pi.wave_add_generic(pulses)
            wid = self.pi.wave_create()
            if wid < 0:
                raise SystemExit(f"pigpio could not create a step wave (error {wid})")
            self.waves.append(wid)

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

    def is_busy(self):
        """True while the motors are moving or a step chain is running."""
        return self.enabled and (self.moving or self.inflight != 0)

    def shutdown(self, go_home=True):
        if not self.enabled:
            return
        if go_home:
            self.set_target_deg(0)
            t0 = time.time()
            while time.time() - t0 < 10:
                with self.lock:
                    done = abs(self.pos_steps) <= STOP_STEPS and not self.inflight
                if done:
                    break
                time.sleep(0.02)
        self.running = False
        self.thread.join(timeout=1.0)
        self.pi.wave_tx_stop()
        for wid in self.waves:
            self.pi.wave_delete(wid)
        self.pi.wave_clear()
        for pin in STEP_PINS:
            self.pi.write(pin, 0)
        for pin in EN_PINS:
            self.pi.write(pin, 1)   # release drivers
        self.pi.stop()

    def _set_direction(self, direction):
        for pin, invert in zip(DIR_PINS, DIR_INVERT):
            level = CW if direction > 0 else CCW
            if invert:
                level = 1 - level
            self.pi.write(pin, level)

    def _plan_chain(self, dist):
        """Build the wave-id list for the next ~CHAIN_TIME_S of motion, with accel / decel limits."""
        slow_zone = SLOW_ZONE_DEG * STEPS_PER_DEG
        freq = self.freq if self.freq > 0 else MIN_FREQ_HZ
        seq = []
        t = 0.0
        steps_left = dist
        while steps_left > STOP_STEPS and len(seq) < MAX_CHAIN_UNITS and t < CHAIN_TIME_S:
            want = MIN_FREQ_HZ + (MAX_FREQ_HZ - MIN_FREQ_HZ) * min(1.0, steps_left / slow_zone)
            if want > freq:
                freq = min(want, freq + ACCEL_HZ_PER_STEP * UNIT_STEPS)
            else:
                freq = max(want, freq - DECEL_HZ_PER_STEP * UNIT_STEPS)
            freq = max(MIN_FREQ_HZ, min(freq, MAX_FREQ_HZ))
            lvl = int(round((freq - MIN_FREQ_HZ) / (MAX_FREQ_HZ - MIN_FREQ_HZ) * (FREQ_LEVELS - 1)))
            lvl = max(0, min(FREQ_LEVELS - 1, lvl))
            seq.append(self.waves[lvl])
            t += UNIT_STEPS / self.level_freqs[lvl]
            steps_left -= UNIT_STEPS
        self.freq = freq
        return seq

    def _run(self):
        while self.running:
            # 1) wait for the running chain to finish, then count its steps exactly once
            if self.inflight:
                if self.pi.wave_tx_busy():
                    time.sleep(0.0005)
                    continue
                with self.lock:
                    self.pos_steps += self.inflight
                self.inflight = 0

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
                self._set_direction(direction)
                time.sleep(DIR_SETUP_S)
                self.direction = direction
                self.freq = 0   # restart slowly after a reversal

            seq = self._plan_chain(dist)
            if not seq:
                self.moving = False
                continue
            rc = self.pi.wave_chain(seq)
            if rc is not None and rc < 0:
                print(f"pigpio wave_chain error {rc}")
                time.sleep(0.05)
                continue
            self.inflight = len(seq) * UNIT_STEPS * direction


# ======================= Motion detection =======================
class MotionDetector:
    """Background-difference motion detector.

    update(frame_bgr, blank=False) -> (blobs, ready, global_change)
      blobs: [(x, y, w, h, area)] in frame pixels, largest first
      ready: False while the background is still being learned at startup
      global_change: True if so much changed that the frame was discarded
      blank: True while the motors move - no detection, background is re-learned
    """

    def __init__(self, frame_w, frame_h):
        self.fw, self.fh = frame_w, frame_h
        self.mw = MOTION_WIDTH
        self.mh = int(frame_h * self.mw / frame_w)
        self.scale = frame_w / self.mw
        self.min_area = MIN_AREA_FRAC * self.mw * self.mh
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_PX, DILATE_PX))
        self.bg = None
        self.frames = 0
        self.mask = None

    def update(self, frame, blank=False):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (self.mw, self.mh), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (BLUR_PX, BLUR_PX), 0)
        self.frames += 1

        if self.bg is None:
            self.bg = small.astype(np.float32)
            return [], False, False

        if self.frames <= WARMUP_FRAMES:
            cv2.accumulateWeighted(small, self.bg, 0.2)   # learn the background quickly
            return [], False, False

        if blank:
            cv2.accumulateWeighted(small, self.bg, 0.5)   # follow the scene while it is shifting
            self.mask = None
            return [], True, False

        diff = cv2.absdiff(small, cv2.convertScaleAbs(self.bg))
        cv2.accumulateWeighted(small, self.bg, BG_ALPHA)
        _, mask = cv2.threshold(diff, MOTION_THRESH, 255, cv2.THRESH_BINARY)
        mask = cv2.dilate(mask, self.kernel, iterations=1)
        for (rx0, ry0, rx1, ry1) in IGNORE_REGIONS:
            mask[int(ry0 * self.mh):int(ry1 * self.mh), int(rx0 * self.mw):int(rx1 * self.mw)] = 0
        self.mask = mask

        if cv2.countNonZero(mask) > MAX_MOTION_FRAC * mask.size:
            # lighting flash / camera bump / exposure change: relearn and ignore this frame
            self.bg = small.astype(np.float32)
            return [], True, True

        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        blobs = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            s = self.scale
            blobs.append((int(x * s), int(y * s), int(w * s), int(h * s), area))
        blobs.sort(key=lambda b: b[4], reverse=True)
        return blobs, True, False


# ======================= Browser streaming =======================
def local_ip():
    """Best guess at this machine's LAN address (what to type into the browser)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))   # no packet is sent; just picks the outgoing interface
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "<pi-address>"


class Streamer:
    """Serves annotated frames as an MJPEG stream you can open in a browser.

      /           page with the live video
      /stream     the raw MJPEG stream
      /snapshot   a single JPEG (handy to test the connection)
    The server starts immediately, before the camera, so the page loads even while
    the rest of the script is still starting up.
    """

    def __init__(self, port):
        self.jpeg = None
        self.cond = threading.Condition()
        streamer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                path = self.path.split("?")[0]
                if path == "/stream":
                    print(f"stream: {self.client_address[0]} opened the video stream")
                    self.send_response(200)
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    try:
                        while True:
                            with streamer.cond:
                                streamer.cond.wait(timeout=5)
                                data = streamer.jpeg
                            if data is None:
                                continue
                            self.wfile.write(
                                b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
                    except OSError:
                        pass   # browser closed the tab
                elif path == "/snapshot":
                    with streamer.cond:
                        data = streamer.jpeg
                    if data is None:
                        body = b"no camera frame yet"
                        self.send_response(503)
                        self.send_header("Content-Type", "text/plain")
                    else:
                        body = data
                        self.send_response(200)
                        self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    print(f"stream: {self.client_address[0]} loaded the page")
                    page = (b'<html><head><title>googley eyes</title></head>'
                            b'<body style="margin:0;background:#000;color:#ccc;font-family:sans-serif">'
                            b'<img src="/stream" style="width:100%;max-width:960px;display:block">'
                            b'<p style="padding:8px">If the picture stays blank, the camera has not produced '
                            b'a frame yet - check the terminal. Still image: <a href="/snapshot" '
                            b'style="color:#8cf">/snapshot</a></p></body></html>')
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(page)))
                    self.end_headers()
                    self.wfile.write(page)

        self.server = None
        self.port = None
        for p in range(port, port + 10):
            try:
                self.server = ThreadingHTTPServer(("0.0.0.0", p), Handler)
                self.port = p
                break
            except OSError as e:
                print(f"stream: port {p} unavailable ({e})")
        if self.server is None:
            raise SystemExit(f"Could not open any stream port from {port} to {port + 9}. "
                             f"Is an old copy still running? Try: pkill -f personTracker.py")
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
    ap.add_argument("--stream", action="store_true", help="serve video in a browser (now the default)")
    ap.add_argument("--no-stream", action="store_true", help="do not serve video in a browser")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--no-motor", action="store_true", help="detect only, never touch the motors")
    ap.add_argument("--test-motor", action="store_true",
                    help="sweep motors +30 / -30 / 0 deg with no camera, then exit")
    args = ap.parse_args()

    if args.test_motor:
        tm = PositionMotor(enabled=True)
        try:
            for deg in (30, -30, 0):
                print(f"moving to {deg:+d} deg ...")
                tm.set_target_deg(deg)
                t0 = time.time()
                while time.time() - t0 < 10:
                    if abs(tm.angle_deg() - deg) * STEPS_PER_DEG <= STOP_STEPS and not tm.is_busy():
                        break
                    time.sleep(0.02)
                print(f"  reached {tm.angle_deg():+.1f} deg")
                time.sleep(0.5)
            print("test done - the motors should now be back where they started and silent")
        except KeyboardInterrupt:
            pass
        finally:
            tm.shutdown(go_home=False)
        return

    cv2.setNumThreads(CV_THREADS)   # leave CPU headroom for the motor thread

    # Start the video server FIRST so the page loads even if something later is slow or fails
    streamer = None if args.no_stream else Streamer(args.port)
    if streamer:
        print(f"Video stream ready: open  http://{local_ip()}:{streamer.port}/  in a browser "
              f"(single still: /snapshot)")

    motor = PositionMotor(enabled=not args.no_motor)

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": FRAME_SIZE, "format": "RGB888"},  # RGB888 = BGR byte order, OpenCV-ready
        raw={"size": SENSOR_SIZE})
    picam2.configure(config)
    picam2.start()
    print("Sensor mode:", picam2.camera_configuration()["sensor"])

    w, h = FRAME_SIZE
    detector = MotionDetector(w, h)
    print(f"Motor home = current position (0 deg). Range +/-{MAX_DEG:.0f} deg = "
          f"+/-{int(MAX_DEG * STEPS_PER_DEG)} steps. Driving {len(STEP_PINS)} motors in sync.")
    print(f"Learning the background for {WARMUP_FRAMES} frames - keep the scene still ...")

    sw = STREAM_WIDTH
    sh = int(h * sw / w)
    sx = sw / w

    last_seen = time.time()     # last time confirmed motion was tracked
    smooth_err = 0.0
    have_target = False         # smoothing state is valid
    active = False              # confirmed motion is being tracked
    hits = 0                    # consecutive frames with consistent motion
    last_hit_cx = None
    homed = False               # already commanded back to 0 after motion stopped
    status = "starting"
    fps = 0.0
    t_prev = time.time()
    last_print = 0.0
    last_status_kind = ""
    last_busy = 0.0

    try:
        while True:
            frame = picam2.capture_array()
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-3))
            t_prev = now

            if motor.is_busy():
                last_busy = now
            blank = BLANK_WHILE_MOVING and (motor.is_busy() or now - last_busy < SETTLE_S)

            blobs, ready, global_change = detector.update(frame, blank)
            tracked = blobs[0] if blobs else None

            if not ready:
                status = "learning background"
                last_seen = now
            elif blank:
                hits = 0
                last_hit_cx = None
                if active:
                    last_seen = now   # still following something; don't start the go-home timer
                status = "motors moving - ignoring motion"
            elif global_change:
                hits = 0
                last_hit_cx = None
                status = "big change - relearning background"
            elif tracked is not None:
                x, y, bw, bh, area = tracked
                cx = x + bw / 2
                if last_hit_cx is not None and abs(cx - last_hit_cx) <= CONFIRM_MAX_JUMP * w:
                    hits += 1
                else:
                    hits = 1
                last_hit_cx = cx
                if hits >= CONFIRM_FRAMES:
                    active = True

                if active:
                    error = (cx - w / 2) / (w / 2)   # -1 = far left ... +1 = far right
                    smooth_err = SMOOTHING * smooth_err + (1 - SMOOTHING) * error if have_target else error
                    have_target = True
                    last_seen = now
                    homed = False
                    angle = -smooth_err if LEFT_IS_CLOCKWISE else smooth_err
                    new_target = angle * TARGET_GAIN * MAX_DEG
                    # Deadband: only retarget when the motion has really moved (ignores wobble)
                    if abs(new_target - motor.target_deg()) >= TARGET_DEADBAND_DEG:
                        motor.set_target_deg(new_target)
                    status = f"motion target {motor.target_deg():+.0f} deg"
                else:
                    status = f"motion candidate {hits}/{CONFIRM_FRAMES}"
            else:
                hits = 0
                last_hit_cx = None
                if active and now - last_seen > LOST_S:
                    active = False
                    have_target = False
                status = "no motion - home" if homed else "no motion - holding"

            # Return to 0 deg after RETURN_HOME_AFTER_S with no confirmed motion
            if (ready and RETURN_HOME_AFTER_S is not None and not homed
                    and now - last_seen > RETURN_HOME_AFTER_S):
                motor.set_target_deg(0)
                homed = True
                active = False
                have_target = False
                status = "no motion - returning home"

            # Console output: once a second, or immediately when the kind of status changes
            kind = status.split(" ")[0] + (" " + status.split(" ")[1] if " " in status else "")
            if now - last_print >= 1.0 or kind != last_status_kind:
                print(f"{status} | motor {motor.angle_deg():+.0f} deg | video {fps:.0f} fps")
                last_print = now
                last_status_kind = kind

            if args.preview or streamer:
                view = cv2.resize(frame, (sw, sh))
                cv2.line(view, (sw // 2, 0), (sw // 2, sh), (255, 255, 0), 1)
                for b in blobs:
                    bx, by, bbw, bbh, _ = b
                    is_target = b is tracked
                    color = (0, 255, 0) if (is_target and active) else (0, 165, 255) if is_target else (0, 0, 160)
                    cv2.rectangle(view, (int(bx * sx), int(by * sx)),
                                  (int((bx + bbw) * sx), int((by + bbh) * sx)), color, 2)
                if detector.mask is not None:   # what the tracker sees as motion (bottom-right inset)
                    mm = cv2.resize(detector.mask, (sw // 4, sh // 4), interpolation=cv2.INTER_NEAREST)
                    mh_, mw_ = mm.shape[:2]
                    view[sh - mh_ - 30:sh - 30, sw - mw_:sw] = cv2.cvtColor(mm, cv2.COLOR_GRAY2BGR)
                    cv2.rectangle(view, (sw - mw_, sh - mh_ - 30), (sw - 1, sh - 31), (255, 255, 255), 1)
                cv2.putText(view, f"{status} | motor {motor.angle_deg():+.0f}", (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                cv2.putText(view, f"video {fps:.0f} fps", (8, sh - 10),
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
        motor.shutdown(go_home=RETURN_HOME_ON_EXIT)
        picam2.stop()
        if args.preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
