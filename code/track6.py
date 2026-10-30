#!/usr/bin/env python3
"""
Motion tracker -> two TMC2208 STEP/DIR steppers (pigpio), POSITION control

  python3 personTracker8.py                          # track motion + drive motors
  python3 personTracker8.py --stream                 # also serve video at http://<pi>:8080
  python3 personTracker8.py --stream --no-motor      # camera/detection test only
  python3 personTracker8.py --stream --mask          # stream shows the motion mask in the corner
  python3 personTracker8.py --preview                # local window (needs a display)

Motor behavior:
  * Both motors receive the same shared STEP pulse wave for tracking and homing.
  * Startup and inactivity homing find both Hall edges and use their midpoint as 0 deg (down).
  * Horizontal position of the motion maps linearly to motor angle:
        left edge -> +90 deg (clockwise) | center -> 0 | right edge -> -90 deg
        (flip with LEFT_IS_CLOCKWISE). Never exceeds +/-MAX_DEG.
  * Target still -> motors still (deadband). After RETURN_HOME_AFTER_S without a target,
    both motors independently seek their Hall sensors and reset to 0 deg.
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
HALL_PINS = [23, 24]
DIR_INVERT = [False, False]
DIR_SETUP_S = 0.001

# Hall sensors are assumed active LOW with pull-ups, as in the working track2.py.
HALL_ACTIVE_LEVEL = 0
HALL_PULL = pigpio.PUD_UP

# Calibration scan direction must be the same logical direction for both motors.
# If the motors physically rotate opposite ways, adjust DIR_INVERT for one motor.
# Three-magnet calibration
CALIBRATION_DIR = [1, 1]
HOME_FINE_FREQ_HZ = 50
CALIBRATION_FREQ_HZ = 100
CALIBRATION_REVOLUTIONS = 1       # One revolution; stop after finding all three magnets
CALIBRATION_RETRIES = 3           # Retry incomplete or inconsistent scans
CALIBRATION_MAX_EXTRA_STEPS = 500
FINE_BACKOFF_STEPS = 30
MAGNET_SPACING_TOLERANCE_DEG = 18.0
MAGNET_DEDUP_TOLERANCE_DEG = 5.0
HOME_OFFSET_DEG = [0.0, 0.0]

STEP_MASK = sum(1 << p for p in STEP_PINS)

# ---- Motor geometry ----
STEPS_PER_REV = 200
MICROSTEPS = 8
GEAR_RATIO = 1.0
MAX_DEG = 90.0
STEPS_PER_DEG = STEPS_PER_REV * MICROSTEPS * GEAR_RATIO / 360.0
REV_STEPS = int(round(STEPS_PER_REV * MICROSTEPS * GEAR_RATIO))

# ---- Motion tuning ----
MIN_FREQ_HZ = 200
MAX_FREQ_HZ = 1200
ACCEL_HZ_PER_CHUNK = 150
SLOW_ZONE_DEG = 25.0
CHUNK_STEPS = 4
OUTER_APPROACH_DEG = 20.0
START_DEG = 1.0
STOP_STEPS = 1
RETURN_HOME_ON_EXIT = False
RETURN_HOME_AFTER_S = 10.0

# ---- Mapping ----
LEFT_IS_CLOCKWISE = True
TARGET_GAIN = 1.0
SMOOTHING = 0.4

CW = 1
CCW = 0

FRAME_SIZE = (640, 480)
SENSOR_SIZE = (1640, 1232)

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


# ======================= Motor (three-magnet position control) =======================
class PositionMotor:
    """
    Two independently referenced motors driven by synchronized pigpio waves.

    Startup: scan until all three magnets have been found on each motor,
    identify the center magnet from the 90/90/180 degree spacing, then perform
    a separate fine edge-to-edge zero calibration.

    Tracking: clamp to +/-90 degrees. The outer Hall magnets are active travel
    stops. If the target disappears, park at the outer magnet on the last-seen
    side. After RETURN_HOME_AFTER_S, approach the center magnet from that side
    and perform another fine zero calibration.
    """

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.lock = threading.Lock()
        self.control_lock = threading.Lock()
        self.running = True
        self.homing = False
        self.pos_steps = [0, 0]
        self.target_steps = [0, 0]
        self.freq = 0
        self.limit_latched = [0, 0]  # -1 or +1 while held against an outer magnet
        self.last_direction = [0, 0]
        self.direction_levels = [None, None]
        self.park_side = [1, 1]
        self.thread = None

        if not enabled:
            return
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise SystemExit("Could not connect to pigpiod - is it running?")
        for pin in STEP_PINS + DIR_PINS + EN_PINS:
            self.pi.set_mode(pin, pigpio.OUTPUT)
        for pin in STEP_PINS:
            self.pi.write(pin, 0)
        for pin in EN_PINS:
            self.pi.write(pin, 0)
        for pin in HALL_PINS:
            self.pi.set_mode(pin, pigpio.INPUT)
            self.pi.set_pull_up_down(pin, HALL_PULL)
        self.pi.wave_clear()

        print("Starting three-magnet one-revolution calibration...")
        try:
            self.calibrate_three_magnets()
        except Exception:
            # Do not leave the drivers energized if calibration aborts.
            for pin in EN_PINS:
                self.pi.write(pin, 1)
            self.pi.stop()
            raise
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _hall_active(self, i):
        return self.pi.read(HALL_PINS[i]) == HALL_ACTIVE_LEVEL

    def _set_direction(self, i, direction):
        level = CW if direction > 0 else CCW
        if DIR_INVERT[i]:
            level = 1 - level
        self.pi.write(DIR_PINS[i], level)

    def _pulse(self, active_dirs, steps, frequency):
        """One waveform; active STEP pins share exactly the same pulse edges."""
        if not active_dirs or steps <= 0:
            return
        mask = 0
        direction_changed = False
        for i, direction in active_dirs.items():
            level = CW if direction > 0 else CCW
            if DIR_INVERT[i]:
                level = 1 - level
            if self.direction_levels[i] != level:
                self.pi.write(DIR_PINS[i], level)
                self.direction_levels[i] = level
                direction_changed = True
            mask |= 1 << STEP_PINS[i]
        if direction_changed:
            time.sleep(DIR_SETUP_S)
        half_us = max(2, int(500000 / max(1, frequency)))
        pulses = []
        for _ in range(steps):
            pulses.append(pigpio.pulse(mask, 0, half_us))
            pulses.append(pigpio.pulse(0, mask, half_us))
        self.pi.wave_clear()
        self.pi.wave_add_generic(pulses)
        wid = self.pi.wave_create()
        if wid < 0:
            raise RuntimeError("pigpio could not create a step waveform")
        self.pi.wave_send_once(wid)
        while self.pi.wave_tx_busy():
            time.sleep(0.0005)
        self.pi.wave_delete(wid)

    def _move_relative(self, deltas, frequency, sample_each_step=False):
        """Move motors by signed relative microsteps, sharing waveform timing."""
        remaining = [int(deltas[0]), int(deltas[1])]
        while any(v != 0 for v in remaining):
            active = {}
            batch = 1 if sample_each_step else min(4, min(abs(v) for v in remaining if v))
            for i, value in enumerate(remaining):
                if value:
                    active[i] = 1 if value > 0 else -1
            self._pulse(active, batch, frequency)
            for i, value in enumerate(remaining):
                if value:
                    moved = min(abs(value), batch)
                    remaining[i] -= moved if value > 0 else -moved

    def _find_three_centers(self, i):
        """Find the three Hall magnets in one revolution and identify the middle one.

        The scan stops immediately after the third complete Hall activation
        window. The three measured positions are then checked for the expected
        90/90/180-degree spacing. This avoids continuing into a second pass,
        where the sensor can still be active at the arbitrary scan endpoint.
        """
        direction = CALIBRATION_DIR[i]
        scan_limit = REV_STEPS + CALIBRATION_MAX_EXTRA_STEPS
        spacing_tolerance = MAGNET_SPACING_TOLERANCE_DEG * STEPS_PER_DEG
        last_error = None

        for attempt in range(1, CALIBRATION_RETRIES + 1):
            print(
                f"Motor {i+1}: starting one-revolution three-magnet calibration "
                f"(attempt {attempt}/{CALIBRATION_RETRIES})..."
            )

            try:
                # Begin from an inactive Hall state so all three magnets are
                # encountered as complete ON/OFF windows during this pass.
                if self._hall_active(i):
                    print(f"Motor {i+1}: clearing active Hall sensor...")
                    for _ in range(CALIBRATION_MAX_EXTRA_STEPS):
                        self._pulse({i: direction}, 1, CALIBRATION_FREQ_HZ)
                        if not self._hall_active(i):
                            break
                    else:
                        raise RuntimeError(
                            f"Motor {i+1}: Hall sensor would not clear."
                        )

                scan_pos = 0
                detections = []
                entry = None

                while scan_pos < scan_limit and len(detections) < 3:
                    self._pulse({i: direction}, 1, CALIBRATION_FREQ_HZ)
                    scan_pos += 1
                    active = self._hall_active(i)

                    if entry is None and active:
                        entry = scan_pos
                    elif entry is not None and not active:
                        center = (entry + scan_pos) / 2.0
                        detections.append(center)
                        print(
                            f"Motor {i+1}: magnet {len(detections)}/3 "
                            f"detected at scan step {center:.1f}."
                        )
                        entry = None

                if entry is not None:
                    raise RuntimeError(
                        f"Motor {i+1}: scan limit reached while the Hall sensor was active."
                    )
                if len(detections) != 3:
                    raise RuntimeError(
                        f"Motor {i+1}: found only {len(detections)} complete magnet "
                        "windows in one revolution; expected 3."
                    )

                # These are three distinct magnets encountered in sequence.
                # Convert to one-revolution coordinates and sort for cyclic gaps.
                centers = sorted(position % REV_STEPS for position in detections)
                gaps = [
                    centers[1] - centers[0],
                    centers[2] - centers[1],
                    REV_STEPS - (centers[2] - centers[0]),
                ]
                quarter = REV_STEPS / 4.0
                half = REV_STEPS / 2.0
                short_gaps = [
                    abs(gap - quarter) <= spacing_tolerance for gap in gaps
                ]
                long_gaps = [
                    abs(gap - half) <= spacing_tolerance for gap in gaps
                ]

                if sum(short_gaps) != 2 or sum(long_gaps) != 1:
                    raise RuntimeError(
                        f"Motor {i+1}: magnet spacing is not 90/90/180 degrees "
                        f"(measured {[round(g / STEPS_PER_DEG, 1) for g in gaps]} degrees)."
                    )

                # The center magnet is the point shared by the two ~90-degree gaps.
                if short_gaps[0] and short_gaps[1]:
                    center_index = 1
                elif short_gaps[1] and short_gaps[2]:
                    center_index = 2
                elif short_gaps[2] and short_gaps[0]:
                    center_index = 0
                else:
                    raise RuntimeError(
                        f"Motor {i+1}: could not identify the center magnet."
                    )

                center_scan_pos = centers[center_index]
                raw_delta = center_scan_pos - scan_pos
                delta_to_center = int(round(
                    (raw_delta + REV_STEPS / 2.0) % REV_STEPS
                    - REV_STEPS / 2.0
                ))

                print(
                    f"Motor {i+1}: identified center magnet at "
                    f"{center_scan_pos / STEPS_PER_DEG:.1f} degrees; "
                    f"cyclic gaps {[round(g / STEPS_PER_DEG, 1) for g in gaps]} degrees."
                )
                print(
                    f"Motor {i+1}: stopping scan after exactly three magnets; "
                    f"moving {delta_to_center:+d} steps to the center magnet."
                )

                return (
                    scan_pos,
                    int(round(center_scan_pos)),
                    delta_to_center,
                    direction,
                )

            except RuntimeError as exc:
                last_error = exc
                print(
                    f"Motor {i+1}: calibration attempt {attempt} failed: {exc}"
                )
                if attempt < CALIBRATION_RETRIES:
                    print(f"Motor {i+1}: retrying calibration...")

        raise RuntimeError(
            f"Motor {i+1}: calibration failed after "
            f"{CALIBRATION_RETRIES} attempts. Last error: {last_error}"
        ) from last_error

    def _fine_zero_at_center(self, i, direction):
        """Back away, re-approach the center magnet one microstep at a time."""
        # Move away far enough to guarantee a fresh ON edge.
        self._move_relative(
            [(-direction * FINE_BACKOFF_STEPS) if j == i else 0 for j in range(2)],
            HOME_FINE_FREQ_HZ, sample_each_step=True
        )
        cleared = False
        for _ in range(CALIBRATION_MAX_EXTRA_STEPS):
            if not self._hall_active(i):
                cleared = True
                break
            self._move_relative(
                [(-direction) if j == i else 0 for j in range(2)],
                HOME_FINE_FREQ_HZ, sample_each_step=True
            )
        if not cleared:
            raise RuntimeError(f"Motor {i+1}: could not clear center magnet before fine homing.")

        entry = None
        width = 0
        for _ in range(CALIBRATION_MAX_EXTRA_STEPS):
            self._move_relative(
                [(direction) if j == i else 0 for j in range(2)],
                HOME_FINE_FREQ_HZ, sample_each_step=True
            )
            if entry is None and self._hall_active(i):
                entry = 1
            elif entry is not None:
                width += 1
                if not self._hall_active(i):
                    if width < 2:
                        raise RuntimeError(f"Motor {i+1}: center magnet window too narrow.")
                    half = int(round(width / 2.0))
                    self._move_relative(
                        [(-direction * half) if j == i else 0 for j in range(2)],
                        HOME_FINE_FREQ_HZ, sample_each_step=True
                    )
                    offset = int(round(HOME_OFFSET_DEG[i] * STEPS_PER_DEG))
                    if offset:
                        self._move_relative(
                            [(1 if offset > 0 else -1) * abs(offset) if j == i else 0
                             for j in range(2)],
                            HOME_FINE_FREQ_HZ, sample_each_step=True
                        )
                    return
        raise RuntimeError(f"Motor {i+1}: fine zero pass could not find both Hall edges.")

    def calibrate_three_magnets(self):
        if not self.enabled:
            return
        with self.control_lock:
            self.homing = True
            try:
                while self.pi.wave_tx_busy():
                    time.sleep(0.001)
                scans = [self._find_three_centers(i) for i in range(2)]
                # Bring each motor back to its identified middle magnet.
                deltas = [scans[i][2] for i in range(2)]
                self._move_relative(deltas, CALIBRATION_FREQ_HZ, sample_each_step=True)
                for i in range(2):
                    if not self._hall_active(i):
                        raise RuntimeError(
                            f"Motor {i+1}: predicted center position does not activate its Hall sensor."
                        )
                # Fine zero is a separate, slow, edge-to-edge approach for each eye.
                for i in range(2):
                    self._fine_zero_at_center(i, scans[i][3])
                self.pos_steps = [0, 0]
                self.target_steps = [0, 0]
                self.limit_latched = [0, 0]
                print("Three-magnet calibration and separate fine 0-degree homing complete.")
            finally:
                self.homing = False

    def _fine_home_from_side(self, side):
        """
        Re-home from a known outer stop. Leave that outer magnet, seek the next
        magnet inward (which is the center), then fine-scan its edges.
        """
        with self.control_lock:
            self.homing = True
            try:
                for i in range(2):
                    inward = -1 if side > 0 else 1
                    # Clear the outer magnet first.
                    for _ in range(CALIBRATION_MAX_EXTRA_STEPS):
                        if not self._hall_active(i):
                            break
                        self._move_relative(
                            [inward if j == i else 0 for j in range(2)],
                            HOME_FINE_FREQ_HZ, sample_each_step=True
                        )
                    else:
                        raise RuntimeError(f"Motor {i+1}: could not leave outer magnet.")
                    # The next magnet in this known inward direction is the center.
                    found = False
                    for _ in range(int(120 * STEPS_PER_DEG)):
                        self._move_relative(
                            [inward if j == i else 0 for j in range(2)],
                            CALIBRATION_FREQ_HZ, sample_each_step=True
                        )
                        if self._hall_active(i):
                            found = True
                            break
                    if not found:
                        raise RuntimeError(f"Motor {i+1}: center magnet not found from {side:+d} stop.")
                    # Move through the center window, then back off and fine-approach.
                    for _ in range(CALIBRATION_MAX_EXTRA_STEPS):
                        if not self._hall_active(i):
                            break
                        self._move_relative(
                            [inward if j == i else 0 for j in range(2)],
                            HOME_FINE_FREQ_HZ, sample_each_step=True
                        )
                    if self._hall_active(i):
                        raise RuntimeError(f"Motor {i+1}: center magnet window did not clear.")
                    # We have just passed the center magnet in 'inward' direction.
                    # Backtrack and re-approach in the same direction.
                    self._move_relative(
                        [-inward * FINE_BACKOFF_STEPS if j == i else 0 for j in range(2)],
                        HOME_FINE_FREQ_HZ, sample_each_step=True
                    )
                    self._fine_zero_from_clear(i, inward)
                self.pos_steps = [0, 0]
                self.target_steps = [0, 0]
                self.limit_latched = [0, 0]
                print("Inactivity zero homing complete from the known outer side.")
            finally:
                self.homing = False

    def _fine_zero_from_clear(self, i, direction):
        entry = None
        width = 0
        for _ in range(CALIBRATION_MAX_EXTRA_STEPS):
            self._move_relative(
                [direction if j == i else 0 for j in range(2)],
                HOME_FINE_FREQ_HZ, sample_each_step=True
            )
            if entry is None and self._hall_active(i):
                entry = 1
            elif entry is not None:
                width += 1
                if not self._hall_active(i):
                    half = int(round(width / 2.0))
                    self._move_relative(
                        [-direction * half if j == i else 0 for j in range(2)],
                        HOME_FINE_FREQ_HZ, sample_each_step=True
                    )
                    return
        raise RuntimeError(f"Motor {i+1}: fine center homing failed.")

    def set_target_deg(self, deg):
        deg = max(-MAX_DEG, min(MAX_DEG, float(deg)))
        steps = int(round(deg * STEPS_PER_DEG))
        with self.lock:
            self.target_steps = [steps, steps]

    def park_at_limit(self, side):
        side = 1 if side >= 0 else -1
        self.park_side = [side, side]
        self.set_target_deg(side * MAX_DEG)

    def angle_deg(self):
        with self.lock:
            return sum(self.pos_steps) / (2.0 * STEPS_PER_DEG)

    def target_deg(self):
        with self.lock:
            return sum(self.target_steps) / (2.0 * STEPS_PER_DEG)

    def _stop_at_outer_magnet(self, i, direction):
        if direction > 0 and self.pos_steps[i] >= int(45 * STEPS_PER_DEG):
            return True
        if direction < 0 and self.pos_steps[i] <= -int(45 * STEPS_PER_DEG):
            return True
        return False

    def _run(self):
        freq = MIN_FREQ_HZ
        while self.running:
            if self.homing:
                time.sleep(0.002)
                continue
            with self.lock:
                remaining = [self.target_steps[i] - self.pos_steps[i] for i in range(2)]
                positions = self.pos_steps[:]
            active = {}
            dist = [abs(v) for v in remaining]
            if max(dist) <= STOP_STEPS:
                time.sleep(0.003)
                continue

            for i, rem in enumerate(remaining):
                if abs(rem) <= STOP_STEPS:
                    continue
                direction = 1 if rem > 0 else -1
                # A latched outer stop blocks only further movement outwards.
                hall_now = self._hall_active(i)
                if self.limit_latched[i] and direction != self.limit_latched[i] and not hall_now:
                    self.limit_latched[i] = 0
                if self.limit_latched[i] == direction:
                    continue
                if hall_now and self._stop_at_outer_magnet(i, direction):
                    self.limit_latched[i] = direction
                    with self.lock:
                        self.target_steps[i] = self.pos_steps[i]
                    print(f"Motor {i+1}: outer Hall stop engaged at {positions[i] / STEPS_PER_DEG:+.1f} deg.")
                    continue
                active[i] = direction

            if not active:
                time.sleep(0.002)
                continue

            # Near either outer magnet, use one-step bursts and inspect the Hall
            # input after every pulse. Else use small shared bursts.
            near_outer = any(abs(positions[i] / STEPS_PER_DEG) >= MAX_DEG - OUTER_APPROACH_DEG
                             for i in active)
            batch = 1 if near_outer else min(CHUNK_STEPS, min(abs(remaining[i]) for i in active))
            max_dist = max(abs(remaining[i]) for i in active)
            want = MIN_FREQ_HZ + (MAX_FREQ_HZ - MIN_FREQ_HZ) * min(
                1.0, max_dist / (SLOW_ZONE_DEG * STEPS_PER_DEG))
            freq = min(want, freq + ACCEL_HZ_PER_CHUNK)
            with self.control_lock:
                if not self.homing:
                    self._pulse(active, batch, int(freq))
                    with self.lock:
                        for i, direction in active.items():
                            self.pos_steps[i] += direction * batch
            # Hall stop is checked on the next loop before another outward pulse.
            time.sleep(0.0005)

    def home_from_park(self, side):
        self._fine_home_from_side(1 if side >= 0 else -1)

    def at_limit(self, side):
        side = 1 if side >= 0 else -1
        return all(value == side for value in self.limit_latched)

    def shutdown(self, go_home=False):
        if not self.enabled:
            return
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        self.pi.wave_tx_stop()
        self.pi.wave_clear()
        for pin in STEP_PINS:
            self.pi.write(pin, 0)
        for pin in EN_PINS:
            self.pi.write(pin, 1)
        self.pi.stop()


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
        main={"size": FRAME_SIZE, "format": "RGB888"},
        raw={"size": SENSOR_SIZE})
    picam2.configure(config)
    picam2.start()
    print("Sensor mode:", picam2.camera_configuration()["sensor"])

    w, h = FRAME_SIZE
    sx = w / DET_W
    detector = MotionDetector()
    print(f"Learning the background for {WARMUP_FRAMES} frames - keep the view clear of movement.")
    print(f"Three-magnet calibration complete. Tracking range is +/-{MAX_DEG:.0f} degrees.")

    hits = 0
    misses = 0
    last_cx = None
    have_lock = False
    smooth_err = 0.0
    last_seen = time.time()
    ever_locked = False
    last_tracking_side = 1
    parked_for_loss = False
    home_after_idle_done = False
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
                hits = 0
                last_cx = None
                have_lock = False
                smooth_err = 0.0

            locked = tgt is not None and hits >= CONFIRM_FRAMES
            if locked:
                error = (tgt.cx / DET_W - 0.5) * 2
                if have_lock:
                    smooth_err = SMOOTHING * smooth_err + (1 - SMOOTHING) * error
                else:
                    smooth_err = error
                have_lock = True
                last_seen = now
                ever_locked = True
                parked_for_loss = False
                home_after_idle_done = False

                angle = -smooth_err if LEFT_IS_CLOCKWISE else smooth_err
                desired_deg = max(-MAX_DEG, min(MAX_DEG, angle * TARGET_GAIN * MAX_DEG))
                if desired_deg > 2.0:
                    last_tracking_side = 1
                elif desired_deg < -2.0:
                    last_tracking_side = -1
                motor.set_target_deg(desired_deg)
                status = f"tracking target {motor.target_deg():+.1f} deg"
            else:
                if tgt is not None:
                    status = f"confirming {hits}/{CONFIRM_FRAMES}"
                elif res.status != "no motion":
                    status = res.status
                else:
                    status = "no motion"

                # After the target has genuinely disappeared, park at the outer
                # magnet on the side where it was last seen.
                if (ever_locked and misses >= LOSE_FRAMES and not parked_for_loss):
                    current = motor.angle_deg()
                    if abs(current) > 2.0 and last_tracking_side == 1 and current < 0:
                        last_tracking_side = -1
                    elif abs(current) > 2.0 and last_tracking_side == -1 and current > 0:
                        last_tracking_side = 1
                    print(
                        f"Target lost: parking at {'+90' if last_tracking_side > 0 else '-90'} "
                        "degree Hall stop."
                    )
                    motor.park_at_limit(last_tracking_side)
                    parked_for_loss = True
                    home_after_idle_done = False
                    status = "target lost - parking at outer magnet"

                # Ten seconds after the last confirmed target, return from the
                # known outer stop to the middle magnet and fine-home at zero.
                if (ever_locked and parked_for_loss and not home_after_idle_done
                        and now - last_seen >= RETURN_HOME_AFTER_S):
                    if motor.at_limit(last_tracking_side):
                        print("10 seconds without target: approaching center magnet for fine zero homing.")
                        try:
                            motor.home_from_park(last_tracking_side)
                            status = "idle - fine homed at zero"
                        except RuntimeError as exc:
                            status = "idle homing error"
                            print(f"ERROR during inactivity zero homing: {exc}")
                    else:
                        status = "idle - outer Hall stop not confirmed; zero homing skipped"
                        print(
                            "WARNING: not both outer Hall sensors confirmed at the park stop; "
                            "refusing to guess which magnet is center."
                        )
                    home_after_idle_done = True

            if now - t_print > 1.0:
                t_print = now
                angle_text = motor.angle_deg() if motor.enabled else 0.0
                print(f"{status} | motor avg {angle_text:+.1f} deg | {fps:.0f} fps | "
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
                cv2.putText(view, f"{status} | motor {motor.angle_deg() if motor.enabled else 0:+.1f}",
                            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(view, f"{fps:.0f} fps", (8, h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                if args.mask:
                    mw, mh = 160, 120
                    inset = cv2.cvtColor(
                        cv2.resize(res.mask, (mw, mh), interpolation=cv2.INTER_NEAREST),
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
        motor.shutdown(go_home=False)
        picam2.stop()
        if args.preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
