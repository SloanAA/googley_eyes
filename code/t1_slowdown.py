#!/usr/bin/env python3
"""
Motion tracker -> two TMC2208 STEP/DIR steppers (pigpio), POSITION control

Usage:
  python3 personTracker14.py
  python3 personTracker14.py --stream
  python3 personTracker14.py --stream --no-motor
  python3 personTracker14.py --stream --mask
  python3 personTracker14.py --preview

Motor behavior:
  * Both motors move together on shared STEP pulse edges.
  * Three-magnet Hall homing.
  * 0 magnet identified using 90/90/180 degree magnet spacing.
  * Travel limits measured during homing.
  * Position control with +/-90 degree mapping.
  * Hall-based outer travel stops.
  * Center magnet drift correction.
  * Edge-exit parking.
  * Motor movement decelerates over final 10%.
  * Homing approaches Hall sensors at reduced speed.
  * Motors run in their own thread.

Camera stabilization:
  * Lucas-Kanade optical flow tracks background features.
  * RANSAC estimates camera movement.
  * Small movements are compensated before MOG2.
  * Larger movements are rejected as camera shake.
  * Background learning freezes during shake.
  * Recovery frames do not count as target misses.

Requires:
  pigpiod running
  Picamera2
  OpenCV
  NumPy
  pigpio
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


# ============================================================
# MOTOR PINS
# ============================================================

STEP_PINS = [18, 21]
DIR_PINS = [14, 16]
EN_PINS = [15, 20]
HALL_PINS = [23, 24]

DIR_INVERT = [False, False]
DIR_SETUP_S = 0.001

HALL_ACTIVE_LEVEL = 0
HALL_PULL = pigpio.PUD_UP

STEP_MASK = sum(1 << p for p in STEP_PINS)


# ============================================================
# HOMING
# ============================================================

HOME_FREQ_HZ = 200
HOME_APPROACH_FREQ_HZ = 100

CALIBRATION_DIR = [1, 1]
CALIBRATION_RETRIES = 3
CALIBRATION_MAX_EXTRA_STEPS = 500

MAGNET_SPACING_TOLERANCE_DEG = 18.0

HOME_OFFSET_DEG = [0.0, 0.0]


# ============================================================
# MOTOR GEOMETRY
# ============================================================

STEPS_PER_REV = 200
MICROSTEPS = 8
GEAR_RATIO = 1.0

MAX_DEG = 90.0

STEPS_PER_DEG = (
    STEPS_PER_REV * MICROSTEPS * GEAR_RATIO / 360.0
)

REV_STEPS = int(round(
    STEPS_PER_REV * MICROSTEPS * GEAR_RATIO
))

MAX_STEPS = int(round(MAX_DEG * STEPS_PER_DEG))


# ============================================================
# CENTER DRIFT CORRECTION
# ============================================================

CENTER_TRACK_DEG = 20.0
CENTER_MAX_CORRECTION_DEG = 15.0
CENTER_BIAS_MAX_DEG = 6.0
CENTER_GAIN = 1.0

CENTER_TRACK_STEPS = int(round(
    CENTER_TRACK_DEG * STEPS_PER_DEG
))

CENTER_MAX_CORRECTION_STEPS = int(round(
    CENTER_MAX_CORRECTION_DEG * STEPS_PER_DEG
))

CENTER_BIAS_MAX_STEPS = int(round(
    CENTER_BIAS_MAX_DEG * STEPS_PER_DEG
))


# ============================================================
# MOTOR MOTION TUNING
# ============================================================

MIN_FREQ_HZ = 200
MAX_FREQ_HZ = 1200

ACCEL_HZ_PER_CHUNK = 150

SLOW_ZONE_DEG = 25.0

CHUNK_STEPS = 4

OUTER_APPROACH_DEG = 30.0
OUTER_FREQ_HZ = 400

START_DEG = 1.0
STOP_STEPS = 1

# Decelerate over the final 10% of each commanded movement.
DECEL_DISTANCE_FRAC = 0.10

# Minimum step frequency at the end of a movement.
END_FREQ_HZ = 200

# Slow down while crossing a Hall sensor during homing.
HOME_APPROACH_FREQ_HZ = 100

RETURN_HOME_ON_EXIT = False
RETURN_HOME_AFTER_S = 10.0


# ============================================================
# TRACKING MAPPING
# ============================================================

LEFT_IS_CLOCKWISE = True

TARGET_GAIN = 1.0
SMOOTHING = 0.4

EDGE_EXIT_FRAC = 0.6

CW = 1
CCW = 0


# ============================================================
# CAMERA
# ============================================================

FRAME_SIZE = (640, 480)
SENSOR_SIZE = (1640, 1232)


# ============================================================
# MOTION DETECTION
# ============================================================

DET_W = 320
DET_H = 240

BLUR_K = 7

BG_VAR_THRESHOLD = 25
BG_LEARNING_RATE = 0.003
WARMUP_LEARNING_RATE = 0.05
WARMUP_FRAMES = 40

OPEN_K = 3
MERGE_K = 25

MIN_MOTION_FRAC = 0.003
GLOBAL_MOTION_FRAC = 0.5

CONFIRM_FRAMES = 3
LOSE_FRAMES = 15

STICKY_MIN_FRAC = 0.5

IGNORE_ZONES = []


# ============================================================
# CAMERA SHAKE COMPENSATION SETTINGS
# ============================================================

SHAKE_TRANSLATION_PX = 1.8
SHAKE_ROTATION_DEG = 0.45
SHAKE_SCALE_CHANGE = 0.012

MAX_FEATURES = 250
FEATURE_QUALITY = 0.01
FEATURE_DISTANCE = 7

MIN_FEATURES = 12
MIN_INLIER_RATIO = 0.40

RANSAC_THRESHOLD = 2.5

UNRELIABLE_FLOW_PX = 4.0

RECOVERY_STABLE_FRAMES = 8


Blob = namedtuple(
    "Blob",
    "px cx cy x y w h"
)

Result = namedtuple(
    "Result",
    "target blobs status mask camera_shake"
)


# ============================================================
# POSITION MOTOR
# ============================================================

class PositionMotor:

    def __init__(self, enabled=True):

        self.enabled = enabled

        self.lock = threading.Lock()
        self.control_lock = threading.Lock()

        self.running = True
        self.homing = False

        self.pos_steps = [0, 0]
        self.target_steps = [0, 0]

        # Original distance of each commanded movement.
        self.move_total_steps = [0, 0]
        self.move_direction = [0, 0]

        self.freq = 0

        self.limit_latched = [0, 0]
        self.direction_levels = [None, None]

        self.limit_steps = [
            [MAX_STEPS, MAX_STEPS],
            [MAX_STEPS, MAX_STEPS]
        ]

        self._found_limits = [None, None]
        self._found_widths = [None, None]

        self.center_width = [None, None]
        self.center_bias = [None, None]
        self._center_pass = [None, None]

        self.thread = None

        if not enabled:
            return

        self.pi = pigpio.pi()

        if not self.pi.connected:
            raise SystemExit(
                "Could not connect to pigpiod - is it running?"
            )

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

        print("Starting startup homing (both motors together)...")

        try:
            self.home()

        except Exception:
            for pin in EN_PINS:
                self.pi.write(pin, 1)

            self.pi.stop()
            raise

        self.thread = threading.Thread(
            target=self._run,
            daemon=True
        )

        self.thread.start()

    def _hall_active(self, i):
        return self.pi.read(HALL_PINS[i]) == HALL_ACTIVE_LEVEL

    # ========================================================
    # STEP PULSES
    # ========================================================

    def _pulse(self, active_dirs, steps, frequency):

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

        half_us = max(
            2,
            int(500000 / max(1, frequency))
        )

        pulses = []

        for _ in range(steps):
            pulses.append(pigpio.pulse(mask, 0, half_us))
            pulses.append(pigpio.pulse(0, mask, half_us))

        self.pi.wave_clear()
        self.pi.wave_add_generic(pulses)

        wid = self.pi.wave_create()

        if wid < 0:
            raise RuntimeError(
                "pigpio could not create a step waveform"
            )

        self.pi.wave_send_once(wid)

        while self.pi.wave_tx_busy():
            time.sleep(0.0005)

        self.pi.wave_delete(wid)

    # ========================================================
    # HOMING MOVEMENT
    # ========================================================

    def _run_together(self, gens, frequency):

        live = dict(enumerate(gens))

        while live:

            active = {}
            decel_frequencies = []

            for i in list(live):

                try:
                    command = next(live[i])

                except StopIteration:
                    del live[i]
                    continue

                # Known-distance movement:
                # (direction, remaining_steps, total_steps)
                if isinstance(command, tuple):

                    direction, remaining, total = command

                    active[i] = direction

                    decel_steps = max(
                        1,
                        int(total * DECEL_DISTANCE_FRAC)
                    )

                    if remaining <= decel_steps:

                        fraction = max(
                            0.0,
                            min(1.0, remaining / decel_steps)
                        )

                        use_freq = (
                            END_FREQ_HZ
                            + (frequency - END_FREQ_HZ) * fraction
                        )

                    else:

                        use_freq = frequency

                    decel_frequencies.append(use_freq)

                else:

                    # Unknown-distance homing sweep.
                    active[i] = command

                    if self._hall_active(i):

                        decel_frequencies.append(
                            HOME_APPROACH_FREQ_HZ
                        )

                    else:

                        decel_frequencies.append(
                            frequency
                        )

            if active:

                use_frequency = max(
                    1,
                    int(min(decel_frequencies))
                )

                self._pulse(
                    active,
                    1,
                    use_frequency
                )

    # ========================================================
    # HOMING GENERATOR
    # ========================================================

    def _home_gen(self, i):

        d = CALIBRATION_DIR[i]

        quarter = REV_STEPS / 4.0
        half = REV_STEPS / 2.0

        tol = MAGNET_SPACING_TOLERANCE_DEG * STEPS_PER_DEG

        def near(a, b):
            return abs(a - b) <= tol

        def gaps_text(c):
            return [
                round(
                    (c[k + 1] - c[k]) / STEPS_PER_DEG,
                    1
                )
                for k in range(len(c) - 1)
            ]

        # Move off any Hall window we start inside.

        n = 0

        while self._hall_active(i):

            if n >= CALIBRATION_MAX_EXTRA_STEPS:
                raise RuntimeError(
                    f"Motor {i+1}: Hall sensor would not clear."
                )

            yield d
            n += 1

        pos = 0
        entry = None

        centers = []
        windows = []

        center_idx = None
        next_is_center = False
        target = None

        limit = 2 * REV_STEPS

        while target is None:

            if pos >= limit:
                raise RuntimeError(
                    f"Motor {i+1}: homing sweep ran too long."
                )

            yield d
            pos += 1

            active = self._hall_active(i)

            if entry is None and active:
                entry = pos

            elif entry is not None and not active:

                centers.append((entry + pos) / 2.0)
                windows.append((entry, pos))

                print(
                    f"Motor {i+1}: magnet window "
                    f"{len(centers)} at sweep step "
                    f"{centers[-1]:.1f} "
                    f"(width {pos-entry} steps)"
                )

                entry = None

                if next_is_center:

                    target = centers[-1]
                    center_idx = len(centers) - 1

                elif len(centers) == 2:

                    g1 = centers[1] - centers[0]

                    if near(g1, half):
                        next_is_center = True

                    elif not near(g1, quarter):
                        raise RuntimeError(
                            f"Motor {i+1}: invalid first magnet gap."
                        )

                elif len(centers) == 3:

                    g1 = centers[1] - centers[0]
                    g2 = centers[2] - centers[1]

                    if near(g1, quarter) and near(g2, quarter):

                        target = centers[1]
                        center_idx = 1

                    elif near(g1, quarter) and near(g2, half):

                        next_is_center = True

                    else:
                        raise RuntimeError(
                            f"Motor {i+1}: magnet spacing "
                            f"{gaps_text(centers)} deg invalid."
                        )

        print(
            f"Motor {i+1}: 0 magnet identified "
            f"(gaps {gaps_text(centers)} deg)."
        )

        # Measure outer magnet limits.

        cmid = centers[center_idx]

        plus_edges = []
        minus_edges = []

        for k, (en, ex) in enumerate(windows):

            if k == center_idx:
                continue

            mid = (en + ex) / 2.0

            rel_mid = (
                (mid - cmid + REV_STEPS / 2.0)
                % REV_STEPS
                - REV_STEPS / 2.0
            )

            if abs(rel_mid) < quarter / 2.0:
                continue

            a = ((en - mid) + rel_mid) * d
            b = ((ex - mid) + rel_mid) * d

            if rel_mid * d > 0:
                plus_edges.append(min(a, b))
            else:
                minus_edges.append(-max(a, b))

        minus_lim = (
            min(MAX_STEPS, int(min(minus_edges)))
            if minus_edges else MAX_STEPS
        )

        plus_lim = (
            min(MAX_STEPS, int(min(plus_edges)))
            if plus_edges else MAX_STEPS
        )

        self._found_limits[i] = [minus_lim, plus_lim]

        self._found_widths[i] = (
            windows[center_idx][1]
            - windows[center_idx][0]
        )

        print(
            f"Motor {i+1}: travel limits "
            f"-{minus_lim} / +{plus_lim} steps "
            f"(-{minus_lim / STEPS_PER_DEG:.1f} / "
            f"+{plus_lim / STEPS_PER_DEG:.1f} deg)."
        )

        # Move to the center of the 0 magnet.
        # This is a known-distance movement, so decelerate
        # over its final 10%.

        delta = int(round(target - pos))

        print(
            f"Motor {i+1}: moving {delta*d:+d} steps "
            "to the middle of the 0 magnet."
        )

        move_dir = d if delta > 0 else -d
        total = abs(delta)

        for remaining in range(total, 0, -1):
            yield (move_dir, remaining, total)

        # Apply home offset with the same deceleration.

        offset = int(round(
            HOME_OFFSET_DEG[i] * STEPS_PER_DEG
        ))

        if offset:

            offset_dir = 1 if offset > 0 else -1
            total_offset = abs(offset)

            for remaining in range(total_offset, 0, -1):
                yield (
                    offset_dir,
                    remaining,
                    total_offset
                )

    # ========================================================
    # HOME
    # ========================================================

    def home(self):

        if not self.enabled:
            return

        with self.control_lock:

            self.homing = True

            try:

                while self.pi.wave_tx_busy():
                    time.sleep(0.001)

                last_error = None

                for attempt in range(
                    1,
                    CALIBRATION_RETRIES + 1
                ):

                    try:

                        self._run_together(
                            [self._home_gen(i) for i in range(2)],
                            HOME_FREQ_HZ
                        )

                        break

                    except RuntimeError as exc:

                        last_error = exc

                        print(
                            f"Homing attempt {attempt}/"
                            f"{CALIBRATION_RETRIES} failed: {exc}"
                        )

                else:

                    raise RuntimeError(
                        f"Homing failed after "
                        f"{CALIBRATION_RETRIES} attempts. "
                        f"Last error: {last_error}"
                    ) from last_error

                self.limit_steps = [
                    list(self._found_limits[i])
                    for i in range(2)
                ]

                self.center_width = list(self._found_widths)

                self.center_bias = [None, None]
                self._center_pass = [None, None]

                with self.lock:
                    self.pos_steps = [0, 0]
                    self.target_steps = [0, 0]
                    self.move_total_steps = [0, 0]
                    self.move_direction = [0, 0]

                self.limit_latched = [0, 0]

                print("Homing complete: both motors at 0 deg.")

            finally:
                self.homing = False

    # ========================================================
    # TRACKING TARGETS
    # ========================================================

    def set_target_deg(self, deg):

        deg = max(-MAX_DEG, min(MAX_DEG, float(deg)))

        steps = int(round(deg * STEPS_PER_DEG))

        with self.lock:

            new_targets = [
                max(
                    -self.limit_steps[i][0],
                    min(self.limit_steps[i][1], steps)
                )
                for i in range(2)
            ]

            for i in range(2):

                current_pos = self.pos_steps[i]

                old_remaining = (
                    self.target_steps[i] - current_pos
                )

                new_remaining = (
                    new_targets[i] - current_pos
                )

                new_direction = (
                    1 if new_remaining > 0
                    else -1 if new_remaining < 0
                    else 0
                )

                # Start a new movement profile when direction
                # changes or the previous movement is complete.

                if (
                    new_direction != self.move_direction[i]
                    or abs(old_remaining) <= STOP_STEPS
                ):

                    self.move_direction[i] = new_direction

                    self.move_total_steps[i] = abs(
                        new_remaining
                    )

                else:

                    # Preserve the original movement distance
                    # if tracking continuously updates the target.

                    self.move_total_steps[i] = max(
                        self.move_total_steps[i],
                        abs(new_remaining)
                    )

            self.target_steps = new_targets

    def park_at_limit(self, side):

        side = 1 if side >= 0 else -1

        self.set_target_deg(side * MAX_DEG)

    def angle_deg(self):

        with self.lock:
            return sum(self.pos_steps) / (
                2.0 * STEPS_PER_DEG
            )

    def target_deg(self):

        with self.lock:
            return sum(self.target_steps) / (
                2.0 * STEPS_PER_DEG
            )

    # ========================================================
    # HALL LIMITS
    # ========================================================

    def _stop_at_outer_magnet(self, i, direction):

        lim = (
            self.limit_steps[i][1]
            if direction > 0
            else self.limit_steps[i][0]
        )

        return self.pos_steps[i] * direction >= 0.5 * lim

    # ========================================================
    # CENTER DRIFT CORRECTION
    # ========================================================

    def _center_update(self, i, direction):

        pos = self.pos_steps[i]

        if abs(pos) > CENTER_TRACK_STEPS:
            self._center_pass[i] = None
            return

        active = self._hall_active(i)
        st = self._center_pass[i]

        if st is None or st["dir"] != direction:

            self._center_pass[i] = {
                "dir": direction,
                "entry": None,
                "ok": not active,
                "prev": active
            }

            return

        if active and not st["prev"]:

            if st["ok"]:
                st["entry"] = pos

        elif st["prev"] and not active:

            if st["ok"] and st["entry"] is not None:

                self._center_crossed(
                    i,
                    direction,
                    st["entry"],
                    pos
                )

            st["entry"] = None
            st["ok"] = True

        st["prev"] = active

    def _center_crossed(self, i, direction, entry, exit_pos):

        width = abs(exit_pos - entry)
        w0 = self.center_width[i]

        if w0 and not (0.5 * w0 <= width <= 2.0 * w0):

            print(
                f"Motor {i+1}: ignored a 0-magnet pass "
                f"(width {width} steps, expected ~{w0})."
            )

            return

        mid = (entry + exit_pos) / 2.0

        if direction == CALIBRATION_DIR[i]:

            ref = 0.0

        else:

            if self.center_bias[i] is None:

                if abs(mid) <= CENTER_BIAS_MAX_STEPS:

                    self.center_bias[i] = mid

                    print(
                        f"Motor {i+1}: learned reverse-direction "
                        f"0 offset {mid:+.1f} steps."
                    )

                return

            ref = self.center_bias[i]

        shift = mid - ref

        if abs(shift) > CENTER_MAX_CORRECTION_STEPS:

            print(
                f"Motor {i+1}: ignored a {shift:+.0f} step "
                "re-zero."
            )

            return

        shift = int(round(shift * CENTER_GAIN))

        if shift:

            with self.lock:
                self.pos_steps[i] -= shift

            if abs(shift) >= 2:

                print(
                    f"Motor {i+1}: crossed the 0 magnet, "
                    f"re-zeroed by {-shift:+d} steps."
                )

    # ========================================================
    # NORMAL MOTOR THREAD
    # ========================================================

    def _run(self):

        freq = MIN_FREQ_HZ

        while self.running:

            if self.homing:
                time.sleep(0.002)
                continue

            with self.lock:

                remaining = [
                    self.target_steps[i] - self.pos_steps[i]
                    for i in range(2)
                ]

                positions = self.pos_steps[:]
                move_totals = self.move_total_steps[:]

            dist = [abs(v) for v in remaining]

            if max(dist) <= STOP_STEPS:

                with self.lock:

                    for i in range(2):

                        if abs(remaining[i]) <= STOP_STEPS:

                            self.move_total_steps[i] = 0
                            self.move_direction[i] = 0

                time.sleep(0.003)
                continue

            active = {}

            for i, rem in enumerate(remaining):

                if abs(rem) <= STOP_STEPS:
                    continue

                direction = 1 if rem > 0 else -1

                lim = (
                    self.limit_steps[i][1]
                    if direction > 0
                    else self.limit_steps[i][0]
                )

                if positions[i] * direction >= lim:

                    with self.lock:
                        self.target_steps[i] = positions[i]

                    continue

                hall_now = self._hall_active(i)

                if (
                    self.limit_latched[i]
                    and direction != self.limit_latched[i]
                    and not hall_now
                ):

                    self.limit_latched[i] = 0

                if self.limit_latched[i] == direction:
                    continue

                if hall_now and self._stop_at_outer_magnet(
                    i, direction
                ):

                    self.limit_latched[i] = direction

                    with self.lock:
                        self.target_steps[i] = self.pos_steps[i]

                    print(
                        f"Motor {i+1}: outer Hall stop engaged "
                        f"at {positions[i] / STEPS_PER_DEG:+.1f} deg."
                    )

                    continue

                active[i] = direction

            if not active:
                time.sleep(0.002)
                continue

            near_outer = any(
                abs(positions[i] / STEPS_PER_DEG)
                >= MAX_DEG - OUTER_APPROACH_DEG
                for i in active
            )

            near_center = any(
                abs(positions[i]) <= CENTER_TRACK_STEPS + 8
                for i in active
            )

            # Normal acceleration profile.

            max_dist = max(
                abs(remaining[i]) for i in active
            )

            want = MIN_FREQ_HZ + (
                MAX_FREQ_HZ - MIN_FREQ_HZ
            ) * min(
                1.0,
                max_dist / (SLOW_ZONE_DEG * STEPS_PER_DEG)
            )

            freq = min(
                want,
                freq + ACCEL_HZ_PER_CHUNK
            )

            # Apply final 10% deceleration.

            decel_frequencies = []
            decelerating = False

            for i in active:

                total = move_totals[i]
                distance = abs(remaining[i])

                decel_steps = max(
                    1,
                    int(total * DECEL_DISTANCE_FRAC)
                )

                if total > 0 and distance <= decel_steps:

                    fraction = max(
                        0.0,
                        min(1.0, distance / decel_steps)
                    )

                    decel_freq = (
                        END_FREQ_HZ
                        + (freq - END_FREQ_HZ) * fraction
                    )

                    decel_frequencies.append(decel_freq)
                    decelerating = True

                else:

                    decel_frequencies.append(freq)

            use_freq = min(decel_frequencies)

            # Use individual steps near the destination.

            if near_outer or near_center or decelerating:

                batch = 1

            else:

                batch = min(
                    CHUNK_STEPS,
                    min(abs(remaining[i]) for i in active)
                )

            with self.control_lock:

                if not self.homing:

                    if near_outer:

                        use_freq = min(
                            use_freq,
                            OUTER_FREQ_HZ
                        )

                    self._pulse(
                        active,
                        batch,
                        max(END_FREQ_HZ, int(use_freq))
                    )

                    with self.lock:

                        for i, direction in active.items():
                            self.pos_steps[i] += direction * batch

                    for i, direction in active.items():
                        self._center_update(i, direction)

            time.sleep(0.0005)

    # ========================================================
    # SHUTDOWN
    # ========================================================

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


# ============================================================
# MOTION DETECTOR WITH CAMERA STABILIZATION
# ============================================================

class MotionDetector:

    def __init__(self):

        self.bg = self._new_background()

        self.open_k = np.ones(
            (OPEN_K, OPEN_K),
            np.uint8
        )

        self.merge_k = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (MERGE_K, MERGE_K)
        )

        self.frames = 0
        self.min_px = MIN_MOTION_FRAC * DET_W * DET_H

        self.prev_gray = None

        self.recovering = False
        self.relearning = False
        self.stable_frames = 0

    def _new_background(self):

        return cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=BG_VAR_THRESHOLD,
            detectShadows=False
        )

    # ========================================================
    # CAMERA MOVEMENT ESTIMATION
    # ========================================================

    def _camera_compensate(self, gray):

        if self.prev_gray is None:

            self.prev_gray = gray.copy()

            return gray, False, "initializing"

        p0 = cv2.goodFeaturesToTrack(
            self.prev_gray,
            maxCorners=MAX_FEATURES,
            qualityLevel=FEATURE_QUALITY,
            minDistance=FEATURE_DISTANCE,
            blockSize=7
        )

        if p0 is None or len(p0) < MIN_FEATURES:

            self.prev_gray = gray.copy()

            return gray, False, "insufficient features"

        p1, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            p0,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS
                | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01
            )
        )

        if p1 is None or status is None:

            self.prev_gray = gray.copy()

            return gray, False, "flow unavailable"

        good0 = p0[status.ravel() == 1].reshape(-1, 2)
        good1 = p1[status.ravel() == 1].reshape(-1, 2)

        if len(good0) < MIN_FEATURES:

            self.prev_gray = gray.copy()

            return gray, False, "insufficient tracked features"

        flow = good1 - good0

        median_flow = float(
            np.median(np.linalg.norm(flow, axis=1))
        )

        matrix, inliers = cv2.estimateAffinePartial2D(
            good0,
            good1,
            method=cv2.RANSAC,
            ransacReprojThreshold=RANSAC_THRESHOLD,
            maxIters=2000,
            confidence=0.99,
            refineIters=10
        )

        if matrix is None or inliers is None:

            self.prev_gray = gray.copy()

            shaking = (
                median_flow >= UNRELIABLE_FLOW_PX
            )

            return (
                gray,
                shaking,
                "unreliable camera movement"
            )

        inlier_count = int(np.count_nonzero(inliers))
        inlier_ratio = float(np.mean(inliers))

        if inlier_count < MIN_FEATURES:

            self.prev_gray = gray.copy()

            shaking = (
                median_flow >= UNRELIABLE_FLOW_PX
            )

            return (
                gray,
                shaking,
                "unreliable optical flow"
            )

        tx = float(matrix[0, 2])
        ty = float(matrix[1, 2])

        translation = float(np.hypot(tx, ty))

        rotation = float(np.degrees(
            np.arctan2(
                matrix[1, 0],
                matrix[0, 0]
            )
        ))

        scale = float(np.sqrt(
            matrix[0, 0] ** 2
            + matrix[1, 0] ** 2
        ))

        reliable = (
            inlier_ratio >= MIN_INLIER_RATIO
        )

        shaking = False

        if reliable:

            shaking = (
                translation >= SHAKE_TRANSLATION_PX
                or abs(rotation) >= SHAKE_ROTATION_DEG
                or abs(scale - 1.0) >= SHAKE_SCALE_CHANGE
            )

        elif median_flow >= UNRELIABLE_FLOW_PX:

            shaking = True

        description = (
            f"move {translation:.1f}px "
            f"rotate {rotation:+.2f}deg "
            f"scale {scale:.3f}"
        )

        if shaking:

            self.prev_gray = gray.copy()

            return gray, True, description

        inverse = cv2.invertAffineTransform(matrix)

        stabilized = cv2.warpAffine(
            gray,
            inverse,
            (DET_W, DET_H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        self.prev_gray = stabilized.copy()

        return stabilized, False, description

    # ========================================================
    # MOTION PROCESSING
    # ========================================================

    def process(self, frame, last_cx=None):

        small = cv2.resize(
            frame,
            (DET_W, DET_H),
            interpolation=cv2.INTER_AREA
        )

        gray = cv2.cvtColor(
            small,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.GaussianBlur(
            gray,
            (BLUR_K, BLUR_K),
            0
        )

        stable_gray, shaking, movement = (
            self._camera_compensate(gray)
        )

        empty_mask = np.zeros(
            (DET_H, DET_W),
            dtype=np.uint8
        )

        if shaking:

            self.recovering = True
            self.relearning = False
            self.stable_frames = 0

            return Result(
                None,
                [],
                f"camera shake - ignoring ({movement})",
                empty_mask,
                True
            )

        if self.recovering:

            self.stable_frames += 1

            if self.stable_frames < RECOVERY_STABLE_FRAMES:

                return Result(
                    None,
                    [],
                    f"camera recovering "
                    f"{self.stable_frames}/"
                    f"{RECOVERY_STABLE_FRAMES}",
                    empty_mask,
                    True
                )

            print(
                "Camera stable - relearning background."
            )

            self.bg = self._new_background()

            self.frames = 0

            self.recovering = False
            self.relearning = True
            self.stable_frames = 0

        lr = (
            WARMUP_LEARNING_RATE
            if self.frames < WARMUP_FRAMES
            else BG_LEARNING_RATE
        )

        mask = self.bg.apply(
            stable_gray,
            learningRate=lr
        )

        self.frames += 1

        if self.frames <= WARMUP_FRAMES:

            return Result(
                None,
                [],
                f"learning background "
                f"{self.frames}/{WARMUP_FRAMES}",
                mask,
                self.relearning
            )

        self.relearning = False

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            self.open_k
        )

        for x0, y0, x1, y1 in IGNORE_ZONES:

            mask[
                int(y0 * DET_H):int(y1 * DET_H),
                int(x0 * DET_W):int(x1 * DET_W)
            ] = 0

        if cv2.countNonZero(mask) > (
            GLOBAL_MOTION_FRAC * DET_W * DET_H
        ):

            return Result(
                None,
                [],
                "global change - ignoring frame",
                mask,
                False
            )

        merged = cv2.dilate(
            mask,
            self.merge_k
        )

        contours, _ = cv2.findContours(
            merged,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        blobs = []

        for c in contours:

            x, y, w, h = cv2.boundingRect(c)

            m = cv2.moments(
                mask[y:y + h, x:x + w],
                binaryImage=True
            )

            px = m["m00"]

            if px < self.min_px:
                continue

            blobs.append(
                Blob(
                    px,
                    x + m["m10"] / px,
                    y + m["m01"] / px,
                    x,
                    y,
                    w,
                    h
                )
            )

        if not blobs:

            return Result(
                None,
                [],
                "no motion",
                mask,
                False
            )

        biggest = max(
            blobs,
            key=lambda b: b.px
        )

        chosen = biggest

        if last_cx is not None:

            near = [
                b for b in blobs
                if b.px >= STICKY_MIN_FRAC * biggest.px
            ]

            chosen = min(
                near,
                key=lambda b: abs(b.cx - last_cx)
            )

        return Result(
            chosen,
            blobs,
            "motion",
            mask,
            False
        )


# ============================================================
# BROWSER STREAMING
# ============================================================

class Streamer:

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

                    self.send_header(
                        "Content-Type",
                        "multipart/x-mixed-replace; boundary=frame"
                    )

                    self.end_headers()

                    try:

                        while True:

                            with streamer.cond:

                                streamer.cond.wait()

                                data = streamer.jpeg

                            if data is None:
                                continue

                            self.wfile.write(
                                b"--frame\r\n"
                                b"Content-Type: image/jpeg\r\n"
                                b"Content-Length: "
                                + str(len(data)).encode()
                                + b"\r\n\r\n"
                                + data
                                + b"\r\n"
                            )

                    except (
                        BrokenPipeError,
                        ConnectionResetError
                    ):
                        pass

                else:

                    page = (
                        b'<html><body style="margin:0;background:#000">'
                        b'<img src="/stream" '
                        b'style="width:100%;max-width:960px">'
                        b'</body></html>'
                    )

                    self.send_response(200)

                    self.send_header(
                        "Content-Type",
                        "text/html"
                    )

                    self.send_header(
                        "Content-Length",
                        str(len(page))
                    )

                    self.end_headers()
                    self.wfile.write(page)

        self.server = ThreadingHTTPServer(
            ("0.0.0.0", port),
            Handler
        )

        threading.Thread(
            target=self.server.serve_forever,
            daemon=True
        ).start()

    def update(self, frame):

        ok, buf = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 70]
        )

        if ok:

            with self.cond:

                self.jpeg = buf.tobytes()
                self.cond.notify_all()


# ============================================================
# MAIN
# ============================================================

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--preview",
        action="store_true",
        help="show local camera window"
    )

    ap.add_argument(
        "--stream",
        action="store_true",
        help="serve preview in a browser"
    )

    ap.add_argument(
        "--mask",
        action="store_true",
        help="show motion mask in preview"
    )

    ap.add_argument(
        "--port",
        type=int,
        default=8080
    )

    ap.add_argument(
        "--no-motor",
        action="store_true",
        help="detect only, never touch motors"
    )

    args = ap.parse_args()

    motor = PositionMotor(
        enabled=not args.no_motor
    )

    streamer = (
        Streamer(args.port)
        if args.stream
        else None
    )

    if streamer:

        print(
            f"Streaming at http://<pi-address>:{args.port} "
            "(find address with: hostname -I)"
        )

    picam2 = Picamera2()

    config = picam2.create_preview_configuration(
        main={
            "size": FRAME_SIZE,
            "format": "RGB888"
        },
        raw={
            "size": SENSOR_SIZE
        }
    )

    picam2.configure(config)
    picam2.start()

    print(
        "Sensor mode:",
        picam2.camera_configuration()["sensor"]
    )

    w, h = FRAME_SIZE

    sx = w / DET_W

    detector = MotionDetector()

    print(
        f"Learning the background for {WARMUP_FRAMES} "
        "frames - keep the view clear of movement."
    )

    print(
        f"Homing complete. Tracking range is "
        f"+/-{MAX_DEG:.0f} degrees."
    )

    print("Camera shake compensation enabled.")
    print("Motor deceleration enabled over final 10%.")

    hits = 0
    misses = 0

    last_cx = None
    have_lock = False

    smooth_err = 0.0

    last_seen = time.time()

    ever_locked = False

    last_desired_deg = 0.0

    loss_handled = False
    home_done = False

    status = "starting"

    fps = 0.0

    t_prev = time.time()
    t_print = 0.0

    try:

        while True:

            frame = picam2.capture_array()

            now = time.time()

            fps = (
                0.9 * fps
                + 0.1 * (
                    1.0 / max(now - t_prev, 1e-3)
                )
            )

            t_prev = now

            res = detector.process(
                frame,
                last_cx
            )

            tgt = res.target

            # =================================================
            # CAMERA SHAKE / RECOVERY
            # =================================================

            if res.camera_shake:

                status = res.status

            else:

                # =============================================
                # NORMAL TARGET DETECTION
                # =============================================

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

                locked = (
                    tgt is not None
                    and hits >= CONFIRM_FRAMES
                )

                if locked:

                    error = (
                        (tgt.cx / DET_W - 0.5) * 2
                    )

                    if have_lock:

                        smooth_err = (
                            SMOOTHING * smooth_err
                            + (1 - SMOOTHING) * error
                        )

                    else:

                        smooth_err = error

                    have_lock = True

                    last_seen = now
                    ever_locked = True

                    loss_handled = False
                    home_done = False

                    angle = (
                        -smooth_err
                        if LEFT_IS_CLOCKWISE
                        else smooth_err
                    )

                    desired_deg = max(
                        -MAX_DEG,
                        min(
                            MAX_DEG,
                            angle * TARGET_GAIN * MAX_DEG
                        )
                    )

                    last_desired_deg = desired_deg

                    motor.set_target_deg(
                        desired_deg
                    )

                    status = (
                        f"tracking target "
                        f"{motor.target_deg():+.1f} deg"
                    )

                else:

                    if tgt is not None:

                        status = (
                            f"confirming "
                            f"{hits}/{CONFIRM_FRAMES}"
                        )

                    elif res.status != "no motion":

                        status = res.status

                    else:

                        status = "no motion"

                    # =========================================
                    # TARGET LOSS / EDGE PARKING
                    # =========================================

                    if (
                        ever_locked
                        and misses >= LOSE_FRAMES
                        and not loss_handled
                    ):

                        if (
                            abs(last_desired_deg)
                            >= EDGE_EXIT_FRAC * MAX_DEG
                        ):

                            side = (
                                1
                                if last_desired_deg > 0
                                else -1
                            )

                            print(
                                f"Target left the frame: "
                                f"going to the "
                                f"{side * 90:+d} deg Hall stop."
                            )

                            motor.park_at_limit(side)

                            status = (
                                f"target left - parking at "
                                f"{side * 90:+d}"
                            )

                        else:

                            print(
                                "Target lost mid-frame: "
                                "holding position."
                            )

                            status = (
                                "target lost - holding"
                            )

                        loss_handled = True

                    # =========================================
                    # AUTOMATIC HOMING
                    # =========================================

                    if (
                        ever_locked
                        and loss_handled
                        and not home_done
                        and now - last_seen >= RETURN_HOME_AFTER_S
                    ):

                        print(
                            f"{RETURN_HOME_AFTER_S:.0f} seconds "
                            "without target: homing."
                        )

                        try:

                            motor.home()

                            status = "idle - homed at 0"

                        except RuntimeError as exc:

                            status = "idle homing error"

                            print(
                                f"ERROR during homing: {exc}"
                            )

                        last_desired_deg = 0.0

                        home_done = True

            # =================================================
            # STATUS OUTPUT
            # =================================================

            if now - t_print > 1.0:

                t_print = now

                angle_text = (
                    motor.angle_deg()
                    if motor.enabled
                    else 0.0
                )

                print(
                    f"{status} | "
                    f"motor avg {angle_text:+.1f} deg | "
                    f"{fps:.0f} fps | "
                    f"{len(res.blobs)} blob(s)"
                )

            # =================================================
            # PREVIEW / STREAM
            # =================================================

            if args.preview or streamer:

                view = frame.copy()

                cv2.line(
                    view,
                    (w // 2, 0),
                    (w // 2, h),
                    (255, 255, 0),
                    1
                )

                for b in res.blobs:

                    color = (160, 160, 160)
                    thick = 1

                    if tgt is not None and b is tgt:

                        color = (
                            (0, 255, 0)
                            if hits >= CONFIRM_FRAMES
                            else (0, 255, 255)
                        )

                        thick = 2

                    cv2.rectangle(
                        view,
                        (
                            int(b.x * sx),
                            int(b.y * sx)
                        ),
                        (
                            int((b.x + b.w) * sx),
                            int((b.y + b.h) * sx)
                        ),
                        color,
                        thick
                    )

                    if thick == 2:

                        cv2.circle(
                            view,
                            (
                                int(b.cx * sx),
                                int(b.cy * sx)
                            ),
                            5,
                            color,
                            -1
                        )

                cv2.putText(
                    view,
                    (
                        f"{status} | motor "
                        f"{motor.angle_deg() if motor.enabled else 0:+.1f}"
                    ),
                    (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    view,
                    f"{fps:.0f} fps",
                    (8, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2
                )

                if res.camera_shake:

                    cv2.putText(
                        view,
                        "CAMERA SHAKE / RECOVERY",
                        (8, 48),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 255),
                        2
                    )

                if args.mask:

                    mw, mh = 160, 120

                    inset = cv2.cvtColor(
                        cv2.resize(
                            res.mask,
                            (mw, mh),
                            interpolation=cv2.INTER_NEAREST
                        ),
                        cv2.COLOR_GRAY2BGR
                    )

                    view[0:mh, w-mw:w] = inset

                    cv2.rectangle(
                        view,
                        (w-mw, 0),
                        (w-1, mh-1),
                        (255, 255, 255),
                        1
                    )

                if streamer:
                    streamer.update(view)

                if args.preview:

                    cv2.imshow(
                        "googley eyes",
                        view
                    )

                    if (
                        cv2.waitKey(1) & 0xFF
                        == ord("q")
                    ):
                        break

    except KeyboardInterrupt:
        pass

    finally:

        motor.shutdown(
            go_home=False
        )

        picam2.stop()

        if args.preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
