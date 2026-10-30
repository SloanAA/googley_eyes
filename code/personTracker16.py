
#!/usr/bin/env python3
"""
Motion tracker -> two TMC2208 STEP/DIR steppers (pigpio), POSITION control

Usage:
  python3 personTracker14.py
  python3 personTracker14.py --stream
  python3 personTracker14.py --stream --no-motor
  python3 personTracker14.py --stream --mask
  python3 personTracker14.py --preview

HOMING:
  1. Both motors sweep together in CALIBRATION_DIR.
  2. Identify the center magnet using the -90 / 0 / +90 spacing.
  3. Measure the coarse center-magnet window.
  4. Move beyond the magnet and reverse to a point before its first edge.
  5. Fine-scan the center magnet one microstep at a time.
  6. Record Hall activation and deactivation positions.
  7. Calculate the midpoint of the measured window.
  8. Return to the midpoint and declare it zero.

TRACKING:
  * Both motors use synchronized STEP pulse edges.
  * Position is mapped to +/-90 degrees.
  * Actual travel limits are measured during homing.
  * Outer Hall magnets provide additional travel protection.
  * Center magnet crossings provide drift correction.
  * Target disappearance at an edge parks at that side's outer magnet.
  * Target disappearance in the middle holds position.
  * Motors operate in their own thread.

Requires:
  sudo systemctl start pigpiod
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


# ==============================================================
# MOTOR PINS
# ==============================================================

STEP_PINS = [18, 21]
DIR_PINS = [14, 16]
EN_PINS = [15, 20]
HALL_PINS = [23, 24]

DIR_INVERT = [False, False]
DIR_SETUP_S = 0.001

HALL_ACTIVE_LEVEL = 0
HALL_PULL = pigpio.PUD_UP

CW = 1
CCW = 0


# ==============================================================
# HOMING CONFIGURATION
# ==============================================================

HOME_FREQ_HZ = 200
FINE_HOME_FREQ_HZ = 80

CALIBRATION_DIR = [1, 1]

CALIBRATION_RETRIES = 3
CALIBRATION_MAX_EXTRA_STEPS = 500

MAGNET_SPACING_TOLERANCE_DEG = 18.0

HOME_OFFSET_DEG = [0.0, 0.0]

FINE_HOME_EXTRA_STEPS = 10
FINE_HOME_MAX_STEPS = 1000

STEP_MASK = sum(1 << p for p in STEP_PINS)


# ==============================================================
# MOTOR GEOMETRY
# ==============================================================

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


# ==============================================================
# CENTER MAGNET DRIFT CORRECTION
# ==============================================================

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


# ==============================================================
# MOTOR MOVEMENT TUNING
# ==============================================================

MIN_FREQ_HZ = 200
MAX_FREQ_HZ = 1200

ACCEL_HZ_PER_CHUNK = 150

SLOW_ZONE_DEG = 25.0

CHUNK_STEPS = 4

OUTER_APPROACH_DEG = 30.0
OUTER_FREQ_HZ = 400

STOP_STEPS = 1

RETURN_HOME_ON_EXIT = False
RETURN_HOME_AFTER_S = 10.0


# ==============================================================
# TRACKING MAPPING
# ==============================================================

LEFT_IS_CLOCKWISE = True

TARGET_GAIN = 1.0
SMOOTHING = 0.4

EDGE_EXIT_FRAC = 0.6


# ==============================================================
# CAMERA
# ==============================================================

FRAME_SIZE = (640, 480)
SENSOR_SIZE = (1640, 1232)


# ==============================================================
# MOTION DETECTION
# ==============================================================

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


Blob = namedtuple("Blob", "px cx cy x y w h")
Result = namedtuple("Result", "target blobs status mask")


# ==============================================================
# POSITION MOTOR
# ==============================================================

class PositionMotor:

    def __init__(self, enabled=True):

        self.enabled = enabled

        self.lock = threading.Lock()
        self.control_lock = threading.Lock()

        self.running = True
        self.homing = False

        self.pos_steps = [0, 0]
        self.target_steps = [0, 0]

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

        print("\nStarting startup homing...")

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

    # ----------------------------------------------------------
    # HALL SENSOR
    # ----------------------------------------------------------

    def _hall_active(self, i):
        return self.pi.read(HALL_PINS[i]) == HALL_ACTIVE_LEVEL

    # ----------------------------------------------------------
    # SYNCHRONIZED MOTOR PULSE
    # ----------------------------------------------------------

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

            pulses.append(
                pigpio.pulse(mask, 0, half_us)
            )

            pulses.append(
                pigpio.pulse(0, mask, half_us)
            )

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

    # ==========================================================
    # HOMING
    # ==========================================================

    def _run_together(self, gens, frequency):

        """
        Run both motor generators together.

        Normal yields:
            +1 or -1

        Fine homing yields:
            (direction, True)

        If either motor is in its fine scan, both motors use
        the slower fine homing frequency for that pulse.

        All active motors still share exactly the same STEP edges.
        """

        live = dict(enumerate(gens))

        while live:

            active = {}
            fine_mode = False

            for i in list(live):

                try:

                    command = next(live[i])

                    if isinstance(command, tuple):

                        direction, fine = command

                        if fine:
                            fine_mode = True

                    else:
                        direction = command

                    active[i] = direction

                except StopIteration:

                    del live[i]

            if active:

                use_freq = (
                    FINE_HOME_FREQ_HZ
                    if fine_mode
                    else frequency
                )

                self._pulse(active, 1, use_freq)

    # ----------------------------------------------------------
    # IDENTIFY MAGNETS AND FINE-HOME
    # ----------------------------------------------------------

    def _home_gen(self, i):

        """
        Position-independent homing.

        Coarse stage:
            Identify the center magnet using 90/90/180 spacing.

        Fine stage:
            Clear the center magnet.
            Reverse beyond its first edge.
            Approach again one step at a time.
            Measure both Hall boundaries.
            Calculate the midpoint.
            Return to the midpoint.

        The generator yields one motor step at a time.
        """

        d = CALIBRATION_DIR[i]

        quarter = REV_STEPS / 4.0
        half = REV_STEPS / 2.0

        tol = (
            MAGNET_SPACING_TOLERANCE_DEG
            * STEPS_PER_DEG
        )

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

        # ------------------------------------------------------
        # 1. Clear any magnet we start inside.
        # ------------------------------------------------------

        print(f"\nMotor {i+1}: clearing initial Hall window.")

        n = 0

        while self._hall_active(i):

            if n >= CALIBRATION_MAX_EXTRA_STEPS:

                raise RuntimeError(
                    f"Motor {i+1}: Hall sensor would not clear."
                )

            yield d

            n += 1

        # ------------------------------------------------------
        # 2. Coarse magnet sweep.
        # ------------------------------------------------------

        print(
            f"Motor {i+1}: beginning coarse magnet identification."
        )

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

                centers.append(
                    (entry + pos) / 2.0
                )

                windows.append(
                    (entry, pos)
                )

                width = pos - entry

                print(
                    f"Motor {i+1}: coarse magnet "
                    f"{len(centers)} at {centers[-1]:.1f}, "
                    f"width {width} steps."
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
                            f"Motor {i+1}: first magnet gap "
                            f"{g1 / STEPS_PER_DEG:.1f} degrees "
                            "is not approximately 90 or 180."
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
                            f"{gaps_text(centers)} degrees "
                            "does not match 90/90/180."
                        )

        print(
            f"Motor {i+1}: center magnet identified. "
            f"Gaps: {gaps_text(centers)} degrees."
        )

        # ------------------------------------------------------
        # 3. Measure outer magnet limits.
        # ------------------------------------------------------

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
            if minus_edges
            else MAX_STEPS
        )

        plus_lim = (
            min(MAX_STEPS, int(min(plus_edges)))
            if plus_edges
            else MAX_STEPS
        )

        self._found_limits[i] = [
            minus_lim,
            plus_lim
        ]

        coarse_width = (
            windows[center_idx][1]
            - windows[center_idx][0]
        )

        print(
            f"Motor {i+1}: travel limits "
            f"-{minus_lim} / +{plus_lim} steps "
            f"(-{minus_lim / STEPS_PER_DEG:.1f} / "
            f"+{plus_lim / STEPS_PER_DEG:.1f} degrees)."
        )

        print(
            f"Motor {i+1}: coarse center window "
            f"{coarse_width} steps."
        )

        # ======================================================
        # 4. CLEAR CENTER MAGNET
        # ======================================================

        # At this point the coarse sweep has just exited
        # the center magnet in direction d.

        backoff = max(
            20,
            coarse_width + FINE_HOME_EXTRA_STEPS
        )

        print(
            f"Motor {i+1}: moving beyond center magnet "
            f"({backoff} steps)."
        )

        for _ in range(backoff):
            yield d

        # ======================================================
        # 5. REVERSE TO BEFORE THE FIRST EDGE
        # ======================================================

        # Reverse twice the backoff distance.
        # This places us beyond the opposite edge.

        print(
            f"Motor {i+1}: reversing for fine approach."
        )

        for _ in range(backoff * 2):
            yield -d

        if self._hall_active(i):

            raise RuntimeError(
                f"Motor {i+1}: Hall sensor still active "
                "before fine scan."
            )

        # ======================================================
        # 6. FINE SCAN
        # ======================================================

        print(
            f"Motor {i+1}: starting fine scan "
            f"at {FINE_HOME_FREQ_HZ} steps/sec."
        )

        fine_pos = 0

        fine_entry = None
        fine_exit = None

        for _ in range(FINE_HOME_MAX_STEPS):

            # Fine scan uses one microstep per pulse.
            # The second value tells _run_together to
            # use the slower homing frequency.

            yield (d, True)

            fine_pos += 1

            active = self._hall_active(i)

            # First edge: Hall activates.

            if fine_entry is None:

                if active:

                    fine_entry = fine_pos

                    print(
                        f"Motor {i+1}: first edge found "
                        f"at {fine_entry} fine steps."
                    )

            # Second edge: Hall deactivates.

            else:

                if not active:

                    fine_exit = fine_pos

                    print(
                        f"Motor {i+1}: second edge found "
                        f"at {fine_exit} fine steps."
                    )

                    break

        if fine_entry is None:

            raise RuntimeError(
                f"Motor {i+1}: fine scan never detected "
                "the center magnet."
            )

        if fine_exit is None:

            raise RuntimeError(
                f"Motor {i+1}: fine scan never exited "
                "the center magnet."
            )

        # ======================================================
        # 7. CALCULATE TRUE MAGNETIC MIDPOINT
        # ======================================================

        fine_width = fine_exit - fine_entry

        if fine_width <= 0:

            raise RuntimeError(
                f"Motor {i+1}: invalid fine magnet width."
            )

        # Check that the fine measurement is reasonably
        # consistent with the coarse measurement.

        if not (
            0.5 * coarse_width
            <= fine_width
            <= 2.0 * coarse_width
        ):

            raise RuntimeError(
                f"Motor {i+1}: fine window width "
                f"{fine_width} differs too much from "
                f"coarse width {coarse_width}."
            )

        fine_midpoint = (
            fine_entry + fine_exit
        ) / 2.0

        self._found_widths[i] = fine_width

        print(
            f"Motor {i+1}: fine window width "
            f"{fine_width} steps "
            f"({fine_width / STEPS_PER_DEG:.2f} degrees)."
        )

        print(
            f"Motor {i+1}: measured midpoint "
            f"{fine_midpoint:.1f} fine steps."
        )

        # ======================================================
        # 8. RETURN TO MIDPOINT
        # ======================================================

        # The motor is currently at fine_exit.
        # Reverse to the measured midpoint.

        delta = int(round(
            fine_midpoint - fine_pos
        ))

        print(
            f"Motor {i+1}: returning to zero midpoint "
            f"({delta:+d} steps)."
        )

        move_dir = d if delta > 0 else -d

        for _ in range(abs(delta)):
            yield (move_dir, True)

        # ======================================================
        # 9. OPTIONAL ZERO OFFSET
        # ======================================================

        offset = int(round(
            HOME_OFFSET_DEG[i] * STEPS_PER_DEG
        ))

        if offset:

            print(
                f"Motor {i+1}: applying zero offset "
                f"{HOME_OFFSET_DEG[i]:+.2f} degrees."
            )

            offset_dir = 1 if offset > 0 else -1

            for _ in range(abs(offset)):
                yield (offset_dir, True)

        print(
            f"Motor {i+1}: fine zero calibration complete."
        )

    # ----------------------------------------------------------
    # HOME BOTH MOTORS
    # ----------------------------------------------------------

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

                        print(
                            f"\nHoming attempt "
                            f"{attempt}/{CALIBRATION_RETRIES}"
                        )

                        self._run_together(
                            [
                                self._home_gen(i)
                                for i in range(2)
                            ],
                            HOME_FREQ_HZ
                        )

                        break

                    except RuntimeError as exc:

                        last_error = exc

                        print(
                            f"Homing attempt {attempt} failed: "
                            f"{exc}"
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

                self.center_width = list(
                    self._found_widths
                )

                self.center_bias = [None, None]

                self._center_pass = [None, None]

                with self.lock:

                    self.pos_steps = [0, 0]
                    self.target_steps = [0, 0]

                self.limit_latched = [0, 0]

                print("\n================================")
                print("HOMING COMPLETE")
                print("Both motors at calibrated zero.")
                print(
                    f"Motor 1 center window: "
                    f"{self.center_width[0]} steps"
                )
                print(
                    f"Motor 2 center window: "
                    f"{self.center_width[1]} steps"
                )
                print("================================\n")

            finally:

                self.homing = False

    # ==========================================================
    # TRACKING TARGET
    # ==========================================================

    def set_target_deg(self, deg):

        deg = max(
            -MAX_DEG,
            min(MAX_DEG, float(deg))
        )

        steps = int(round(
            deg * STEPS_PER_DEG
        ))

        with self.lock:

            self.target_steps = [

                max(
                    -self.limit_steps[i][0],
                    min(
                        self.limit_steps[i][1],
                        steps
                    )
                )

                for i in range(2)
            ]

    def park_at_limit(self, side):

        side = 1 if side >= 0 else -1

        self.set_target_deg(
            side * MAX_DEG
        )

    def angle_deg(self):

        with self.lock:

            return (
                sum(self.pos_steps)
                / (2.0 * STEPS_PER_DEG)
            )

    def target_deg(self):

        with self.lock:

            return (
                sum(self.target_steps)
                / (2.0 * STEPS_PER_DEG)
            )

    # ----------------------------------------------------------
    # OUTER HALL STOP
    # ----------------------------------------------------------

    def _stop_at_outer_magnet(self, i, direction):

        lim = (
            self.limit_steps[i][1]
            if direction > 0
            else self.limit_steps[i][0]
        )

        return (
            self.pos_steps[i] * direction
            >= 0.5 * lim
        )

    # ----------------------------------------------------------
    # CENTER MAGNET DRIFT CORRECTION
    # ----------------------------------------------------------

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

    def _center_crossed(
        self,
        i,
        direction,
        entry,
        exit_pos
    ):

        width = abs(exit_pos - entry)

        w0 = self.center_width[i]

        if w0 and not (
            0.5 * w0 <= width <= 2.0 * w0
        ):

            print(
                f"Motor {i+1}: ignored center pass "
                f"(width {width}, expected ~{w0})."
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
                        f"Motor {i+1}: learned reverse "
                        f"zero offset {mid:+.1f} steps."
                    )

                return

            ref = self.center_bias[i]

        shift = mid - ref

        if abs(shift) > CENTER_MAX_CORRECTION_STEPS:

            print(
                f"Motor {i+1}: ignored {shift:+.0f} "
                "step re-zero."
            )

            return

        shift = int(round(
            shift * CENTER_GAIN
        ))

        if shift:

            with self.lock:

                self.pos_steps[i] -= shift

            if abs(shift) >= 2:

                print(
                    f"Motor {i+1}: center correction "
                    f"{-shift:+d} steps."
                )

    # ==========================================================
    # MOTOR CONTROL THREAD
    # ==========================================================

    def _run(self):

        freq = MIN_FREQ_HZ

        while self.running:

            if self.homing:

                time.sleep(0.002)
                continue

            with self.lock:

                remaining = [
                    self.target_steps[i]
                    - self.pos_steps[i]
                    for i in range(2)
                ]

                positions = self.pos_steps[:]

            active = {}

            dist = [
                abs(v)
                for v in remaining
            ]

            if max(dist) <= STOP_STEPS:

                time.sleep(0.003)
                continue

            for i, rem in enumerate(remaining):

                if abs(rem) <= STOP_STEPS:
                    continue

                direction = (
                    1 if rem > 0 else -1
                )

                # Hard software travel limit.

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

                # Release a latched outer stop only when
                # moving inward and the Hall sensor clears.

                if (
                    self.limit_latched[i]
                    and direction != self.limit_latched[i]
                    and not hall_now
                ):

                    self.limit_latched[i] = 0

                if self.limit_latched[i] == direction:
                    continue

                # Outer magnet detection.

                if (
                    hall_now
                    and self._stop_at_outer_magnet(
                        i,
                        direction
                    )
                ):

                    self.limit_latched[i] = direction

                    with self.lock:
                        self.target_steps[i] = self.pos_steps[i]

                    print(
                        f"Motor {i+1}: outer Hall stop "
                        f"at {positions[i] / STEPS_PER_DEG:+.1f} deg."
                    )

                    continue

                active[i] = direction

            if not active:

                time.sleep(0.002)
                continue

            # Slow down near the outer magnets and center magnet.

            near_outer = any(
                abs(positions[i] / STEPS_PER_DEG)
                >= MAX_DEG - OUTER_APPROACH_DEG
                for i in active
            )

            near_center = any(
                abs(positions[i])
                <= CENTER_TRACK_STEPS + 8
                for i in active
            )

            if near_outer or near_center:

                batch = 1

            else:

                batch = min(
                    CHUNK_STEPS,
                    min(
                        abs(remaining[i])
                        for i in active
                    )
                )

            max_dist = max(
                abs(remaining[i])
                for i in active
            )

            want = (
                MIN_FREQ_HZ
                + (MAX_FREQ_HZ - MIN_FREQ_HZ)
                * min(
                    1.0,
                    max_dist
                    / (SLOW_ZONE_DEG * STEPS_PER_DEG)
                )
            )

            freq = min(
                want,
                freq + ACCEL_HZ_PER_CHUNK
            )

            with self.control_lock:

                if not self.homing:

                    use_freq = (
                        min(int(freq), OUTER_FREQ_HZ)
                        if near_outer
                        else int(freq)
                    )

                    self._pulse(
                        active,
                        batch,
                        use_freq
                    )

                    with self.lock:

                        for i, direction in active.items():

                            self.pos_steps[i] += (
                                direction * batch
                            )

                    for i, direction in active.items():

                        self._center_update(
                            i,
                            direction
                        )

            time.sleep(0.0005)

    # ==========================================================
    # SHUTDOWN
    # ==========================================================

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


# ==============================================================
# MOTION DETECTOR
# ==============================================================

class MotionDetector:

    def __init__(self):

        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=BG_VAR_THRESHOLD,
            detectShadows=False
        )

        self.open_k = np.ones(
            (OPEN_K, OPEN_K),
            np.uint8
        )

        self.merge_k = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (MERGE_K, MERGE_K)
        )

        self.frames = 0

        self.min_px = (
            MIN_MOTION_FRAC * DET_W * DET_H
        )

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

        lr = (
            WARMUP_LEARNING_RATE
            if self.frames < WARMUP_FRAMES
            else BG_LEARNING_RATE
        )

        mask = self.bg.apply(
            gray,
            learningRate=lr
        )

        self.frames += 1

        if self.frames <= WARMUP_FRAMES:

            return Result(
                None,
                [],
                f"learning background {self.frames}/{WARMUP_FRAMES}",
                mask
            )

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

        if (
            cv2.countNonZero(mask)
            > GLOBAL_MOTION_FRAC * DET_W * DET_H
        ):

            return Result(
                None,
                [],
                "global change - ignoring frame",
                mask
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
                mask[y:y+h, x:x+w],
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
                mask
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
            mask
        )


# ==============================================================
# BROWSER STREAMING
# ==============================================================

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


# ==============================================================
# MAIN
# ==============================================================

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--preview",
        action="store_true",
        help="Show local camera window."
    )

    ap.add_argument(
        "--stream",
        action="store_true",
        help="Serve preview in a browser."
    )

    ap.add_argument(
        "--mask",
        action="store_true",
        help="Show motion mask in preview."
    )

    ap.add_argument(
        "--port",
        type=int,
        default=8080
    )

    ap.add_argument(
        "--no-motor",
        action="store_true",
        help="Disable motors."
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
            f"Streaming at http://<pi-address>:{args.port}"
        )

        print(
            "Find Pi address with: hostname -I"
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
        f"Learning background for {WARMUP_FRAMES} frames."
    )

    print(
        f"Tracking range: +/-{MAX_DEG:.0f} degrees."
    )

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

            # --------------------------------------------------
            # TARGET TRACKING
            # --------------------------------------------------

            if locked:

                error = (
                    (tgt.cx / DET_W - 0.5)
                    * 2
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

            # --------------------------------------------------
            # TARGET LOST
            # --------------------------------------------------

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

                # Determine whether target left through an edge.

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
                            f"Target left frame: "
                            f"parking at {side * 90:+d} degrees."
                        )

                        motor.park_at_limit(side)

                        status = (
                            f"target left - parking "
                            f"at {side * 90:+d}"
                        )

                    else:

                        print(
                            "Target lost mid-frame: holding position."
                        )

                        status = "target lost - holding"

                    loss_handled = True

                # Automatic return to calibrated zero.

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

                        status = "idle - homed at zero"

                    except RuntimeError as exc:

                        status = "idle homing error"

                        print(
                            f"ERROR during homing: {exc}"
                        )

                    last_desired_deg = 0.0

                    home_done = True

            # --------------------------------------------------
            # STATUS OUTPUT
            # --------------------------------------------------

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

            # --------------------------------------------------
            # VIDEO DISPLAY
            # --------------------------------------------------

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
                            if locked
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

                    view[
                        0:mh,
                        w-mw:w
                    ] = inset

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
                    ) == ord("q"):

                        break

    except KeyboardInterrupt:

        pass

    finally:

        print("\nShutting down...")

        motor.shutdown(
            go_home=False
        )

        picam2.stop()

        if args.preview:

            cv2.destroyAllWindows()


if __name__ == "__main__":

    main()
