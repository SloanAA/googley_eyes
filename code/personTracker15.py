#!/usr/bin/env python3
"""
Motion tracker -> two TMC2208 STEP/DIR steppers (pigpio), POSITION control

  python3 personTracker15.py                         # track motion + drive motors
  python3 personTracker15.py --stream                # also serve video at http://<pi>:8080
  python3 personTracker15.py --stream --no-motor     # camera/detection test only
  python3 personTracker15.py --stream --mask         # stream shows the motion mask in the corner
  python3 personTracker15.py --preview               # local window (needs a display)

Motor behavior:
  * Both motors always move at the same time on shared STEP pulse edges
    (tracking AND homing).
  * Homing (at startup, and RETURN_HOME_AFTER_S after the target is last seen) has two phases:

      COARSE: both motors sweep in the same direction at HOME_FREQ_HZ (200 steps/s), watching
        their Hall sensors. The 90/90/180 degree spacing of the three magnets (-90 / 0 / +90)
        identifies the 0 magnet, so homing does NOT depend on where the motors think they are.
        Each motor then moves (by counting steps) to the midpoint of the 0 magnet's Hall window
        as measured on the sweep.

      FINE: a Hall switch has hysteresis, so the window seen travelling one way is shifted from
        the window seen travelling the other way. The coarse midpoint therefore carries a
        direction-dependent error. The fine phase (at FINE_FREQ_HZ, one step at a time) measures
        ALL FOUR edges of the 0 magnet (on/off going forward, on/off going back). The mean of the
        four edges is the true center: hysteresis shifts the two midpoints in opposite directions
        by the same amount, so it cancels. The motor then approaches that point from the
        negative side (to take up backlash the same way every time) and calls it 0 deg (down).
        The forward/reverse midpoint offsets measured here are also stored and used as the
        reference for drift correction while tracking, in both directions.

  * Horizontal position of the motion maps linearly to motor angle:
        left edge -> +90 deg (clockwise) | center -> 0 | right edge -> -90 deg
        (flip with LEFT_IS_CLOCKWISE). Never exceeds +/-MAX_DEG, and the outer
        Hall magnets act as hard travel stops.
  * Travel limits: during homing each motor MEASURES how far (in steps) the outer magnets are
    from 0 (about 400 steps = 90 deg). The tracking target is clamped to that, never more than
    +/-MAX_DEG, and _run() refuses to step outward past it even if the Hall sensor is missed.
    If the Hall sensor sees an outer magnet on the way there, the motor also stops right there.
  * Drift correction: every time a motor passes fully through the 0 magnet's Hall window while
    tracking, the midpoint of that pass is compared with where it should be (the per-direction
    reference measured by fine homing) and the motor's step count is re-zeroed to match
    (see _center_update). Passes that look wrong (odd window width, huge jump) are ignored.
  * If the target leaves the frame off one side (it was near that edge when last seen),
    the motors go to and stop at that side's outer magnet (+90 or -90).
    If it just disappears mid-frame, the motors hold where they are.
  * Motors run in their own thread so they move smoothly.

Motion detection design: see MotionDetector (MOG2 background model, blob merging,
confirm/lose frame counts, sticky target choice).

Stream colors: green box = locked target, yellow = candidate not yet confirmed,
gray = other motion. Cyan line = center of view. --mask shows what the detector sees.

Requires: pigpiod running (sudo systemctl start pigpiod)
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

# Hall sensors are assumed active LOW with pull-ups.
HALL_ACTIVE_LEVEL = 0
HALL_PULL = pigpio.PUD_UP

# ---- Homing (coarse) ----
HOME_FREQ_HZ = 200                # step rate during the coarse homing sweep
CALIBRATION_DIR = [1, 1]          # sweep direction; must be the same logical direction for both motors.
                                  # If the motors physically rotate opposite ways, adjust DIR_INVERT.
CALIBRATION_RETRIES = 3           # retry a failed homing sweep
CALIBRATION_MAX_EXTRA_STEPS = 500
MAGNET_SPACING_TOLERANCE_DEG = 18.0
HOME_OFFSET_DEG = [0.0, 0.0]      # trim where "0" lands relative to the true middle of the magnet window.
                                  # Use this per motor if an eye does not look straight down at 0.

# ---- Homing (fine) ----
FINE_HOMING = True                # measure all four edges of the 0 magnet and center on their mean
FINE_FREQ_HZ = 100                # slow, single-step moves for the fine phase
FINE_MARGIN_STEPS = 4             # steps past each edge before reversing
FINE_SEARCH_STEPS = 40            # if the Hall is off after the coarse move, look this far each side
FINE_MAX_SPREAD = 3.0             # abort an edge search after this many times the coarse window width
FINE_MAX_SHIFT_DEG = 8.0          # the fine center must be within this of the coarse center
FINE_BACKLASH_TAKEUP_STEPS = 6    # only used if the final move has to go in the negative direction

STEP_MASK = sum(1 << p for p in STEP_PINS)

# ---- Motor geometry ----
STEPS_PER_REV = 200
MICROSTEPS = 8
GEAR_RATIO = 1.0
MAX_DEG = 90.0
STEPS_PER_DEG = STEPS_PER_REV * MICROSTEPS * GEAR_RATIO / 360.0
REV_STEPS = int(round(STEPS_PER_REV * MICROSTEPS * GEAR_RATIO))
MAX_STEPS = int(round(MAX_DEG * STEPS_PER_DEG))   # ~400 steps from 0 to an outer magnet
FINE_MAX_SHIFT_STEPS = int(round(FINE_MAX_SHIFT_DEG * STEPS_PER_DEG))

# ---- Re-zeroing on the 0 magnet while tracking ----
CENTER_TRACK_DEG = 20.0           # watch the 0 magnet's Hall window while within this of believed 0
                                  # (single-step moves in this zone so edges are seen exactly)
CENTER_MAX_CORRECTION_DEG = 15.0  # ignore a single correction bigger than this (probably a bad read)
CENTER_BIAS_MAX_DEG = 6.0         # largest direction-dependent midpoint offset we will accept/learn
CENTER_GAIN = 1.0                 # 1.0 = snap fully to the measured middle each pass
CENTER_TRACK_STEPS = int(round(CENTER_TRACK_DEG * STEPS_PER_DEG))
CENTER_MAX_CORRECTION_STEPS = int(round(CENTER_MAX_CORRECTION_DEG * STEPS_PER_DEG))
CENTER_BIAS_MAX_STEPS = int(round(CENTER_BIAS_MAX_DEG * STEPS_PER_DEG))

# ---- Motion tuning ----
MIN_FREQ_HZ = 200
MAX_FREQ_HZ = 1200
ACCEL_HZ_PER_CHUNK = 150
SLOW_ZONE_DEG = 25.0
CHUNK_STEPS = 4
OUTER_APPROACH_DEG = 30.0   # within this many degrees of the outer stop: single-step moves, Hall checked every step
OUTER_FREQ_HZ = 400         # speed cap in that zone
START_DEG = 1.0
STOP_STEPS = 1
RETURN_HOME_ON_EXIT = False
RETURN_HOME_AFTER_S = 10.0

# ---- Mapping ----
LEFT_IS_CLOCKWISE = True
TARGET_GAIN = 1.0
SMOOTHING = 0.4
EDGE_EXIT_FRAC = 0.6        # last commanded angle must be at least this fraction of MAX_DEG
                            # (i.e. target was near the screen edge) to count as "left the frame"

CW = 1
CCW = 0

FRAME_SIZE = (640, 480)
SENSOR_SIZE = (1640, 1232)

# ---- Motion detection ----
DET_W, DET_H = 320, 240
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

Blob = namedtuple("Blob", "px cx cy x y w h")   # all in detection-resolution pixels
Result = namedtuple("Result", "target blobs status mask")


# ======================= Motor (three-magnet position control) =======================
class PositionMotor:
    """
    Two motors driven by synchronized pigpio waves (shared STEP edges).

    home(): position-independent homing. Both motors sweep together, each
    motor's generator (_home_gen) identifies the 0 magnet from the magnet
    spacing and lands near the middle of its Hall window. Then _fine_gen
    measures all four window edges and centers on their mean.

    Tracking: clamp to +/-90 degrees. The outer Hall magnets are travel stops.
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
        self.direction_levels = [None, None]
        # Per-motor travel limit from 0, in steps: [minus side, plus side]. Replaced by the
        # measured outer-magnet positions after each successful homing.
        self.limit_steps = [[MAX_STEPS, MAX_STEPS], [MAX_STEPS, MAX_STEPS]]
        self._found_limits = [None, None]
        self._found_widths = [None, None]
        self._fine = [None, None]          # fine-homing results per motor (dict) or None
        self.center_width = [None, None]   # width (steps) of the 0 magnet window, measured at homing
        # Expected midpoint of a pass through the 0 window, relative to true 0, per travel
        # direction: center_ref[i][+1] / center_ref[i][-1]. Measured by fine homing; if a
        # direction is None it is learned from the first pass (see _center_crossed).
        self.center_ref = [{1: None, -1: None}, {1: None, -1: None}]
        self._center_pass = [None, None]   # per-motor state of the pass through the 0 window in progress
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

        print("Starting startup homing (both motors together)...")
        try:
            self.home()
        except Exception:
            # Do not leave the drivers energized if homing aborts.
            for pin in EN_PINS:
                self.pi.write(pin, 1)
            self.pi.stop()
            raise
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _hall_active(self, i):
        return self.pi.read(HALL_PINS[i]) == HALL_ACTIVE_LEVEL

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

    # ---------- Homing ----------
    def _run_together(self, gens, frequency):
        """
        Drive one generator per motor in lockstep. Each next() yields +1/-1 meaning
        "take one microstep that way"; the generator reads its Hall sensor when it
        resumes, i.e. right after the previous step finished. All requested steps in a
        tick go out in ONE waveform, so both motors step on the same edges. A motor whose
        generator has finished simply stops stepping while the other carries on.
        """
        live = dict(enumerate(gens))
        while live:
            active = {}
            for i in list(live):
                try:
                    active[i] = next(live[i])
                except StopIteration:
                    del live[i]
            if active:
                self._pulse(active, 1, frequency)

    def _home_gen(self, i):
        """
        COARSE position-independent homing for motor i (a generator, see _run_together).

        1. Step off any Hall window we start inside (its true center is unknown).
        2. Sweep in one direction, recording the midpoint of each full Hall window.
        3. Identify the 0 magnet from the spacing (-90 / 0 / +90 => gaps 90, 90, 180):
             first gap ~180      -> first=+90, second=-90, so the NEXT window is 0
             gaps 90 then 90     -> windows were -90, 0, +90, so the SECOND window is 0
             gaps 90 then 180    -> windows were 0, +90, -90, so the NEXT window is 0
        4. Move (by counting steps) to the middle of that window as measured on the sweep.
           This is only approximate (single direction => hysteresis error). _fine_gen then
           refines it.
        """
        d = CALIBRATION_DIR[i]
        quarter = REV_STEPS / 4.0
        half = REV_STEPS / 2.0
        tol = MAGNET_SPACING_TOLERANCE_DEG * STEPS_PER_DEG

        def near(a, b):
            return abs(a - b) <= tol

        def gaps_text(c):
            return [round((c[k + 1] - c[k]) / STEPS_PER_DEG, 1) for k in range(len(c) - 1)]

        # 1. leave the window we may be sitting in
        n = 0
        while self._hall_active(i):
            if n >= CALIBRATION_MAX_EXTRA_STEPS:
                raise RuntimeError(f"Motor {i+1}: Hall sensor would not clear.")
            yield d
            n += 1

        # 2./3. sweep until the 0 magnet is identified
        pos = 0
        entry = None
        centers = []
        windows = []                # (entry, exit) of every full window seen, in sweep steps
        center_idx = None
        next_is_center = False
        target = None
        limit = 2 * REV_STEPS

        while target is None:
            if pos >= limit:
                raise RuntimeError(f"Motor {i+1}: homing sweep ran too long without finding the 0 magnet.")
            yield d
            pos += 1
            active = self._hall_active(i)
            if entry is None and active:
                entry = pos
            elif entry is not None and not active:
                centers.append((entry + pos) / 2.0)
                windows.append((entry, pos))
                print(f"Motor {i+1}: magnet window {len(centers)} at sweep step "
                      f"{centers[-1]:.1f} (width {pos - entry} steps)")
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
                            f"Motor {i+1}: first magnet gap {g1 / STEPS_PER_DEG:.1f} deg "
                            "is not ~90 or ~180.")
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
                            f"Motor {i+1}: magnet spacing {gaps_text(centers)} deg "
                            "does not match 90/90/180.")

        print(f"Motor {i+1}: 0 magnet identified (gaps {gaps_text(centers)} deg).")

        # Measure where the two outer magnets are relative to 0, in this motor's pos_steps
        # sign. Use each outer window's edge closest to 0 (where the Hall first turns on
        # when travelling outward), so the limit is on the safe side of the magnet.
        cmid = centers[center_idx]
        plus_edges = []
        minus_edges = []
        for k, (en, ex) in enumerate(windows):
            if k == center_idx:
                continue
            mid = (en + ex) / 2.0
            rel_mid = (mid - cmid + REV_STEPS / 2.0) % REV_STEPS - REV_STEPS / 2.0
            if abs(rel_mid) < quarter / 2.0:      # another sighting of the 0 magnet itself
                continue
            a = ((en - mid) + rel_mid) * d
            b = ((ex - mid) + rel_mid) * d
            if rel_mid * d > 0:
                plus_edges.append(min(a, b))
            else:
                minus_edges.append(-max(a, b))
        minus_lim = min(MAX_STEPS, int(min(minus_edges))) if minus_edges else MAX_STEPS
        plus_lim = min(MAX_STEPS, int(min(plus_edges))) if plus_edges else MAX_STEPS
        self._found_limits[i] = [minus_lim, plus_lim]
        self._found_widths[i] = windows[center_idx][1] - windows[center_idx][0]
        print(f"Motor {i+1}: travel limits -{minus_lim} / +{plus_lim} steps "
              f"(-{minus_lim / STEPS_PER_DEG:.1f} / +{plus_lim / STEPS_PER_DEG:.1f} deg).")

        # 4. go to the (approximate) middle of the 0 magnet window
        delta = int(round(target - pos))
        print(f"Motor {i+1}: coarse move {delta * d:+d} steps to the middle of the 0 magnet.")
        move_dir = d if delta > 0 else -d
        for _ in range(abs(delta)):
            yield move_dir

    def _step_until(self, i, direction, want_active, max_steps):
        """
        Sub-generator (use with `yield from`): step one microstep at a time in `direction`
        until the Hall sensor reads `want_active`. Returns the number of steps taken, or
        None if it was not seen within max_steps. If the sensor already reads `want_active`
        it takes no steps and returns 0.
        """
        n = 0
        while self._hall_active(i) != want_active:
            if n >= max_steps:
                return None
            yield direction
            n += 1
        return n

    def _fine_gen(self, i):
        """
        FINE homing for motor i. Starts at the coarse center (pos = 0 here, in logical
        step directions, the same sign convention as pos_steps).

        Measures the four edges of the 0 magnet's Hall window:
            going +1:  l_on  (sensor turns on),  u_off (sensor turns off)
            going -1:  u_on  (turns on),         l_off (turns off)
        Hysteresis moves the +1 midpoint one way and the -1 midpoint the other way by the
        same amount, so true center = mean of all four edges. The motor then goes there,
        approaching in the + direction so backlash is taken up the same way every time,
        and HOME_OFFSET_DEG is applied as a final trim.
        """
        offset = int(round(HOME_OFFSET_DEG[i] * STEPS_PER_DEG))
        self._fine[i] = None
        pos = 0

        if not FINE_HOMING:
            for _ in range(abs(offset)):
                yield 1 if offset > 0 else -1
            return

        width0 = self._found_widths[i] or 0
        span = int(FINE_MAX_SPREAD * width0) + 2 * FINE_MARGIN_STEPS + 10

        # Make sure we start inside the window (the coarse move can land on its edge).
        if not self._hall_active(i):
            n = yield from self._step_until(i, 1, True, FINE_SEARCH_STEPS)
            if n is None:
                for _ in range(FINE_SEARCH_STEPS):
                    yield -1
                n = yield from self._step_until(i, -1, True, FINE_SEARCH_STEPS)
                if n is None:
                    raise RuntimeError(f"Motor {i+1}: fine homing could not find the 0 magnet window.")
                pos = -n
            else:
                pos = n

        # Exit through the + side: u_off.
        n = yield from self._step_until(i, 1, False, span)
        if n is None:
            raise RuntimeError(f"Motor {i+1}: fine homing never saw the window's + edge turn off.")
        pos += n
        u_off = pos
        for _ in range(FINE_MARGIN_STEPS):
            yield 1
        pos += FINE_MARGIN_STEPS

        # Come back: sensor turns on at u_on, off again at l_off.
        n = yield from self._step_until(i, -1, True, span + FINE_MARGIN_STEPS)
        if n is None:
            raise RuntimeError(f"Motor {i+1}: fine homing never saw the window's + edge turn on.")
        pos -= n
        u_on = pos
        n = yield from self._step_until(i, -1, False, span)
        if n is None:
            raise RuntimeError(f"Motor {i+1}: fine homing never saw the window's - edge turn off.")
        pos -= n
        l_off = pos
        for _ in range(FINE_MARGIN_STEPS):
            yield -1
        pos -= FINE_MARGIN_STEPS

        # Forward again: sensor turns on at l_on.
        n = yield from self._step_until(i, 1, True, span + FINE_MARGIN_STEPS)
        if n is None:
            raise RuntimeError(f"Motor {i+1}: fine homing never saw the window's - edge turn on.")
        pos += n
        l_on = pos

        mid_f = (l_on + u_off) / 2.0       # midpoint seen travelling +1
        mid_r = (u_on + l_off) / 2.0       # midpoint seen travelling -1
        center = (mid_f + mid_r) / 2.0     # hysteresis cancels
        if abs(center) > FINE_MAX_SHIFT_STEPS:
            raise RuntimeError(
                f"Motor {i+1}: fine center is {center / STEPS_PER_DEG:+.1f} deg from the coarse "
                f"center (limit {FINE_MAX_SHIFT_DEG:.0f}).")

        final = int(round(center)) + offset
        delta = final - pos
        if delta >= 0:
            for _ in range(delta):
                yield 1
        else:
            for _ in range(-delta + FINE_BACKLASH_TAKEUP_STEPS):
                yield -1
            for _ in range(FINE_BACKLASH_TAKEUP_STEPS):
                yield 1

        self._fine[i] = {
            "shift": final,                          # how far the zero moved from the coarse center
            "off_fwd": mid_f - final,                # +1 pass midpoint relative to true 0
            "off_rev": mid_r - final,                # -1 pass midpoint relative to true 0
            "width_fwd": u_off - l_on,
            "width_rev": u_on - l_off,
        }
        f = self._fine[i]
        print(f"Motor {i+1}: fine homing: zero moved {final:+d} steps from coarse; "
              f"window width fwd/rev {f['width_fwd']}/{f['width_rev']} steps; "
              f"midpoint offset fwd/rev {f['off_fwd']:+.1f}/{f['off_rev']:+.1f} steps "
              f"(hysteresis ~{abs(f['off_fwd'] - f['off_rev']):.1f} steps).")

    def home(self):
        """Re-zero both motors together. Works from any starting position."""
        if not self.enabled:
            return
        with self.control_lock:
            self.homing = True
            try:
                while self.pi.wave_tx_busy():
                    time.sleep(0.001)
                last_error = None
                for attempt in range(1, CALIBRATION_RETRIES + 1):
                    try:
                        self._run_together([self._home_gen(i) for i in range(2)], HOME_FREQ_HZ)
                        self._run_together([self._fine_gen(i) for i in range(2)], FINE_FREQ_HZ)
                        break
                    except RuntimeError as exc:
                        last_error = exc
                        print(f"Homing attempt {attempt}/{CALIBRATION_RETRIES} failed: {exc}")
                else:
                    raise RuntimeError(
                        f"Homing failed after {CALIBRATION_RETRIES} attempts. Last error: {last_error}"
                    ) from last_error

                # Limits were measured relative to the coarse center; the fine phase may have
                # moved zero by `shift` logical steps, so move the limits with it.
                limits = []
                for i in range(2):
                    minus_lim, plus_lim = self._found_limits[i]
                    f = self._fine[i]
                    if f:
                        shift = f["shift"]
                        minus_lim = min(MAX_STEPS, max(0, minus_lim + shift))
                        plus_lim = min(MAX_STEPS, max(0, plus_lim - shift))
                    limits.append([minus_lim, plus_lim])
                self.limit_steps = limits

                self.center_width = []
                self.center_ref = []
                for i in range(2):
                    f = self._fine[i]
                    if f:
                        self.center_width.append((f["width_fwd"] + f["width_rev"]) / 2.0)
                        self.center_ref.append({1: f["off_fwd"], -1: f["off_rev"]})
                    else:
                        self.center_width.append(self._found_widths[i])
                        self.center_ref.append({1: None, -1: None})
                self._center_pass = [None, None]
                with self.lock:
                    self.pos_steps = [0, 0]
                    self.target_steps = [0, 0]
                self.limit_latched = [0, 0]
                print("Homing complete: both motors at 0 deg.")
            finally:
                self.homing = False

    # ---------- Tracking ----------
    def set_target_deg(self, deg):
        deg = max(-MAX_DEG, min(MAX_DEG, float(deg)))
        steps = int(round(deg * STEPS_PER_DEG))
        with self.lock:
            self.target_steps = [
                max(-self.limit_steps[i][0], min(self.limit_steps[i][1], steps))
                for i in range(2)]

    def park_at_limit(self, side):
        side = 1 if side >= 0 else -1
        self.set_target_deg(side * MAX_DEG)

    def angle_deg(self):
        with self.lock:
            return sum(self.pos_steps) / (2.0 * STEPS_PER_DEG)

    def target_deg(self):
        with self.lock:
            return sum(self.target_steps) / (2.0 * STEPS_PER_DEG)

    def _stop_at_outer_magnet(self, i, direction):
        """A Hall hit only means 'outer magnet' once we are at least half way to the limit
        (otherwise it is the 0 magnet)."""
        lim = self.limit_steps[i][1] if direction > 0 else self.limit_steps[i][0]
        return self.pos_steps[i] * direction >= 0.5 * lim

    def _center_update(self, i, direction):
        """
        Called after every tracking step of motor i. Near believed 0, watch the 0 magnet's Hall
        window; each time the motor passes completely through it (entering from outside in a
        constant direction) hand the entry/exit step counts to _center_crossed.
        """
        pos = self.pos_steps[i]
        if abs(pos) > CENTER_TRACK_STEPS:
            self._center_pass[i] = None
            return
        active = self._hall_active(i)
        st = self._center_pass[i]
        if st is None or st["dir"] != direction:
            # New pass. Only valid if we start outside the window (otherwise we never saw its entry).
            self._center_pass[i] = {"dir": direction, "entry": None, "ok": not active, "prev": active}
            return
        if active and not st["prev"]:
            if st["ok"]:
                st["entry"] = pos
        elif st["prev"] and not active:
            if st["ok"] and st["entry"] is not None:
                self._center_crossed(i, direction, st["entry"], pos)
            st["entry"] = None
            st["ok"] = True
        st["prev"] = active

    def _center_crossed(self, i, direction, entry, exit_pos):
        """A full pass through the 0 window: re-zero pos_steps[i] to the measured middle."""
        width = abs(exit_pos - entry)
        w0 = self.center_width[i]
        if w0 and not (0.5 * w0 <= width <= 2.0 * w0):
            print(f"Motor {i+1}: ignored a 0-magnet pass (width {width} steps, expected ~{w0:.0f}).")
            return
        mid = (entry + exit_pos) / 2.0
        ref = self.center_ref[i][direction]
        if ref is None:
            # Fine homing did not give a reference for this direction.
            if direction == CALIBRATION_DIR[i]:
                ref = 0.0
            else:
                # First pass the other way since homing: learn how far its midpoint sits from 0.
                if abs(mid) <= CENTER_BIAS_MAX_STEPS:
                    self.center_ref[i][direction] = mid
                    print(f"Motor {i+1}: learned reverse-direction 0 offset {mid:+.1f} steps.")
                return
        shift = mid - ref
        if abs(shift) > CENTER_MAX_CORRECTION_STEPS:
            print(f"Motor {i+1}: ignored a {shift:+.0f} step re-zero (larger than "
                  f"{CENTER_MAX_CORRECTION_DEG:.0f} deg).")
            return
        shift = int(round(shift * CENTER_GAIN))
        if shift:
            with self.lock:
                self.pos_steps[i] -= shift
            if abs(shift) >= 2:
                print(f"Motor {i+1}: crossed the 0 magnet, re-zeroed by {-shift:+d} steps.")

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
                # Hard travel limit (~400 steps from 0, measured at homing): never step
                # outward past it, whether or not the Hall sensor was seen.
                lim = self.limit_steps[i][1] if direction > 0 else self.limit_steps[i][0]
                if positions[i] * direction >= lim:
                    with self.lock:
                        self.target_steps[i] = positions[i]
                    continue
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
            near_center = any(abs(positions[i]) <= CENTER_TRACK_STEPS + 8 for i in active)
            batch = 1 if (near_outer or near_center) else min(
                CHUNK_STEPS, min(abs(remaining[i]) for i in active))
            max_dist = max(abs(remaining[i]) for i in active)
            want = MIN_FREQ_HZ + (MAX_FREQ_HZ - MIN_FREQ_HZ) * min(
                1.0, max_dist / (SLOW_ZONE_DEG * STEPS_PER_DEG))
            freq = min(want, freq + ACCEL_HZ_PER_CHUNK)
            with self.control_lock:
                if not self.homing:
                    use_freq = min(int(freq), OUTER_FREQ_HZ) if near_outer else int(freq)
                    self._pulse(active, batch, use_freq)
                    with self.lock:
                        for i, direction in active.items():
                            self.pos_steps[i] += direction * batch
                    for i, direction in active.items():
                        self._center_update(i, direction)
            time.sleep(0.0005)

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

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.open_k)
        for x0, y0, x1, y1 in IGNORE_ZONES:
            mask[int(y0 * DET_H):int(y1 * DET_H), int(x0 * DET_W):int(x1 * DET_W)] = 0

        if cv2.countNonZero(mask) > GLOBAL_MOTION_FRAC * DET_W * DET_H:
            return Result(None, [], "global change - ignoring frame", mask)

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
    print(f"Homing complete. Tracking range is +/-{MAX_DEG:.0f} degrees.")

    hits = 0
    misses = 0
    last_cx = None
    have_lock = False
    smooth_err = 0.0
    last_seen = time.time()
    ever_locked = False
    last_desired_deg = 0.0       # last angle commanded while locked
    loss_handled = False         # edge-exit parking (or hold) already decided for this loss
    home_done = False            # already homed for this loss
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
                loss_handled = False
                home_done = False

                angle = -smooth_err if LEFT_IS_CLOCKWISE else smooth_err
                desired_deg = max(-MAX_DEG, min(MAX_DEG, angle * TARGET_GAIN * MAX_DEG))
                last_desired_deg = desired_deg
                motor.set_target_deg(desired_deg)
                status = f"tracking target {motor.target_deg():+.1f} deg"
            else:
                if tgt is not None:
                    status = f"confirming {hits}/{CONFIRM_FRAMES}"
                elif res.status != "no motion":
                    status = res.status
                else:
                    status = "no motion"

                # Target gone. If it was near a screen edge it walked off that side:
                # go to (and stop at) that side's outer magnet. Otherwise just hold.
                if ever_locked and misses >= LOSE_FRAMES and not loss_handled:
                    if abs(last_desired_deg) >= EDGE_EXIT_FRAC * MAX_DEG:
                        side = 1 if last_desired_deg > 0 else -1
                        print(f"Target left the frame: going to the {side * 90:+d} deg Hall stop.")
                        motor.park_at_limit(side)
                        status = f"target left - parking at {side * 90:+d}"
                    else:
                        print("Target lost mid-frame: holding position.")
                        status = "target lost - holding"
                    loss_handled = True

                # RETURN_HOME_AFTER_S after the last confirmed target: run the homing
                # sequence (both motors together, identifies the 0 magnet itself, then
                # fine-centers on it).
                if (ever_locked and loss_handled and not home_done
                        and now - last_seen >= RETURN_HOME_AFTER_S):
                    print(f"{RETURN_HOME_AFTER_S:.0f} seconds without target: homing.")
                    try:
                        motor.home()
                        status = "idle - homed at 0"
                    except RuntimeError as exc:
                        status = "idle homing error"
                        print(f"ERROR during homing: {exc}")
                    last_desired_deg = 0.0
                    home_done = True

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
