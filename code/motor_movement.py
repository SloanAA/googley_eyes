#!/usr/bin/env python3
"""
TMC2208 STEP/DIR simple scheduler on Raspberry Pi (pigpio)
Interactive control: press keys to jog CW/CCW
Requires: pigpiod running (sudo systemctl start pigpiod)
"""

import pigpio
import time

STEP_PIN = 18   # must be a hardware PWM pin: 12, 13, 18, or 19
DIR_PIN  = 14
EN_PIN   = 15

STEPS_PER_REV = 200
MICROSTEPS    = 8   # floating MS1/MS2 default on most TMC2208 boards

DIR_SETUP_S = 0.001   # min time after DIR change before STEP starts (1ms, generous)
SETTLE_S    = 0.01    # gap after stopping PWM before next move can begin

DEFAULT_FREQ_HZ = 500  # step rate — start conservative, raise once stable

pi = pigpio.pi()
if not pi.connected:
    raise SystemExit("Could not connect to pigpiod — is it running?")

pi.set_mode(STEP_PIN, pigpio.OUTPUT)
pi.set_mode(DIR_PIN, pigpio.OUTPUT)
pi.set_mode(EN_PIN, pigpio.OUTPUT)
pi.write(EN_PIN, 0)  # enable driver


def move(steps, direction, freq_hz=DEFAULT_FREQ_HZ):
    """direction: True = CW, False = CCW. Always rewrites DIR, no caching."""
    dir_level = 1 if direction else 0
    pi.write(DIR_PIN, dir_level)
    time.sleep(DIR_SETUP_S)

    print(f"  DIR pin set to {dir_level}, running {steps} steps @ {freq_hz}Hz")

    pi.hardware_PWM(STEP_PIN, freq_hz, 500000)
    time.sleep(steps / freq_hz)
    pi.hardware_PWM(STEP_PIN, 0, 0)
    pi.write(STEP_PIN, 0)
    time.sleep(SETTLE_S)


def cw(steps=None, freq_hz=DEFAULT_FREQ_HZ):
    if steps is None:
        steps = STEPS_PER_REV * MICROSTEPS
    move(steps, True, freq_hz)


def ccw(steps=None, freq_hz=DEFAULT_FREQ_HZ):
    if steps is None:
        steps = STEPS_PER_REV * MICROSTEPS
    move(steps, False, freq_hz)


def check_dir_pin():
    """Toggle DIR a few times with pauses so you can probe it with a multimeter/LED."""
    print("Toggling DIR pin every 2s — probe GPIO14 now (should alternate 0V / 3.3V)")
    for i in range(6):
        level = i % 2
        pi.write(DIR_PIN, level)
        print(f"  DIR = {level}")
        time.sleep(2)


def shutdown():
    pi.hardware_PWM(STEP_PIN, 0, 0)
    pi.write(EN_PIN, 1)
    pi.stop()


def main():
    print("Motor jog control")
    print("  w = clockwise (small)      s = counterclockwise (small)")
    print("  W = clockwise (full rev)   S = counterclockwise (full rev)")
    print("  d = check DIR pin with multimeter")
    print("  q = quit")
    print()

    small_step = MICROSTEPS * 10

    try:
        while True:
            key = input("> ").strip()

            if key == "w":
                cw(small_step)
            elif key == "s":
                ccw(small_step)
            elif key == "W":
                cw()
            elif key == "S":
                ccw()
            elif key == "d":
                check_dir_pin()
            elif key == "q":
                break
            else:
                print("unrecognized — use w/s/W/S/d/q")

    except KeyboardInterrupt:
        pass
    finally:
        shutdown()


if __name__ == "__main__":
    main()

