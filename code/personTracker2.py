#!/usr/bin/env python3
"""
Person tracker -> TMC2208 STEP/DIR motor (pigpio)

  python3 personTracker.py                          # track + drive motor
  python3 personTracker.py --stream                 # also serve video at http://<pi>:8080
  python3 personTracker.py --stream --no-motor      # camera/detection test only
  python3 personTracker.py --preview                # local window (needs a display)

Person on the LEFT  -> clockwise; on the RIGHT -> counterclockwise.
Speed scales with distance from center.
Requires: pigpiod running (sudo systemctl start pigpiod)
"""

import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
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

# ---- Tracking settings ----
DEADZONE = 0.08
LEFT_IS_CLOCKWISE = True
LOST_TIMEOUT = 0.5     # seconds without a person before stopping (try 1.0 if jerky)
DETECT_WIDTH = 320
SMOOTHING = 0.5        # 0 = no smoothing, closer to 1 = smoother but laggier

CW = 1                 # DIR level for clockwise
CCW = 0


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
        print(f"Streaming at http://<pi-address>:{args.port}  (try: hostname -I)")

    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}))  # RGB888 = BGR order, OpenCV-ready
    picam2.start()

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    last_seen = 0.0
    smooth_err = 0.0

    try:
        while True:
            frame = picam2.capture_array()
            h, w = frame.shape[:2]
            small_w = DETECT_WIDTH
            small_h = int(h * small_w / w)
            scale = w / small_w

            gray = cv2.cvtColor(cv2.resize(frame, (small_w, small_h)), cv2.COLOR_BGR2GRAY)
            boxes, _ = hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05)

            status = "no person"
            box = None
            if len(boxes) > 0:
                box = max(boxes, key=lambda b: b[2] * b[3])
                x, y, bw, bh = box
                cx = x + bw / 2
                half = small_w / 2
                error = (cx - half) / half  # -1 = far left ... +1 = far right
                smooth_err = SMOOTHING * smooth_err + (1 - SMOOTHING) * error
                last_seen = time.time()

                if abs(smooth_err) < DEADZONE:
                    motor.drive(0)
                    status = f"centered ({smooth_err:+.2f})"
                else:
                    mag = (abs(smooth_err) - DEADZONE) / (1 - DEADZONE)  # 0..1 past deadzone
                    freq = MIN_FREQ_HZ + mag * (MAX_FREQ_HZ - MIN_FREQ_HZ)
                    left = smooth_err < 0
                    clockwise = left if LEFT_IS_CLOCKWISE else not left
                    motor.drive(freq if clockwise else -freq)
                    status = f"{'CW' if clockwise else 'CCW'} {motor.freq:.0f}Hz ({smooth_err:+.2f})"
            elif time.time() - last_seen > LOST_TIMEOUT:
                motor.drive(0)
                smooth_err = 0.0

            print(status)

            if args.preview or streamer:
                view = frame.copy()
                cv2.line(view, (w // 2, 0), (w // 2, h), (255, 255, 0), 1)
                dz = int(DEADZONE * (w / 2))
                cv2.rectangle(view, (w // 2 - dz, 0), (w // 2 + dz, h), (80, 80, 80), 1)
                if box is not None:
                    bx, by, bbw, bbh = [int(v * scale) for v in box]
                    cv2.rectangle(view, (bx, by), (bx + bbw, by + bbh), (0, 255, 0), 2)
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
