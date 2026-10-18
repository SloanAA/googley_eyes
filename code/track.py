import time
import cv2
from picamera2 import Picamera2
from gpiozero import DigitalOutputDevice, PWMOutputDevice

# ---- Settings ----
STEP_PIN, DIR_PIN, EN_PIN = 18, 23, 24   # change to your wiring
MAX_STEPS_PER_SEC = 800                  # speed at full deflection
DEADZONE = 0.08                          # ignore tiny offsets
LEFT_IS_CLOCKWISE = True                 # flip if direction is backwards
LOST_TIMEOUT = 0.5                       # seconds without a person -> stop
DETECT_WIDTH = 320                       # smaller = faster detection

# ---- Motor (STEP/DIR stepper driver, e.g. A4988/DRV8825) ----
step = PWMOutputDevice(STEP_PIN, frequency=100, initial_value=0)
direction = DigitalOutputDevice(DIR_PIN)
enable = DigitalOutputDevice(EN_PIN, initial_value=True)  # many drivers: LOW = enabled

def set_motor(speed):
    """speed in [-1, 1]; positive = clockwise."""
    if abs(speed) < 0.01:
        step.value = 0
        return
    direction.value = 1 if speed > 0 else 0
    step.frequency = max(10, int(abs(speed) * MAX_STEPS_PER_SEC))
    step.value = 0.5

# ---- Camera + detector ----
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}))
picam2.start()

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

last_seen = 0.0
try:
    enable.off()  # enable driver (active low)
    while True:
        frame = picam2.capture_array()
        small = cv2.resize(frame, (DETECT_WIDTH, int(frame.shape[0] * DETECT_WIDTH / frame.shape[1])))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        boxes, _ = hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05)

        if len(boxes) > 0:
            x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])  # largest person
            cx = x + w / 2
            half = small.shape[1] / 2
            error = (cx - half) / half            # -1 = far left, +1 = far right

            if abs(error) < DEADZONE:
                set_motor(0)
            else:
                speed = -error if LEFT_IS_CLOCKWISE else error
                set_motor(speed)
            last_seen = time.time()
            print(f"person at error={error:+.2f}")
        elif time.time() - last_seen > LOST_TIMEOUT:
            set_motor(0)
finally:
    set_motor(0)
    enable.on()
    picam2.stop()