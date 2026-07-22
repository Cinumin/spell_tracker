"""
Test script: compares RAW fingertip tracking vs ONE-EURO FILTERED
incremental line drawing, side by side.

Left window  = raw landmark position, drawn with plain incremental cv2.line
Right window = OneEuroFilter-smoothed position, drawn the same way

Controls:
  - Pinch (thumb tip + index tip close together) = pen down
  - Release pinch = pen up (breaks the stroke, no connecting line)
  - Press 'c' to clear both canvases
  - Press 'q' to quit

Requirements:
  pip install mediapipe opencv-python numpy --break-system-packages

You'll need a hand_landmarker.task model file in the same directory.
Download it from:
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
"""

import math
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ----------------------------
# One Euro Filter implementation
# ----------------------------
class OneEuroFilter:
    def __init__(self, freq=30.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def _alpha(self, cutoff):
        te = 1.0 / self.freq
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def filter(self, x, timestamp=None):
        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = timestamp
            return x

        if timestamp is not None and self.t_prev is not None:
            dt = timestamp - self.t_prev
            if dt > 0:
                self.freq = 1.0 / dt
            self.t_prev = timestamp

        dx = (x - self.x_prev) * self.freq
        a_d = self._alpha(self.d_cutoff)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff)
        x_hat = a * x + (1 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None


# ----------------------------
# Pinch-based pen state with hysteresis
# ----------------------------
PINCH_ENTER = 0.045
PINCH_EXIT = 0.075
pen_down = False


def update_pen_state(hand_landmarks):
    global pen_down
    thumb_tip = hand_landmarks[4]
    index_tip = hand_landmarks[8]
    distance = math.sqrt(
        (thumb_tip.x - index_tip.x) ** 2
        + (thumb_tip.y - index_tip.y) ** 2
        + (thumb_tip.z - index_tip.z) ** 2
    )
    if not pen_down and distance < PINCH_ENTER:
        pen_down = True
    elif pen_down and distance > PINCH_EXIT:
        pen_down = False
    return pen_down


# ----------------------------
# Shared state between callback thread and main loop
# ----------------------------
latest_result = None


def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result


def main():
    width, height = 640, 480

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path="models/hand_landmarker.task"),
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=1,
        result_callback=result_callback,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    raw_canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
    filtered_canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

    x_filter = OneEuroFilter(min_cutoff=1.0, beta=0.01)
    y_filter = OneEuroFilter(min_cutoff=1.0, beta=0.01)

    raw_prev_point = None
    filtered_prev_point = None

    start_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)  # mirror for natural selfie-view drawing
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))  # guard against cameras ignoring the requested resolution
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.time() - start_time) * 1000)
        landmarker.detect_async(mp_image, timestamp_ms)

        if latest_result is not None and latest_result.hand_landmarks:
            hand_landmarks = latest_result.hand_landmarks[0]
            currently_pen_down = update_pen_state(hand_landmarks)

            index_tip = hand_landmarks[8]
            raw_x = index_tip.x * width
            raw_y = index_tip.y * height

            # --- RAW drawing (no smoothing) ---
            raw_point = (int(raw_x), int(raw_y))
            if currently_pen_down and raw_prev_point is not None:
                cv2.line(raw_canvas, raw_prev_point, raw_point, (0, 0, 0), 3, cv2.LINE_AA)
            raw_prev_point = raw_point if currently_pen_down else None

            # --- FILTERED drawing (OneEuroFilter) ---
            timestamp_s = timestamp_ms / 1000.0
            smoothed_x = x_filter.filter(raw_x, timestamp_s)
            smoothed_y = y_filter.filter(raw_y, timestamp_s)
            filtered_point = (int(smoothed_x), int(smoothed_y))

            if currently_pen_down and filtered_prev_point is not None:
                cv2.line(filtered_canvas, filtered_prev_point, filtered_point, (0, 0, 0), 3, cv2.LINE_AA)
            filtered_prev_point = filtered_point if currently_pen_down else None

            if not currently_pen_down:
                # reset filter state on pen-up so the next stroke doesn't
                # drift/snap from the previous stroke's last position
                x_filter.reset()
                y_filter.reset()

            # visual pen-state indicator on the live camera feed
            color = (0, 255, 0) if currently_pen_down else (0, 0, 255)
            cv2.circle(frame, (int(raw_x), int(raw_y)), 8, color, -1)
        else:
            # hand tracking lost -> break any in-progress stroke so we don't
            # draw a false connecting line once tracking resumes
            raw_prev_point = None
            filtered_prev_point = None
            x_filter.reset()
            y_filter.reset()

        # combine both canvases into one comparison window
        bottom_row = np.hstack([raw_canvas, filtered_canvas])

        cv2.putText(bottom_row, "RAW (no filter)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(bottom_row, "ONE-EURO FILTERED", (width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 0), 2)

        cv2.imshow("Camera (green dot = pen down, red = pen up)", frame)
        cv2.imshow("Raw vs Filtered Drawing", bottom_row)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            raw_canvas[:] = 255
            filtered_canvas[:] = 255
            raw_prev_point = None
            filtered_prev_point = None
            x_filter.reset()
            y_filter.reset()

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
