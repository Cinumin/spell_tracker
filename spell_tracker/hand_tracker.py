import time
from collections import deque
from pathlib import Path

import cv2 as cv
import mediapipe as mp
import numpy as np

from .utils import draw_landmarks_on_frame
from . import geometry

from OneEuroFilter import OneEuroFilter

DEFAULT_FILTER_CONFIG = {
    'freq': 120,       # Hz
    'mincutoff': 1.0,  # Hz
    'beta': 0.1,
    'dcutoff': 1.0,
}

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Landmark index triple (MCP, PIP, TIP) for the index finger, used by
# is_pinch()/get_pinch_position() to locate the fingertip.
INDEX_FINGER_JOINTS = (5, 6, 8)

# How long the pinch gesture must be continuously absent before a stroke is
# considered finished. Absorbs brief single-frame detection dropouts (e.g.
# from motion blur during fast hand movement) so they don't fragment an
# otherwise continuous stroke.
PEN_UP_GRACE_SECONDS = 0.15

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"
MODEL_PATH = PACKAGE_DIR.parent / "models" / "hand_landmarker.task"
TARGET_CHARACTER = "火"


def is_pinch(result: HandLandmarkerResult) -> bool:
    """Determine whether the tracked hand is making a "pinch" gesture.

    The gesture is recognized when the index finger and thumb are close together,
    while the other fingers are curled (bent).

    Args:
        result: The latest hand-landmarker detection result.

    Returns:
        True if the pinch gesture is detected, False otherwise.
    """
    if not result.hand_landmarks:
        return False

    # Get the positions of the index fingertip and thumb tip
    index_tip = result.hand_landmarks[0][INDEX_FINGER_JOINTS[2]]
    thumb_tip = result.hand_landmarks[0][4]  # Thumb tip landmark index is 4

    index_dip = result.hand_landmarks[0][7]  # Index finger DIP joint
    thumb_ip = result.hand_landmarks[0][3]  # Thumb IP joint

    distanceThumbTipIndexTip = np.sqrt((index_tip.x - thumb_tip.x) ** 2 + (index_tip.y - thumb_tip.y) ** 2)
    distanceThumbTipThumbIp = np.sqrt((thumb_tip.x - thumb_ip.x) ** 2 + (thumb_tip.y - thumb_ip.y) ** 2)
    distanceIndexTipIndexDip = np.sqrt((index_tip.x - index_dip.x) ** 2 + (index_tip.y - index_dip.y) ** 2)

    relativeDistance = (distanceThumbTipIndexTip * 10) / (0.5 * (distanceThumbTipThumbIp + distanceIndexTipIndexDip))

    return relativeDistance < 6.0  # Adjust the threshold as needed


def get_pinch_position(result: HandLandmarkerResult, frame_width: int, frame_height: int):
    """Return the pinch midpoint's pixel coordinates on the frame.

    Uses the normalized image-space landmarks (not the 3D world landmarks)
    so the point lines up with where the finger appears in the rendered
    frame, matching the convention used by draw_landmarks_on_frame() in
    utils.py.

    Returns:
        An (x, y) pixel-coordinate tuple, or None if no hand was detected.
    """
    if not result.hand_landmarks:
        return None
    index_tip = result.hand_landmarks[0][INDEX_FINGER_JOINTS[2]]
    thumb_tip = result.hand_landmarks[0][4]
    mid_x = (index_tip.x + thumb_tip.x) / 2
    mid_y = (index_tip.y + thumb_tip.y) / 2
    return int(mid_x * frame_width), int(mid_y * frame_height)


class StrokeTracker:
    """Owns per-session hand-tracking state: buffered strokes, the pinch
    smoothing filters, and the most recent async detection result.

    Fully testable without a webcam or MediaPipe (see test/test_hand_tracker.py).
    """

    def __init__(self, filter_config: dict | None = None):
        cfg = filter_config or DEFAULT_FILTER_CONFIG
        self.fx = OneEuroFilter(**cfg)
        self.fy = OneEuroFilter(**cfg)
        self.latest_result: HandLandmarkerResult | None = None
        self.latest_timestamp_ms: int = 0
        self.strokes: list[deque] = []
        self.current_stroke: deque = deque()
        self._last_pen_down_time: float = 0.0

    def on_result(self, result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int) -> None:
        """Store the latest async detection result.

        Passed directly as the `result_callback` for the live-stream
        HandLandmarker and invoked automatically once detection completes.
        """
        self.latest_result = result
        self.latest_timestamp_ms = timestamp_ms

    def add_point(self, point) -> None:
        """Append a fingertip position to the in-progress stroke."""
        self.current_stroke.append(point)

    def end_stroke(self) -> None:
        """Complete the in-progress stroke, moving it into the strokes list.

        Called on a pen-down -> pen-up transition so each pinch-drag becomes
        its own entry rather than merging into one continuous trail.
        """
        if self.current_stroke:
            self.strokes.append(self.current_stroke)
            self.current_stroke = deque()

    def clear(self) -> None:
        """Clear all strokes, e.g. in response to a 'c' keypress."""
        self.strokes.clear()
        self.current_stroke = deque()

    def draw(self, canvas: np.ndarray) -> None:
        """Draw all completed strokes plus the in-progress stroke on the canvas.

        Each stroke is drawn independently so no line is drawn connecting the
        end of one stroke to the start of the next.
        """
        for stroke in self.strokes + [self.current_stroke]:
            prev = None
            for x, y in stroke:
                point = (int(x), int(y))
                cv.circle(canvas, point, radius=2, color=(0, 0, 0), thickness=-1)
                if prev is not None:
                    cv.line(canvas, prev, point, color=(0, 0, 0), thickness=3)
                prev = point

    def update(self, frame_width: int, frame_height: int) -> None:
        """Advance tracking state by one frame: buffer a point while pinching,
        or end the in-progress stroke once the pinch has been absent for
        longer than PEN_UP_GRACE_SECONDS."""
        pen_down = self.latest_result and is_pinch(self.latest_result)
        if pen_down:
            self._last_pen_down_time = time.time()
            pinch_position = get_pinch_position(self.latest_result, frame_width, frame_height)
            if pinch_position:
                x, y = pinch_position
                timestamp_s = self.latest_timestamp_ms / 1000.0
                filtered = (self.fx(x, timestamp_s), self.fy(y, timestamp_s))
                self.add_point(filtered)
        elif time.time() - self._last_pen_down_time > PEN_UP_GRACE_SECONDS:
            self.end_stroke()

    def compare_to(self, reference_medians):
        """Compare the collected strokes against a reference character's
        median strokes via geometry.compare_to_reference().

        Raises ValueError (propagated from geometry) if the number of
        collected strokes doesn't match the reference stroke count.
        """
        user_strokes = [list(stroke) for stroke in self.strokes]
        return geometry.compare_to_reference(user_strokes, reference_medians)


def main() -> None:
    """Run the live webcam capture and hand-tracking loop until 'q' is pressed."""
    reference_medians = geometry.load_reference(TARGET_CHARACTER, path=str(DATA_DIR / "huo.json"))
    tracker = StrokeTracker()

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=tracker.on_result,
    )

    with HandLandmarker.create_from_options(options) as landmarker:
        # Start capturing from the webcam.
        cap = cv.VideoCapture(0, cv.CAP_AVFOUNDATION)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Empty camera frame.")
                    break

                # MediaPipe expects an RGB image wrapped in its own Image type.
                rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                # Detection runs asynchronously; tracker.on_result() stores
                # the result once it completes.
                landmarker.detect_async(mp_image, int(time.time() * 1000))

                annotated_frame = (
                    draw_landmarks_on_frame(frame, tracker.latest_result)
                    if tracker.latest_result else frame
                )

                frame_height, frame_width = frame.shape[:2]
                tracker.update(frame_width, frame_height)
                tracker.draw(annotated_frame)

                cv.imshow("frame", annotated_frame)
                key = cv.waitKey(1)
                if key == ord("c"):
                    tracker.clear()
                elif key == ord("r"):
                    try:
                        total, per_stroke = tracker.compare_to(reference_medians)
                        print(f"score vs {TARGET_CHARACTER}: total={total:.3f} per_stroke={per_stroke}")
                    except ValueError as e:
                        print(f"can't compare yet: {e}")
                if key == ord("q"):
                    break
        finally:
            cap.release()
            cv.destroyAllWindows()


if __name__ == "__main__":
    main()
