import time

import cv2 as cv
from collections import deque
import mediapipe as mp
import numpy as np

from utils import draw_landmarks_on_frame

from OneEuroFilter import OneEuroFilter

config = {
    'freq': 120,       # Hz
    'mincutoff': 1.0,  # Hz
    'beta': 0.1,       
    'dcutoff': 1.0    
    }

fx = OneEuroFilter(**config)
fy = OneEuroFilter(**config)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Landmark index triples (MCP, PIP, TIP) for each finger, used to measure
# how curled that finger is via calculate_joint_angle().
INDEX_FINGER_JOINTS = (5, 6, 8)
MIDDLE_FINGER_JOINTS = (9, 10, 12)
RING_FINGER_JOINTS = (13, 14, 16)
PINKY_FINGER_JOINTS = (17, 18, 20)

# A finger is considered "bent" below this angle, and "straight" above it.
BENT_ANGLE_THRESHOLD_DEG = 100

# How long the pinch gesture must be continuously absent before a stroke is
# considered finished. Absorbs brief single-frame detection dropouts (e.g.
# from motion blur during fast hand movement) so they don't fragment an
# otherwise continuous stroke.
PEN_UP_GRACE_SECONDS = 0.15

# Holds the most recent detection result produced by the async landmarker
# callback, so the main loop can read it without blocking on detection.
latest_result = None


def calculate_joint_angle(result: HandLandmarkerResult, mcp_index: int, pip_index: int, tip_index: int) -> float:
    """Calculate the angle (in degrees) at a finger's PIP joint.

    The angle is formed between the vector from PIP to MCP and the vector
    from PIP to TIP, so a straight finger measures close to 180 degrees and
    a fully curled finger measures close to 0 degrees.

    Args:
        result: The latest hand-landmarker detection result.
        mcp_index: Landmark index of the metacarpophalangeal joint.
        pip_index: Landmark index of the proximal interphalangeal joint.
        tip_index: Landmark index of the fingertip.

    Returns:
        The joint angle in degrees, or None if no hand was detected.
    """
    if not result.hand_world_landmarks:
        return None

    # hand_world_landmarks is a list of hands, each a list of Landmark objects.
    hand = result.hand_world_landmarks[0]
    mcp = np.array([hand[mcp_index].x, hand[mcp_index].y, hand[mcp_index].z])
    pip = np.array([hand[pip_index].x, hand[pip_index].y, hand[pip_index].z])
    tip = np.array([hand[tip_index].x, hand[tip_index].y, hand[tip_index].z])

    v1 = mcp - pip
    v2 = tip - pip

    dot_product = np.dot(v1, v2)
    v1_magnitude = np.linalg.norm(v1)
    v2_magnitude = np.linalg.norm(v2)

    angle_rad = np.arccos(np.clip(dot_product / (v1_magnitude * v2_magnitude), -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def save_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int) -> None:
    """Store the latest async detection result in the module-level cache.

    This is passed as the `result_callback` for the live-stream
    HandLandmarker and is invoked automatically once detection completes.
    """
    global latest_result, latest_timestamp_ms
    latest_result = result
    latest_timestamp_ms = timestamp_ms

def is_pen_down(result: HandLandmarkerResult) -> bool:
    """Determine whether the tracked hand is making a "pen down" gesture.

    The gesture is recognized when the index finger is extended (straight)
    while the middle, ring, and pinky fingers are curled (bent), mimicking
    the grip used to hold a pen.

    Args:
        result: The latest hand-landmarker detection result.

    Returns:
        True if the pen-down gesture is detected, False otherwise.
    """
    angle_index = calculate_joint_angle(result, *INDEX_FINGER_JOINTS)
    angle_middle = calculate_joint_angle(result, *MIDDLE_FINGER_JOINTS)
    angle_ring = calculate_joint_angle(result, *RING_FINGER_JOINTS)
    angle_pinky = calculate_joint_angle(result, *PINKY_FINGER_JOINTS)

    if None in (angle_index, angle_middle, angle_ring, angle_pinky):
        return False

    return (
        angle_index > BENT_ANGLE_THRESHOLD_DEG
        and angle_middle < BENT_ANGLE_THRESHOLD_DEG
        and angle_ring < BENT_ANGLE_THRESHOLD_DEG
        and angle_pinky < BENT_ANGLE_THRESHOLD_DEG
    )

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

    relativeDistance = (distanceThumbTipIndexTip * 10) / (0.5 *(distanceThumbTipThumbIp + distanceIndexTipIndexDip))
   
    return relativeDistance < 6.0  # Adjust the threshold as needed

# Completed strokes, each a deque of (x, y) pixel positions traced while the
# pen was "down" during one pen-down -> pen-up cycle.
strokes = []

# Pixel positions of the index fingertip recorded during the in-progress
# stroke (empty when the pen is up).
current_stroke = deque()


def get_pinch_position(result: HandLandmarkerResult, frame_width: int, frame_height: int):
    """Return the index fingertip's pixel coordinates on the frame.

    Uses the normalized image-space landmarks (not the 3D world landmarks
    used for angle calculations) so the point lines up with where the
    finger appears in the rendered frame, matching the convention used by
    draw_landmarks_on_frame() in utils.py.

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


def add_to_buffer(point) -> None:
    """Append a fingertip position to the in-progress stroke."""
    current_stroke.append(point)


def end_stroke() -> None:
    """Complete the in-progress stroke, moving it into the strokes list.

    Called on a pen-down -> pen-up transition so each pinch-drag becomes its
    own entry rather than merging into one continuous trail.
    """
    global current_stroke
    if current_stroke:
        strokes.append(current_stroke)
        current_stroke = deque()


def clear_stroke(key: int) -> None:
    """Clear all strokes if the 'c' key was pressed this frame."""
    global current_stroke
    if key == ord("c"):
        strokes.clear()
        current_stroke = deque()

def draw_stroke(canvas: np.ndarray) -> None:
    """Draw all completed strokes plus the in-progress stroke on the canvas.

    Each stroke is drawn independently so no line is drawn connecting the
    end of one stroke to the start of the next.
    """
    for stroke in strokes + [current_stroke]:
        prev = None
        for x, y in stroke:
            point = (int(x), int(y))
            cv.circle(canvas, point, radius=2, color=(0, 0, 0), thickness=-1)
            if prev is not None:
                cv.line(canvas, prev, point, color=(0, 0, 0), thickness=3)
            prev = point

def main() -> None:
    """Run the live webcam capture and hand-tracking loop until 'q' is pressed."""
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="models/hand_landmarker.task"),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=save_result,
    )

    with HandLandmarker.create_from_options(options) as landmarker:
        # Start capturing from the webcam.
        cap = cv.VideoCapture(0, cv.CAP_AVFOUNDATION)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        last_pen_down_time = 0.0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Empty camera frame.")
                    break

                # MediaPipe expects an RGB image wrapped in its own Image type.
                rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                # Detection runs asynchronously; save_result() stores the
                # result in `latest_result` once it completes.
                landmarker.detect_async(mp_image, int(time.time() * 1000))

                annotated_frame = (
                    draw_landmarks_on_frame(frame, latest_result) if latest_result else frame
                )

                pen_down = latest_result and is_pinch(latest_result)
                if pen_down:
                    last_pen_down_time = time.time()
                    frame_height, frame_width = frame.shape[:2]
                    pinch_position = get_pinch_position(latest_result, frame_width, frame_height)
                    if pinch_position:
                        x, y = pinch_position
                        timestamp_s = latest_timestamp_ms / 1000.0
                        filtered = (fx(x, timestamp_s), fy(y, timestamp_s))
                        add_to_buffer(filtered)
                elif time.time() - last_pen_down_time > PEN_UP_GRACE_SECONDS:
                    end_stroke()

                draw_stroke(annotated_frame)

                cv.imshow("frame", annotated_frame)
                key = cv.waitKey(1)
                clear_stroke(key)
                if key == ord("q"):
                    break
        finally:
            cap.release()
            cv.destroyAllWindows()


if __name__ == "__main__":
    main()
