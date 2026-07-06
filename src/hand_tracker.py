import time

import cv2 as cv
import mediapipe as mp
import numpy as np

from utils import draw_landmarks_on_frame

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
    global latest_result
    latest_result = result


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

                if latest_result and is_pen_down(latest_result):
                    print("Pen down.")

                cv.imshow("frame", annotated_frame)
                if cv.waitKey(1) == ord("q"):
                    break
        finally:
            cap.release()
            cv.destroyAllWindows()


if __name__ == "__main__":
    main()
