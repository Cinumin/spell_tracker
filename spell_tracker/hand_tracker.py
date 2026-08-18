import time
from collections import deque
from pathlib import Path

import cv2 as cv
import mediapipe as mp
import numpy as np

from .utils import draw_landmarks_on_frame
from . import geometry

from OneEuroFilter import OneEuroFilter

import struct
import moderngl_window as mglw

from .shader import ShaderWindow

DEFAULT_FILTER_CONFIG = {
    'freq': 120,       # Hz
    'mincutoff': 0.3,  # Hz -- lower = more smoothing of jitter while holding still
    'beta': 0.4,       # higher = less lag when the fingertip moves quickly
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

# Distance-ratio thresholds below which the pinch gesture is considered
# active. Two-level (hysteresis) design: PINCH_ENGAGE_THRESHOLD is the
# stricter bar to *start* a new pinch; PINCH_SUSTAIN_THRESHOLD is the
# looser bar used to *continue* an already-active pinch, so a transient
# single-frame noisy reading (e.g. from motion blur during fast hand
# movement) doesn't get misread as a real pen-lift mid-stroke. Starting
# points to hand-tune on the real webcam -- raise PINCH_SUSTAIN_THRESHOLD
# further if strokes still drop out, or lower PINCH_ENGAGE_THRESHOLD if
# pinch false-triggers when fingers are merely close.
PINCH_ENGAGE_THRESHOLD = 6.0
PINCH_SUSTAIN_THRESHOLD = 7.5

# Curve-smoothing render parameters (see smooth_stroke()).
SPLINE_TARGET_PIXELS_PER_SAMPLE = 4  # ~1 interpolated sample per 4px of chord length
SPLINE_MIN_SAMPLES_PER_SEGMENT = 2
SPLINE_MAX_SAMPLES_PER_SEGMENT = 20  # bounds per-frame draw cost on long fast-motion jumps

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"
MODEL_PATH = PACKAGE_DIR.parent / "models" / "hand_landmarker.task"

# Characters the app tries to recognize a drawn stroke set against.
CANDIDATE_CHARACTERS = ("火", "土")


def is_pinch(result: HandLandmarkerResult, threshold: float = PINCH_ENGAGE_THRESHOLD) -> bool:
    """Determine whether the tracked hand is making a "pinch" gesture.

    The gesture is recognized when the index finger and thumb are close together,
    while the other fingers are curled (bent).

    Args:
        result: The latest hand-landmarker detection result.
        threshold: The relative-distance ceiling below which the gesture
            counts as a pinch. Callers doing hysteresis (see StrokeTracker)
            pass a looser threshold to sustain an already-active pinch.

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

    return relativeDistance < threshold


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


def _catmull_rom_point(p0, p1, p2, p3, t):
    """Evaluate the uniform Catmull-Rom spline segment (p1 -> p2, shaped by
    neighbors p0/p3) at parameter t in [0, 1]. q(0)==p1 and q(1)==p2 exactly,
    so the resulting curve still passes through every real input point."""
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2 * p1)
        + (-p0 + p2) * t
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
        + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
    )


def smooth_stroke(points) -> list:
    """Fit a uniform Catmull-Rom spline through `points` and return a denser
    list of (x, y) pixel points approximating a smooth curve for rendering.

    Degrades gracefully for short strokes:
      - 0 points -> []
      - 1 point  -> [that point]
      - 2+ points -> spline-interpolated curve; endpoint segments reuse the
        nearest real point as the missing p0/p3 neighbor, so the curve still
        passes through the first and last real points exactly.
    """
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n == 0:
        return []
    if n == 1:
        return [(int(pts[0, 0]), int(pts[0, 1]))]

    smoothed = [tuple(pts[0])]
    for i in range(n - 1):
        p0 = pts[i - 1] if i - 1 >= 0 else pts[i]
        p1 = pts[i]
        p2 = pts[i + 1]
        p3 = pts[i + 2] if i + 2 < n else pts[i + 1]

        segment_length = np.linalg.norm(p2 - p1)
        samples = int(np.clip(
            round(segment_length / SPLINE_TARGET_PIXELS_PER_SAMPLE),
            SPLINE_MIN_SAMPLES_PER_SEGMENT, SPLINE_MAX_SAMPLES_PER_SEGMENT,
        ))

        for t in np.linspace(0, 1, samples, endpoint=True)[1:]:  # skip t=0: duplicate of previous append
            smoothed.append(tuple(_catmull_rom_point(p0, p1, p2, p3, t)))

    return [(int(x), int(y)) for x, y in smoothed]


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
        self._pinching: bool = False

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
            curve_points = smooth_stroke(list(stroke))
            prev = None
            for point in curve_points:
                if prev is not None:
                    cv.line(canvas, prev, point, color=(0, 0, 0), thickness=6, lineType=cv.LINE_AA)
                prev = point

        if self.current_stroke:
            last = self.current_stroke[-1]
            cv.circle(canvas, (int(last[0]), int(last[1])), radius=4, color=(0, 0, 0), thickness=-1, lineType=cv.LINE_AA)

    def update(self, frame_width: int, frame_height: int) -> None:
        """Advance tracking state by one frame: buffer a point while pinching,
        or end the in-progress stroke once the pinch has been absent for
        longer than PEN_UP_GRACE_SECONDS.

        Uses hysteresis (PINCH_ENGAGE_THRESHOLD vs. PINCH_SUSTAIN_THRESHOLD)
        so a single noisy frame mid-drag doesn't get misread as a pen lift.
        """
        threshold = PINCH_SUSTAIN_THRESHOLD if self._pinching else PINCH_ENGAGE_THRESHOLD
        pen_down = bool(self.latest_result and is_pinch(self.latest_result, threshold=threshold))
        self._pinching = pen_down
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

    def identify(self, candidates):
        """Rank candidate characters by how closely they match the
        collected strokes, via geometry.identify_character().

        `candidates` maps character -> reference medians. Returns a list
        of (character, total_distance) tuples sorted best-match first;
        empty if no candidate has the same stroke count as the drawing.
        """
        user_strokes = [list(stroke) for stroke in self.strokes]
        return geometry.identify_character(user_strokes, candidates)


def main() -> None:
    """Run the live webcam capture and hand-tracking loop until 'q' is pressed."""
    candidates = geometry.load_references(CANDIDATE_CHARACTERS, path=str(DATA_DIR / "characters.json"))
    tracker = StrokeTracker()

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.LIVE_STREAM,
        # Lowered from the 0.5 default: fewer "no landmarks this frame"
        # drops on motion-blurred frames during fast hand movement, and
        # keeps MediaPipe's lightweight frame-to-frame tracker engaged
        # instead of falling back to full re-detection.
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
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
                    ranked = tracker.identify(candidates)
                    if not ranked:
                        stroke_count = len(tracker.strokes)
                        print(f"no candidate has {stroke_count} strokes -- can't identify yet")
                    else:
                        best_character, best_distance = ranked[0]
                        print(f"looks like {best_character} (distance={best_distance:.3f}); ranking={ranked}")
                        if best_character == "火":
                            ShaderWindow.run_shader()
                if key == ord("q"):
                    break
        finally:
            cap.release()
            cv.destroyAllWindows()

class Test(mglw.WindowConfig):
    gl_version = (3, 3)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        """Run the live webcam capture and hand-tracking loop until 'q' is pressed."""
        self.candidates = geometry.load_references(CANDIDATE_CHARACTERS, path=str(DATA_DIR / "characters.json"))
        self.tracker = StrokeTracker()
        
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=VisionRunningMode.LIVE_STREAM,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
            result_callback=self.tracker.on_result,
            )
        
        self.landmarker = HandLandmarker.create_from_options(options)
        self.cap = cv.VideoCapture(0, cv.CAP_AVFOUNDATION)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            return
        ret, frame = self.cap.read()
                
        self.texture = self.ctx.texture(
                    size=(frame.shape[1], frame.shape[0]),
                    components=3,
                    data=None
                )
        #quad vertices for a full-screen quad
        vertices = np.array([
            #x, y, u, v
            -1.0 , 1.0, 0.0, 1.0, #top left
            -1.0, -1.0, 0.0, 0.0, #bottom left
            1.0, 1.0, 1.0, 1.0, #top right
            1.0, 1.0, 1.0, 1.0, #top right
            -1.0, -1.0, 0.0, 0.0, #bottom left
            1.0, -1.0, 1.0, 0.0, #bottom right
        ], dtype=np.float32)
        vertices_buffer = struct.pack('f' * len(vertices), *vertices)
        # put the array into a VBO
        vbo = self.ctx.buffer(vertices_buffer)
        render_program = self.ctx.program(
        vertex_shader='''
            #version 330
            in vec2 in_vert;
            in vec2 in_uv;
            out vec2 uv;
            void main() {
                gl_Position = vec4(in_vert, 0.0, 1.0);
                uv = vec2(in_uv.x, 1.0 - in_uv.y); // flip y for texture coordinates
            }
        ''',
        fragment_shader = 
    '''
            #version 330
            in vec2 uv;
            out vec4 fragColor;
            uniform sampler2D tex;

            void main() {
                fragColor = texture(tex, uv);
            }
        ''',
    )
        self.vao = self.ctx.vertex_array(render_program, [(vbo, '2f 2f', 'in_vert', 'in_uv')])

    #FIX ME: Write the fragment shader for the fireball effect.
    #shader_program = self.ctx.program(
    #    vertex_shader='''
    #        #version 330
    #        in vec2 in_vert;
    #        in vec2 in_uv;
    #        out vec2 uv;
    #        void main() {
    #            gl_Position = vec4(in_vert, 0.0, 1.0);
    #            uv = in_uv;
    #        }
    #    ''',
    #    fragment_shader='''
    #        #version 330
    #        uniform float u_time;
    #        uniform floatu_intesity;
    #        uniform u_position;
    #        void main() {
    #            fragColor = texture(tex, uv);
    #        }
    #    ''',
    #)
    

    def on_render(self, t: float, frametime: float):
        self.ctx.clear(1.0, 0.0, 0.0, 0.0)
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Empty camera frame.")
            return

        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        self.landmarker.detect_async(mp_image, int(time.time() * 1000))
        annotated_frame = (
            draw_landmarks_on_frame(frame, self.tracker.latest_result)
            if self.tracker.latest_result else frame
        )

        frame_height, frame_width = frame.shape[:2]
        self.tracker.update(frame_width, frame_height)
        self.tracker.draw(annotated_frame)

        rgb_annotated_frame = cv.cvtColor(annotated_frame, cv.COLOR_BGR2RGB)
        self.texture.write(rgb_annotated_frame.tobytes())
        self.texture.use()
        self.vao.render()

    def on_key_event(self, key, action, modifiers):
        if action == self.wnd.keys.ACTION_PRESS:
            if key == self.wnd.keys.C:
                self.tracker.clear()
            elif key == self.wnd.keys.R:
                ranked = self.tracker.identify(self.candidates)
                if not ranked:
                    stroke_count = len(self.tracker.strokes)
                    print(f"no candidate has {stroke_count} strokes -- can't identify yet")
                else:
                    best_character, best_distance = ranked[0]
                    print(f"looks like {best_character} (distance={best_distance:.3f}); ranking={ranked}")
                    if best_character == "火":
                        ShaderWindow.run_shader()

    def on_close(self):
        self.cap.release()
        cv.destroyAllWindows()
        self.landmarker.close()


if __name__ == "__main__":
    Test.run()
