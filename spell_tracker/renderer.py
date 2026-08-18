# test.py
import struct
import cv2 as cv

import moderngl_window as mglw
import numpy as np

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
        ret, frame = self.cap.read()
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            return
                
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
                uv = in_uv;
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

    def on_close(self):
        self.cap.release()
        cv.destroyAllWindows()
        self.landmarker.close()

Test.run()

