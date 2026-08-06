# test.py
import struct

import moderngl_window as mglw
import numpy as np

class Test(mglw.WindowConfig):
    gl_version = (3, 3)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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

            void main() {
                fragColor = vec4(uv, 0.0, 1.0);
            }
        ''',
    )
        self.vao = self.ctx.vertex_array(render_program, [(vbo, '2f 2f', 'in_vert', 'in_uv')])


    def on_render(self, time: float, frametime: float):
        self.ctx.clear(1.0, 0.0, 0.0, 0.0)
        self.vao.render()

Test.run()

