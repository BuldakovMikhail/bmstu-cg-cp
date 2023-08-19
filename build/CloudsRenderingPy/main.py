import moderngl_window as mglw
import math as m

from PIL import Image
import io
import numpy as np


class App(mglw.WindowConfig):
    window_size = 900, 600
    resource_dir = 'programs'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.quad = mglw.geometry.quad_fs()
        self.program = self.load_program(
            vertex_shader='vertex.glsl', fragment_shader='fragment.glsl')
        # uniforms
        self.program['u_resolution'] = self.window_size
        self.theta = 0

        image = Image.open("textures/weatherMap2.png")
        img_data = np.array(list(image.getdata()), np.uint8)
        texture = self.ctx.texture(image.size, 3, img_data)
        # self.program['u_weatherMap'] = 0

        # texture.use(0)

    def render(self, time, frame_time):
        self.ctx.clear()
        # self.program['u_time'] = time
       # x, y, z = map(int, input().split(' '))
        self.theta += 0.0125

        self.program['u_sun_pos'] = (0, 0.5, 1)
        # print(self.y)
        self.quad.render(self.program)

    def mouse_position_event(self, x, y, dx, dy):
        # self.program['u_mouse'] = (x, y)
        pass


if __name__ == '__main__':
    mglw.run_window_config(App)