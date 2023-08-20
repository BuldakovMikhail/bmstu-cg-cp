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
        self.safe_uniform('u_resolution', self.window_size)
        self.theta = 0

        image = Image.open("textures/weatherMap.png")
        img_data = np.array(list(image.getdata()), np.uint8)
        texture = self.ctx.texture(image.size, 4, img_data)
        self.safe_uniform('u_weatherMap', 0)

        #   struct ufStructProperties{
        # 	float density;
        # 	float coverage;
        # 	float phaseInfluence;
        # 	float eccentrisy;

        # 	float phaseInfluence2;
        # 	float eccentrisy2;
        # 	float attenuation;
        # 	float attenuation2;
        # 	float sunIntensity;
        # 	float fog;
        # 	float ambient;
        # };

        self.safe_uniform('u_density', 10)
        self.safe_uniform('u_coverage', 0.7)
        self.safe_uniform('u_phaseInfluence', 1)
        self.safe_uniform('u_eccentrisy', 0.996)
        self.safe_uniform('u_phaseInfluence2', 1)
        self.safe_uniform('u_eccentrisy2', 0.49)
        self.safe_uniform('u_attenuation', 0.2)
        self.safe_uniform('u_attenuation2', 0.2)
        self.safe_uniform('u_sunIntensity', 42)
        # self.safe_uniform('u_fog', 1)
        self.safe_uniform('u_ambient',  1)

        texture.use(0)
        self.fog = 1

    def safe_uniform(self, name, value):
        try:
            self.program[name] = value
        except:
            print(name, ' not used')

    def render(self, time, frame_time):
        self.ctx.clear()
        self.safe_uniform('u_time', time)
       # x, y, z = map(int, input().split(' '))
        self.theta += 0.0125

        self.safe_uniform('u_sun_pos', (0, 0.5, 1))
        # print(self.y)
        self.quad.render(self.program)

    def mouse_position_event(self, x, y, dx, dy):
        # self.program['u_mouse'] = (x, y)
        pass


if __name__ == '__main__':
    mglw.run_window_config(App)
