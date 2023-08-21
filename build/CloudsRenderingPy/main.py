import moderngl_window as mglw
import math as m

from pythonworley import worley
from pythonperlin import perlin

from PIL import Image
import io
import numpy as np


def remap(originalValue, originalMin, originalMax, newMin, newMax):
    return newMin + ((originalValue - originalMin) / (originalMax - originalMin)) * (newMax - newMin)


def get_lf_noise():
    dens = 32
    stacks = []
    mul = [2, 4, 8, 16, 32]

    for i in mul:
        shape = (4 * i, 4 * i, 4 * i)
        w, c = worley(shape, dens=dens // i, seed=0)
        w = w[0].T
        stacks.append(w)

    worleyFBM0 = stacks[0]*0.625 + stacks[1]*0.25 + stacks[2]*0.125
    worleyFBM1 = stacks[1]*0.625 + stacks[2]*0.25 + stacks[3]*0.125
    worleyFBM2 = stacks[2]*0.75 + stacks[3]*0.25

    noise = perlin((4, 4, 4), dens=32, seed=0)
    fbm = worleyFBM0 * 0.625 + worleyFBM1 * 0.25 + worleyFBM2 * 0.125
    return remap(noise, -(1 - fbm), 1, 0, 1).astype(np.float32)


# np.stack((noise, worleyFBM0, worleyFBM1, worleyFBM2), axis=-1)


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

        self.safe_uniform('u_phaseInfluence', 0.5)
        self.safe_uniform('u_eccentrisy', 0.996)
        self.safe_uniform('u_phaseInfluence2', 0.5)
        self.safe_uniform('u_eccentrisy2', 0.49)
        self.safe_uniform('u_attenuation', 0.1)
        self.safe_uniform('u_attenuation2', 0.02)
        self.safe_uniform('u_sunIntensity', 42)
        # self.safe_uniform('u_fog', 1)
        self.safe_uniform('u_ambient',  0.2)
        self.fog = 1

        image = Image.open("textures/weatherMap.png")
        img_data = np.array(list(image.getdata()), np.uint8)
        texture = self.ctx.texture(image.size, 4, img_data)
        self.safe_uniform('u_weatherMap', 0)
        texture.use(0)

        self.safe_uniform('u_lfNoise', 1)
        lf_noise = get_lf_noise()
        print(lf_noise.shape)
        texture2 = self.ctx.texture3d((128, 128, 128), 1, lf_noise, dtype='f4')
        texture2.use(1)

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
