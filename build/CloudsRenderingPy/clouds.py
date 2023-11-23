import moderngl
import moderngl_window as mglw
import math as m

from pythonworley import worley
from pythonperlin import perlin

from PIL import Image
import io
import numpy as np


def remap(originalValue, originalMin, originalMax, newMin, newMax):
    return newMin + ((originalValue - originalMin) / (originalMax - originalMin)) * (
        newMax - newMin
    )


def get_lf_noise():
    dens = 32
    stacks = []
    mul = [2, 4, 16, 32]

    for i in mul:
        shape = (4 * i, 4 * i, 4 * i)
        w, c = worley(shape, dens=dens // i, seed=0)
        w = w[0].T
        stacks.append(w)
    worleyFBM0 = stacks[0] * 0.625 + stacks[1] * 0.25 + stacks[2] * 0.125
    worleyFBM1 = stacks[1] * 0.625 + stacks[2] * 0.25 + stacks[3] * 0.125
    worleyFBM2 = stacks[2] * 0.75 + stacks[3] * 0.25

    noise = perlin((4, 4, 4), dens=32, seed=0)
    perlinWorley = remap(noise * 1.9, worleyFBM0, 1.0, 0.0, 1.0)

    fbm = worleyFBM0 * 0.625 + worleyFBM1 * 0.25 + worleyFBM2 * 0.125
    return remap(perlinWorley, -(1 - fbm), 1, 0, 1).astype(np.float32)


def get_hf_noise():
    dens = 32
    mul = [4, 8, 16, 32]
    # shape = np.array([128, 128])
    stacks = []

    for i in mul:
        shape = (i, i, i)
        w, c = worley(shape, dens=dens // i, seed=0)
        w = w[0].T
        stacks.append(w)

    worleyFBM0 = stacks[0] * 0.625 + stacks[1] * 0.25 + stacks[2] * 0.125
    worleyFBM1 = stacks[1] * 0.625 + stacks[2] * 0.25 + stacks[3] * 0.125
    worleyFBM2 = stacks[2] * 0.75 + stacks[3] * 0.25
    fbm = worleyFBM0 * 0.625 + worleyFBM1 * 0.25 + worleyFBM2 * 0.125

    return np.asarray(fbm, dtype=np.float32, order="C")


class Clouds:
    resource_dir = "programs"
    window_size = 900, 600

    def __init__(self, ctx):
        self.ctx = ctx

        self.quad = mglw.geometry.quad_fs()

        vertex_source = ""
        with open(f"{self.resource_dir}/vertex.glsl", mode="r") as v:
            vertex_source = v.readlines()

        vertex_source = "".join(vertex_source)

        fragment_source = ""
        with open(f"{self.resource_dir}/fragment.glsl", mode="r") as v:
            fragment_source = v.readlines()

        fragment_source = "".join(fragment_source)

        self.program = self.ctx.program(
            vertex_shader=vertex_source, fragment_shader=fragment_source
        )
        # uniforms
        self.safe_uniform("u_resolution", self.window_size)
        self.theta = 0
        self.safe_uniform("u_density", 150)
        self.safe_uniform("u_coverage", 1.2)

        self.safe_uniform("u_phaseInfluence", 0.5)
        self.safe_uniform("u_eccentrisy", 0.996)
        self.safe_uniform("u_phaseInfluence2", 0.5)
        self.safe_uniform("u_eccentrisy2", 0.49)
        self.safe_uniform("u_attenuation", 0.2)
        self.safe_uniform("u_attenuation2", 0.1)
        self.safe_uniform("u_sunIntensity", 42)
        self.safe_uniform("u_sun_pos", (0, 0.5, 1))
        self.safe_uniform("u_look_at", (0, 0.5, 1))

        # self.safe_uniform('u_fog', 1)
        self.safe_uniform("u_ambient", 1)
        self.fog = 1

        self.safe_uniform("u_hfNoise", 2)
        hf_noise = get_hf_noise()
        # print(hf_noise)
        # print(hf_noise.shape)
        texture3 = self.ctx.texture3d((32, 32, 32), 1, hf_noise, dtype="f4")
        texture3.use(2)

        image = Image.open("textures/weatherMap3.png")
        img_data = np.array(list(image.getdata()), np.uint8)
        texture = self.ctx.texture(image.size, 4, img_data)
        self.safe_uniform("u_weatherMap", 0)
        texture.use(0)

        self.safe_uniform("u_lfNoise", 1)
        lf_noise = get_lf_noise()
        # print(lf_noise.shape)
        texture2 = self.ctx.texture3d((128, 128, 128), 1, lf_noise, dtype="f4")
        texture2.use(1)

    def safe_uniform(self, name, value):
        try:
            self.program[name] = value
        except:
            print(name, " not used")

    def clear(self, color=(0, 0, 0, 0)):
        self.ctx.clear(*color)

    def render(self, time, frame_time):
        # print("render clouds")
        self.clear()
        self.safe_uniform("u_time", time)
        # x, y, z = map(int, input().split(' '))
        self.theta += 0.0125

        # print(self.y)
        self.quad.render(self.program)
