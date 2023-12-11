import matplotlib.pyplot as plt
import numpy as np


def plot_density():
    dens = []
    with open("time_measure_density.txt", "r") as src:
        while x := src.readline():
            dens.append(round(float(x.strip())))

    ticks = list(range(0, 160, 10))

    for i, d in enumerate(dens):
        print(f"{ticks[i]} & {d} \\\\")


def plot_coverage():
    dens = []
    with open("time_measure_coverage.txt", "r") as src:
        while x := src.readline():
            dens.append(round(float(x.strip())))

    # ticks = list(range(0, 160, 10))
    ticks = np.arange(1, 2.1, 0.1)
    for i, d in enumerate(dens):
        print(f"{round(ticks[i], 1)} & {d} \\\\")
        print(r"\hline")


def plot_resolution():
    dens = []
    with open("time_measure_resolution2.txt", "r") as src:
        while x := src.readline():
            dens.append(round(float(x.strip())))

    # ticks = list(range(0, 160, 10))
    ticks = [(640, 360), (800, 480), (1280, 720), (1920, 1080), (3840, 2160)]
    for i, d in enumerate(dens):
        print(f"{i + 1} & {ticks[i][0]} & {ticks[i][1]} & {d} \\\\")
        print(r"\hline")


plot_resolution()
