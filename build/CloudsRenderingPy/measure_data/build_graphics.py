import matplotlib.pyplot as plt
import numpy as np


def plot_density():
    dens = []
    with open("time_density.txt", "r") as src:
        while x := src.readline():
            dens.append(float(x.strip()))

    ticks = list(range(0, 160, 20))

    for i, d in enumerate(dens):
        print(f"{ticks[i]} & {round(d, 3)} \\\\")
        print(r"\hline")

    fig1 = plt.figure(figsize=(10, 7))
    plot = fig1.add_subplot()
    plot.plot(ticks, dens)
    # plt.legend()
    plt.grid()
    plt.title("Зависимость количества кадров в секунду от плотности")
    plt.ylabel("Среднее количество кадров в секунду")
    plt.xlabel("Плотность")
    # plt.yscale("log")
    # plt.xticks(list(range(1, 6)))
    # plt.yticks(list(set(dens)))
    plt.show()


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

    fig1 = plt.figure(figsize=(10, 7))
    plot = fig1.add_subplot()
    plot.plot(ticks, dens)
    # plt.legend()
    plt.grid()
    plt.title("Зависимость количества кадров в секунду от размера изображения")
    plt.ylabel("Количество кадров в секунду")
    plt.xlabel("Разрешение экрана")
    # plt.yscale("log")
    # plt.xticks(list(range(1, 6)))
    plt.yticks(list(set(dens)))
    plt.show()


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

    fig1 = plt.figure(figsize=(10, 7))
    plot = fig1.add_subplot()
    plot.plot(list(range(1, 6)), dens)
    # plt.legend()
    plt.grid()
    # plt.title("Зависимость количества кадров в секунду от размера изображения")
    plt.ylabel("Количество кадров в секунду")
    plt.xlabel("Размер изображения")
    # plt.yscale("log")
    plt.xticks(list(range(1, 6)))
    plt.show()


# plot_density()
# plot_coverage()
plot_resolution()
