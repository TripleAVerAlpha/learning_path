import matplotlib.pyplot as plt
import numpy as np


def plot(X, Y, title="График", label=("x", "y"), color=('#0d1117', "#161b22", "#c9d1d9"), file=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(color[0])
    ax.patch.set_facecolor(color[0])
    ax.stackplot(X, Y, color=color[1])

    for i in ax.spines:
        ax.spines[i].set_color(color[2])
    ax.tick_params(colors=color[2], which='both')

    plt.xlabel(label[0], fontsize=14, fontweight="bold", color=color[2])
    plt.ylabel(label[1], fontsize=14, fontweight="bold", color=color[2])
    plt.title(title, fontsize=18, fontweight="bold", color=color[2])

    if file is None:
        plt.show()
    else:
        plt.savefig(file)


def multiplot(Y, title="График", label=("x", "y"), color=('#0d1117', "#161b22", "#c9d1d9"), file=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(color[0])
    ax.patch.set_facecolor(color[0])
    for i in range(len(Y)):
        X = range(len(Y[i]))
        ax.plot(X, Y[i])

    for i in ax.spines:
        ax.spines[i].set_color(color[2])
    ax.tick_params(colors=color[2], which='both')

    plt.xlabel(label[0], fontsize=14, fontweight="bold", color=color[2])
    plt.ylabel(label[1], fontsize=14, fontweight="bold", color=color[2])
    plt.title(title, fontsize=18, fontweight="bold", color=color[2])

    if file is None:
        plt.show()
    else:
        plt.savefig(file)


def plotImage(file, data):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(data)
    if file is None:
        plt.show()
    else:
        plt.savefig(file)


def plotSVM(x, o, title="График", label=("x", "y"), color=('#0d1117', "#373b42", "#c9d1d9"), file=None):
    x = list(x)
    x1 = []
    x2 = []
    o1 = []
    o2 = []
    for i in range(len(x)):
        if i in o:
            if x[i][0] == 0:
                o1.append(x[i])
                x1.append(x[i])
            else:
                o2.append(x[i])
                x2.append(x[i])
        else:
            if x[i][0] == 0:
                x1.append(x[i])
            else:
                x2.append(x[i])

    if len(o1) > len(o2):
        x_ = (o1[0][1] - o1[1][1])
        y_ = (o1[0][2] - o1[1][2])
    else:
        x_ = (o2[0][1] - o2[1][1])
        y_ = (o2[0][2] - o2[1][2])
    k = y_ / x_

    b1 = o1[0][2] - o1[0][1] * k
    b2 = o2[0][2] - o2[0][1] * k

    l1 = np.arange(0, 2) * k + b1
    l2 = np.arange(0, 2) * k + b2

    x1 = np.array(x1).transpose()
    x2 = np.array(x2).transpose()

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(color[0])
    ax.patch.set_facecolor(color[0])
    for i in ax.spines:
        ax.spines[i].set_color(color[2])
    ax.tick_params(colors=color[2], which='both')

    ax.scatter(x1[1], x1[2], s=20, color=color[2])
    ax.scatter(x2[1], x2[2], s=20, color=color[1])
    ax.plot(range(2), l1, color=color[2])
    ax.plot(range(2), l2, color=color[1])

    plt.xlabel(label[0], fontsize=14, fontweight="bold", color=color[2])
    plt.ylabel(label[1], fontsize=14, fontweight="bold", color=color[2])
    plt.title = title
    if file is None:
        plt.show()
    else:
        plt.savefig(file)
