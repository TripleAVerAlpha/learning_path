import matplotlib.pyplot as plt


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
