from matplotlib import pyplot as plt


def plot(x, y, filename, ylabel, xlabel):
    plt.figure()
    plt.scatter(x, y)
    plt.plot(
        x,
        y,
        linestyle="-",
        label="batched inference",
    )
    baseline = [y[0] * i for i in x]
    plt.scatter(x, baseline)
    plt.plot(
        x,
        baseline,
        linestyle="-",
        label="non-batched inference",
    )
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend()
    plt.savefig("../../results/" + filename + ".png")
