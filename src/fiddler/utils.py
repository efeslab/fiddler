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


def plot_batch(batch_size):
    with open(f"../../results/pp_{batch_size}.txt") as f:
        data = f.readlines()
        print(data)
        data = [x.strip() for x in data]
        print(data)


if __name__ == "__main__":
    plot_batch(8)
