from matplotlib import pyplot as plt
import random


categories = [
    "writing",
    "roleplay",
    "reasoning",
    "math",
    "coding",
    "extraction",
    "stem",
    "humanities",
]


def generate_inputid(model, num_tokens, batch_size):
    # Sample pool of words and punctuation
    # more words in words

    # Initialize an empty list to hold the prompt tokens
    input_ids = []
    batch_num = num_tokens // batch_size
    # Keep adding words until the target token count is reached
    for i in range(batch_num):
        token_ids = []
        while len(token_ids) < batch_size:
            token_ids.append(random.randint(0, model.vocab_size - 1))
        input_ids.append(token_ids)
    return input_ids
    # Join the list of tokens into a single string to form the prompt


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
