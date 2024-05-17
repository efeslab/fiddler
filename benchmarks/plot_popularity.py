import json
import matplotlib.pyplot as plt
import numpy as np

subjects = ['high_school_us_history', 'high_school_biology', 'college_computer_science', 'machine_learning', 'college_medicine', 'international_law', 'high_school_mathematics']
fig, axes = plt.subplots(len(subjects), 1, figsize=(10, 15), sharex=True)
for subject, ax in zip(subjects, axes):
    with open(f"results/popularity/expert_popularity_{subject}.json", "r") as f:
        results = json.load(f)
        # print(f"Expert popularity for {subject}: {results}")
        expert_popularity = results["expert_popularity"]
    num_elayers, num_experts = len(expert_popularity), len(expert_popularity[0])
    skewness = np.zeros(num_elayers)
    for layer_idx in range(num_elayers):
        xs = np.arange(num_experts) / num_experts + layer_idx
        ys = np.array(expert_popularity[layer_idx])
        ys = ys / ys.sum()
        ys = np.sort(ys)[::-1]
        skewness[layer_idx] = np.abs(ys[0] / ys[-1])
        # print(xs, ys)
        ax.bar(xs, ys, width=1/num_experts, align="edge", label=f"Expert Layer {layer_idx}")
    print(f"Skewness for {subject}: {skewness} mean: {skewness.mean()} max {skewness.max()} min {skewness.min()}")  
    ax.set_xlabel(f"{num_experts} Experts on each of {num_elayers} Layers")
    ax.set_ylabel("Frequency")
    ax.set_xlim(0, num_elayers)
    ax.title.set_text(subject)
fig.tight_layout()
# fig.savefig("results/popularity/expert_popularity.png")