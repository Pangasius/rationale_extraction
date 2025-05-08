import matplotlib.pyplot as plt
import numpy as np

# Data
species = ("1 - 3", "3 - 6", "6 - 10", "11+")
penguin_means = {
    r"Threshold Bert-large": [(0.61, 0.53, 0.54, 0.37), (0.11, 0.10, 0.12, 0.11), "^"],
    r"Threshold SFR": [(0.66, 0.61, 0.53, 0.47), (0.11, 0.10, 0.14, 0.11), ">"],
    r"Threshold TF-IDF": [(0.69, 0.67, 0.66, 0.52), (0.09, 0.09, 0.11, 0.14), "o"],
    r"Top-K TF-IDF": [(0.71, 0.71, 0.63, 0.44), (0.08, 0.09, 0.12, 0.14), "."],
    r"Top-K Bert-large": [(0.70, 0.64, 0.56, 0.40), (0.09, 0.09, 0.12, 0.15), "v"],
    r"Top-K SFR": [(0.70, 0.70, 0.60, 0.50), (0.09, 0.08, 0.12, 0.14), "<"],
    r"Attention Gemma base": [(0.76, 0.50, 0.41, 0.29), (0.07, 0.07, 0.11, 0.14), "+"],
    r"Attention Gemma FT": [(0.70, 0.63, 0.56, 0.30), (0.09, 0.09, 0.14, 0.16), "p"],
    r"Attention Gemma RL": [(0.74, 0.61, 0.59, 0.30), (0.08, 0.09, 0.13, 0.16), "P"],
    r"Classifier DistilBERT": [(0.80, 0.58, 0.63, 0.48), (0.10, 0.11, 0.13, 0.18), "d"],
    r"Classifier RoBERTa": [(0.86, 0.74, 0.77, 0.50), (0.08, 0.10, 0.11, 0.17), "D"],
    r"Classifier Gemma": [(0.89, 0.79, 0.74, 0.62), (0.07, 0.08, 0.10, 0.14), "*"],
}

# Plotting
fig, ax = plt.subplots(figsize=(12, 8))

# Define markers for different methods

# Colors for different methods
colors = plt.colormaps["gist_yarg"]

# Horizontal offset for spreading markers
offset = np.linspace(-0.2, 0.2, len(penguin_means))

# Plot each method with horizontal offset
for idx, (method, data) in enumerate(penguin_means.items()):
    means, std_devs, marker = data
    x = np.arange(len(species)) + offset[idx]  # Apply the offset to each method
    ax.errorbar(x, means, yerr=std_devs, fmt=marker, label=method, color=colors(50 + idx * 15), capsize=5)

# Customize the plot
ax.set_title('IoU Score Comparison by method and sentence count')
ax.set_xlabel('Number of sentences')
ax.set_ylabel('IoU Score')
ax.set_xticks(np.arange(len(species)))
ax.set_xticklabels(species)
ax.legend(loc='best')
ax.grid(True)
plt.xticks(rotation=45)

# Show plot
plt.tight_layout()
plt.show()

# Save plot
fig.savefig('iou_score_comparison.pdf')
