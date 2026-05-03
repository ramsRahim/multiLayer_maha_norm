from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DPI = 600
FIGSIZE = (13, 11)
TEXT_COLOR = "#111111"
GRID_COLOR = "#b8b8b8"

plt.rcParams.update({
    "font.size": 24,
    "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "text.color": TEXT_COLOR,
})

original_datasets = ["ImageNet-O", "NINCO", "OpenImage-O", "Places365", "Textures"]
datasets = ["OpenImage-O", "ImageNet-O", "NINCO", "Places365", "Textures"]

results = {
    "MSP":      [73.63, 82.72, 80.44, 92.57, 84.75],
    "ODIN":     [65.42, 72.78, 69.69, 85.80, 77.51],
    "Energy":   [75.97, 83.53, 73.16, 91.59, 83.25],
    "ReAct":    [79.20, 85.53, 78.37, 95.54, 84.81],
    "KNN":      [84.98, 89.56, 85.15, 96.57, 87.48],
    "Maha":     [83.65, 85.43, 84.57, 96.98, 85.97],
    "rMaha":    [84.42, 86.89, 85.28, 97.61, 87.41],
    "Maha++":   [86.42, 89.26, 85.75, 98.76, 88.90],
    "rMaha++":  [86.45, 89.31, 85.77, 98.77, 88.93],
    "X-Maha":   [93.76, 96.65, 92.04, 99.40, 89.64],
    "MM++":     [88.76, 97.00, 87.05, 98.43, 89.64],
}

N = len(datasets)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=FIGSIZE, subplot_kw=dict(polar=True))

palette = {
    "MSP":     "#4c72b0",
    "ODIN":    "#c68642",
    "Energy":  "#d1504b",
    "ReAct":   "#4a9ea0",
    "KNN":     "#bfbfbf",
    "Maha":    "#8c8c3c",
    "rMaha":   "#2ec4b6",
    "Maha++":  "#e07b00",
    "rMaha++": "#e377c2",
    "X-Maha":  "#1b7837",
    "MM++":    "#d62728",
}

for method, vals in results.items():
    values_by_dataset = dict(zip(original_datasets, vals))
    values = [values_by_dataset[dataset] / 100.0 for dataset in datasets]
    values += values[:1]
    is_ours = method == "MM++"
    label = "MM++ (Ours)" if is_ours else method
    ax.plot(
        angles, values,
        color=palette[method],
        linewidth=1.8,
        label=label,
        zorder=5 if is_ours else 2,
    )

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(datasets, fontsize=24, color=TEXT_COLOR)

ax.set_ylim(0.60, 1.0)
ax.set_yticks([0.7, 0.8, 0.9, 1.0])
ax.set_yticklabels(["0.7", "0.8", "0.9", "1.0"], fontsize=24, color=TEXT_COLOR)
ax.yaxis.grid(True, linestyle="-", color=GRID_COLOR, linewidth=1.0)
ax.xaxis.grid(True, linestyle="-", color=GRID_COLOR, linewidth=1.0)
ax.spines["polar"].set_color("#999999")
ax.spines["polar"].set_linewidth(1.2)

ax.legend(
    loc="center left",
    bbox_to_anchor=(1.18, 0.5),
    frameon=True,
    fontsize=24,
    labelcolor=TEXT_COLOR,
)

# fig.text(
#     0.98, 0.02,
#     "ID: ImageNet-1K   |   Model: ViT-B/16",
#     ha="right", va="bottom",
#     fontsize=24, color=TEXT_COLOR,
# )

plt.tight_layout()
repo_root = Path(__file__).resolve().parents[2]
out_png = repo_root / "assets" / "radarplot_imagenetLT_vitb16.png"
out_pdf = repo_root / "assets" / "radarplot_imagenetLT_vitb16.pdf"
plt.savefig(out_png, dpi=OUTPUT_DPI, bbox_inches="tight")
plt.savefig(out_pdf, dpi=OUTPUT_DPI, bbox_inches="tight")
print(f"saved: {out_png}")
print(f"saved: {out_pdf}")
