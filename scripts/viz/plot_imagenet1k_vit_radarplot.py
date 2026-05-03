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

original_datasets = ["NINCO", "OpenImage-O", "ImageNet-C", "ImageNet-ES", "ImageNet-R", "ImageNet-V2"]
datasets = ["OpenImage-O", "NINCO", "ImageNet-C", "ImageNet-ES", "ImageNet-R", "ImageNet-V2"]

results = {
    "MSP":      [83.28, 89.23, 71.21, 69.35, 77.63, 58.73],
    "ODIN":     [74.28, 81.93, 64.61, 63.22, 67.90, 54.82],
    "Energy":   [83.23, 89.16, 72.34, 67.54, 72.87, 57.29],
    "ReAct":    [85.62, 92.76, 73.55, 69.57, 78.19, 58.65],
    "KNN":      [40.44, 50.34, 57.73, 55.88, 65.61, 49.80],
    "Maha":     [85.77, 93.54, 70.94, 69.53, 86.23, 58.29],
    "rMaha":    [88.51, 94.62, 71.73, 71.30, 85.02, 59.18],
    "Maha++":   [90.37, 95.81, 74.86, 73.51, 87.22, 59.52],
    "rMaha++":  [90.48, 95.47, 72.49, 71.90, 85.93, 59.08],
    "X-Maha":   [94.98, 98.21, 83.96, 76.78, 88.49, 58.36],
    "MM++":     [91.34, 96.43, 84.35, 81.70, 88.84, 60.78],
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
    "MM++":    "#ff0000",
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

ax.set_ylim(0.40, 1.0)
ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
ax.set_yticklabels(["0.5", "0.6", "0.7", "0.8", "0.9"], fontsize=24, color=TEXT_COLOR)
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
#     "ID: ImageNet-LT   |   Model: ViT-B/16",
#     ha="right", va="bottom",
#     fontsize=24, color=TEXT_COLOR,
# )

plt.tight_layout()
repo_root = Path(__file__).resolve().parents[2]
out_png = repo_root / "assets" / "radarplot_imagenet1k_vitb16.png"
out_pdf = repo_root / "assets" / "radarplot_imagenet1k_vitb16.pdf"
plt.savefig(out_png, dpi=OUTPUT_DPI, bbox_inches="tight")
plt.savefig(out_pdf, dpi=OUTPUT_DPI, bbox_inches="tight")
print(f"saved: {out_png}")
print(f"saved: {out_pdf}")
