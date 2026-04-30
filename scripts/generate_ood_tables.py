#!/usr/bin/env python3
"""Generate OOD benchmark tables from cache/scores/metrics.json."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


METHODS = [
    "MSP",
    "Energy",
    "Energy+React",
    "ODIN",
    "Mahalanobis",
    "Mahalanobis_norm",
    "Relative_Mahalanobis",
    "knn",
    "MM_plus_plus",
    "MM_plus_plus_topk_cat",
]

METRICS = {
    "auroc": "AUROC",
    "fpr_at_95": "FPR95",
}

ID_DATASET_LABELS = {
    "ImageNet1K": "ImageNet-1K",
}

OOD_DATASETS = [
    ("imagenet_o", "ImageNet-O"),
    ("NINCO_OOD_classes", "NINCO"),
    ("openimages_o", "OpenImage-O"),
    ("places365", "Places365"),
    ("texture", "Textures"),
]

LATEX_METHOD_GROUPS = [
    [
        ("MSP", "MSP", False),
        ("ODIN", "ODIN", False),
        ("Energy", "Energy", False),
        ("Energy+React", "ReAct", False),
        ("knn", "KNN", False),
    ],
    [
        ("Mahalanobis", "Maha", False),
        ("Relative_Mahalanobis", "rMaha", False),
        ("Mahalanobis_norm", "Maha++", False),
    ],
    [
        ("MM_plus_plus_topk_cat", "MM++ (Ours)", True),
    ],
]

MODEL_CHECKPOINTS = {
    # "ViT-B16-In21k-augreg": "vit_base_patch16_224.augreg_in21k_ft_in1k",
    # "ViT-L16-In21k-augreg": "vit_large_patch16_224.augreg_in21k_ft_in1k",
    # "ViT-T16-In21k-augreg": "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
    # "ViT-S16-In21k-augreg": "vit_small_patch16_224.augreg_in21k_ft_in1k",
    # "ViT-B16-augreg": "vit_base_patch16_224.augreg_in1k",
    # "ViT-S16-augreg": "vit_small_patch16_224.augreg_in1k",
    # "ViT-so400M-SigLip": "vit_so400m_patch14_siglip.378.webli_ft_in1k",
    # "ViT-H14-CLIP-L2b-In12k": "vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k",
    # "ViT-L14-CLIP-L2b-In12k": "vit_large_patch14_clip_336.laion2b_ft_in12k_in1k",
    # "ViT-B16-In21k-orig": "vit_base_patch16_224.orig_in21k_ft_in1k",
    # "ViT-L16-In21k-orig": "vit_large_patch32_384.orig_in21k_ft_in1k",
    # "ViT-B16-In21k-mil": "vit_base_patch16_224.mil_in21k_ft_in1k",
    # "ViT-B16-In21k-augreg2": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
    # "ViT-B16-CLIP-L2b-In12k": "vit_base_patch16_clip_224.laion2b_ft_in12k_in1k",
    # "EVA02-L14-M38m-In21k": "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
    # "EVA02-B14-In21k": "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k",
    # "EVA02-S14": "eva02_small_patch14_336.mim_in22k_ft_in1k",
    # "EVA02-T14": "eva02_tiny_patch14_336.mim_in22k_ft_in1k",
    # "DeiT3-B16": "deit3_base_patch16_224",
    # "DeiT3-B16-In21k": "deit3_base_patch16_224_in21ft1k",
    # "DeiT3-L16-In21k": "deit3_large_patch16_384.fb_in22k_ft_in1k",
    # "DeiT3-L16": "deit3_large_patch16_384.fb_in1k",
    "DeiT3-B16-In1k": "deit3_base_patch16_384.fb_in1k",
    # "DeiT3-S16-In21k": "deit3_small_patch16_384.fb_in22k_ft_in1k",
    # "DeiT3-S16": "deit3_small_patch16_384.fb_in1k",
    "Swin-T": "swin_tiny_patch4_window7_224.ms_in1k",
    # "SwinV2-S": "swinv2_small_window16_256.ms_in1k",
    # "SwinV2-B": "swinv2_base_window16_256.ms_in1k",
    # "SwinV2-L-In21k": "swinv2_large_window12to24_192to384.ms_in22k_ft_in1k",
    # "SwinV2-B-In21k": "swinv2_base_window12to24_192to384.ms_in22k_ft_in1k",
    # "ResNet50": "resnet50.tv2_in1k",
    # "ResNet101": "resnet101.tv2_in1k",
    # "ResNet152": "resnet152.tv2_in1k",
    # "ResNet50-supcon": "r50supcon",
    "ConvNeXt-T": "convnext_tiny.fb_in1k",
    # "ConvNeXt-B": "convnext_base.fb_in1k",
    # "ConvNeXt-B-In21k": "convnext_base.fb_in22k_ft_in1k",
    # "ConvNeXtV2-L-In21k": "convnextv2_large.fcmae_ft_in22k_in1k_384",
    # "ConvNeXtV2-B-In21k": "convnextv2_base.fcmae_ft_in22k_in1k_384",
    # "ConvNeXtV2-T-In21k": "convnextv2_tiny.fcmae_ft_in22k_in1k_384",
    # "ConvNeXtV2-T": "convnextv2_tiny.fcmae_ft_in1k",
    # "ConvNeXtV2-B": "convnextv2_base.fcmae_ft_in1k",
    # "ConvNeXtV2-L": "convnextv2_large.fcmae_ft_in1k",
    # "Mixer-B16-In21k": "mixer_b16_224.goog_in21k_ft_in1k",
    # "EffNetV2-M": "tf_efficientnetv2_m.in1k",
    # "EffNetV2-S": "tf_efficientnetv2_s.in1k",
    # "EffNetV2-L": "tf_efficientnetv2_l.in1k",
}

CHECKPOINT_ALIASES = {
    "vit_so400m_patch14_siglip.378.webli_ft_in1k": [
        "vit_so400m_patch14_siglip_378.webli_ft_in1k",
    ],
    "vit_base_patch16_224.mil_in21k_ft_in1k": [
        "vit_base_patch16_224_miil.in21k_ft_in1k",
    ],
    "deit3_base_patch16_224_in21ft1k": [
        "deit3_base_patch16_224.fb_in22k_ft_in1k",
    ],
    "r50supcon": [
        "rn50supcon",
    ],
}


@dataclass(frozen=True)
class ModelRow:
    label: str
    checkpoint: str


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_model_names(path: Path) -> set[str]:
    if not path.exists():
        return set()
    text = path.read_text(encoding="utf-8")
    return set(re.findall(r'"([^"]+)"', text))


def available_models(metrics: dict[str, Any], dataset_keys: list[str]) -> set[str]:
    models: set[str] = set()
    for dataset_key in dataset_keys:
        models.update(metrics.get(dataset_key, {}).keys())
    return models


def checkpoint_candidates(checkpoint: str) -> list[str]:
    return [checkpoint, *CHECKPOINT_ALIASES.get(checkpoint, [])]


def resolve_model_rows(
    metrics: dict[str, Any],
    dataset_keys: list[str],
    all_model_names_path: Path,
    include_extra_models: bool,
) -> tuple[list[ModelRow], list[str]]:
    available = available_models(metrics, dataset_keys)
    script_models = parse_model_names(all_model_names_path)
    rows: list[ModelRow] = []
    notes: list[str] = []
    used: set[str] = set()

    for label, checkpoint in MODEL_CHECKPOINTS.items():
        match = next((candidate for candidate in checkpoint_candidates(checkpoint) if candidate in available), None)
        if match is None:
            known_candidates = [candidate for candidate in checkpoint_candidates(checkpoint) if candidate in script_models]
            if known_candidates:
                notes.append(f"Missing metrics for {label}: {', '.join(known_candidates)}")
            continue
        rows.append(ModelRow(label=label, checkpoint=match))
        used.add(match)
        if match != checkpoint:
            notes.append(f"Resolved {label}: {checkpoint} -> {match}")

    extras = sorted(available - used)
    if include_extra_models:
        for checkpoint in extras:
            rows.append(ModelRow(label=checkpoint, checkpoint=checkpoint))
    if extras:
        notes.append(
            "Extra cache models not in MODEL_CHECKPOINTS: "
            + ", ".join(extras)
            + (" (included)" if include_extra_models else " (skipped)")
        )

    return rows, notes


def metric_entry(
    metrics: dict[str, Any],
    dataset_key: str,
    checkpoint: str,
    method: str,
) -> dict[str, Any] | None:
    entry = metrics.get(dataset_key, {}).get(checkpoint, {}).get(method)
    return entry if isinstance(entry, dict) else None


def collect_val_acc(metrics: dict[str, Any], checkpoint: str, dataset_keys: list[str]) -> float | None:
    values: list[float] = []
    for dataset_key in dataset_keys:
        methods = metrics.get(dataset_key, {}).get(checkpoint, {})
        for entry in methods.values():
            if isinstance(entry, dict) and entry.get("val_acc") is not None:
                values.append(float(entry["val_acc"]))
    return values[0] if values else None


def format_number(value: float | None, scale: str, decimals: int, missing: str) -> str:
    if value is None:
        return missing
    if scale == "percent":
        value *= 100.0
    return f"{value:.{decimals}f}"


def metric_value(
    metrics: dict[str, Any],
    dataset_key: str,
    checkpoint: str,
    method: str,
    metric_name: str,
) -> float | None:
    entry = metric_entry(metrics, dataset_key, checkpoint, method)
    if entry is None or entry.get(metric_name) is None:
        return None
    return float(entry[metric_name])


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def latex_label(text: str) -> str:
    label = re.sub(r"[^A-Za-z0-9]+", "_", text.lower()).strip("_")
    return label or "model"


def flatten_latex_methods() -> list[tuple[str, str, bool]]:
    return [method for group in LATEX_METHOD_GROUPS for method in group]


def average(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def latex_metric_row_values(
    metrics: dict[str, Any],
    model: ModelRow,
    method: str,
    dataset_pairs: list[tuple[str, str]],
) -> list[float | None]:
    values: list[float | None] = []
    aurocs: list[float | None] = []
    fprs: list[float | None] = []
    for dataset_key, _ in dataset_pairs:
        auroc = metric_value(metrics, dataset_key, model.checkpoint, method, "auroc")
        fpr = metric_value(metrics, dataset_key, model.checkpoint, method, "fpr_at_95")
        aurocs.append(auroc)
        fprs.append(fpr)
        values.extend([auroc, fpr])
    values.extend([average(aurocs), average(fprs)])
    return values


def latex_value(value: float | None, scale: str, decimals: int, missing: str) -> str:
    if value is None:
        return latex_escape(missing)
    return latex_escape(format_number(value, scale, decimals, missing))


def latex_caption(model: ModelRow, id_dataset: str, scale: str) -> str:
    id_dataset_label = ID_DATASET_LABELS.get(id_dataset, id_dataset)
    value_sentence = (
        r"All values are reported in \%."
        if scale == "percent"
        else "All values are reported as fractions."
    )
    return (
        "OOD detection performance using "
        f"{latex_escape(model.label)} ({latex_escape(model.checkpoint)}) "
        f"pretrained on {latex_escape(id_dataset_label)}.\n"
        rf"($\uparrow$: higher is better; $\downarrow$: lower is better. {value_sentence})"
    )


def latex_table(
    metrics: dict[str, Any],
    model: ModelRow,
    dataset_pairs: list[tuple[str, str]],
    id_dataset: str,
    scale: str,
    decimals: int,
    missing: str,
) -> str:
    method_specs = flatten_latex_methods()
    row_values = [
        latex_metric_row_values(metrics, model, method, dataset_pairs)
        for method, _, _ in method_specs
    ]

    tabular_columns = "l " + " ".join(["cc"] * (len(dataset_pairs) + 1))
    multicolumn_header = [
        r"\multirow{2}{*}{Method} &",
        " &\n".join(
            rf"\multicolumn{{2}}{{c}}{{{latex_escape(dataset_label)}}}"
            for _, dataset_label in dataset_pairs
        ),
        "&",
        r"\multicolumn{2}{c}{Average} \\ ",
    ]
    clines = " ".join(
        rf"\cline{{{start}-{start + 1}}}"
        for start in range(2, 2 * (len(dataset_pairs) + 1) + 1, 2)
    )
    metric_header = "\n".join(
        [r"& AUROC $\uparrow$ & FPR95 $\downarrow$" for _ in range(len(dataset_pairs) + 1)]
    )

    lines = [
        r"\begin{table*}[htb]",
        r"\centering",
        rf"\caption{{{latex_caption(model, id_dataset, scale)}}} ",
        rf"\label{{tab:{latex_label(model.label)}_results}}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{tabular_columns}}}",
        r"\toprule",
        " ".join(multicolumn_header),
        clines + " ",
        r"\noalign{\vskip 3pt}",
        metric_header + r" \\  ",
        r"\midrule",
    ]

    row_index = 0
    for group_index, group in enumerate(LATEX_METHOD_GROUPS):
        if group_index > 0:
            lines.append(r"\midrule")
        for method, display_name, is_ours in group:
            values = row_values[row_index]
            styled_values = [
                latex_value(value, scale, decimals, missing)
                for value in values
            ]
            method_label = latex_escape(display_name)
            lines.append(f"{method_label} & " + " & ".join(styled_values) + r" \\")
            row_index += 1

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines)


def write_latex_tables(
    metrics: dict[str, Any],
    model_rows: list[ModelRow],
    output_dir: Path,
    dataset_pairs: list[tuple[str, str]],
    id_dataset: str,
    scale: str,
    decimals: int,
    missing: str,
) -> Path:
    path = output_dir / "ood_latex_tables.txt"
    tables = [
        latex_table(metrics, model, dataset_pairs, id_dataset, scale, decimals, missing)
        for model in model_rows
    ]
    path.write_text("\n\n".join(tables) + "\n", encoding="utf-8")
    return path


def make_dataset_metric_table(
    metrics: dict[str, Any],
    model_rows: list[ModelRow],
    dataset_key: str,
    metric_name: str,
    dataset_keys: list[str],
    scale: str,
    decimals: int,
    missing: str,
) -> tuple[list[str], list[list[str]]]:
    val_acc_label = "Val Acc (%)" if scale == "percent" else "Val Acc"
    header = ["Model", "Checkpoint", val_acc_label, *METHODS]
    rows: list[list[str]] = []
    for model in model_rows:
        val_acc = collect_val_acc(metrics, model.checkpoint, dataset_keys)
        row = [
            model.label,
            model.checkpoint,
            format_number(val_acc, scale, decimals, missing),
        ]
        for method in METHODS:
            row.append(
                format_number(
                    metric_value(metrics, dataset_key, model.checkpoint, method, metric_name),
                    scale,
                    decimals,
                    missing,
                )
            )
        rows.append(row)
    return header, rows


def make_method_metric_table(
    metrics: dict[str, Any],
    model_rows: list[ModelRow],
    method: str,
    metric_name: str,
    dataset_pairs: list[tuple[str, str]],
    dataset_keys: list[str],
    scale: str,
    decimals: int,
    missing: str,
) -> tuple[list[str], list[list[str]]]:
    val_acc_label = "Val Acc (%)" if scale == "percent" else "Val Acc"
    header = ["Model", "Checkpoint", val_acc_label, *[label for _, label in dataset_pairs]]
    rows: list[list[str]] = []
    for model in model_rows:
        val_acc = collect_val_acc(metrics, model.checkpoint, dataset_keys)
        row = [
            model.label,
            model.checkpoint,
            format_number(val_acc, scale, decimals, missing),
        ]
        for dataset_key, _ in dataset_pairs:
            row.append(
                format_number(
                    metric_value(metrics, dataset_key, model.checkpoint, method, metric_name),
                    scale,
                    decimals,
                    missing,
                )
            )
        rows.append(row)
    return header, rows


def write_by_dataset_tables(
    metrics: dict[str, Any],
    model_rows: list[ModelRow],
    output_dir: Path,
    dataset_pairs: list[tuple[str, str]],
    scale: str,
    decimals: int,
    missing: str,
) -> list[Path]:
    dataset_keys = [key for key, _ in dataset_pairs]
    csv_paths: list[Path] = []

    for dataset_key, _ in dataset_pairs:
        for metric_name in METRICS:
            header, rows = make_dataset_metric_table(
                metrics,
                model_rows,
                dataset_key,
                metric_name,
                dataset_keys,
                scale,
                decimals,
                missing,
            )
            csv_path = output_dir / f"by_dataset_{dataset_key}_{metric_name}.csv"
            write_csv(csv_path, header, rows)
            csv_paths.append(csv_path)

    return csv_paths


def write_by_method_tables(
    metrics: dict[str, Any],
    model_rows: list[ModelRow],
    output_dir: Path,
    dataset_pairs: list[tuple[str, str]],
    scale: str,
    decimals: int,
    missing: str,
) -> list[Path]:
    dataset_keys = [key for key, _ in dataset_pairs]
    csv_paths: list[Path] = []

    for method in METHODS:
        safe_method = re.sub(r"[^A-Za-z0-9_.-]+", "_", method)
        for metric_name in METRICS:
            header, rows = make_method_metric_table(
                metrics,
                model_rows,
                method,
                metric_name,
                dataset_pairs,
                dataset_keys,
                scale,
                decimals,
                missing,
            )
            csv_path = output_dir / f"by_method_{safe_method}_{metric_name}.csv"
            write_csv(csv_path, header, rows)
            csv_paths.append(csv_path)

    return csv_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-json", type=Path, default=Path("cache/scores/metrics.json"))
    parser.add_argument("--id-dataset", default="ImageNet1K")
    parser.add_argument("--all-model-names", type=Path, default=Path("scripts/all_model_names.sh"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--layout",
        choices=["by-dataset", "by-method", "both"],
        default="by-dataset",
        help=(
            "Controls CSV exports. by-dataset writes 10 CSV files: 5 OOD datasets x 2 metrics. "
            "by-method writes one CSV file per method and metric with datasets as columns. "
            "The LaTeX text file is always written."
        ),
    )
    parser.add_argument(
        "--scale",
        choices=["percent", "fraction"],
        default="percent",
        help="Format metric values and val_acc as percentages or raw fractions.",
    )
    parser.add_argument("--decimals", type=int, default=2)
    parser.add_argument("--missing", default="NA")
    parser.add_argument(
        "--skip-extra-models",
        action="store_true",
        help="Only include models that resolve from MODEL_CHECKPOINTS.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_json(args.metrics_json)
    if args.id_dataset not in data:
        raise KeyError(f"{args.id_dataset!r} not found in {args.metrics_json}")

    metrics = data[args.id_dataset]
    dataset_pairs = [(key, label) for key, label in OOD_DATASETS if key in metrics]
    if not dataset_pairs:
        raise KeyError(f"None of the expected OOD datasets were found under {args.id_dataset!r}")

    dataset_keys = [key for key, _ in dataset_pairs]
    model_rows, notes = resolve_model_rows(
        metrics=metrics,
        dataset_keys=dataset_keys,
        all_model_names_path=args.all_model_names,
        include_extra_models=not args.skip_extra_models,
    )
    if not model_rows:
        raise RuntimeError("No model rows resolved from metrics cache.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    if args.layout in {"by-dataset", "both"}:
        output_paths.extend(
            write_by_dataset_tables(
                metrics,
                model_rows,
                args.output_dir,
                dataset_pairs,
                args.scale,
                args.decimals,
                args.missing,
            )
        )
    if args.layout in {"by-method", "both"}:
        output_paths.extend(
            write_by_method_tables(
                metrics,
                model_rows,
                args.output_dir,
                dataset_pairs,
                args.scale,
                args.decimals,
                args.missing,
            )
        )
    output_paths.append(
        write_latex_tables(
            metrics,
            model_rows,
            args.output_dir,
            dataset_pairs,
            args.id_dataset,
            args.scale,
            args.decimals,
            args.missing,
        )
    )

    print("Wrote:")
    for output_path in output_paths:
        print(f"  {output_path}")
    if notes:
        print("\nNotes:")
        for note in notes:
            print(f"  {note}")


if __name__ == "__main__":
    main()
