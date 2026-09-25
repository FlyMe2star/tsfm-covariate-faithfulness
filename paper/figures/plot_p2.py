"""Render the frozen P2 robustness figures with Matplotlib.

The source tables are checked against the archived P2 report before drawing.
All distances and intervals remain descriptive post-primary evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import ScaledTranslation

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42})
plt.rcParams["font.size"] = 7.5
plt.rcParams["axes.titlesize"] = 8
plt.rcParams["axes.labelsize"] = 7.5
plt.rcParams["xtick.labelsize"] = 7
plt.rcParams["ytick.labelsize"] = 7
plt.rcParams["legend.fontsize"] = 7
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.linewidth"] = 0.7
plt.rcParams["legend.frameon"] = False
plt.rcParams["figure.facecolor"] = "white"

ORACLE = "#333A40"
MODEL = "#0F4D92"
TEAL = "#2A817B"
BOUNDARY = "#A74F2B"
MECHANISMS = {
    "biphasic_rebound": ("Rebound", "#0F4D92"),
    "dispersed_delayed_pulse": ("Delayed pulse", "#2A817B"),
    "rate_asymmetric_hysteresis": ("Hysteresis", "#7F628D"),
    "two_covariate_synergy": ("Synergy", "#A07131"),
}
WIDTH_DOUBLE_IN = 183.0 / 25.4
WIDTH_SINGLE_IN = 89.0 / 25.4


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_verified(source: Path) -> dict[str, Any]:
    report = json.loads((source / "p2_robustness_report.json").read_text(encoding="utf-8"))
    if report["result_status"] != "verified_post_primary_analysis_only":
        raise RuntimeError("P2 report does not have verified post-primary status")
    for name, manifest in report["artifacts"].items():
        path = source / name
        if _sha256(path) != manifest["sha256"]:
            raise RuntimeError(f"P2 artifact hash mismatch: {name}")
        count = len(_read_csv(path)) if name.endswith(".csv") else len(
            json.loads(path.read_text(encoding="utf-8"))
        )
        expected = manifest.get("row_count", manifest.get("item_count"))
        if count != expected:
            raise RuntimeError(f"P2 artifact count mismatch: {name}")
    representatives = json.loads(
        (source / "representative_responses.json").read_text(encoding="utf-8")
    )
    if len(representatives) != 3:
        raise RuntimeError("expected all three registered passing-cell representatives")
    return {
        "report": report,
        "representatives": representatives,
        "monotone": _read_csv(source / "monotone_cell_summary.csv"),
        "thresholds": _read_csv(source / "threshold_sensitivity.csv"),
    }


def _panel_label(ax: Any, label: str) -> None:
    offset = ScaledTranslation(-4 / 72, 5 / 72, ax.figure.dpi_scale_trans)
    ax.text(
        0,
        1,
        label,
        transform=ax.transAxes + offset,
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
    )


def _audit_and_save(
    fig: Any,
    name: str,
    output: Path,
    previews: Path,
    audit_helper_dir: Path | None,
) -> None:
    fig.canvas.draw()
    if audit_helper_dir is not None:
        sys.path.insert(0, str(audit_helper_dir.resolve()))
        alignment = importlib.import_module("audit_panel_alignment")
        alignment.require_matplotlib_panel_alignment(
            fig,
            json_out=output / f"{name}.alignment.json",
            overlay_svg=previews / f"{name}.alignment.svg",
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            require_panel_labels=len(fig.axes) > 1,
            strict=True,
        )
    fig.savefig(output / f"{name}.pdf")
    fig.savefig(output / f"{name}.svg")
    fig.savefig(previews / f"{name}.png", dpi=600)
    plt.close(fig)


def representative_paths(
    data: dict[str, Any], output: Path, previews: Path, audit: Path | None
) -> None:
    rows = data["representatives"]
    expected_cells = [
        "chronos_2/biphasic_rebound",
        "chronos_2/dispersed_delayed_pulse",
        "timesfm_3/biphasic_rebound",
    ]
    if [row["cell"] for row in rows] != expected_cells:
        raise RuntimeError("unexpected representative-cell order")

    fig, axes = plt.subplots(2, 3, figsize=(WIDTH_DOUBLE_IN, 3.30), dpi=300)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.16, top=0.81, wspace=0.34, hspace=0.42)
    titles = ["Chronos-2 / rebound", "Chronos-2 / delayed", "TimesFM-3 / rebound"]
    normalized: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for row in rows:
        oracle = np.asarray(row["oracle_response"], dtype=float)
        model = np.asarray(row["predicted_response"], dtype=float)
        if oracle.size != 24 or model.size != 24:
            raise RuntimeError("representative response must have 24 horizons")
        oracle_coarse = np.asarray(row["oracle_width8"], dtype=float)
        model_coarse = np.asarray(row["predicted_width8"], dtype=float)
        normalized.append(
            (
                oracle / np.sum(np.abs(oracle)),
                model / np.sum(np.abs(model)),
                oracle_coarse / np.sum(np.abs(oracle_coarse)),
                model_coarse / np.sum(np.abs(model_coarse)),
            )
        )

    fine_max = max(float(np.max(np.abs(item))) for row in normalized for item in row[:2])
    coarse_max = max(float(np.max(np.abs(item))) for row in normalized for item in row[2:])
    for column, row in enumerate(rows):
        oracle, model, oracle_coarse, model_coarse = normalized[column]
        upper, lower = axes[:, column]
        upper.plot(np.arange(1, 25), oracle, color=ORACLE, linewidth=1.35, label="Oracle")
        upper.plot(
            np.arange(1, 25), model, color=MODEL, linewidth=1.35, linestyle="--", label="Model"
        )
        for boundary in (8.5, 16.5):
            upper.axvline(boundary, color="#C8CDD2", linewidth=0.55, linestyle=":")
        upper.axhline(0, color="#9BA2A8", linewidth=0.6)
        upper.set_xlim(1, 24)
        upper.set_ylim(-fine_max * 1.12, fine_max * 1.12)
        upper.set_xticks([1, 8, 16, 24])
        upper.set_title(
            f"{titles[column]}\n"
            f"D1={float(row['d1']):.3f}; "
            f"G={float(row['registered_resolution_contrast']):.3f}",
            pad=7,
        )
        _panel_label(upper, "abc"[column])
        if column == 0:
            upper.set_ylabel("Fine signed profile")

        centers = np.arange(1, 4)
        lower.bar(centers - 0.17, oracle_coarse, width=0.31, color=ORACLE)
        lower.bar(centers + 0.17, model_coarse, width=0.31, color=MODEL)
        lower.axhline(0, color="#9BA2A8", linewidth=0.6)
        lower.set_xlim(0.5, 3.5)
        lower.set_ylim(-coarse_max * 1.12, coarse_max * 1.12)
        lower.set_xticks([1, 2, 3])
        lower.set_xlabel("8-step block")
        _panel_label(lower, "def"[column])
        if column == 0:
            lower.set_ylabel("Coarse signed profile")

    fig.legend(
        [Line2D([], [], color=ORACLE, linewidth=1.4),
         Line2D([], [], color=MODEL, linewidth=1.4, linestyle="--")],
        ["Oracle", "Model"],
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.99),
    )
    _audit_and_save(fig, "p2_representative_paths", output, previews, audit)


def resolution_curves(
    data: dict[str, Any], output: Path, previews: Path, audit: Path | None
) -> None:
    rows = data["monotone"]
    if len(rows) != 8 or not all(row["all_series_curves_nonincreasing"] == "True" for row in rows):
        raise RuntimeError("incomplete or nonmonotone P2 cell summary")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_DOUBLE_IN, 2.98), dpi=300, sharey=True)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.35, top=0.87, wspace=0.12)
    passing = {
        "chronos_2/biphasic_rebound",
        "chronos_2/dispersed_delayed_pulse",
        "timesfm_3/biphasic_rebound",
    }
    widths = (1, 2, 4, 8)
    for ax, backbone, label, title in zip(
        axes,
        ("chronos_2", "timesfm_3"),
        ("a", "b"),
        ("Chronos-2", "TimesFM-3"),
        strict=True,
    ):
        selected = [row for row in rows if row["backbone"] == backbone]
        if len(selected) != 4:
            raise RuntimeError(f"incomplete mechanism set for {backbone}")
        for row in selected:
            mechanism = row["mechanism"]
            color = MECHANISMS[mechanism][1]
            center = np.asarray([float(row[f"monotone_c{w}_estimate"]) for w in widths])
            lower = np.asarray([float(row[f"monotone_c{w}_lower"]) for w in widths])
            upper = np.asarray([float(row[f"monotone_c{w}_upper"]) for w in widths])
            if np.any(lower > center) or np.any(center > upper):
                raise RuntimeError("invalid P2 confidence interval")
            passed = f"{backbone}/{mechanism}" in passing
            ax.errorbar(
                np.arange(4),
                center,
                yerr=np.vstack((center - lower, upper - center)),
                color=color,
                linestyle="-" if passed else "--",
                linewidth=1.25,
                marker="o" if passed else "s",
                markersize=2.8,
                capsize=1.5,
                elinewidth=0.65,
            )
        ax.set_xticks(range(4), [str(w) for w in widths])
        ax.set_xlim(-0.17, 3.17)
        ax.set_ylim(0, 0.49)
        ax.set_xlabel("Block width (steps)")
        ax.set_title(title)
        _panel_label(ax, label)
        ax.spines["left"].set_visible(True)
        if ax is axes[0]:
            ax.set_ylabel("Fixed-fine signed distance")
    handles = [
        Line2D([], [], color=color, marker="o", linewidth=1.2, markersize=3)
        for _, color in MECHANISMS.values()
    ]
    handles.extend(
        [
            Line2D([], [], color=ORACLE, linestyle="-", marker="o", linewidth=1),
            Line2D([], [], color=ORACLE, linestyle="--", marker="s", linewidth=1),
        ]
    )
    labels = [item[0] for item in MECHANISMS.values()] + ["P1 passing", "Other"]
    fig.legend(handles, labels, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.01))
    _audit_and_save(fig, "p2_resolution_curves", output, previews, audit)


def threshold_sensitivity(
    data: dict[str, Any], output: Path, previews: Path, audit: Path | None
) -> None:
    rows = data["thresholds"]
    d1_values = sorted({float(row["d1_lower_threshold"]) for row in rows})
    g_values = sorted({float(row["registered_g_lower_threshold"]) for row in rows})
    if len(rows) != 56 or len(d1_values) != 7 or len(g_values) != 8:
        raise RuntimeError("threshold grid is incomplete")
    lookup = {
        (float(row["d1_lower_threshold"]), float(row["registered_g_lower_threshold"])): row
        for row in rows
    }
    colors = {0: "#F1F3F5", 1: "#B9CDE0", 2: "#6E9BC1", 3: "#0F4D92"}
    fig, ax = plt.subplots(figsize=(WIDTH_SINGLE_IN, 2.75), dpi=300)
    fig.subplots_adjust(left=0.20, right=0.99, bottom=0.24, top=0.83)
    for y, d1 in enumerate(d1_values):
        for x, g in enumerate(g_values):
            row = lookup[(d1, g)]
            count = int(row["passing_cell_count"])
            ax.add_patch(
                Rectangle((x, y), 1, 1, facecolor=colors[count], edgecolor="white", linewidth=0.65)
            )
            ax.text(
                x + 0.5,
                y + 0.5,
                str(count),
                ha="center",
                va="center",
                color="white" if count == 3 else ORACLE,
                fontsize=7.5,
            )
            if d1 == 0.08 and g == 0.03:
                ax.add_patch(
                    Rectangle(
                        (x + 0.025, y + 0.025),
                        0.95,
                        0.95,
                        fill=False,
                        edgecolor=BOUNDARY,
                        linewidth=1.55,
                    )
                )
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(8) + 0.5, [f"{value:.2f}" for value in g_values])
    ax.set_yticks(np.arange(7) + 0.5, [f"{value:.2f}" for value in d1_values])
    ax.set_xlabel("Registered G lower-bound threshold")
    ax.set_ylabel("D1 lower-bound threshold")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.legend(
        [Patch(facecolor="#6E9BC1"), Patch(facecolor="#F1F3F5"),
         Patch(facecolor="white", edgecolor=BOUNDARY, linewidth=1.5)],
        ["2-3 cells", "0-1 cell", "Frozen"],
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.60, 0.985),
    )
    _audit_and_save(fig, "p2_threshold_sensitivity", output, previews, audit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview-dir", type=Path, required=True)
    parser.add_argument("--audit-helper-dir", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.preview_dir.mkdir(parents=True, exist_ok=True)
    data = load_verified(args.source)
    representative_paths(data, args.output, args.preview_dir, args.audit_helper_dir)
    resolution_curves(data, args.output, args.preview_dir, args.audit_helper_dir)
    threshold_sensitivity(data, args.output, args.preview_dir, args.audit_helper_dir)
    print("Rendered three verified P2 figure pairs (PDF and SVG).")


if __name__ == "__main__":
    main()
