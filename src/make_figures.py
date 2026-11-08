"""Produce every figure used in the report.

Run after the training scripts:
    python src/make_figures.py
"""
from __future__ import annotations

import csv
import json
import os
import shutil
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = "figures"
RES = "results"

# A clean, print-friendly palette.
C_PRIMARY = "#2b6cb0"
C_ACCENT = "#c05621"
C_GOOD = "#2f855a"
C_BAD = "#c53030"
C_GREY = "#718096"
C_PURPLE = "#6b46c1"


def _read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig, name: str) -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, name), dpi=160)
    plt.close(fig)
    print(f"[figures] {name}")


def _copy(src: str, name: str) -> None:
    if not os.path.exists(src):
        print(f"[figures] MISSING source for {name}: {src}")
        return
    os.makedirs(FIG_DIR, exist_ok=True)
    shutil.copyfile(src, os.path.join(FIG_DIR, name))
    print(f"[figures] {name}")


# --------------------------------------------------------------------------------------
# Task 1
# --------------------------------------------------------------------------------------
def fig_task1_curves() -> None:
    rows = _read_csv(os.path.join(RES, "task1", "log.csv"))
    if not rows:
        return
    ep = [int(r["epoch"]) for r in rows]
    tr = [float(r["train_nll_bits"]) for r in rows]
    te = [float(r["test_nll_bits"]) for r in rows]

    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    ax.plot(ep, tr, color=C_PRIMARY, lw=1.8, label="train")
    ax.plot(ep, te, color=C_ACCENT, lw=1.8, label="test")
    ax.set_xlabel("epoch")
    ax.set_ylabel("NLL (bits / dim)")
    ax.set_title("Task 1 --- FVSBN with ordered conditionals")
    ax.legend(frameon=False)
    _style(ax)
    _save(fig, "task1_curve.png")


def fig_task1_gibbs() -> None:
    """Flip rate of the Gibbs chain of the *unordered* FVSBN."""
    s = _read_json(os.path.join(RES, "task1_fullvis", "sampling_study", "summary.json"))
    if not s or not s.get("flip_traces"):
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    colors = [C_GREY, C_PRIMARY, C_ACCENT, C_BAD]
    for i, (key, trace) in enumerate(s["flip_traces"].items()):
        ax.plot(range(len(trace)), trace, lw=1.6,
                color=colors[i % len(colors)], label=key.replace("_", ", "))
    ax.set_xlabel("Gibbs sweep")
    ax.set_ylabel("fraction of pixels flipped per sweep")
    ax.set_title("Unordered FVSBN --- the Gibbs chain settles into a noisy mode")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    _save(fig, "task1_gibbs_flips.png")


def fig_task1_ablation() -> None:
    rows = _read_csv(os.path.join(RES, "ablation_task1.csv"))
    if not rows:
        return
    labels, vals = [], []
    for r in rows:
        if r["run"] == "t1_base":
            labels.append("full-rank $W$")
            vals.append(float(r["test_bits_per_dim"]))
        elif r["run"].startswith("t1_rank"):
            labels.append(f"rank {r['rank']}")
            vals.append(float(r["test_bits_per_dim"]))
        elif r["run"].startswith("t1_binar"):
            labels.append(f"binar. {r['binarization']}")
            vals.append(float(r["test_bits_per_dim"]))
    if len(labels) < 2:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.bar(range(len(vals)), vals, color=C_PRIMARY, alpha=0.85, width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=8.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("test pseudo-NLL (bits / dim)")
    ax.set_title("Unordered FVSBN --- weight rank and binarisation scheme")
    _style(ax)
    _save(fig, "ablation_fvsbn.png")


# --------------------------------------------------------------------------------------
# Task 2
# --------------------------------------------------------------------------------------
def _curve(folder: str, title: str, out: str) -> None:
    rows = _read_csv(os.path.join(RES, folder, "log.csv"))
    if not rows:
        return
    ep = [int(r["epoch"]) for r in rows]
    tr = [float(r["train_bits"]) for r in rows]
    te = [float(r["test_bits_fixed"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    ax.plot(ep, tr, color=C_PRIMARY, lw=1.8, label="train")
    ax.plot(ep, te, color=C_ACCENT, lw=1.8, label="test (fixed binarisation)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("NLL (bits / dim)")
    ax.set_title(title)
    ax.legend(frameon=False)
    _style(ax)
    _save(fig, out)


def fig_task2_curves() -> None:
    _curve("task2_made", "Task 2 --- deep AR, fixed raster order", "task2_made_curve.png")
    _curve("task2", "Task 2 --- NADE, order-agnostic training", "task2_curve.png")


def fig_ablation_nade() -> None:
    rows = _read_csv(os.path.join(RES, "ablation_task2.csv"))
    # the two reference configurations live outside results/ablation_*, so merge
    # them in from their own metrics.json
    for run_id, folder in (("t2_made", "task2_made"), ("t2_nade_oa", "task2"),
                           ("t2_gruar", "task2_gruar")):
        m = _read_json(os.path.join(RES, folder, "metrics.json"))
        if m:
            rows.append({"run": run_id,
                         "test_bits_per_dim": f"{m['test_nll_bits_per_dim']:.4f}"})
    if not rows:
        return
    order = ["t2_made", "t2_hidden100", "t2_nhidden1", "t2_no_V", "t2_nade_oa", "t2_gruar"]
    nice = {
        "t2_made": "deep AR,\nfixed order",
        "t2_hidden100": "hidden\n100",
        "t2_nhidden1": "1 hidden\nlayer",
        "t2_no_V": "no direct\ninput weights",
        "t2_nade_oa": "NADE\n(order-agnostic)",
        "t2_gruar": "GRU AR",
    }
    by_id = {r["run"]: r for r in rows}
    labels, vals, cols = [], [], []
    for k in order:
        if k not in by_id:
            continue
        labels.append(nice[k])
        vals.append(float(by_id[k]["test_bits_per_dim"]))
        cols.append(C_GREY if k == "t2_gruar" else C_PRIMARY)
    if len(labels) < 2:
        return
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.bar(range(len(vals)), vals, color=cols, alpha=0.9, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=8.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("test NLL (bits / dim)")
    ax.set_title("Task 2 ablation --- capacity, connectivity and pixel order")
    _style(ax)
    _save(fig, "ablation_nade.png")


# --------------------------------------------------------------------------------------
# Task 3
# --------------------------------------------------------------------------------------
def fig_task3_curves() -> None:
    for name, lab in [("task3_dcgan", "DCGAN"), ("task3_wgan", "WGAN-GP")]:
        rows = _read_csv(os.path.join(RES, name, "log.csv"))
        if not rows:
            continue
        ep = [int(r["epoch"]) for r in rows]
        d = [float(r["d_loss"]) for r in rows]
        g = [float(r["g_loss"]) for r in rows]
        gp = [float(r["gp"]) if r.get("gp") not in ("", None) else np.nan for r in rows]

        fig, ax = plt.subplots(figsize=(6.2, 3.3))
        ax.plot(ep, d, color=C_PRIMARY, lw=1.6, label="discriminator loss")
        ax.plot(ep, g, color=C_ACCENT, lw=1.6, label="generator loss")
        if not np.all(np.isnan(gp)):
            ax2 = ax.twinx()
            ax2.plot(ep, gp, color=C_PURPLE, lw=1.2, ls="--", label="gradient penalty")
            ax2.set_ylabel("gradient penalty", color=C_PURPLE)
            ax2.spines["top"].set_visible(False)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title(f"Task 3 --- {lab} training curve")
        ax.legend(frameon=False, loc="upper right", fontsize=9)
        _style(ax)
        _save(fig, f"{name}_curve.png")


def fig_task3_collapse() -> None:
    variants = [("task3_dcgan", "DCGAN", C_PRIMARY), ("task3_wgan", "WGAN-GP", C_PURPLE)]
    have = []
    for key, lab, col in variants:
        m = _read_json(os.path.join(RES, key, "metrics.json"))
        if m and m.get("collapse_metrics_final", {}).get("histogram"):
            have.append((lab, col, m["collapse_metrics_final"]))
    if not have:
        return
    fig, ax = plt.subplots(figsize=(6.8, 3.3))
    width = 0.8 / len(have)
    x = np.arange(10)
    for i, (lab, col, cm) in enumerate(have):
        h = np.array(cm["histogram"], dtype=float)
        h = 100 * h / h.sum()
        ax.bar(x + i * width - 0.4 + width / 2, h, width=width, label=lab, color=col, alpha=0.85)
    ax.axhline(10, color=C_GREY, ls="--", lw=1, label="uniform (10 %)")
    ax.set_xticks(x)
    ax.set_xlabel("digit class assigned by the evaluator CNN")
    ax.set_ylabel("share of generated samples (%)")
    ax.set_title("Task 3 --- mode coverage of the generator")
    ax.legend(frameon=False, ncol=3, fontsize=9)
    _style(ax)
    _save(fig, "task3_collapse.png")


def fig_task3_stability() -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    plotted = False
    for key, lab, col in [("task3_dcgan", "DCGAN", C_PRIMARY), ("task3_wgan", "WGAN-GP", C_PURPLE)]:
        rows = _read_csv(os.path.join(RES, key, "log.csv"))
        pts = [(int(r["epoch"]), float(r["mode_entropy"])) for r in rows
               if r.get("mode_entropy") not in ("", None)]
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                color=col, lw=1.8, marker="o", ms=3, label=lab)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.axhline(1.0, color=C_GREY, ls="--", lw=1, label="perfect coverage")
    ax.set_xlabel("epoch")
    ax.set_ylabel("normalised mode entropy")
    ax.set_ylim(0.5, 1.02)
    ax.set_title("Task 3 --- mode coverage during training")
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    _save(fig, "task3_stability.png")


# --------------------------------------------------------------------------------------
# Cross-model comparison
# --------------------------------------------------------------------------------------
def fig_nll_comparison() -> None:
    base = _read_json(os.path.join(RES, "baselines.json"))
    if not base:
        return
    labels = ["constant\npredictor", "independent\npixels"]
    vals = [base["constant_bits_per_dim"], base["independent_pixels_bits_per_dim"]]
    cols = [C_GREY, C_GREY]

    t1 = _read_json(os.path.join(RES, "task1", "metrics.json"))
    if t1:
        labels.append("FVSBN\n(linear AR)")
        vals.append(t1["test_nll_bits_per_dim"])
        cols.append(C_ACCENT)
    t2m = _read_json(os.path.join(RES, "task2_made", "metrics.json"))
    if t2m:
        labels.append("deep AR\n(fixed order)")
        vals.append(t2m["test_nll_bits_per_dim"])
        cols.append(C_PRIMARY)
    t2 = _read_json(os.path.join(RES, "task2", "metrics.json"))
    if t2:
        labels.append("NADE\n(order-agnostic)")
        vals.append(t2["test_nll_bits_per_dim"])
        cols.append(C_GOOD)
    gru = _read_json(os.path.join(RES, "task2_gruar", "metrics.json"))
    if gru:
        labels.append("GRU AR")
        vals.append(gru["test_nll_bits_per_dim"])
        cols.append(C_PURPLE)

    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    ax.bar(range(len(vals)), vals, color=cols, alpha=0.9, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.008, f"{v:.4f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("bits / dim")
    ax.set_title("Binarized MNIST --- test likelihood (lower is better)")
    _style(ax)
    _save(fig, "nll_comparison.png")


def fig_copy_samples() -> None:
    _copy(os.path.join(RES, "task1", "training_progress.png"), "task1_progress.png")
    _copy(os.path.join(RES, "task1", "final_samples.png"), "task1_samples.png")
    _copy(os.path.join(RES, "task1_fullvis", "sampling_improved", "anneal.png"),
          "task1_fullvis_samples.png")
    _copy(os.path.join(RES, "task2_made", "training_progress.png"), "task2_progress.png")
    _copy(os.path.join(RES, "task2_made", "final_samples.png"), "task2_samples.png")
    _copy(os.path.join(RES, "task2", "final_samples.png"), "task2_nade_samples.png")
    for key in ("task3_dcgan", "task3_wgan"):
        _copy(os.path.join(RES, key, "training_progress.png"), f"{key}_progress.png")
        _copy(os.path.join(RES, key, "final_samples.png"), f"{key}_samples.png")


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_task1_curves()
    fig_task1_gibbs()
    fig_task1_ablation()
    fig_task2_curves()
    fig_ablation_nade()
    fig_task3_curves()
    fig_task3_collapse()
    fig_task3_stability()
    fig_nll_comparison()
    fig_copy_samples()
    print(f"[figures] written to {FIG_DIR}/")


if __name__ == "__main__":
    main()
