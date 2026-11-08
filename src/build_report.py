"""Assemble the LaTeX report: numeric macros and tables from the result files.

Everything the report quotes is generated from ``results/`` so that the text and
the experiments can never drift apart.

Outputs (into ``report/``)
--------------------------
    numbers.tex       \\newcommand definitions for every number used in the text
    tab_*.tex         the LaTeX tables of the report

Usage:  python src/build_report.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
from typing import Dict, List, Optional

RES = "results"
REPORT = "report"


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def jload(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def cload(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def n(x, nd: int = 4, dash: str = "n/a") -> str:
    if x is None:
        return dash
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return dash


def i(x, dash: str = "n/a") -> str:
    if x is None:
        return dash
    try:
        return f"{int(x):,}".replace(",", "{,}")
    except (TypeError, ValueError):
        return dash


def secs(x, dash: str = "n/a") -> str:
    if x is None:
        return dash
    x = float(x)
    if x < 90:
        return f"{x:.1f}~s"
    return f"{x / 60:.1f}~min"


def esc(s: str) -> str:
    return (s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
             .replace("#", r"\#").replace("$", r"\$"))


def write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[report] wrote {path}")


# --------------------------------------------------------------------------------------
# numbers.tex
# --------------------------------------------------------------------------------------
def build_numbers() -> Dict[str, str]:
    m: Dict[str, str] = {}

    base = jload(os.path.join(RES, "baselines.json")) or {}
    m["ConstBits"] = n(base.get("constant_bits_per_dim"))
    m["IndepBits"] = n(base.get("independent_pixels_bits_per_dim"))
    m["GlobalMean"] = n(base.get("global_mean_foreground"), 4)

    # ---- Task 1: ordered (main) and unordered (comparison)
    t1 = jload(os.path.join(RES, "task1", "metrics.json")) or {}
    m["ToneBits"] = n(t1.get("test_nll_bits_per_dim"))
    m["ToneTime"] = secs(t1.get("training_time_s"))
    m["ToneGenTime"] = n(t1.get("inference_time_per_sample_s"), 3)
    m["ToneGenBatch"] = n(t1.get("inference_time_batch64_s"), 2)
    m["ToneParams"] = i(t1.get("params"))
    m["ToneEpochs"] = i(t1.get("epochs"))
    m["ToneSweeps"] = i(t1.get("gibbs_sweeps"))
    m["ToneBlock"] = i(t1.get("gibbs_block_size"))
    m["ToneTrainSize"] = i(t1.get("train_size"))

    fv = jload(os.path.join(RES, "task1_fullvis", "metrics.json")) or {}
    m["ToneFullBits"] = n(fv.get("test_pseudo_nll_bits_per_dim"))

    try:
        gain = 100.0 * (1 - float(t1["test_nll_bits_per_dim"]) /
                        float(base["independent_pixels_bits_per_dim"]))
        m["ToneGain"] = f"{gain:.1f}"
    except (KeyError, TypeError, ZeroDivisionError):
        m["ToneGain"] = "n/a"

    # ---- Task 2
    t2 = jload(os.path.join(RES, "task2", "metrics.json")) or {}
    m["TtwoBits"] = n(t2.get("test_nll_bits_per_dim"))
    m["TtwoTime"] = secs(t2.get("training_time_s"))
    m["TtwoGenTime"] = n(t2.get("inference_time_batch64_s"), 3)
    m["TtwoParams"] = i(t2.get("params"))
    m["TtwoEpochs"] = i(t2.get("epochs"))
    m["TtwoHidden"] = i(t2.get("hidden"))
    m["TtwoLayers"] = i(t2.get("n_hidden"))

    tm = jload(os.path.join(RES, "task2_made", "metrics.json")) or {}
    m["TtwoMadeBits"] = n(tm.get("test_nll_bits_per_dim"))
    m["TtwoMadeTime"] = secs(tm.get("training_time_s"))
    m["TtwoMadeEpochs"] = i(tm.get("epochs"))

    gr = jload(os.path.join(RES, "task2_gruar", "metrics.json")) or {}
    m["GruBits"] = n(gr.get("test_nll_bits_per_dim"))
    m["GruTime"] = secs(gr.get("training_time_s"))

    # ---- Task 3
    for tag, key in (("Dcgan", "task3_dcgan"), ("Wgan", "task3_wgan")):
        g = jload(os.path.join(RES, key, "metrics.json")) or {}
        cm = g.get("collapse_metrics_final", {}) or {}
        m[f"{tag}Modes"] = i(cm.get("num_modes_covered"))
        m[f"{tag}Entropy"] = n(cm.get("mode_entropy"), 4)
        m[f"{tag}ChiSq"] = n(cm.get("mode_chi2_vs_uniform"), 1)
        m[f"{tag}TopOne"] = n(cm.get("top1_share"), 4)
        m[f"{tag}Time"] = secs(g.get("training_time_s"))
        m[f"{tag}GenTime"] = n(g.get("inference_time_batch64_s"), 3)
        m[f"{tag}GParams"] = i(g.get("G_params"))
        m[f"{tag}DParams"] = i(g.get("D_params"))
        m[f"{tag}Epochs"] = i(g.get("epochs"))
        m[f"{tag}TrainSize"] = i(g.get("train_size"))

    acc = jload(os.path.join(RES, "task3_dcgan", "classifier_accuracy.json")) or {}
    if not acc:
        acc = jload(os.path.join(RES, "task3_wgan", "classifier_accuracy.json")) or {}
    m["ClfAcc"] = n(acc.get("test_accuracy"), 4)
    m["ClfNTest"] = i(acc.get("n_test"))

    return m


def write_numbers(m: Dict[str, str]) -> None:
    def _safe(name: str) -> str:
        # LaTeX's \newcommand refuses names that contain digits
        for d, w in (("0", "Zero"), ("1", "One"), ("2", "Two"), ("3", "Three")):
            name = name.replace(d, w)
        return name

    body = "\n".join(f"\\newcommand{{\\{_safe(k)}}}{{{v}}}" for k, v in m.items())
    write(os.path.join(REPORT, "numbers.tex"),
          "% Auto-generated by src/build_report.py -- do not edit by hand.\n" + body + "\n")


# --------------------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------------------
def tab_runtime() -> None:
    t1 = jload(os.path.join(RES, "task1", "metrics.json")) or {}
    fv = jload(os.path.join(RES, "task1_fullvis", "metrics.json")) or {}
    tm = jload(os.path.join(RES, "task2_made", "metrics.json")) or {}
    t2 = jload(os.path.join(RES, "task2", "metrics.json")) or {}
    d = jload(os.path.join(RES, "task3_dcgan", "metrics.json")) or {}
    w = jload(os.path.join(RES, "task3_wgan", "metrics.json")) or {}

    rows = [
        ("Task 1 -- FVSBN (ordered)", i(t1.get("params")), i(t1.get("epochs")),
         secs(t1.get("training_time_s")), n(t1.get("inference_time_batch64_s"), 2) + "~s"),
        ("Task 1 -- FVSBN (unordered)", i(fv.get("params")), i(fv.get("epochs")),
         secs(fv.get("training_time_s")), n(fv.get("inference_time_batch64_s"), 2) + "~s"),
        ("Task 2 -- deep AR (fixed order)", i(tm.get("params")), i(tm.get("epochs")),
         secs(tm.get("training_time_s")), n(tm.get("inference_time_batch64_s"), 2) + "~s"),
        ("Task 2 -- NADE (order-agnostic)", i(t2.get("params")), i(t2.get("epochs")),
         secs(t2.get("training_time_s")), n(t2.get("inference_time_batch64_s"), 2) + "~s"),
        ("Task 3 -- DCGAN", i(d.get("G_params")) + " + " + i(d.get("D_params")),
         i(d.get("epochs")), secs(d.get("training_time_s")),
         n(d.get("inference_time_batch64_s"), 3) + "~s"),
        ("Task 3 -- WGAN-GP", i(w.get("G_params")) + " + " + i(w.get("D_params")),
         i(w.get("epochs")), secs(w.get("training_time_s")),
         n(w.get("inference_time_batch64_s"), 3) + "~s"),
    ]
    tex = [r"\begin{tabular}{l r r l l}", r"\toprule",
           r"Model & \# params & Epochs & Training time & Generation (64 images) \\",
           r"\midrule"]
    for r in rows:
        tex.append(f"{r[0]} & {r[1]} & {r[2]} & {r[3]} & {r[4]} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    write(os.path.join(REPORT, "tab_runtime.tex"), "\n".join(tex) + "\n")


def tab_likelihood() -> None:
    base = jload(os.path.join(RES, "baselines.json")) or {}
    t1 = jload(os.path.join(RES, "task1", "metrics.json")) or {}
    fv = jload(os.path.join(RES, "task1_fullvis", "metrics.json")) or {}
    tm = jload(os.path.join(RES, "task2_made", "metrics.json")) or {}
    t2 = jload(os.path.join(RES, "task2", "metrics.json")) or {}
    gr = jload(os.path.join(RES, "task2_gruar", "metrics.json")) or {}

    rows = (
        ("Constant predictor (no model)", n(base.get("constant_bits_per_dim")),
         "global foreground frequency"),
        ("Independent pixels (per-position marginal)",
         n(base.get("independent_pixels_bits_per_dim")), "no spatial structure"),
        (r"\textbf{FVSBN, ordered (Task 1)}",
         r"\textbf{" + n(t1.get("test_nll_bits_per_dim")) + "}",
         "exact NLL, ancestral sampling"),
        ("FVSBN, unordered (pseudo-NLL)", n(fv.get("test_pseudo_nll_bits_per_dim")),
         "not comparable: pseudo-likelihood"),
        (r"\textbf{Deep AR, fixed order (Task 2)}",
         r"\textbf{" + n(tm.get("test_nll_bits_per_dim")) + "}",
         "exact NLL, ancestral sampling"),
        ("NADE, order-agnostic (Task 2)", n(t2.get("test_nll_bits_per_dim")),
         "exact NLL, ancestral sampling"),
    )
    tex = [r"\begin{tabular}{l r l}", r"\toprule",
           r"Model & Test bits/dim & Comment \\", r"\midrule"]
    for a, b, c in rows:
        tex.append(f"{a} & {b} & {c} \\\\")
    if gr:
        tex.append("GRU autoregressive & " + n(gr.get("test_nll_bits_per_dim")) +
                   " & exact NLL, ancestral sampling \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    write(os.path.join(REPORT, "tab_likelihood.tex"), "\n".join(tex) + "\n")


def tab_collapse() -> None:
    d = jload(os.path.join(RES, "task3_dcgan", "metrics.json")) or {}
    w = jload(os.path.join(RES, "task3_wgan", "metrics.json")) or {}
    rows = [("DCGAN", d.get("collapse_metrics_final") or {}, d.get("training_time_s")),
            ("WGAN-GP", w.get("collapse_metrics_final") or {}, w.get("training_time_s"))]
    tex = [r"\begin{tabular}{l r r r r r}", r"\toprule",
           r"Objective & Modes & Mode entropy & $\chi^2$ vs.\ unif. & Top-1 share & Train time \\",
           r"\midrule"]
    for lab, cm, t in rows:
        if not cm:
            continue
        tex.append(f"{lab} & {cm.get('num_modes_covered', 'n/a')} & "
                   f"{n(cm.get('mode_entropy'), 4)} & {n(cm.get('mode_chi2_vs_uniform'), 1)} & "
                   f"{n(cm.get('top1_share'), 4)} & {secs(t)} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    write(os.path.join(REPORT, "tab_collapse.tex"), "\n".join(tex) + "\n")


# ---- ablation tables -----------------------------------------------------------------
NICE1 = {
    "t1_base": "unordered, full-rank $W$",
    "t1_rank200": "unordered, rank 200",
    "t1_rank50": "unordered, rank 50",
    "t1_block1": "Gibbs block 1 (exact)",
    "t1_block784": "Gibbs block 784",
    "t1_sweeps20": "20 Gibbs sweeps",
    "t1_sweeps50": "50 Gibbs sweeps",
    "t1_binar_static": "static binarisation",
    "t1_binar_mean": "mean binarisation",
}
NICE2 = {
    "t2_made": "deep AR, fixed raster order",
    "t2_nade_oa": "NADE, order-agnostic",
    "t2_hidden100": "hidden units 100",
    "t2_hidden250": "hidden units 250",
    "t2_nhidden1": "1 hidden layer",
    "t2_nhidden3": "3 hidden layers",
    "t2_no_V": "no direct input weights",
}


def _collect(task: int) -> List[dict]:
    out = []
    if task == 1:
        for d in sorted(glob.glob(os.path.join(RES, "ablation_t1*"))):
            p = os.path.join(d, "metrics.json")
            if not os.path.exists(p):
                continue
            r = jload(p)
            out.append({
                "run": os.path.basename(d).replace("ablation_", ""),
                "bits": r.get("test_pseudo_nll_bits_per_dim"),
                "params": r.get("params"),
                "sweeps": r.get("gibbs_sweeps"), "block": r.get("gibbs_block_size"),
                "time": r.get("training_time_s"),
            })
    else:
        for run_id, path in (("t2_made", os.path.join(RES, "task2_made", "metrics.json")),
                             ("t2_nade_oa", os.path.join(RES, "task2", "metrics.json"))):
            r = jload(path)
            if r:
                out.append({"run": run_id, "bits": r.get("test_nll_bits_per_dim"),
                            "params": r.get("params"),
                            "hidden": r.get("hidden"), "layers": r.get("n_hidden"),
                            "time": r.get("training_time_s")})
        for d in sorted(glob.glob(os.path.join(RES, "ablation_t2*"))):
            p = os.path.join(d, "metrics.json")
            if not os.path.exists(p):
                continue
            r = jload(p)
            out.append({
                "run": os.path.basename(d).replace("ablation_", ""),
                "bits": r.get("test_nll_bits_per_dim"),
                "params": r.get("params"),
                "hidden": r.get("hidden"), "layers": r.get("n_hidden"),
                "time": r.get("training_time_s"),
            })
    return out


def _ablation_table(task: int) -> None:
    rows = _collect(task)
    if task == 1:
        head = [r"Configuration & Params & bits/dim & Sweeps & Block & Train time \\",
                r"\midrule"]
    else:
        head = [r"Configuration & Params & bits/dim & Hidden & Layers & Train time \\",
                r"\midrule"]
    tex = [r"\begin{tabular}{l r r r r r}", r"\toprule"] + head
    for r in rows:
        nice = (NICE1 if task == 1 else NICE2).get(r["run"], esc(r["run"]))
        if task == 1:
            tex.append(f"{nice} & {i(r['params'])} & {n(r['bits'])} & {i(r['sweeps'])} & "
                       f"{i(r['block'])} & {secs(r['time'])} \\\\")
        else:
            tex.append(f"{nice} & {i(r['params'])} & {n(r['bits'])} & {i(r['hidden'])} & "
                       f"{i(r['layers'])} & {secs(r['time'])} \\\\")
    if not rows:
        tex.append(r"\multicolumn{6}{c}{results pending} \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    write(os.path.join(REPORT, f"tab_ablation{task}.tex"), "\n".join(tex) + "\n")


def main() -> None:
    write_numbers(build_numbers())
    tab_runtime()
    tab_likelihood()
    tab_collapse()
    _ablation_table(1)
    _ablation_table(2)
    print("[report] done")


if __name__ == "__main__":
    main()
