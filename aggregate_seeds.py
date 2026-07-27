"""Aggregate a multi-seed benchmark into mean +/- std, a significance test, and an
error-bar figure.

Run after a multi-seed benchmark:

    python train_3dv2.py --mesh nefertiti.obj --epochs 500 --seeds 0 1 2
    python aggregate_seeds.py

Reads results_3d_seeds.csv (the Seed-columned file the training script writes for
multi-seed runs) and emits:

  - thesis3/generated_tables_seeds.tex   mean +/- std table, best in bold
  - thesis3/Figures/results/iou_error_bars.png   dot plot with +/- std bars
  - a significance summary, printed and written to thesis3/generated_stats.tex

Design of the test (see the printed caveats): with a handful of seeds a formal
test is underpowered, so the honest deliverables are the error bars and a global
ANOVA. We report both, and identify the "leading cluster" as the methods that a
Holm-corrected Welch t-test cannot separate from the best mean.
"""

import csv
import os
import sys
from collections import defaultdict

import numpy as np

CSV_PATH = "results_3d_seeds.csv"
THESIS_DIR = os.environ.get("THESIS_DIR", "thesis4")   # override with THESIS_DIR=... if renamed
TEX_TABLE = os.path.join(THESIS_DIR, "generated_tables_seeds.tex")
TEX_STATS = os.path.join(THESIS_DIR, "generated_stats.tex")
FIG_PATH = os.path.join(THESIS_DIR, "Figures", "results", "iou_error_bars.png")

DISPLAY = {
    "fourier": "Fourier Features", "incode": "INCODE", "fr": "FR", "siren": "SIREN",
    "finer": "FINER", "wire": "WIRE", "mfn": "MFN", "gauss": "GAUSS",
}
INK = "#1A1A1A"        # primary text
MUTED = "#6B6B6B"      # axes / secondary
MARK = "#3B6BA5"       # single series hue (colorblind-safe mid blue)
GRID = "#E6E6E6"


def load(path=CSV_PATH):
    """Return {model: {metric: np.array over seeds}} from the seed CSV."""
    if not os.path.exists(path):
        sys.exit(f"[error] {path} not found. Run a multi-seed benchmark first:\n"
                 f"  python train_3dv2.py --mesh nefertiti.obj --epochs 500 --seeds 0 1 2")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "Seed" not in rows[0]:
        sys.exit(f"[error] {path} has no 'Seed' column; is this a multi-seed file?")

    acc = defaultdict(lambda: defaultdict(list))
    seen = defaultdict(set)
    for r in rows:
        m, s = r["Model"], r["Seed"]
        if s in seen[m]:
            continue  # a repeated (model, seed) -> keep the first, warn later
        seen[m].add(s)
        for key, col in (("eval", "Eval_IoU"), ("train", "Final_Train_IoU"),
                         ("chamfer", "Chamfer_L1"), ("nc", "Normal_Consistency")):
            v = r.get(col, "")
            acc[m][key].append(float(v) if v not in ("", "N/A") else np.nan)
    return {m: {k: np.array(v) for k, v in d.items()} for m, d in acc.items()}


def summarize(data):
    """Per-model n / mean / std of eval IoU, ranked by mean descending."""
    out = []
    for m, d in data.items():
        ev = d["eval"]
        out.append({
            "model": m, "n": len(ev),
            "mean": float(np.mean(ev)),
            "std": float(np.std(ev, ddof=1)) if len(ev) > 1 else 0.0,
            "chamfer": float(np.nanmean(d["chamfer"])),
            "nc": float(np.nanmean(d["nc"])),
        })
    return sorted(out, key=lambda r: -r["mean"])


def significance(data, ranked):
    """Global ANOVA plus Holm-corrected Welch t-tests of the best vs. each other.

    Returns (lines_for_print, leading_cluster_models, tex_summary)."""
    from scipy import stats

    ns = {r["n"] for r in ranked}
    if ns == {1}:
        return (["Only one seed per model: no variance, no test possible."],
                [r["model"] for r in ranked], "single seed per configuration")

    groups = [data[r["model"]]["eval"] for r in ranked]
    f, p_all = stats.f_oneway(*groups)
    lines = [f"Global one-way ANOVA across {len(ranked)} methods: "
             f"F={f:.2f}, p={p_all:.2e}  ->  {'differences exist' if p_all < 0.05 else 'no differences detected'}."]

    best = ranked[0]
    best_ev = data[best["model"]]["eval"]
    others = ranked[1:]
    raw = []
    for r in others:
        _, p = stats.ttest_ind(best_ev, data[r["model"]]["eval"], equal_var=False)
        raw.append((r["model"], p))

    # Holm-Bonferroni step-down correction over the family of best-vs-each tests.
    order = sorted(range(len(raw)), key=lambda i: raw[i][1])
    m = len(raw)
    sig = {}
    for rank, i in enumerate(order):
        adj = raw[i][1] * (m - rank)
        sig[raw[i][0]] = adj < 0.05

    cluster = [best["model"]] + [mdl for mdl, _ in raw if not sig[mdl]]
    lines.append(f"Best method: {DISPLAY[best['model']]} ({best['mean']:.4f}).")
    lines.append("Welch t-test of best vs. each other (Holm-corrected):")
    for mdl, pr in raw:
        lines.append(f"  {DISPLAY[mdl]:17s} p_raw={pr:.3f}  "
                     f"{'DIFFERENT' if sig[mdl] else 'not separable'}")
    lines.append(f"Leading cluster (indistinguishable from best): "
                 f"{', '.join(DISPLAY[c] for c in cluster)}.")

    tex = (f"A one-way ANOVA across the {len(ranked)} methods "
           f"{'rejects' if p_all < 0.05 else 'does not reject'} equality of means "
           f"($F={f:.2f}$, $p={p_all:.2g}$). Holm-corrected Welch $t$-tests of the "
           f"best method against each other place {len(cluster)} methods in a "
           f"leading cluster that cannot be separated at $\\alpha=0.05$: "
           f"{', '.join(DISPLAY[c] for c in cluster)}.")
    return lines, cluster, tex


def write_table(ranked, cluster):
    best_mean = ranked[0]["mean"]
    L = [
        "% AUTO-GENERATED by aggregate_seeds.py -- do not edit by hand.",
        r"\begin{table}[htbp]", r"\centering",
        rf"\caption{{Evaluation IoU over {ranked[0]['n']} seeds (mean $\pm$ standard "
        rf"deviation), ranked by mean. Methods in the leading cluster --- those a "
        rf"Holm-corrected Welch $t$-test cannot separate from the best --- are marked "
        rf"$\dagger$. Chamfer-L1 and normal consistency are seed means.}}",
        r"\label{tab:seed_results}",
        r"\begin{tabular}{lccc}", r"\toprule",
        r"\textbf{Method} & \textbf{Eval IoU (mean $\pm$ std)} $\uparrow$ & "
        r"\textbf{Chamfer-L1} $\downarrow$ & \textbf{Normal Cons.} $\uparrow$ \\",
        r"\midrule",
    ]
    for r in ranked:
        dag = r"$^\dagger$" if r["model"] in cluster else ""
        bold = r["mean"] == best_mean
        cell = rf"{r['mean']:.4f} $\pm$ {r['std']:.4f}"
        if bold:
            cell = rf"\textbf{{{cell}}}"
        L.append(f"{DISPLAY[r['model']]}{dag} & {cell} & "
                 f"{r['chamfer']:.6f} & {r['nc']:.4f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    os.makedirs(os.path.dirname(TEX_TABLE), exist_ok=True)
    with open(TEX_TABLE, "w") as f:
        f.write("\n".join(L) + "\n")


def write_figure(ranked, cluster):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranked = list(reversed(ranked))            # best at top of the axis
    names = [DISPLAY[r["model"]] for r in ranked]
    means = np.array([r["mean"] for r in ranked])
    stds = np.array([r["std"] for r in ranked])
    y = np.arange(len(ranked))

    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=200)
    # Dot plot, not bars: values sit in a narrow band, so a zero baseline would
    # hide the differences. Points encode position and need no zero.
    ax.errorbar(means, y, xerr=stds, fmt="o", ms=7, color=MARK,
                ecolor=MUTED, elinewidth=1.4, capsize=4, capthick=1.4, zorder=3)

    for yi, (mu, sd, r) in enumerate(zip(means, stds, ranked)):
        ax.annotate(f"{mu:.4f}", (mu, yi), xytext=(0, 9),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=8.5, color=INK)

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{n}$\\dagger$" if r["model"] in cluster else n
         for n, r in zip(names, ranked)], fontsize=10, color=INK)
    ax.set_xlabel("Evaluation IoU  (mean $\\pm$ std over seeds)", fontsize=10, color=INK)

    lo = float(min(means - stds)) - 0.005
    hi = float(max(means + stds)) + 0.008
    ax.set_xlim(lo, hi)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=0)
    ax.margins(y=0.08)
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    fig.savefig(FIG_PATH, bbox_inches="tight", facecolor="white")
    print(f"-> wrote {FIG_PATH}")


def main():
    data = load()
    ranked = summarize(data)

    print("=== Evaluation IoU over seeds ===")
    for r in ranked:
        print(f"{DISPLAY[r['model']]:17s} n={r['n']}  "
              f"mean={r['mean']:.4f}  std={r['std']:.4f}")
    seeds_ok = all(r["n"] >= 2 for r in ranked)
    if not seeds_ok:
        print("\n[warn] some models have <2 seeds; std and tests are not meaningful for those.")

    print()
    lines, cluster, tex = significance(data, ranked)
    print("\n".join(lines))

    write_table(ranked, cluster)
    os.makedirs(os.path.dirname(TEX_STATS), exist_ok=True)
    with open(TEX_STATS, "w") as f:
        f.write("% AUTO-GENERATED by aggregate_seeds.py\n" + tex + "\n")
    write_figure(ranked, cluster)
    print(f"-> wrote {TEX_TABLE}\n-> wrote {TEX_STATS}")
    print("\nCaveat: with few seeds these tests have low power; a 'not separable' "
          "result means the data does not resolve a difference, not that none exists.")


if __name__ == "__main__":
    main()
