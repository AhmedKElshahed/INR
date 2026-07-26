"""Generate the thesis result tables directly from results_3d_comparison.csv.

Run after every benchmark run so the report can never drift from the data:

    python make_tables.py

Writes LaTeX to thesis3/generated_tables.tex (input it from Chapter 5) and prints
a summary plus the grouped property analysis used in the Discussion.

If a model appears more than once in the CSV (e.g. after re-running a single
model), the LAST row wins -- the CSV is append-only, so the newest run supersedes.
"""

import csv
import os
import statistics as st

CSV_PATH = "results_3d_comparison.csv"
OUT_PATH = os.path.join("thesis3", "generated_tables.tex")

DISPLAY = {
    "fourier": "Fourier Features", "incode": "INCODE", "fr": "FR", "siren": "SIREN",
    "finer": "FINER", "wire": "WIRE", "mfn": "MFN", "gauss": "GAUSS",
}

# Taxonomy of Essakine et al. (2024); see Table 2.1 in the report.
FREQ_COMPACT = {"finer", "incode", "gauss", "wire"}
ADAPTIVE = {"finer", "incode", "fr"}


def load(path=CSV_PATH):
    """Read the append-only CSV, keeping the most recent row per model."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    latest = {}
    for r in rows:
        latest[r["Model"]] = r  # later rows overwrite earlier ones

    out = []
    for name, r in latest.items():
        out.append({
            "model": name,
            "epochs": int(r["Epochs"]),
            "time": float(r["Time(s)"]),
            "train": float(r["Final_Train_IoU"]),
            "eval": float(r["Eval_IoU"]),
            "chamfer": float(r["Chamfer_L1"]),
            "nc": float(r["Normal_Consistency"]),
            "config": r["Config_Params"],
        })

    n_dupes = len(rows) - len(latest)
    if n_dupes:
        print(f"[note] {n_dupes} superseded row(s) ignored; kept newest per model\n")
    return sorted(out, key=lambda r: -r["eval"])


def fmt_config(cfg):
    """Turn the stored dict-string into compact LaTeX, dropping hidden_layers."""
    body = cfg.strip("{}").replace("'", "")
    parts = [p.strip() for p in body.split(",") if "hidden_layers" not in p]
    return ", ".join(parts).replace("_", r"\_")


def bold_min(vals, i, fmt):
    return rf"\textbf{{{fmt % vals[i]}}}" if vals[i] == min(vals) else fmt % vals[i]


def bold_max(vals, i, fmt):
    return rf"\textbf{{{fmt % vals[i]}}}" if vals[i] == max(vals) else fmt % vals[i]


def table_main(rows):
    ev = [r["eval"] for r in rows]
    ch = [r["chamfer"] for r in rows]
    nc = [r["nc"] for r in rows]
    epochs = rows[0]["epochs"]

    L = [
        r"\begin{table}[htbp]", r"\centering",
        rf"\caption{{Quantitative results for 3D occupancy reconstruction on the Nefertiti bust "
        rf"({epochs} epochs). Evaluation IoU is measured on the held-out 50{{,}}000-point "
        rf"validation split; Chamfer-L1 and normal consistency are computed against the ground-truth "
        rf"mesh from a $256^3$ marching-cubes extraction. Best results in bold.}}",
        r"\label{tab:nefertiti_results}",
        r"\begin{tabular}{lccccl}", r"\toprule",
        r"\textbf{Method} & \textbf{Eval IoU} $\uparrow$ & \textbf{Train IoU} & "
        r"\textbf{Chamfer-L1} $\downarrow$ & \textbf{Normal Cons.} $\uparrow$ & \textbf{Config} \\",
        r"\midrule",
    ]
    for i, r in enumerate(rows):
        L.append(
            f"{DISPLAY[r['model']]} & {bold_max(ev, i, '%.4f')} & {r['train']:.4f} & "
            f"{bold_min(ch, i, '%.6f')} & {bold_max(nc, i, '%.4f')} & {fmt_config(r['config'])} \\\\"
        )
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def table_efficiency(rows):
    rows = sorted(rows, key=lambda r: r["time"])
    eff = [r["eval"] / (r["time"] / 60) for r in rows]
    tm = [r["time"] for r in rows]

    L = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Training time and quality-efficiency trade-off, ordered by wall-clock time. "
        r"IoU/minute is evaluation IoU divided by training time in minutes.}",
        r"\label{tab:training_time}",
        r"\begin{tabular}{lccc}", r"\toprule",
        r"\textbf{Method} & \textbf{Time (s)} $\downarrow$ & \textbf{Eval IoU} $\uparrow$ & "
        r"\textbf{IoU/minute} $\uparrow$ \\", r"\midrule",
    ]
    for i, r in enumerate(rows):
        L.append(
            f"{DISPLAY[r['model']]} & {bold_min(tm, i, '%.1f')} & {r['eval']:.4f} & "
            f"{bold_max(eff, i, '%.4f')} \\\\"
        )
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def property_analysis(rows):
    by = {r["model"]: r["eval"] for r in rows}
    mean = lambda ks: st.mean(by[k] for k in ks if k in by)
    present = set(by)

    fc = sorted(present & FREQ_COMPACT)
    nfc = sorted(present - FREQ_COMPACT)
    out = ["=== Property analysis (recompute the Discussion prose from these) ===",
           f"frequency-compact     {fc}: {mean(fc):.4f}",
           f"not frequency-compact {nfc}: {mean(nfc):.4f}",
           f"  -> difference: {100 * (mean(fc) - mean(nfc)):+.2f} pp in favour of frequency-compact"]

    # Sensitivity: the grouped means are dominated by whichever method is the outlier.
    worst = min(by, key=by.get)
    nfc2 = [m for m in nfc if m != worst]
    fc2 = [m for m in fc if m != worst]
    if nfc2 and fc2:
        out += ["", f"Sensitivity check -- dropping the weakest method ({DISPLAY[worst]}, {by[worst]:.4f}):",
                f"  frequency-compact     {fc2}: {mean(fc2):.4f}",
                f"  not frequency-compact {nfc2}: {mean(nfc2):.4f}",
                f"  -> difference: {100 * (mean(fc2) - mean(nfc2)):+.2f} pp"]

    ad = sorted(present & FREQ_COMPACT & ADAPTIVE)
    nad = sorted((present & FREQ_COMPACT) - ADAPTIVE)
    if ad and nad:
        out += ["", f"adaptive + frequency-compact     {ad}: {mean(ad):.4f}",
                f"non-adaptive + frequency-compact {nad}: {mean(nad):.4f}",
                f"  -> difference: {100 * (mean(ad) - mean(nad)):+.2f} pp"]
    return "\n".join(out)


def main():
    rows = load()

    print("=== Results, ranked by evaluation IoU ===")
    for r in rows:
        gap = r["train"] - r["eval"]
        flag = "  <-- memorizes" if gap > 0.03 else ("  <-- underfits" if r["train"] < 0.95 else "")
        print(f"{DISPLAY[r['model']]:17s} eval={r['eval']:.4f}  train={r['train']:.4f}  "
              f"gap={gap:+.4f}  {r['time']:7.1f}s{flag}")

    print(f"\n{property_analysis(rows)}\n")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write("% AUTO-GENERATED by make_tables.py -- do not edit by hand.\n")
        f.write("% Regenerate with: python make_tables.py\n\n")
        f.write(table_main(rows) + "\n\n" + table_efficiency(rows) + "\n")
    print(f"-> wrote {OUT_PATH}  ({len(rows)} methods)")
    print("   In Chapter 5, replace both hand-written tables with: \\input{generated_tables}")


if __name__ == "__main__":
    main()
