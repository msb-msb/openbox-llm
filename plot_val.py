"""
plot_val.py — overlay the nsa vs full val-loss curves. THIS PLOT IS THE RESULT.

Reads runs/{nsa,full}_metrics.csv and writes plots/val_loss.png (and a train-loss
companion). Success for rung 2a = the NSA val curve TRACKS the full-attention
curve within a small margin (no pathology / no divergence).
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


def load(path):
    steps, tr, va = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            tr.append(float(row["train_loss"]))
            va.append(float(row["val_loss"]))
    return steps, tr, va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="runs")
    ap.add_argument("--out_dir", default="plots")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    curves = {}
    for attn in ["full", "nsa"]:
        p = os.path.join(args.run_dir, f"{attn}_metrics.csv")
        if os.path.exists(p):
            curves[attn] = load(p)
        else:
            print(f"WARNING: missing {p}")
    if not curves:
        raise SystemExit("no metrics CSVs found — run training first")

    colors = {"full": "tab:blue", "nsa": "tab:red"}

    # --- the headline: val-loss overlay ----------------------------------
    plt.figure(figsize=(8, 5))
    for attn, (steps, tr, va) in curves.items():
        plt.plot(steps, va, label=f"{attn} (val)", color=colors[attn], lw=2)
    plt.xlabel("step"); plt.ylabel("val loss (cross-entropy)")
    plt.title("Rung 2a — NSA vs full attention: held-out val loss")
    plt.legend(); plt.grid(alpha=0.3)
    if len(curves) == 2:
        f_end = curves["full"][2][-1]; n_end = curves["nsa"][2][-1]
        plt.annotate(f"final gap (nsa-full): {n_end - f_end:+.4f}",
                     xy=(0.5, 0.95), xycoords="axes fraction", ha="center",
                     fontsize=10, bbox=dict(boxstyle="round", fc="wheat", alpha=0.6))
    out = os.path.join(args.out_dir, "val_loss.png")
    plt.tight_layout(); plt.savefig(out, dpi=130); print("wrote", out)

    # --- companion: train + val, both arms -------------------------------
    plt.figure(figsize=(8, 5))
    for attn, (steps, tr, va) in curves.items():
        plt.plot(steps, tr, "--", color=colors[attn], lw=1.3, alpha=0.7,
                 label=f"{attn} (train)")
        plt.plot(steps, va, "-", color=colors[attn], lw=2, label=f"{attn} (val)")
    plt.xlabel("step"); plt.ylabel("loss (cross-entropy)")
    plt.title("Rung 2a — train (dashed) & val (solid)")
    plt.legend(); plt.grid(alpha=0.3)
    out2 = os.path.join(args.out_dir, "train_val_loss.png")
    plt.tight_layout(); plt.savefig(out2, dpi=130); print("wrote", out2)

    if len(curves) == 2:
        print(f"\nfinal val loss  full={curves['full'][2][-1]:.4f}  "
              f"nsa={curves['nsa'][2][-1]:.4f}  "
              f"gap(nsa-full)={curves['nsa'][2][-1]-curves['full'][2][-1]:+.4f}")


if __name__ == "__main__":
    main()
