import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Measured, RTX 3090, fused path, 1.5B, fwd+bwd (no optimizer), atomic->FSA selection backward
seq   = [512, 1024, 2048]
ratio = [1.083, 1.177, 1.259]     # batch 1
b2_seq, b2_ratio = 1024, 1.359    # batch 2 point

fig, ax = plt.subplots(figsize=(5.0, 3.4))
ax.plot(seq, ratio, "-o", color="#1f77b4", lw=2, ms=6, label="batch 1")
ax.plot([b2_seq], [b2_ratio], "s", color="#d62728", ms=8, label="batch 2")
ax.annotate("1.36×", (b2_seq, b2_ratio), textcoords="offset points",
            xytext=(8, -2), fontsize=9, color="#d62728")
for x, y in zip(seq, ratio):
    ax.annotate(f"{y:.2f}×", (x, y), textcoords="offset points",
                xytext=(6, 6), fontsize=9, color="#1f77b4")
ax.axhline(1.63, ls="--", color="#888888", lw=1)
ax.annotate("1.63× (prior Amdahl projection)", (512, 1.63),
            textcoords="offset points", xytext=(4, 5), fontsize=8, color="#888888")
ax.set_xscale("log", base=2)
ax.set_xticks(seq); ax.set_xticklabels([str(s) for s in seq])
ax.set_xlabel("sequence length")
ax.set_ylabel("end-to-end fwd+bwd speedup\n(atomic → FSA selection backward)")
ax.set_ylim(1.0, 1.72)
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right", fontsize=9, frameon=False)
fig.tight_layout()
fig.savefig("e2e_speedup_curve.png", dpi=200)
print("wrote e2e_speedup_curve.png")
