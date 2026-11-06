"""Showcase figure: public vs private score of all 49 scored submissions, coloured by how each was built.

Reads Kaggle's submission list cached by `kgkit ledger sync-lb` (.kaggle-gm/kaggle_submissions.json) and the ledger
tags. Writes docs/figures/public_vs_private_all.png.

    python src/plot_showcase.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f1f1e", "#6b6a63", "#e6e5e0"
GROUPS = [  # fixed categorical order (validated palette), plus a marker shape as secondary encoding
    ("honest", "Honest stack / blend (CV-chosen)", "#2a78d6", "o"),
    ("lb-blend", "Blend with public submission files", "#eb6834", "s"),
    ("fitted", "Tilted toward public labels", "#1baf7a", "^"),
]
BRONZE = 0.96237


def main():
    subs = json.loads((ROOT / ".kaggle-gm/kaggle_submissions.json").read_text())["scored"]
    tags = {}
    for line in (ROOT / ".kaggle-gm/ledger.jsonl").read_text().splitlines():
        r = json.loads(line)
        tags[r["id"]] = set(r.get("tags") or [])
    pts = {g[0]: [] for g in GROUPS}
    for s in subs:
        rid = s["file"].removesuffix(".csv")
        t = tags.get(rid, set())
        g = "fitted" if "public-lb-fitted" in t else "lb-blend" if "lb-blend" in t else "honest"
        pts[g].append((float(s["public_score"]), float(s["private_score"]), rid))

    fig, ax = plt.subplots(figsize=(8.4, 5.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for key, label, color, marker in GROUPS:
        xs, ys = [p[0] for p in pts[key]], [p[1] for p in pts[key]]
        ax.scatter(xs, ys, s=46, c=color, marker=marker, edgecolors=SURFACE, linewidths=1.5, zorder=3,
                   label=f"{label} (n={len(xs)})")
    ax.axhline(BRONZE, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(0.96472, BRONZE + 0.00004, "bronze cut-off 0.96237 (private)", color=MUTED, fontsize=8, ha="right")

    def note(rid, text, tx, ty, ha="left"):
        for group in pts.values():
            for x, y, r in group:
                if r == rid:
                    ax.annotate(text, (x, y), xytext=(tx, ty), fontsize=8.5, color=INK, ha=ha, va="center",
                                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
                    return

    note("0078_tilt-v16-exp-t1k18-eta1", "0078: public 1st (0.96473)\nprivate 0.96027, rank ~1,981 if picked",
         0.96468, 0.96125, ha="right")
    note("0020_stack-lr-pool319-kaggle", "0020 final: private 0.96236, rank 426", 0.96255, 0.96228)
    note("0057_lbb-settf16-10", "0057 final: private 0.96222", 0.96255, 0.96206)

    ax.set_xlabel("Public score (20% of test)", color=INK)
    ax.set_ylabel("Private score (80% of test)", color=INK)
    ax.set_title("The higher the public score, the lower the private score", color=INK, loc="left", fontsize=12)
    ax.grid(color=GRID, lw=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.legend(loc="lower left", frameon=False, fontsize=8.5, labelcolor=INK)
    fig.tight_layout()
    out = ROOT / "docs/figures/public_vs_private_all.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
