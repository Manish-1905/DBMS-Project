import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET = (
    "#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2")
GRAY, TEXT, MUTED, GRID = "#a6a6a6", "#191919", "#666666", "#e6e6e6"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.spines.left": False, "axes.grid": True,
    "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False, "font.family": "DejaVu Sans",
})


def label_bars(ax, bars, fmt="{:.1f}%", pad=1.0):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + pad, fmt.format(h),
                ha="center", va="bottom", fontsize=9, color=TEXT)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(name, bbox_inches="tight")
    plt.close(fig)


systems = ["S0\nno schema", "S1\nfull schema", "S2\nretrieved\nschema", "S3\n+ value\ngrounding",
           "S5\n+ query\nplanning", "S7\n+ repair"]
acc = [32.8, 51.5, 55.1, 56.2, 53.9, 54.7]
fig, ax = plt.subplots(figsize=(6.4, 3.0))
bars = ax.bar(range(6), acc, width=0.55, color=BLUE)
label_bars(ax, bars)
ax.set_xticks(range(6), systems, fontsize=8.5)
ax.set_ylim(0, 70)
ax.set_ylabel("correct answers (%)")
save(fig, "fig_accuracy.pdf")

noise = ["0% fake", "80% fake", "95% fake"]
full, retr = [11.0, 5.0, 7.0], [15.0, 12.0, 7.5]
fig, ax = plt.subplots(figsize=(5.2, 2.8))
x = range(3)
b1 = ax.bar([i - 0.15 for i in x], full, width=0.28, color=ORANGE, label="full schema in prompt")
b2 = ax.bar([i + 0.15 for i in x], retr, width=0.28, color=BLUE, label="retrieved schema in prompt")
label_bars(ax, b1, pad=0.3)
label_bars(ax, b2, pad=0.3)
ax.set_xticks(list(x), noise)
ax.set_ylim(0, 19)
ax.set_ylabel("correct answers (%)")
ax.legend(loc="upper right", fontsize=8.5)
save(fig, "fig_noise.pdf")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.9), gridspec_kw={"width_ratios": [3, 2]})
v = a1.bar([0, 1, 2], [28.1, 0.0, 27.3], width=0.55, color=[GRAY, ORANGE, BLUE])
label_bars(a1, v, pad=0.6)
a1.set_xticks([0, 1, 2], ["before the\nchange", "after: values\nfrom question", "after: values\nfrom graph"],
              fontsize=8.5)
a1.set_title("Stored values changed (139 questions)", fontsize=9.5, color=TEXT)
a1.set_ylim(0, 35)
a1.set_ylabel("correct answers (%)")
s = a2.bar([0, 1], [0.0, 27.9], width=0.55, color=[ORANGE, BLUE])
label_bars(a2, s, pad=0.6)
a2.set_xticks([0, 1], ["outdated\nschema", "current\nschema"], fontsize=8.5)
a2.set_title("Schema renamed (624 questions)", fontsize=9.5, color=TEXT)
a2.set_ylim(0, 35)
fig.legend(handles=[Patch(color=GRAY, label="original graph"),
                    Patch(color=ORANGE, label="static (does not look at the graph)"),
                    Patch(color=BLUE, label="graph-aware (reads the current graph)")],
           loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig("fig_change.pdf", bbox_inches="tight")
plt.close(fig)

hops = ["0 hops\n(313)", "1 hop\n(360)", "2 hops\n(165)", "3+ hops\n(18)"]
s3, s5 = [97, 49, 1, 0], [71, 51, 32, 28]
fig, ax = plt.subplots(figsize=(5.4, 2.8))
x = range(4)
b1 = ax.bar([i - 0.15 for i in x], s3, width=0.28, color=ORANGE, label="S3: LLM writes Cypher")
b2 = ax.bar([i + 0.15 for i in x], s5, width=0.28, color=BLUE, label="S5: LLM writes a plan")
label_bars(ax, b1, fmt="{:.0f}%", pad=1.2)
label_bars(ax, b2, fmt="{:.0f}%", pad=1.2)
ax.set_xticks(list(x), hops, fontsize=8.5)
ax.set_ylim(0, 112)
ax.set_ylabel("correct answers (%)")
ax.legend(loc="upper right", fontsize=8.5)
save(fig, "fig_hops.pdf")

cats = ["correct", "syntax error", "label/relationship\ndoes not exist", "wrong direction",
        "property does\nnot exist", "wrong path", "wrong aggregation", "other"]
colors = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET, GREEN, GRAY]
data = {
    "S0": [281, 114, 250, 44, 153, 0, 9, 5],
    "S1": [441, 122, 0, 62, 171, 9, 28, 23],
    "S2": [472, 123, 0, 39, 155, 7, 26, 34],
    "S3": [481, 124, 0, 31, 162, 6, 26, 26],
    "S5": [461, 85, 0, 6, 96, 84, 63, 61],
}
fig, ax = plt.subplots(figsize=(6.4, 3.0))
rows = list(data)[::-1]
for i, sysn in enumerate(rows):
    left = 0
    for j, c in enumerate(data[sysn]):
        w = c / 856 * 100
        if w <= 0:
            continue
        ax.barh(i, w - 0.3, left=left, height=0.55, color=colors[j])
        if w >= 6:
            ax.text(left + w / 2, i, f"{w:.0f}%", ha="center", va="center", fontsize=8,
                    color="white" if j in (0, 5, 6) else TEXT)
        left += w
ax.set_yticks(range(len(rows)), rows)
ax.set_xlim(0, 100)
ax.set_xlabel("share of all 856 questions (%)")
ax.grid(axis="y", visible=False)
ax.grid(axis="x", visible=True)
ax.legend(handles=[Patch(color=c, label=l.replace("\n", " ")) for c, l in zip(colors, cats)],
          loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize=7.8)
save(fig, "fig_errors.pdf")
print("figures written")
