"""Generate A/B comparison plots from evolution/ab_results.json."""

import json
import statistics
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).parent / "evolution"
DATA = OUT_DIR / "ab_results.json"
PLOT = OUT_DIR / "ab_plot.png"


def main():
    data = json.loads(DATA.read_text())
    games = [g for g in data["games"] if g.get("completed")]
    if not games:
        print("no completed games"); return

    treated_surv, control_surv = [], []
    treated_r1, control_r1 = [], []
    for g in games:
        for name, surv in g["survival"].items():
            (treated_surv if name == g["treated"] else control_surv).append(surv)
        for name, v in g["votes_round1"].items():
            (treated_r1 if name == g["treated"] else control_r1).append(v)

    # ---- figure ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.patch.set_facecolor("#0a0908")
    text = "#ede8e1"
    dim = "#948b80"
    accent = "#f5b800"
    pink = "#e36ba7"
    grid = "#2b2724"
    teal = "#5fbcb0"

    for ax in axes:
        ax.set_facecolor("#14110f")
        ax.tick_params(colors=text)
        for spine in ax.spines.values():
            spine.set_color(grid)
        ax.grid(True, color=grid, linewidth=0.5, axis="y")

    # ---- panel 1: avg survival round, bar comparison ----
    ax = axes[0]
    t_mean = statistics.mean(treated_surv) if treated_surv else 0
    c_mean = statistics.mean(control_surv) if control_surv else 0
    t_std  = statistics.pstdev(treated_surv) if len(treated_surv) > 1 else 0
    c_std  = statistics.pstdev(control_surv) if len(control_surv) > 1 else 0
    bars = ax.bar(
        ["TREATED\n(playbook)", "CONTROL\n(empty)"],
        [t_mean, c_mean],
        color=[accent, dim],
        edgecolor="#0a0908", linewidth=2,
        yerr=[t_std, c_std], ecolor=text, capsize=10,
    )
    for b, v, n in zip(bars, [t_mean, c_mean], [len(treated_surv), len(control_surv)]):
        ax.text(b.get_x() + b.get_width()/2, v + 0.05, f"{v:.2f}",
                ha="center", color=text, fontsize=13, fontweight="bold")
        ax.text(b.get_x() + b.get_width()/2, -0.15, f"n={n}",
                ha="center", color=dim, fontsize=10)
    ax.set_ylabel("avg round when eliminated", color=text, fontsize=11)
    ax.set_title("Survival round — treated vs control",
                 color=text, fontsize=13, loc="left", pad=12)

    # ---- panel 2: distribution of survival rounds ----
    ax = axes[1]
    all_rounds = sorted(set(treated_surv + control_surv))
    if all_rounds:
        x = np.arange(len(all_rounds))
        width = 0.4
        t_counts = [Counter(treated_surv).get(r, 0) for r in all_rounds]
        c_counts = [Counter(control_surv).get(r, 0) for r in all_rounds]
        # normalize to fraction of group
        t_n = sum(t_counts) or 1
        c_n = sum(c_counts) or 1
        t_frac = [x / t_n for x in t_counts]
        c_frac = [x / c_n for x in c_counts]
        ax.bar(x - width/2, t_frac, width, color=accent, label="Treated", edgecolor="#0a0908")
        ax.bar(x + width/2, c_frac, width, color=dim, label="Control", edgecolor="#0a0908")
        ax.set_xticks(x)
        ax.set_xticklabels([f"R{r}" for r in all_rounds])
        ax.set_ylabel("share of agents", color=text, fontsize=11)
        ax.set_title("Where agents got eliminated (R5 = survived to final 2)",
                     color=text, fontsize=13, loc="left", pad=12)
        ax.legend(facecolor="#14110f", edgecolor=grid, labelcolor=text)

    # ---- panel 3: round-1 votes received ----
    ax = axes[2]
    t_v = statistics.mean(treated_r1) if treated_r1 else 0
    c_v = statistics.mean(control_r1) if control_r1 else 0
    t_vs = statistics.pstdev(treated_r1) if len(treated_r1) > 1 else 0
    c_vs = statistics.pstdev(control_r1) if len(control_r1) > 1 else 0
    bars = ax.bar(
        ["TREATED", "CONTROL"],
        [t_v, c_v],
        color=[accent, dim],
        edgecolor="#0a0908", linewidth=2,
        yerr=[t_vs, c_vs], ecolor=text, capsize=10,
    )
    for b, v in zip(bars, [t_v, c_v]):
        ax.text(b.get_x() + b.get_width()/2, v + 0.05, f"{v:.2f}",
                ha="center", color=text, fontsize=13, fontweight="bold")
    ax.set_ylabel("avg round-1 votes received", color=text, fontsize=11)
    ax.set_title("Round 1 suspicion — votes received",
                 color=text, fontsize=13, loc="left", pad=12)

    # ---- caption ----
    cost = data.get("final_cost", 0)
    n_games = len(games)
    fig.suptitle(
        f"Pass The Turing Test — playbook A/B test  ·  {n_games} games, 1 treated agent per game  ·  ${cost:.2f}",
        color=accent, fontsize=15, y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(PLOT, dpi=150, facecolor="#0a0908", bbox_inches="tight")
    print(f"saved {PLOT}")


if __name__ == "__main__":
    main()
