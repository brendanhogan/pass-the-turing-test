"""Generate plots from evolution/results.json."""

import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT_DIR = Path(__file__).parent / "evolution"
DATA = OUT_DIR / "results.json"
PLOT = OUT_DIR / "plot.png"
TACTICS_HISTORY = OUT_DIR / "tactics_history.txt"


def main():
    data = json.loads(DATA.read_text())
    gens = data["generations"]

    # Per-generation aggregates
    gen_labels = []
    entropies = []        # list of lists (one per generation)
    modal_shares = []     # list of lists
    completed_rates = []
    survived_to_final = []

    for g in gens:
        gen_labels.append(g["gen"])
        e = [game["round1_entropy"] for game in g["games"] if game["round1_entropy"] is not None]
        m = [game["round1_modal_share"] for game in g["games"] if game["round1_modal_share"] is not None]
        c = [int(game["completed"]) for game in g["games"]]
        # "survived to final" = how many made it past last round per game (always 2 if completed)
        # Better signal: mean survival round across all agents/games
        all_survival = []
        for game in g["games"]:
            for name, r in game["survival_round"].items():
                all_survival.append(r)
        entropies.append(e)
        modal_shares.append(m)
        completed_rates.append(sum(c) / max(len(c), 1))
        survived_to_final.append(statistics.mean(all_survival) if all_survival else 0)

    # ---- big figure with 3 subplots stacked ----
    fig, axes = plt.subplots(3, 1, figsize=(11, 13))
    fig.patch.set_facecolor("#0a0908")
    accent = "#f5b800"
    text_color = "#ede8e1"
    dim = "#948b80"
    grid_color = "#2b2724"

    for ax in axes:
        ax.set_facecolor("#14110f")
        ax.tick_params(colors=text_color)
        for spine in ax.spines.values():
            spine.set_color(grid_color)
        ax.grid(True, color=grid_color, linewidth=0.5, axis="y")

    # ---- panel 1: round-1 entropy ----
    ax = axes[0]
    means = [statistics.mean(e) if e else 0 for e in entropies]
    stds = [statistics.pstdev(e) if len(e) > 1 else 0 for e in entropies]
    # individual points scattered
    for i, e_list in enumerate(entropies):
        ax.scatter([i] * len(e_list), e_list, color=accent, alpha=0.35, s=40, zorder=2)
    # mean line
    ax.plot(gen_labels, means, color=accent, linewidth=2.5, marker="o",
            markersize=10, markerfacecolor=accent, markeredgecolor="#0a0908",
            markeredgewidth=2, zorder=3, label="Mean")
    # ±1 std shaded
    upper = [m + s for m, s in zip(means, stds)]
    lower = [m - s for m, s in zip(means, stds)]
    ax.fill_between(gen_labels, lower, upper, color=accent, alpha=0.12, zorder=1)
    ax.set_title("Round-1 vote entropy (higher = group is more confused = better imposters)",
                 color=text_color, fontsize=13, pad=15, loc="left")
    ax.set_ylabel("Shannon entropy (bits)", color=text_color, fontsize=11)
    ax.set_xticks(gen_labels)
    ax.set_xlabel("Generation", color=dim, fontsize=10)
    ax.legend(facecolor="#14110f", edgecolor=grid_color, labelcolor=text_color)

    # ---- panel 2: modal share ----
    ax = axes[1]
    m_means = [statistics.mean(m) if m else 0 for m in modal_shares]
    for i, m_list in enumerate(modal_shares):
        ax.scatter([i] * len(m_list), m_list, color="#e36ba7", alpha=0.35, s=40, zorder=2)
    ax.plot(gen_labels, m_means, color="#e36ba7", linewidth=2.5, marker="o",
            markersize=10, markerfacecolor="#e36ba7", markeredgecolor="#0a0908",
            markeredgewidth=2, zorder=3)
    ax.set_title("Round-1 modal vote share (lower = group can't agree on the AI)",
                 color=text_color, fontsize=13, pad=15, loc="left")
    ax.set_ylabel("share of votes on top target", color=text_color, fontsize=11)
    ax.set_xticks(gen_labels)
    ax.set_xlabel("Generation", color=dim, fontsize=10)
    ax.set_ylim(0, 1)

    # ---- panel 3: tactics over time ----
    ax = axes[2]
    n_tactics = [len(g["tactics_out"]) for g in gens]
    ax.bar(gen_labels, n_tactics, color="#5fbcb0", edgecolor="#0a0908", linewidth=1.5)
    ax.set_title("Tactics carried into next generation",
                 color=text_color, fontsize=13, pad=15, loc="left")
    ax.set_ylabel("count", color=text_color, fontsize=11)
    ax.set_xticks(gen_labels)
    ax.set_xlabel("Generation", color=dim, fontsize=10)

    # ---- caption ----
    cost = data.get("final_cost", 0)
    games = sum(len(g["games"]) for g in gens)
    fig.suptitle(
        f"Pass The Turing Test — prompt evolution  ·  {games} games, {len(gens)} generations  ·  ${cost:.2f}",
        color=accent, fontsize=15, y=0.995, x=0.5,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(PLOT, dpi=150, facecolor="#0a0908", bbox_inches="tight")
    print(f"saved {PLOT}")

    # ---- tactics evolution writeup ----
    lines = []
    lines.append(f"Total games: {games} across {len(gens)} generations")
    lines.append(f"Final cost: ${cost:.4f}")
    lines.append(f"Final tactics ({len(data.get('final_tactics', []))}):")
    for t in data.get("final_tactics", []):
        lines.append(f"  - {t}")
    lines.append("")
    lines.append("=" * 60)
    lines.append("Per-generation evolution")
    lines.append("=" * 60)
    for g in gens:
        lines.append(f"\n[Gen {g['gen']}] mean R1 entropy: "
                     f"{statistics.mean([x['round1_entropy'] for x in g['games'] if x['round1_entropy'] is not None]) if any(x['round1_entropy'] for x in g['games']) else 'n/a':.3f} | "
                     f"tactics carried in: {len(g['tactics_in'])} → out: {len(g['tactics_out'])}")
        for m in g.get("meta", []):
            lines.append(f"  · {m['name']} R{m['round']}: {m.get('diagnosis', '')}")
            for t in m.get("add", []):
                lines.append(f"     + {t}")
            for t in m.get("remove", []):
                lines.append(f"     - {t}")
    TACTICS_HISTORY.write_text("\n".join(lines))
    print(f"saved {TACTICS_HISTORY}")


if __name__ == "__main__":
    main()
