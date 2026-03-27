#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║      CHROMOSOME COMPARISON TOOL — GA-Evolved vs. Hand-Tuned            ║
║                                                                        ║
║  Runs both chromosomes through identical scenarios, collects detailed   ║
║  metrics, and generates publication-ready comparison figures.           ║
╚══════════════════════════════════════════════════════════════════════════╝

USAGE
─────
  python compare_chromosomes.py \
      --ga_file   final_best_agent_2_15.pkl \
      --hand_file hand_tuned_chromosome.pkl \
      [--scenarios training1 training2 training3 crush aim_trainer] \
      [--runs_per_scenario 5] \
      [--output_dir comparison_results]

  All flags have sensible defaults — you can just run:
      python compare_chromosomes.py

OUTPUT
──────
  comparison_results/
    ├── comparison_report.txt          Full text report
    ├── metric_bars.png                Bar chart: all metrics side by side
    ├── radar_comparison.png           Radar/spider chart of normalized metrics
    ├── fitness_by_scenario.png        Per-scenario fitness breakdown
    ├── response_surface_diff.png      Side-by-side + difference heatmaps
    ├── input_sweep_overlay.png        Full-tree input sweeps overlaid
    ├── tree_structures.png            Both tree topologies
    ├── statistical_summary.csv        Raw numbers for LaTeX tables
    └── per_run_details.csv            Every individual run result
"""

import argparse
import os
import sys
import time
import math
import csv
import traceback
import warnings
from collections import defaultdict
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — works on headless systems
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from algorithms import (
    InputNode, FISNode, compile_chromosome, load_chromosome,
    fuzzy_tree_output, get_tree_size, gather_fis_nodes, gather_leaf_nodes,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_NAMES = {
    0: "Heading\nAlignment",
    1: "Closure\nRate",
    2: "Asteroid\nRadius",
    3: "Log\nDistance",
    4: "Collision\nUrgency",
}

INPUT_NAMES_SHORT = {
    0: "Heading", 1: "Closure", 2: "Radius", 3: "Distance", 4: "Collision",
}

AGENT_COLORS = {
    "GA-Evolved":  "#1565C0",   # blue
    "Hand-Tuned":  "#E65100",   # orange
}

AGENT_COLORS_LIGHT = {
    "GA-Evolved":  "#90CAF9",
    "Hand-Tuned":  "#FFCC80",
}


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 1: GAME EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def run_scenario(chromosome, scenario, game_settings):
    """
    Run a single chromosome on a single scenario.

    Returns a dict with detailed metrics, or an error dict on failure.
    """
    try:
        from kesslergame import TrainerEnvironment
        from redone_controller import FuzzyController
        from algorithms import SafeControllerWrapper

        compile_chromosome(chromosome)
        game = TrainerEnvironment(settings=game_settings)
        controller = FuzzyController(chromosome)
        safe_ctrl = SafeControllerWrapper(controller)

        t0 = time.perf_counter()
        score, perf_data = game.run(scenario=scenario, controllers=[safe_ctrl])
        wall_time = time.perf_counter() - t0

        team = score.teams[0]
        fitness = (team.asteroids_hit * team.accuracy) - 20 * team.deaths

        return {
            "asteroids_hit": team.asteroids_hit,
            "accuracy":      team.accuracy,
            "deaths":        team.deaths,
            "shots_fired":   team.shots_fired,
            "fitness":       fitness,
            "wall_time":     wall_time,
            "mean_eval_ms":  team.mean_eval_time * 1000 if team.mean_eval_time else 0.0,
            "error":         None,
        }
    except Exception as e:
        return {
            "asteroids_hit": 0, "accuracy": 0.0, "deaths": 0,
            "shots_fired": 0, "fitness": -99999.0, "wall_time": 0.0,
            "mean_eval_ms": 0.0, "error": str(e),
        }


def evaluate_agent(chromosome, agent_name, scenario_dict, runs_per_scenario, game_settings):
    """
    Evaluate one agent across all scenarios with multiple runs each.

    Returns:
        all_runs  — list of dicts (one per individual run)
        summary   — dict of {scenario_name: {metric: {mean, std, min, max}}}
    """
    all_runs = []
    summary = {}

    for s_name, scenario in scenario_dict.items():
        runs = []
        for run_idx in range(runs_per_scenario):
            print(f"    {agent_name} | {s_name} | run {run_idx+1}/{runs_per_scenario}...",
                  end=" ", flush=True)
            result = run_scenario(chromosome, scenario, game_settings)
            result["agent"] = agent_name
            result["scenario"] = s_name
            result["run"] = run_idx
            runs.append(result)

            if result["error"]:
                print(f"ERROR: {result['error']}")
            else:
                print(f"fitness={result['fitness']:.2f}")

        all_runs.extend(runs)

        # Compute summary statistics for this scenario
        metrics = ["asteroids_hit", "accuracy", "deaths", "shots_fired",
                    "fitness", "wall_time", "mean_eval_ms"]
        scenario_summary = {}
        for m in metrics:
            vals = [r[m] for r in runs if r["error"] is None]
            if vals:
                scenario_summary[m] = {
                    "mean": np.mean(vals),
                    "std":  np.std(vals),
                    "min":  np.min(vals),
                    "max":  np.max(vals),
                    "n":    len(vals),
                }
            else:
                scenario_summary[m] = {"mean": 0, "std": 0, "min": 0, "max": 0, "n": 0}
        summary[s_name] = scenario_summary

    return all_runs, summary


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 2: AGGREGATE STATISTICS
# ═══════════════════════════════════════════════════════════════════════════

def compute_aggregate(all_runs):
    """Compute overall means/stds across ALL runs for an agent."""
    valid = [r for r in all_runs if r["error"] is None]
    if not valid:
        return {}
    metrics = ["asteroids_hit", "accuracy", "deaths", "shots_fired",
               "fitness", "wall_time", "mean_eval_ms"]
    agg = {}
    for m in metrics:
        vals = [r[m] for r in valid]
        agg[m] = {"mean": np.mean(vals), "std": np.std(vals),
                   "min": np.min(vals), "max": np.max(vals), "n": len(vals)}
    return agg


def count_tree_params(root):
    """Count total tunable parameters in the tree."""
    fis_nodes = []
    gather_fis_nodes(root, fis_nodes)
    # Each FIS node: 2 MF centers + 9 rule constants = 11 params
    return len(fis_nodes) * 11


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 3: TEXT REPORT
# ═══════════════════════════════════════════════════════════════════════════

def write_text_report(filepath, ga_runs, ga_summary, ga_agg, ga_chrom,
                      ht_runs, ht_summary, ht_agg, ht_chrom, scenario_names):
    """Write a comprehensive text comparison report."""
    lines = []
    W = 76

    lines.append("=" * W)
    lines.append("  CHROMOSOME COMPARISON REPORT")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * W)

    # --- Tree complexity ---
    lines.append("\n┌─ TREE COMPLEXITY")
    lines.append(f"│  {'':20s} {'GA-Evolved':>15s} {'Hand-Tuned':>15s}")
    lines.append(f"│  {'Tree size (nodes)':20s} {get_tree_size(ga_chrom):>15d} {get_tree_size(ht_chrom):>15d}")
    ga_fis, ht_fis = [], []
    gather_fis_nodes(ga_chrom, ga_fis)
    gather_fis_nodes(ht_chrom, ht_fis)
    lines.append(f"│  {'FIS nodes':20s} {len(ga_fis):>15d} {len(ht_fis):>15d}")
    lines.append(f"│  {'Tunable params':20s} {count_tree_params(ga_chrom):>15d} {count_tree_params(ht_chrom):>15d}")
    lines.append("└" + "─" * (W - 1))

    # --- Overall aggregates ---
    lines.append("\n┌─ OVERALL PERFORMANCE (mean ± std across all scenarios & runs)")
    lines.append(f"│  {'Metric':20s} {'GA-Evolved':>20s} {'Hand-Tuned':>20s} {'Δ (GA−HT)':>12s}")
    lines.append(f"│  {'─'*20} {'─'*20} {'─'*20} {'─'*12}")

    display_metrics = [
        ("Fitness",        "fitness"),
        ("Asteroids Hit",  "asteroids_hit"),
        ("Accuracy",       "accuracy"),
        ("Deaths",         "deaths"),
        ("Shots Fired",    "shots_fired"),
        ("Wall Time (s)",  "wall_time"),
        ("Eval Time (ms)", "mean_eval_ms"),
    ]

    for label, key in display_metrics:
        ga_m = ga_agg.get(key, {}).get("mean", 0)
        ga_s = ga_agg.get(key, {}).get("std", 0)
        ht_m = ht_agg.get(key, {}).get("mean", 0)
        ht_s = ht_agg.get(key, {}).get("std", 0)
        delta = ga_m - ht_m
        lines.append(f"│  {label:20s} {ga_m:>8.2f} ± {ga_s:<8.2f} {ht_m:>8.2f} ± {ht_s:<8.2f} {delta:>+10.2f}")
    lines.append("└" + "─" * (W - 1))

    # --- Per-scenario breakdown ---
    lines.append("\n┌─ PER-SCENARIO BREAKDOWN (Fitness: mean ± std)")
    lines.append(f"│  {'Scenario':22s} {'GA-Evolved':>20s} {'Hand-Tuned':>20s} {'Winner':>10s}")
    lines.append(f"│  {'─'*22} {'─'*20} {'─'*20} {'─'*10}")

    ga_wins, ht_wins, ties = 0, 0, 0
    for s_name in scenario_names:
        ga_f = ga_summary.get(s_name, {}).get("fitness", {})
        ht_f = ht_summary.get(s_name, {}).get("fitness", {})
        ga_m, ga_s = ga_f.get("mean", 0), ga_f.get("std", 0)
        ht_m, ht_s = ht_f.get("mean", 0), ht_f.get("std", 0)
        if abs(ga_m - ht_m) < 0.01:
            winner = "TIE"
            ties += 1
        elif ga_m > ht_m:
            winner = "GA ✓"
            ga_wins += 1
        else:
            winner = "HAND ✓"
            ht_wins += 1
        lines.append(f"│  {s_name:22s} {ga_m:>8.2f} ± {ga_s:<8.2f} {ht_m:>8.2f} ± {ht_s:<8.2f} {winner:>10s}")

    lines.append(f"│")
    lines.append(f"│  Scenario wins: GA={ga_wins}, Hand={ht_wins}, Tie={ties}")
    lines.append("└" + "─" * (W - 1))

    # --- Per-scenario detailed metrics ---
    for s_name in scenario_names:
        lines.append(f"\n  ── {s_name} ──")
        lines.append(f"  {'Metric':18s} {'GA-Evolved':>20s} {'Hand-Tuned':>20s}")
        for label, key in display_metrics[:5]:
            ga_v = ga_summary.get(s_name, {}).get(key, {})
            ht_v = ht_summary.get(s_name, {}).get(key, {})
            lines.append(
                f"  {label:18s} {ga_v.get('mean',0):>8.2f} ± {ga_v.get('std',0):<8.2f}"
                f" {ht_v.get('mean',0):>8.2f} ± {ht_v.get('std',0):<8.2f}"
            )

    lines.append("\n" + "=" * W)
    lines.append("  END OF REPORT")
    lines.append("=" * W)

    report_text = "\n".join(lines)
    # NEW (Fixed)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(report_text)
    return report_text


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 4: CSV EXPORT
# ═══════════════════════════════════════════════════════════════════════════

def write_per_run_csv(filepath, ga_runs, ht_runs):
    all_runs = ga_runs + ht_runs
    if not all_runs:
        return
    keys = ["agent", "scenario", "run", "fitness", "asteroids_hit",
            "accuracy", "deaths", "shots_fired", "wall_time", "mean_eval_ms", "error"]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        for r in all_runs:
            writer.writerow(r)


def write_summary_csv(filepath, ga_summary, ht_summary, ga_agg, ht_agg, scenario_names):
    rows = []
    metrics = ["fitness", "asteroids_hit", "accuracy", "deaths", "shots_fired"]
    # Per-scenario
    for s_name in scenario_names:
        for m in metrics:
            ga_v = ga_summary.get(s_name, {}).get(m, {})
            ht_v = ht_summary.get(s_name, {}).get(m, {})
            rows.append({
                "scope": s_name, "metric": m,
                "ga_mean": f"{ga_v.get('mean',0):.4f}", "ga_std": f"{ga_v.get('std',0):.4f}",
                "ht_mean": f"{ht_v.get('mean',0):.4f}", "ht_std": f"{ht_v.get('std',0):.4f}",
            })
    # Overall
    for m in metrics:
        ga_v = ga_agg.get(m, {})
        ht_v = ht_agg.get(m, {})
        rows.append({
            "scope": "OVERALL", "metric": m,
            "ga_mean": f"{ga_v.get('mean',0):.4f}", "ga_std": f"{ga_v.get('std',0):.4f}",
            "ht_mean": f"{ht_v.get('mean',0):.4f}", "ht_std": f"{ht_v.get('std',0):.4f}",
        })

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scope","metric","ga_mean","ga_std","ht_mean","ht_std"])
        writer.writeheader()
        writer.writerows(rows)


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 5: VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════

# --- 5a. Grouped bar chart of key metrics ---

def plot_metric_bars(ga_agg, ht_agg, output_path):
    """Side-by-side bar chart for overall aggregate metrics."""
    metrics = [
        ("Fitness",       "fitness"),
        ("Asteroids Hit", "asteroids_hit"),
        ("Accuracy",      "accuracy"),
        ("Deaths",        "deaths"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 5))
    fig.suptitle("Overall Performance Comparison", fontsize=16, fontweight='bold', y=1.02)

    for ax, (label, key) in zip(axes, metrics):
        ga_m = ga_agg.get(key, {}).get("mean", 0)
        ga_s = ga_agg.get(key, {}).get("std", 0)
        ht_m = ht_agg.get(key, {}).get("mean", 0)
        ht_s = ht_agg.get(key, {}).get("std", 0)

        x = np.array([0, 1])
        bars = ax.bar(x, [ga_m, ht_m],
                       yerr=[ga_s, ht_s],
                       color=[AGENT_COLORS["GA-Evolved"], AGENT_COLORS["Hand-Tuned"]],
                       edgecolor='black', linewidth=0.8,
                       capsize=6, width=0.55, zorder=3)

        # Value labels on bars
        for bar, val, std in zip(bars, [ga_m, ht_m], [ga_s, ht_s]):
            y = bar.get_height()
            sign = 1 if y >= 0 else -1
            ax.text(bar.get_x() + bar.get_width() / 2, y + sign * std + sign * abs(y) * 0.03,
                    f'{val:.2f}', ha='center', va='bottom' if y >= 0 else 'top',
                    fontsize=10, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(["GA-Evolved", "Hand-Tuned"], fontsize=10)
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3, zorder=0)
        ax.axhline(0, color='black', linewidth=0.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


# --- 5b. Per-scenario fitness grouped bar chart ---

def plot_fitness_by_scenario(ga_summary, ht_summary, scenario_names, output_path):
    """Grouped bar chart: fitness per scenario, GA vs Hand-Tuned."""
    n = len(scenario_names)
    x = np.arange(n)
    width = 0.35

    ga_means = [ga_summary.get(s, {}).get("fitness", {}).get("mean", 0) for s in scenario_names]
    ga_stds  = [ga_summary.get(s, {}).get("fitness", {}).get("std", 0) for s in scenario_names]
    ht_means = [ht_summary.get(s, {}).get("fitness", {}).get("mean", 0) for s in scenario_names]
    ht_stds  = [ht_summary.get(s, {}).get("fitness", {}).get("std", 0) for s in scenario_names]

    fig, ax = plt.subplots(figsize=(max(10, 2.5 * n), 6))

    bars1 = ax.bar(x - width/2, ga_means, width, yerr=ga_stds, label='GA-Evolved',
                    color=AGENT_COLORS["GA-Evolved"], edgecolor='black', linewidth=0.6,
                    capsize=5, zorder=3)
    bars2 = ax.bar(x + width/2, ht_means, width, yerr=ht_stds, label='Hand-Tuned',
                    color=AGENT_COLORS["Hand-Tuned"], edgecolor='black', linewidth=0.6,
                    capsize=5, zorder=3)

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            y = bar.get_height()
            if abs(y) > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2, y + (abs(y)*0.02 + 0.5) * (1 if y >= 0 else -1),
                        f'{y:.1f}', ha='center', va='bottom' if y >= 0 else 'top', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names, rotation=30, ha='right', fontsize=10)
    ax.set_ylabel("Fitness Score", fontsize=12)
    ax.set_title("Fitness by Scenario", fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, axis='y', alpha=0.3, zorder=0)
    ax.axhline(0, color='black', linewidth=0.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


# --- 5c. Radar / Spider chart ---

def plot_radar(ga_agg, ht_agg, output_path):
    """
    Radar chart comparing normalized metrics.
    Metrics are scaled so that "better" always points outward.
    """
    metric_defs = [
        ("Fitness",       "fitness",       1),   # higher is better
        ("Asteroids Hit", "asteroids_hit", 1),
        ("Accuracy",      "accuracy",      1),
        ("Survival",      "deaths",       -1),   # fewer deaths is better → invert
    ]

    labels = [m[0] for m in metric_defs]
    ga_vals = []
    ht_vals = []

    for label, key, direction in metric_defs:
        ga_m = ga_agg.get(key, {}).get("mean", 0)
        ht_m = ht_agg.get(key, {}).get("mean", 0)
        if direction == -1:
            # Invert: fewer deaths → higher radar value
            # Use max(ga, ht) + 1 as ceiling to avoid division by zero
            ceiling = max(abs(ga_m), abs(ht_m), 1)
            ga_vals.append(max(0, 1 - ga_m / ceiling))
            ht_vals.append(max(0, 1 - ht_m / ceiling))
        else:
            ga_vals.append(ga_m)
            ht_vals.append(ht_m)

    # Normalize to [0, 1] per metric
    for i in range(len(ga_vals)):
        vmax = max(abs(ga_vals[i]), abs(ht_vals[i]), 1e-6)
        ga_vals[i] /= vmax
        ht_vals[i] /= vmax

    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    ga_vals += ga_vals[:1]
    ht_vals += ht_vals[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)

    ax.plot(angles, ga_vals, 'o-', linewidth=2.5, color=AGENT_COLORS["GA-Evolved"],
            label='GA-Evolved', markersize=7)
    ax.fill(angles, ga_vals, alpha=0.15, color=AGENT_COLORS["GA-Evolved"])

    ax.plot(angles, ht_vals, 's-', linewidth=2.5, color=AGENT_COLORS["Hand-Tuned"],
            label='Hand-Tuned', markersize=7)
    ax.fill(angles, ht_vals, alpha=0.15, color=AGENT_COLORS["Hand-Tuned"])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.set_title("Normalized Performance Profile", fontsize=14, fontweight='bold', pad=25)
    ax.legend(loc='lower right', bbox_to_anchor=(1.25, 0), fontsize=11)

    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


# --- 5d. Input sweep overlay ---

def plot_input_sweep_overlay(ga_chrom, ht_chrom, output_path, resolution=120):
    """Overlay both trees' marginal responses to each input."""
    compile_chromosome(ga_chrom)
    compile_chromosome(ht_chrom)

    n_inputs = 5
    defaults = np.full(n_inputs, 0.5)
    x = np.linspace(0, 1, resolution)

    fig, axes = plt.subplots(1, n_inputs, figsize=(4.2 * n_inputs, 4.5), sharey=True)
    fig.suptitle("Full-Tree Output: Single-Input Sweeps (others at 0.5)",
                 fontsize=14, fontweight='bold')

    for idx in range(n_inputs):
        batch = np.tile(defaults, (resolution, 1))
        batch[:, idx] = x

        ga_out = fuzzy_tree_output(ga_chrom, batch)
        ht_out = fuzzy_tree_output(ht_chrom, batch)

        ax = axes[idx]
        ax.plot(x, ga_out, lw=2.5, color=AGENT_COLORS["GA-Evolved"],
                label='GA-Evolved', zorder=3)
        ax.plot(x, ht_out, lw=2.5, color=AGENT_COLORS["Hand-Tuned"],
                label='Hand-Tuned', ls='--', zorder=3)

        ax.fill_between(x, ga_out, ht_out, alpha=0.12, color='gray')

        ax.set_xlabel(INPUT_NAMES.get(idx, f"Input {idx}"), fontsize=9)
        if idx == 0:
            ax.set_ylabel("Tree Output", fontsize=10)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Sweep [{idx}]", fontsize=10, fontweight='bold')
        if idx == n_inputs - 1:
            ax.legend(fontsize=9, loc='best')

    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


# --- 5e. Response surface comparison heatmaps ---

def _compute_mfs(x, center):
    c = np.clip(center, 0.001, 0.999)
    low  = np.maximum(0.0, 1.0 - x / c)
    high = np.maximum(0.0, 1.0 - (1.0 - x) / (1.0 - c))
    med  = np.maximum(0.0, 1.0 - low - high)
    return low, med, high


def _fis_surface(c1, c2, rules, res=80):
    x = np.linspace(0, 1, res)
    X, Y = np.meshgrid(x, x)
    Z = np.zeros_like(X)
    la, ma, ha = _compute_mfs(X, c1)
    lb, mb, hb = _compute_mfs(Y, c2)
    for i, a_mf in enumerate([la, ma, ha]):
        for j, b_mf in enumerate([lb, mb, hb]):
            Z += a_mf * b_mf * rules[i * 3 + j]
    return X, Y, Z


def plot_response_surface_diff(ga_chrom, ht_chrom, output_path):
    """
    Side-by-side + difference heatmaps for pairwise input sweeps
    through the full tree (not individual FIS nodes).
    """
    compile_chromosome(ga_chrom)
    compile_chromosome(ht_chrom)

    # Pick the most informative input pairs
    pairs = [(0, 3), (2, 3), (0, 4), (1, 2)]  # heading×dist, radius×dist, heading×coll, closure×radius
    n_pairs = len(pairs)
    res = 60

    fig, axes = plt.subplots(n_pairs, 3, figsize=(15, 4.2 * n_pairs))
    fig.suptitle("Full-Tree Response Surfaces: GA vs. Hand-Tuned vs. Difference",
                 fontsize=15, fontweight='bold', y=1.01)

    defaults = np.full(5, 0.5)
    x = np.linspace(0, 1, res)

    for row, (i, j) in enumerate(pairs):
        X, Y = np.meshgrid(x, x)
        batch = np.tile(defaults, (res * res, 1))
        batch[:, i] = X.ravel()
        batch[:, j] = Y.ravel()

        ga_z = fuzzy_tree_output(ga_chrom, batch).reshape(res, res)
        ht_z = fuzzy_tree_output(ht_chrom, batch).reshape(res, res)
        diff = ga_z - ht_z

        vmin = min(ga_z.min(), ht_z.min())
        vmax = max(ga_z.max(), ht_z.max())

        titles = ["GA-Evolved", "Hand-Tuned", "Difference (GA − Hand)"]
        data = [ga_z, ht_z, diff]
        cmaps = [cm.viridis, cm.viridis, cm.RdBu_r]

        for col in range(3):
            ax = axes[row, col] if n_pairs > 1 else axes[col]
            if col < 2:
                im = ax.imshow(data[col], origin='lower', extent=[0,1,0,1],
                               aspect='auto', cmap=cmaps[col], vmin=vmin, vmax=vmax)
            else:
                abs_max = max(abs(diff.min()), abs(diff.max()), 0.01)
                im = ax.imshow(data[col], origin='lower', extent=[0,1,0,1],
                               aspect='auto', cmap=cmaps[col], vmin=-abs_max, vmax=abs_max)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xlabel(f"[{i}] {INPUT_NAMES_SHORT[i]}", fontsize=9)
            ax.set_ylabel(f"[{j}] {INPUT_NAMES_SHORT[j]}", fontsize=9)
            if row == 0:
                ax.set_title(titles[col], fontsize=11, fontweight='bold')

    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


# --- 5f. Tree structure side-by-side ---

def _draw_tree_on_ax(ax, root, title):
    """Draw a tree diagram onto a matplotlib axes."""
    positions = {}
    labels = {}
    colors = {}
    edges = []
    node_id = [0]

    def layout(node, x, y, dx):
        nid = node_id[0]
        node_id[0] += 1
        positions[nid] = (x, y)

        if isinstance(node, InputNode):
            name = INPUT_NAMES_SHORT.get(node.index, f"In{node.index}")
            labels[nid] = f"[{node.index}] {name}"
            colors[nid] = '#FFCDD2'
            return nid

        lbl = getattr(node, '_label', '').split('\n')[0] if hasattr(node, '_label') else 'FIS'
        labels[nid] = f"{lbl}\nc1={node.medium1_center:.2f}\nc2={node.medium2_center:.2f}"
        colors[nid] = '#C8E6C9'

        left_id = layout(node.left, x - dx, y - 1, dx / 2)
        right_id = layout(node.right, x + dx, y - 1, dx / 2)
        edges.append((nid, left_id))
        edges.append((nid, right_id))
        return nid

    layout(root, 0, 0, 2.5)

    for (p, c) in edges:
        px, py = positions[p]
        cx, cy = positions[c]
        ax.plot([px, cx], [py, cy], 'k-', lw=1.2, alpha=0.5, zorder=1)

    for nid, (x, y) in positions.items():
        bbox = dict(boxstyle="round,pad=0.4", facecolor=colors[nid],
                    edgecolor='#555', linewidth=1.2, alpha=0.9)
        ax.text(x, y, labels[nid], ha='center', va='center',
                fontsize=7, fontweight='bold', bbox=bbox, zorder=2)

    margin = 1.2
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)


def plot_tree_structures(ga_chrom, ht_chrom, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle("Tree Structure Comparison", fontsize=16, fontweight='bold', y=0.98)

    _draw_tree_on_ax(ax1, ga_chrom, f"GA-Evolved (nodes={get_tree_size(ga_chrom)}, params={count_tree_params(ga_chrom)})")
    _draw_tree_on_ax(ax2, ht_chrom, f"Hand-Tuned (nodes={get_tree_size(ht_chrom)}, params={count_tree_params(ht_chrom)})")

    input_patch = mpatches.Patch(color='#FFCDD2', label='Input Node')
    fis_patch   = mpatches.Patch(color='#C8E6C9', label='FIS Node')
    fig.legend(handles=[input_patch, fis_patch], loc='lower center', ncol=2, fontsize=11)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


# --- 5g. Per-scenario multi-metric grouped bars ---

def plot_detailed_scenario_bars(ga_summary, ht_summary, scenario_names, output_path):
    """For each scenario: asteroids_hit, accuracy, deaths as grouped bars."""
    metrics = [
        ("Asteroids Hit", "asteroids_hit"),
        ("Accuracy",      "accuracy"),
        ("Deaths",        "deaths"),
    ]
    n_scen = len(scenario_names)
    n_met  = len(metrics)

    fig, axes = plt.subplots(n_met, 1, figsize=(max(10, 2.5 * n_scen), 4.5 * n_met), sharex=True)
    if n_met == 1:
        axes = [axes]

    fig.suptitle("Detailed Per-Scenario Metrics", fontsize=15, fontweight='bold', y=1.01)

    x = np.arange(n_scen)
    width = 0.35

    for ax, (label, key) in zip(axes, metrics):
        ga_m = [ga_summary.get(s, {}).get(key, {}).get("mean", 0) for s in scenario_names]
        ga_s = [ga_summary.get(s, {}).get(key, {}).get("std", 0)  for s in scenario_names]
        ht_m = [ht_summary.get(s, {}).get(key, {}).get("mean", 0) for s in scenario_names]
        ht_s = [ht_summary.get(s, {}).get(key, {}).get("std", 0)  for s in scenario_names]

        ax.bar(x - width/2, ga_m, width, yerr=ga_s, label='GA-Evolved',
               color=AGENT_COLORS["GA-Evolved"], edgecolor='black', lw=0.5, capsize=4, zorder=3)
        ax.bar(x + width/2, ht_m, width, yerr=ht_s, label='Hand-Tuned',
               color=AGENT_COLORS["Hand-Tuned"], edgecolor='black', lw=0.5, capsize=4, zorder=3)

        ax.set_ylabel(label, fontsize=11, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3, zorder=0)
        ax.legend(fontsize=9)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(scenario_names, rotation=30, ha='right', fontsize=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Compare GA-Evolved vs Hand-Tuned fuzzy tree chromosomes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ga_file", type=str, default="final_best_agent_2_16.pkl",
                        help="Path to GA-evolved chromosome .pkl")
    parser.add_argument("--hand_file", type=str, default="final_best_agent_2_28.pkl",
                        help="Path to hand-tuned chromosome .pkl")
    parser.add_argument("--scenarios", nargs="+",
                        default=["training1", "training2", "training3", "crush", "aim_trainer"],
                        help="Scenario names to evaluate on")
    parser.add_argument("--runs_per_scenario", type=int, default=3,
                        help="Number of runs per scenario (for variance estimation)")
    parser.add_argument("--output_dir", type=str, default="comparison_results",
                        help="Directory for output files")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip game evaluation; only generate structural/response plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load chromosomes ---
    print("\n" + "=" * 65)
    print("  CHROMOSOME COMPARISON TOOL")
    print("=" * 65)

    print(f"\n  Loading GA-Evolved:  {args.ga_file}")
    if not os.path.isfile(args.ga_file):
        print(f"  ERROR: File not found: {args.ga_file}")
        print(f"  Tip: Run the GA first, or point --ga_file to your .pkl")
        sys.exit(1)
    ga_chrom = load_chromosome(args.ga_file)

    print(f"  Loading Hand-Tuned:  {args.hand_file}")
    if not os.path.isfile(args.hand_file):
        print(f"  ERROR: File not found: {args.hand_file}")
        print(f"  Tip: Run hand_tune_chromosome.py first to create it")
        sys.exit(1)
    ht_chrom = load_chromosome(args.hand_file)

    compile_chromosome(ga_chrom)
    compile_chromosome(ht_chrom)
    print(f"  GA tree: {get_tree_size(ga_chrom)} nodes, {count_tree_params(ga_chrom)} params")
    print(f"  HT tree: {get_tree_size(ht_chrom)} nodes, {count_tree_params(ht_chrom)} params")

    # --- Structural / response plots (no game needed) ---
    print(f"\n  Generating structural comparison plots...")

    plot_tree_structures(
        ga_chrom, ht_chrom,
        os.path.join(args.output_dir, "tree_structures.png"))
    print("    ✓ tree_structures.png")

    plot_input_sweep_overlay(
        ga_chrom, ht_chrom,
        os.path.join(args.output_dir, "input_sweep_overlay.png"))
    print("    ✓ input_sweep_overlay.png")

    plot_response_surface_diff(
        ga_chrom, ht_chrom,
        os.path.join(args.output_dir, "response_surface_diff.png"))
    print("    ✓ response_surface_diff.png")

    if args.skip_eval:
        print("\n  --skip_eval: Skipping game evaluation. Done.\n")
        return

    # --- Load scenarios ---
    try:
        from scenarios import scenarios as all_scenarios
    except ImportError:
        print("  ERROR: Cannot import scenarios.py — is it on your path?")
        sys.exit(1)

    scenario_dict = {}
    for s_name in args.scenarios:
        if s_name in all_scenarios:
            scenario_dict[s_name] = all_scenarios[s_name]
        else:
            print(f"  WARNING: Scenario '{s_name}' not found, skipping.")
    
    if not scenario_dict:
        print("  ERROR: No valid scenarios found.")
        sys.exit(1)

    scenario_names = list(scenario_dict.keys())
    print(f"\n  Scenarios: {scenario_names}")
    print(f"  Runs per scenario: {args.runs_per_scenario}")
    print(f"  Total evaluations: {2 * len(scenario_names) * args.runs_per_scenario}")

    game_settings = {
        "frequency": 30,
        "perf_tracker": True,
        "prints_on": False,
        "graphics_type": 0,
        "realtime_multiplier": 0,
        "time_limit": 120.0,
    }

    # --- Evaluate both agents ---
    print(f"\n{'─'*65}")
    print("  Evaluating GA-Evolved agent...")
    print(f"{'─'*65}")
    ga_runs, ga_summary = evaluate_agent(
        ga_chrom, "GA-Evolved", scenario_dict, args.runs_per_scenario, game_settings)

    print(f"\n{'─'*65}")
    print("  Evaluating Hand-Tuned agent...")
    print(f"{'─'*65}")
    ht_runs, ht_summary = evaluate_agent(
        ht_chrom, "Hand-Tuned", scenario_dict, args.runs_per_scenario, game_settings)

    ga_agg = compute_aggregate(ga_runs)
    ht_agg = compute_aggregate(ht_runs)

    # --- Generate all outputs ---
    print(f"\n{'─'*65}")
    print("  Generating outputs...")
    print(f"{'─'*65}")

    # Text report
    write_text_report(
        os.path.join(args.output_dir, "comparison_report.txt"),
        ga_runs, ga_summary, ga_agg, ga_chrom,
        ht_runs, ht_summary, ht_agg, ht_chrom,
        scenario_names)
    print("  ✓ comparison_report.txt")

    # CSVs
    write_per_run_csv(os.path.join(args.output_dir, "per_run_details.csv"), ga_runs, ht_runs)
    print("  ✓ per_run_details.csv")

    write_summary_csv(os.path.join(args.output_dir, "statistical_summary.csv"),
                      ga_summary, ht_summary, ga_agg, ht_agg, scenario_names)
    print("  ✓ statistical_summary.csv")

    # Performance plots
    plot_metric_bars(ga_agg, ht_agg, os.path.join(args.output_dir, "metric_bars.png"))
    print("  ✓ metric_bars.png")

    plot_fitness_by_scenario(ga_summary, ht_summary, scenario_names,
                             os.path.join(args.output_dir, "fitness_by_scenario.png"))
    print("  ✓ fitness_by_scenario.png")

    plot_radar(ga_agg, ht_agg, os.path.join(args.output_dir, "radar_comparison.png"))
    print("  ✓ radar_comparison.png")

    plot_detailed_scenario_bars(ga_summary, ht_summary, scenario_names,
                                os.path.join(args.output_dir, "detailed_scenario_bars.png"))
    print("  ✓ detailed_scenario_bars.png")

    # --- Final summary ---
    ga_fit = ga_agg.get("fitness", {}).get("mean", 0)
    ht_fit = ht_agg.get("fitness", {}).get("mean", 0)
    winner = "GA-Evolved" if ga_fit > ht_fit else "Hand-Tuned" if ht_fit > ga_fit else "TIE"

    print(f"\n{'='*65}")
    print(f"  RESULT SUMMARY")
    print(f"{'='*65}")
    print(f"  GA-Evolved avg fitness:  {ga_fit:>10.2f}")
    print(f"  Hand-Tuned avg fitness:  {ht_fit:>10.2f}")
    print(f"  Difference (GA − Hand):  {ga_fit - ht_fit:>+10.2f}")
    print(f"  Overall winner:          {winner}")
    print(f"\n  All outputs saved to: {os.path.abspath(args.output_dir)}/")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()