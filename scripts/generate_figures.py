#!/usr/bin/env python3
"""Generate publication figures from experiment_results.json. CPU-only, fast."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ─── Load results ───
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "results")
with open(os.path.join(RESULTS_DIR, "tables", "experiment_results.json")) as f:
    data = json.load(f)

FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─── Color palette ───
C_BLUE = "#2563EB"
C_RED = "#DC2626"
C_GREEN = "#059669"
C_ORANGE = "#D97706"
C_PURPLE = "#7C3AED"
C_GRAY = "#6B7280"
BG = "#FAFBFC"

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.facecolor': BG,
    'figure.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

datasets = list(data.keys())
short_names = {
    "AI4I": "AI4I",
    "C-MAPSS": "C-MAPSS",
    "Synthetic (Cascading)": "Synthetic",
    "SMD": "SMD",
    "JM1": "JM1",
    "PC1": "PC1",
    "CM1": "CM1",
    "MC2": "MC2"
}

def get_short_name(ds):
    return short_names.get(ds, ds.split()[0] if " " in ds else ds)

# ════════════════════════════════════════════════
# FIG 1: PID Decomposition — Stacked Bars
# ════════════════════════════════════════════════
n_ds = len(datasets)
ncols = 2          # always 2 columns → forces 2-row layout
nrows = (n_ds + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4.5 * nrows), squeeze=False)
axes = axes.flatten()
fig.suptitle("PID Decomposition: Information Atoms per Dataset", fontsize=14, fontweight='bold', y=1.02)

components = [
    ("Redundancy", C_BLUE),
    ("Unique₁", C_GREEN),
    ("Unique₂", C_ORANGE),
    ("Synergy", C_RED),
]

for idx, ds in enumerate(datasets):
    ax = axes[idx]
    pids = data[ds]["pairwise_pids"]
    pair_labels = [p["pair"].replace("×", " ×\n") for p in pids]

    x_pos = np.arange(len(pids))
    bottom = np.zeros(len(pids))

    for label, color in components:
        if label == "Redundancy":
            vals = [p["redundancy"] for p in pids]
        elif label == "Unique₁":
            vals = [p["unique_i"] for p in pids]
        elif label == "Unique₂":
            vals = [p["unique_j"] for p in pids]
        else:
            vals = [p["synergy"] for p in pids]

        vals = np.array(vals)
        ax.bar(x_pos, vals, bottom=bottom, color=color, width=0.6, label=label, edgecolor='white', linewidth=0.5)
        bottom += vals

    ax.set_xticks(x_pos)
    ax.set_xticklabels(pair_labels, fontsize=7, ha='center')
    sr = data[ds]["synergy_ratio"]
    ax.set_title(f"{get_short_name(ds)}\nSR = {sr:.3f}", fontsize=11, fontweight='bold')
    if idx % ncols == 0:
        ax.set_ylabel("bits")

# Hide unused subplots
for idx in range(n_ds, len(axes)):
    axes[idx].set_visible(False)

handles = [mpatches.Patch(color=c, label=l) for l, c in components]
fig.legend(handles=handles, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.0), fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "fig1_pid_decomposition.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ fig1_pid_decomposition.png")


# ════════════════════════════════════════════════
# FIG 2: SR vs Gap to XGBoost (THE KEY FIGURE)
# ════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8.5, 5.8))
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

# ── Professional style overrides for this figure ──────────────
plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         11,
    'axes.facecolor':    'white',
    'axes.grid':         False,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

# ── Dataset visual style ───────────────────────────────────────
style = {
    "AI4I":      {"c": "#C0392B", "marker": "o", "s": 310, "ec": "#8B0000"},
    "C-MAPSS":   {"c": "#1A6EA8", "marker": "s", "s": 250, "ec": "#0D3B6B"},
    "SMD":       {"c": "#1A8A55", "marker": "D", "s": 240, "ec": "#0D4D30"},
    "Synthetic": {"c": "#7B3DAE", "marker": "^", "s": 270, "ec": "#4B1880"},
    "JM1":       {"c": "#E67E22", "marker": "v", "s": 260, "ec": "#A04000"},
    "PC1":       {"c": "#F1C40F", "marker": "p", "s": 250, "ec": "#B7950B"},
    "CM1":       {"c": "#E74C3C", "marker": "h", "s": 250, "ec": "#943126"},
    "MC2":       {"c": "#9B59B6", "marker": "8", "s": 250, "ec": "#633974"},
}

# ── Collect data ───────────────────────────────────────────────
srs, gaps, names = [], [], []
for ds in datasets:
    sr  = data[ds]["synergy_ratio"]
    xgb_entry = data[ds]["baselines"].get("XGBoost (GB)")
    if xgb_entry:
        gap = xgb_entry["auc_mean"] - data[ds]["full_auc_mean"]
        srs.append(sr)
        gaps.append(gap)
        names.append(get_short_name(ds))

# ── Background region (very subtle) ───────────────────────────
if len(srs) > 0:
    max_sr = max(srs)
    ax.axvspan(-0.15, 0.15, color='#EBF5FB', alpha=0.55, zorder=0, lw=0)
    ax.axvspan( 0.15, max(2.85, max_sr * 1.1), color='#FDEDEC', alpha=0.45, zorder=0, lw=0)

# ── Threshold line ─────────────────────────────────────────────
ax.axvline(0.15, color='#999999', lw=1.4, ls=(0, (6, 3)), alpha=0.9, zorder=2)

# ── Region labels (top, near threshold) ───────────────────────
ax.text(0.07, 0.1185,  "Low SR\n(simple model\nsufficient)",
        fontsize=8.2, color="#1A6EA8", ha='center', va='top',
        linespacing=1.5,
        bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#AED6F1',
                  alpha=0.85, lw=0.8))
ax.text(1.50, 0.1185, "High SR\n(fusion\njustified)",
        fontsize=8.2, color="#C0392B", ha='center', va='top',
        linespacing=1.5,
        bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#F1948A',
                  alpha=0.85, lw=0.8))
ax.text(0.156, 0.001, "SR=0.15", fontsize=7.5, color='#888888',
        ha='left', va='bottom', style='italic')

# ── Trend line (linear fit) ────────────────────────────────────
if len(srs) > 1:
    try:
        coef = np.polyfit(srs, gaps, 1)
        poly1d_fn = np.poly1d(coef)
        xs = np.linspace(-0.05, max(srs) * 1.05, 100)
        ax.plot(xs, poly1d_fn(xs), color='#BBBBBB', lw=1.8,
                ls=(0, (7, 4)), alpha=0.75, zorder=1)
    except Exception:
        pass

# ── Scatter points ─────────────────────────────────────────────
for i, name in enumerate(names):
    st = style.get(name, {"c": C_BLUE, "marker": "o", "s": 250, "ec": C_GRAY})
    ax.scatter(srs[i], gaps[i],
               c=st["c"], marker=st["marker"], s=st["s"],
               edgecolors=st["ec"], linewidth=1.2, zorder=6)

# ── Label positions (data coords) — carefully fanned out ──────
label_cfg = {
    #        text anchor           connection point   ha
    "AI4I":      ((1.95,  0.1070), (2.387, 0.101),  "right"),
    "C-MAPSS":   ((0.65, -0.003 ), (0.006, 0.000),  "left"),
    "SMD":       ((0.65,  0.0175), (0.008, 0.006),  "left"),
    "Synthetic": ((0.65,  0.0380), (0.166, 0.002),  "left"),
    "JM1":       ((0.65,  0.0113), (0.496, 0.011),  "left"),
    "PC1":       ((0.65,  0.0480), (0.012, 0.038),  "left"),
    "CM1":       ((0.65,  0.0580), (0.020, 0.037),  "left"),
    "MC2":       ((0.65,  0.0280), (0.017, 0.022),  "left"),
}

for name in names:
    if name not in label_cfg:
        continue
    st = style.get(name, {"c": C_BLUE})
    txt_xy, pt_xy, ha = label_cfg[name]
    ax.annotate(
        name,
        xy=pt_xy,
        xytext=txt_xy,
        xycoords='data', textcoords='data',
        fontsize=10.5, fontweight='bold', color=st["c"],
        ha=ha, va='center',
        bbox=dict(boxstyle='round,pad=0.28', fc='white',
                  ec=st["c"], alpha=0.92, lw=0.9),
        arrowprops=dict(
            arrowstyle='-',
            color=st["c"],
            lw=0.9, alpha=0.55,
            connectionstyle='arc3,rad=0.0',
        ),
        zorder=7,
    )

ax.set_ylim(-0.008, 0.12)
ax.set_xlabel("Synergy Ratio (SR)")
ax.set_ylabel("XGBoost AUC - Ours (Full) AUC")
ax.set_title("Synergy Ratio predicts Gap to XGBoost")

plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "fig2_sr_vs_gap.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ fig2_sr_vs_gap.png")


# ════════════════════════════════════════════════
# FIG 3: Model Comparison Bars per Dataset
# ════════════════════════════════════════════════
# ncols / nrows are carried from FIG 1 (ncols=2 → 2-row layout)
fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 5 * nrows), squeeze=False)
axes = axes.flatten()
fig.suptitle("AUC-ROC Comparison Across Datasets", fontsize=14, fontweight='bold', y=1.02)

methods = ["Ours (Full)", "Ours (Ablation)", "Naive Bayes", "Logistic Reg.", "Random Forest", "XGBoost (GB)"]
method_colors = [C_RED, C_ORANGE, C_GRAY, C_GRAY, C_BLUE, C_PURPLE]

for idx, ds in enumerate(datasets):
    ax = axes[idx]
    aucs = [
        data[ds]["full_auc_mean"],
        data[ds]["ablation_auc_mean"],
        data[ds]["baselines"]["Naive Bayes"]["auc_mean"],
        data[ds]["baselines"]["Logistic Reg."]["auc_mean"],
        data[ds]["baselines"]["Random Forest"]["auc_mean"],
        data[ds]["baselines"]["XGBoost (GB)"]["auc_mean"],
    ]
    stds = [
        data[ds]["full_auc_std"],
        0,  # no std for ablation in JSON
        data[ds]["baselines"]["Naive Bayes"]["auc_std"],
        data[ds]["baselines"]["Logistic Reg."]["auc_std"],
        data[ds]["baselines"]["Random Forest"]["auc_std"],
        data[ds]["baselines"]["XGBoost (GB)"]["auc_std"],
    ]

    y_pos = np.arange(len(methods))
    bars = ax.barh(y_pos, aucs, xerr=stds, color=method_colors, height=0.6,
                   edgecolor='white', linewidth=0.5, capsize=3)

    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f'{auc:.3f}', va='center', fontsize=8, color='#374151')

    sr = data[ds]["synergy_ratio"]
    ax.set_title(f"{get_short_name(ds)} (SR={sr:.3f})", fontsize=11, fontweight='bold')
    ax.set_xlim(0.65, 1.02)
    if idx % ncols == 0:
        ax.set_yticks(y_pos)
        ax.set_yticklabels(methods, fontsize=9)
    else:
        ax.set_yticks([])

# Hide unused subplots
for idx in range(n_ds, len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "fig3_model_comparison.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ fig3_model_comparison.png")


# ════════════════════════════════════════════════
# FIG 4: Ablation Effect — Entropy Adds Value?
# ════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 11))

# Left: Delta AUC bars with significance
deltas = [data[ds]["ablation_delta_auc"] for ds in datasets]
pvals = [data[ds]["ablation_p_value"] for ds in datasets]
bar_colors = [C_GREEN if p < 0.05 else C_GRAY for p in pvals]

y_pos = np.arange(len(datasets))
bars = ax1.barh(y_pos, deltas, color=bar_colors, height=0.5, edgecolor='white')

# Dynamic x-axis limits — add 40% padding so all bars + labels are visible
min_delta = min(deltas)
max_delta = max(deltas)
pad_left  = abs(min_delta) * 0.55   # extra room for negative bar labels
pad_right = abs(max_delta) * 0.55
xlim_left  = min(min_delta - pad_left,  -0.002)
xlim_right = max(max_delta + pad_right,  0.005)

for i, (bar, delta, pval) in enumerate(zip(bars, deltas, pvals)):
    sig = "✓ p<0.05" if pval < 0.05 else f"p={pval:.2f}"
    # Place label to the right of positive bars, to the LEFT of negative bars
    if delta >= 0:
        x_text = delta + (xlim_right - xlim_left) * 0.01
        ha = 'left'
    else:
        x_text = delta - (xlim_right - xlim_left) * 0.01
        ha = 'right'
    ax1.text(x_text, bar.get_y() + bar.get_height() / 2,
             f'Δ={delta:+.4f} ({sig})', va='center', ha=ha, fontsize=9,
             fontweight='bold' if pval < 0.05 else 'normal')

ax1.axvline(x=0, color='black', linewidth=0.8, linestyle='-')
ax1.set_yticks(y_pos)
ax1.set_yticklabels([get_short_name(ds) for ds in datasets], fontsize=11)
ax1.set_xlabel("Δ AUC (Full − Ablation)", fontsize=12)
ax1.set_title("Entropy Feature Contribution", fontsize=13, fontweight='bold')
ax1.set_xlim(xlim_left, xlim_right)

# Bottom: SR vs Ablation Delta
# Per-dataset label offsets (in points) — fanned out to avoid overlap near SR≈0 cluster
label_offsets = {
    "AI4I":      ( 12,    8),
    "C-MAPSS":   ( 55,   28),   # far right + up
    "SMD":       ( 55,  -28),   # far right + down
    "Synthetic": (-80,   28),   # far left  + up
    "MC2":       (-80,  -28),   # far left  + down
    "JM1":       ( 12,    8),
    "PC1":       (-75,    8),   # left
    "CM1":       ( 12,  -18),   # down
}

for i, ds in enumerate(datasets):
    name = get_short_name(ds)
    sr = data[ds]["synergy_ratio"]
    delta = data[ds]["ablation_delta_auc"]
    st = style.get(name, {"c": C_BLUE})
    c = st["c"]
    ax2.scatter(sr, delta, s=180, c=c, zorder=5,
                edgecolors='white', linewidth=2)
    offset = label_offsets.get(name, (10, 6))
    ax2.annotate(
        name, (sr, delta),
        textcoords="offset points", xytext=offset,
        fontsize=11, fontweight='bold', color=c,
        arrowprops=dict(arrowstyle="-", color=c, lw=0.8, alpha=0.6),
    )

ax2.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
ax2.set_xlabel("Imin-SR  (Synergy Ratio)", fontsize=12)
ax2.set_ylabel("Δ AUC (Full − Ablation)", fontsize=12)
ax2.set_title("SR Predicts Entropy Feature Value", fontsize=13, fontweight='bold')

plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "fig4_ablation.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ fig4_ablation.png")


# ════════════════════════════════════════════════
# FIG 5: Synergy Ratio Overview — The Hero Figure
# ════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

sr_vals = [data[ds]["synergy_ratio"] for ds in datasets]
bar_colors_sr = [C_RED if sr > 0.15 else (C_ORANGE if sr > 0.05 else C_BLUE) for sr in sr_vals]

bars = ax.bar(range(len(datasets)), sr_vals, color=bar_colors_sr, width=0.5,
              edgecolor='white', linewidth=1.5)

# Add value labels
for bar, sr in zip(bars, sr_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'SR = {sr:.3f}', ha='center', fontsize=12, fontweight='bold')

# Threshold lines
ax.axhline(y=0.05, color=C_ORANGE, linewidth=1.5, linestyle='--', alpha=0.6)
ax.axhline(y=0.15, color=C_RED, linewidth=1.5, linestyle='--', alpha=0.6)
ax.text(len(datasets) - 0.3, 0.06, 'Moderate\nthreshold', fontsize=8, color=C_ORANGE, ha='center')
ax.text(len(datasets) - 0.3, 0.16, 'High\nthreshold', fontsize=8, color=C_RED, ha='center')

# Zone labels
ax.axhspan(-0.1, 0.05, alpha=0.04, color=C_BLUE)
ax.axhspan(0.05, 0.15, alpha=0.04, color=C_ORANGE)

ax.set_xticks(range(len(datasets)))
ax.set_xticklabels([get_short_name(ds) for ds in datasets], fontsize=12, fontweight='bold')
ax.set_ylabel(r"I$_{\min}$-SR", fontsize=13)
ax.set_title(r"Cross-Dataset I$_{\min}$-SR Comparison", fontsize=14, fontweight='bold')
ax.set_ylim(0, max(sr_vals) * 1.15)

# Legend
legend_patches = [
    mpatches.Patch(color=C_BLUE, label='Low SR — Simple model sufficient'),
    mpatches.Patch(color=C_ORANGE, label='Moderate SR — Fusion optional'),
    mpatches.Patch(color=C_RED, label='High SR — Fusion justified'),
]
ax.legend(handles=legend_patches, loc='upper right', fontsize=10)

plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "fig5_synergy_overview.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ fig5_synergy_overview.png")

print(f"\n  All 5 figures saved to {FIGURES_DIR}/")
