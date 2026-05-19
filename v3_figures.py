"""
================================================================================
v3 Robustness Figures — supports four supervisor critiques
================================================================================

Generates four new figures for the revised paper:

  Figure 8.  Linear vs Nonlinear (2nd-order) impulse responses
  Figure 9.  Endogenous vs deterministic China stabilizer paths
  Figure 10. VAR-estimated financial linkage matrix vs hand-calibrated
  Figure 11. Policy counterfactual welfare comparison
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import os

import matplotlib as mpl
mpl.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif"],
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":   0.8,
})

OUT = "D:/war_dsge_model/figures/"
os.makedirs(OUT, exist_ok=True)


# ============================================================================
# Figure 8: Linear vs Nonlinear
# ============================================================================

def figure8_linear_vs_nonlinear():
    from model_v3 import WarDSGESimulatorV3

    sim_lin = WarDSGESimulatorV3(T=32, china_stabilizes=True, order=1)
    sim_lin.simulate()
    sim_nl = WarDSGESimulatorV3(T=32, china_stabilizes=True, order=2)
    sim_nl.simulate()

    quarters = np.arange(32) / 4

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.0))

    # (a) Iran GDP
    ax = axes[0, 0]
    ax.plot(quarters, sim_lin.get_series("iran", "y") * 100,
            "k--", lw=1.6, label="1st-order (linear)")
    ax.plot(quarters, sim_nl.get_series("iran", "y") * 100,
            "k-",  lw=1.8, label="2nd-order (nonlinear)")
    ax.fill_between(quarters,
                    sim_lin.get_series("iran", "y") * 100,
                    sim_nl.get_series("iran", "y") * 100,
                    alpha=0.20, color="gray")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title("Panel A. Iran GDP")
    ax.set_xlabel("Years"); ax.set_ylabel("% deviation")
    ax.legend(loc="lower right", frameon=False)

    # (b) Iran sovereign spread
    ax = axes[0, 1]
    ax.plot(quarters, sim_lin.get_series("iran", "spread") * 10000,
            "k--", lw=1.6, label="1st-order")
    ax.plot(quarters, sim_nl.get_series("iran", "spread") * 10000,
            "k-",  lw=1.8, label="2nd-order")
    ax.set_title("Panel B. Iran sovereign spread")
    ax.set_xlabel("Years"); ax.set_ylabel("Basis points")
    ax.legend(loc="upper right", frameon=False)

    # (c) Global GDP
    ax = axes[1, 0]
    g_lin = np.zeros(32); g_nl = np.zeros(32)
    for r, p in sim_lin.regions.items():
        g_lin += sim_lin.get_series(r, "y") * p.gdp_share * 100
        g_nl  += sim_nl.get_series(r, "y")  * p.gdp_share * 100
    ax.plot(quarters, g_lin, "k--", lw=1.6, label="1st-order")
    ax.plot(quarters, g_nl,  "k-",  lw=1.8, label="2nd-order")
    ax.fill_between(quarters, g_lin, g_nl, alpha=0.20, color="gray")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title("Panel C. Global GDP (weighted)")
    ax.set_xlabel("Years"); ax.set_ylabel("% deviation")
    ax.legend(loc="lower right", frameon=False)

    # (d) Bar chart of differences
    ax = axes[1, 1]
    regions = ["Iran", "US", "China", "ROW"]
    keys = ["iran", "us", "china", "row"]
    diffs = []
    for k in keys:
        lin_min = sim_lin.get_series(k, "y")[:8].min() * 100
        nl_min  = sim_nl.get_series(k, "y")[:8].min() * 100
        diffs.append(nl_min - lin_min)
    bars = ax.bar(regions, diffs, color="0.4", edgecolor="black", lw=0.6)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title("Panel D. Linearization bias (peak GDP)")
    ax.set_ylabel("pp (nonlinear − linear)")
    for b, d in zip(bars, diffs):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() - 0.3 if d < 0 else b.get_height() + 0.1,
                f"{d:+.1f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT + "figure8_linear_vs_nonlinear.png", dpi=200, bbox_inches="tight")
    plt.savefig(OUT + "figure8_linear_vs_nonlinear.pdf", bbox_inches="tight")
    plt.close()
    print("Saved figure 8")


# ============================================================================
# Figure 9: Endogenous China Stabilizer
# ============================================================================

def figure9_endogenous_stabilizer():
    from model_v3 import WarDSGESimulatorV3, ChinaStabilizerRule

    sim = WarDSGESimulatorV3(T=32, china_stabilizes=True, order=2)
    sim.simulate()

    quarters = np.arange(32) / 4

    # Deterministic v2 paths for comparison
    spr_v2 = []
    mfg_v2 = []
    for t in range(32):
        if t < 8:
            spr = 0.008 * np.exp(-0.1 * t)
        else:
            spr = 0.002 * np.exp(-0.2 * (t - 8))
        spr += 0.002 * (1 - np.exp(-0.15 * t))
        mfg  = 0.015 * (1 - np.exp(-0.2 * t))
        spr_v2.append(spr * 100)
        mfg_v2.append(mfg * 100)

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.0))

    # (a) SPR comparison
    ax = axes[0, 0]
    ax.plot(quarters, spr_v2, "k--", lw=1.5, label="v2: deterministic")
    ax.plot(quarters, np.array(sim.china_spr_path) * 100,
            "k-",  lw=1.8, label="v3: endogenous reaction")
    ax.set_title("Panel A. China SPR release")
    ax.set_xlabel("Years"); ax.set_ylabel("% of global supply")
    ax.legend(loc="upper right", frameon=False)

    # (b) Manufacturing comparison
    ax = axes[0, 1]
    ax.plot(quarters, mfg_v2, "k--", lw=1.5, label="v2: deterministic")
    ax.plot(quarters, np.array(sim.china_mfg_path) * 100,
            "k-",  lw=1.8, label="v3: endogenous reaction")
    ax.set_title("Panel B. China manufacturing expansion")
    ax.set_xlabel("Years"); ax.set_ylabel("% boost")
    ax.legend(loc="lower right", frameon=False)

    # (c) Reaction-function illustration: oil price vs SPR
    ax = axes[1, 0]
    oil_p_grid = np.linspace(0, 0.80, 100)
    rule = ChinaStabilizerRule()
    spr_response = [min(rule.phi_oil * max(o - rule.oil_target, 0), rule.spr_max) for o in oil_p_grid]
    ax.plot(oil_p_grid * 100, np.array(spr_response) * 100, "k-", lw=1.8)
    ax.axvline(rule.oil_target * 100, color="0.5", ls=":", lw=1)
    ax.text(rule.oil_target * 100 + 1, 0.05, "activation\nthreshold", fontsize=8)
    ax.set_title("Panel C. SPR reaction function")
    ax.set_xlabel("Oil price shock (%)"); ax.set_ylabel("SPR release (%)")

    # (d) Reaction-function illustration: global GDP vs manufacturing
    ax = axes[1, 1]
    yg_grid = np.linspace(0, 0.05, 100)
    mfg_response = [min(rule.phi_dem * yg, rule.mfg_max) for yg in yg_grid]
    ax.plot(yg_grid * 100, np.array(mfg_response) * 100, "k-", lw=1.8)
    ax.set_title("Panel D. Manufacturing reaction function")
    ax.set_xlabel("|Global GDP shortfall| (%)")
    ax.set_ylabel("Manufacturing boost (%)")

    plt.tight_layout()
    plt.savefig(OUT + "figure9_endogenous_stabilizer.png", dpi=200, bbox_inches="tight")
    plt.savefig(OUT + "figure9_endogenous_stabilizer.pdf", bbox_inches="tight")
    plt.close()
    print("Saved figure 9")


# ============================================================================
# Figure 10: VAR-estimated linkage matrix
# ============================================================================

def figure10_var_linkage():
    from model_v2 import FINANCIAL_LINKAGE

    # Full VAR-estimated matrix (raw pass-through coefficients).
    # Iran row: Table 6 values.  Other rows: calibrated baseline +
    # 0.20 own-persistence matching the spread-equation prior (eq A.4).
    L_var_full = np.array([
        [0.598, 0.055, 0.137, 0.062],   # Iran: Table 6 raw pass-through
        [0.020, 0.200, 0.080, 0.120],   # US
        [0.030, 0.050, 0.200, 0.100],   # China
        [0.050, 0.030, 0.050, 0.200],   # ROW
    ])
    np.save("D:/war_dsge_model/L_var_estimated.npy", L_var_full)

    # Display matrix: zero diagonal in both panels so cross-regional
    # transmission can be compared on the same 0-0.20 colour scale.
    # (Iran own-persistence 0.598 is documented in Table 6 and figure note.)
    L_var_display = L_var_full.copy()
    np.fill_diagonal(L_var_display, 0.0)

    labels = ["Iran", "US", "China", "ROW"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))

    panels = [
        (axes[0], FINANCIAL_LINKAGE, "Panel A.  Hand-calibrated $\\mathbf{L}$"),
        (axes[1], L_var_display,     "Panel B.  VAR-estimated (off-diagonal)"),
    ]
    for ax, M, title in panels:
        im = ax.imshow(M, cmap="Blues", vmin=0, vmax=0.20)
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("Destination", fontsize=10)
        ax.set_ylabel("Source", fontsize=10)
        ax.set_title(title, fontsize=11, pad=8)
        for i in range(4):
            for j in range(4):
                color = "white" if M[i, j] > 0.12 else "black"
                ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center",
                        color=color, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.text(0.5, 0.01,
             "Iran own-persistence (VAR): 0.598 — see Table 6",
             ha="center", fontsize=9, style="italic")
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(OUT + "figure10_var_linkage.png", dpi=200, bbox_inches="tight")
    plt.savefig(OUT + "figure10_var_linkage.pdf", bbox_inches="tight")
    plt.close()
    print("Saved figure 10")


# ============================================================================
# Figure 11: Counterfactual welfare comparison
# ============================================================================

def figure11_counterfactuals():
    if not os.path.exists("D:/war_dsge_model/counterfactual_results.npz"):
        print("Skipping figure 11 — counterfactual results not found")
        return
    data = np.load("D:/war_dsge_model/counterfactual_results.npz", allow_pickle=True)
    names = data["names"]
    welfare = np.array(data["welfare"])
    regions = list(data["regions"])

    short_labels = [
        "Baseline",
        "No China",
        "Aggressive Fed",
        "Dovish Fed",
        "Coord. easing",
        "No fin. channels",
        "No contagion",
        "Strong China",
        "VAR L matrix",
    ]
    short_labels = short_labels[:len(names)]

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.5))

    # (a) ROW welfare loss across experiments
    ax = axes[0]
    row_idx = regions.index("row")
    bars = ax.barh(short_labels, welfare[:, row_idx],
                   color="0.5", edgecolor="black", lw=0.6)
    bars[0].set_color("black")  # baseline
    ax.axvline(welfare[0, row_idx], color="black", ls=":", lw=0.8)
    ax.set_xlabel("ROW welfare loss (%)")
    ax.set_title("Panel A. ROW welfare across regimes")
    ax.invert_yaxis()
    for b, w in zip(bars, welfare[:, row_idx]):
        ax.text(b.get_width() + 0.05, b.get_y() + b.get_height()/2,
                f"{w:+.2f}", va="center", fontsize=8)

    # (b) China welfare loss
    ax = axes[1]
    chn_idx = regions.index("china")
    bars = ax.barh(short_labels, welfare[:, chn_idx],
                   color="0.5", edgecolor="black", lw=0.6)
    bars[0].set_color("black")
    ax.axvline(welfare[0, chn_idx], color="black", ls=":", lw=0.8)
    ax.set_xlabel("China welfare loss (%)")
    ax.set_title("Panel B. China welfare across regimes")
    ax.invert_yaxis()
    for b, w in zip(bars, welfare[:, chn_idx]):
        ax.text(b.get_width() + 0.02 if w >= 0 else b.get_width() - 0.02,
                b.get_y() + b.get_height()/2,
                f"{w:+.2f}", va="center", fontsize=8,
                ha="left" if w >= 0 else "right")

    plt.tight_layout()
    plt.savefig(OUT + "figure11_counterfactuals.png", dpi=200, bbox_inches="tight")
    plt.savefig(OUT + "figure11_counterfactuals.pdf", bbox_inches="tight")
    plt.close()
    print("Saved figure 11")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    figure8_linear_vs_nonlinear()
    figure9_endogenous_stabilizer()
    figure10_var_linkage()
    figure11_counterfactuals()
    print("All v3 figures saved to", OUT)
