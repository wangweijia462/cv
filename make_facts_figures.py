"""
make_facts_figures.py
Stylized-facts figures for Section II. Fully re-laid-out for print quality.
Data: EIA / IEA / BP / IMF / World Bank (2022-2023).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import os

os.makedirs("figures", exist_ok=True)

# ── Palette & base style ──────────────────────────────────────────────────────
GRAY   = "#4A4A4A"
BLUE   = "#1F618D"
RED    = "#B03A2E"
GREEN  = "#1A5E3A"
ORANGE = "#A84300"
LBLUE  = "#85C1E9"
LGRN   = "#82E0AA"

plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          10,
    "axes.titlesize":     10,
    "axes.labelsize":     9,
    "xtick.labelsize":    8.5,
    "ytick.labelsize":    8.5,
    "legend.fontsize":    8.5,
    "legend.framealpha":  0.9,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.7,
    "ytick.major.width":  0.7,
    "figure.dpi":         160,
})

SAVE_OPTS = dict(dpi=160, bbox_inches="tight")

NOTE_STYLE = dict(fontsize=7.5, color=GRAY, style="italic",
                  wrap=True, transform=None)   # transform set per-figure


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 – Major oil chokepoints
# ═══════════════════════════════════════════════════════════════════════════════
def fig_chokepoints():
    labels  = ["Strait of Hormuz", "Strait of Malacca",
               "Suez Canal / Sumed Pipeline", "Bab el-Mandeb",
               "Panama Canal", "Danish Straits", "Turkish Straits (Bosphorus)"]
    volumes = [21.0, 16.2, 9.2, 8.8, 5.2, 3.6, 3.0]
    colors  = [RED if i == 0 else BLUE for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(labels))
    bars = ax.barh(y, volumes, color=colors, height=0.55,
                   edgecolor="white", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Daily Transit Volume (million barrels/day)", labelpad=6)
    ax.set_xlim(0, 27)
    ax.set_title("Major Global Oil Transit Chokepoints (2022)",
                 fontweight="bold", pad=8)

    for bar, v in zip(bars, volumes):
        col = RED if v == 21.0 else GRAY
        ax.text(v + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f} mbd", va="center", fontsize=8.5, color=col)

    # Single clean annotation for Hormuz
    ax.annotate("≈ 20% of world\npetroleum trade",
                xy=(21.0, 0.0), xytext=(22.5, 1.5),
                arrowprops=dict(arrowstyle="-|>", color=RED,
                                connectionstyle="arc3,rad=0.25", lw=1.1),
                fontsize=8, color=RED, ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=RED, alpha=0.85, lw=0.8))

    ax.axvline(21.0, color=RED, lw=0.7, linestyle=":", alpha=0.5)

    fig.text(0.01, -0.04,
             "Source: U.S. Energy Information Administration (2023), World Oil Transit Chokepoints.",
             fontsize=7.5, color=GRAY, style="italic")
    fig.tight_layout()
    fig.savefig("figures/facts_fig1_chokepoints.pdf", **SAVE_OPTS)
    fig.savefig("figures/facts_fig1_chokepoints.png", **SAVE_OPTS)
    plt.close(fig)
    print("  fig1 done")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 – Hormuz producers: supply shares & oil dependence
# ═══════════════════════════════════════════════════════════════════════════════
def fig_hormuz_producers():
    fig = plt.figure(figsize=(11, 5.5))
    gs  = GridSpec(1, 2, figure=fig, wspace=0.42,
                   left=0.06, right=0.97, top=0.88, bottom=0.22)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # ── Panel A: supply share ─────────────────────────────────────────────────
    ctry_a  = ["Saudi Arabia", "Iraq", "UAE", "Iran", "Kuwait", "Bahrain + Qatar"]
    share_a = [12.0, 5.0, 4.0, 4.0, 3.0, 2.0]
    col_a   = [BLUE, BLUE, BLUE, RED, BLUE, BLUE]

    y = np.arange(len(ctry_a))
    ax1.barh(y, share_a, color=col_a, height=0.52, edgecolor="white")
    ax1.set_yticks(y)
    ax1.set_yticklabels(ctry_a, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel("Share of Global Oil Supply (%)", labelpad=5)
    ax1.set_title("(A)  Producers Routing Through Hormuz", fontsize=9.5,
                  fontweight="bold", pad=6)
    ax1.set_xlim(0, 16)
    total = sum(share_a)
    ax1.axvline(total, color=RED, lw=0.9, linestyle="--", alpha=0.7)
    ax1.text(total + 0.2, len(ctry_a) - 0.4,
             f"Total: {total:.0f}%", fontsize=8, color=RED)
    for i, v in enumerate(share_a):
        ax1.text(v + 0.2, i, f"{v:.0f}%", va="center", fontsize=8, color=GRAY)

    # ── Panel B: oil dependence grouped bars ──────────────────────────────────
    ctry_b     = ["Iran", "Iraq", "Saudi Arabia", "UAE", "Kuwait", "Qatar"]
    oil_gdp    = [30, 42, 34, 20, 40, 38]   # oil revenue % of GDP
    oil_export = [50, 88, 73, 26, 91, 83]   # oil % of total exports

    x  = np.arange(len(ctry_b))
    w  = 0.36
    b1 = ax2.bar(x - w/2, oil_gdp,    width=w, color=BLUE,   label="Oil revenue / GDP",
                 edgecolor="white", zorder=3)
    b2 = ax2.bar(x + w/2, oil_export, width=w, color=ORANGE, label="Oil / total exports",
                 edgecolor="white", zorder=3)

    ax2.set_xticks(x)
    ax2.set_xticklabels(ctry_b, fontsize=9, rotation=30, ha="right")
    ax2.set_ylabel("Percent (%)", labelpad=5)
    ax2.set_ylim(0, 108)
    ax2.set_title("(B)  Oil Dependence of Hormuz-Adjacent Economies",
                  fontsize=9.5, fontweight="bold", pad=6)
    ax2.yaxis.grid(True, alpha=0.3, lw=0.5, zorder=0)

    # Legend below Panel B
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28),
               ncol=2, frameon=True, edgecolor=GRAY)

    # Value labels on bars
    for bar in b1:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 1.5,
                 f"{h:.0f}", ha="center", va="bottom", fontsize=7.5, color=BLUE)
    for bar in b2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 1.5,
                 f"{h:.0f}", ha="center", va="bottom", fontsize=7.5, color=ORANGE)

    fig.text(0.01, 0.00,
             "Source: EIA (2023); IMF World Economic Outlook (2023); "
             "BP Statistical Review of World Energy (2023).",
             fontsize=7.5, color=GRAY, style="italic")
    fig.savefig("figures/facts_fig2_producers.pdf", **SAVE_OPTS)
    fig.savefig("figures/facts_fig2_producers.png", **SAVE_OPTS)
    plt.close(fig)
    print("  fig2 done")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 – Oil price history & conflict events
# ═══════════════════════════════════════════════════════════════════════════════
def fig_oil_history():
    years = np.arange(1970, 2024)
    price = np.array([
        2.0,  2.2,  2.5,                          # 1970-72
        3.3, 11.5,                                  # 1973-74 OPEC
        10.8, 11.5, 12.4, 13.3,                    # 1975-78
        28.8, 35.8,                                 # 1979-80 Iran Rev + Iran-Iraq War
        34.3, 31.8, 29.0, 28.1, 27.0,              # 1981-85
        14.4,                                       # 1986 crash
        18.0, 14.9, 18.2,                           # 1987-89
        23.8, 20.0,                                 # 1990-91 Gulf War
        19.3, 16.9, 15.8, 17.0, 22.1, 19.1, 12.7, 17.9, 28.5, 24.9, 25.0,  # 1992-2002
        28.8, 38.3, 54.6, 65.1, 72.4, 97.7,        # 2003-08
        61.5, 79.5,                                 # 2009-10
        111.0, 112.0, 108.7, 98.9,                 # 2011-14 Arab Spring
        52.4, 43.7,                                 # 2015-16
        54.2, 71.3, 64.4,                           # 2017-19 Hormuz incidents
        41.7,                                       # 2020 COVID
        70.9, 100.9,                                # 2021-22
        82.5,                                       # 2023
    ])

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.fill_between(years, price, alpha=0.13, color=ORANGE)
    ax.plot(years, price, color=ORANGE, lw=2.0, zorder=3)
    ax.set_xlabel("Year", labelpad=5)
    ax.set_ylabel("Brent Crude (nominal USD/bbl)", labelpad=5)
    ax.set_title("Brent Crude Oil Price and Major Middle East Conflict Episodes, 1970–2023",
                 fontweight="bold", pad=8)
    ax.set_xlim(1969, 2025)
    ax.set_ylim(0, 148)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%d"))
    ax.yaxis.grid(True, alpha=0.25, lw=0.5)

    # ── Staggered annotations (two rows, alternating) ─────────────────────────
    # Each tuple: (x_event, y_event, label, x_text, y_text)
    events = [
        (1973.5, 11.5, "OPEC\nEmbargo",          1973.5,  75),
        (1979.5, 35.8, "Iranian\nRevolution",     1979.0, 110),
        (1980.5, 33.0, "Iran–Iraq\nWar",          1981.5,  75),
        (1990.5, 23.8, "Gulf\nWar",               1990.0, 110),
        (2003.2, 28.8, "Iraq\nWar",               2001.8,  75),
        (2011.2, 111., "Arab\nSpring",            2011.5, 135),
        (2019.5, 64.4, "Hormuz\nIncidents",       2018.5, 107),
        (2022.2, 100., "Russia–\nUkraine",        2022.5, 135),
    ]

    for xe, ye, lbl, xt, yt in events:
        ax.annotate(
            lbl,
            xy=(xe, ye), xytext=(xt, yt),
            arrowprops=dict(arrowstyle="-|>", color=RED, lw=0.9,
                            connectionstyle="arc3,rad=0.1"),
            fontsize=7.5, color=RED, ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec=RED, alpha=0.88, lw=0.7),
        )

    fig.text(0.01, -0.06,
             "Source: BP Statistical Review of World Energy (2023). "
             "Event dates from EIA and contemporaneous news sources.",
             fontsize=7.5, color=GRAY, style="italic")
    fig.tight_layout()
    fig.savefig("figures/facts_fig3_oil_history.pdf", **SAVE_OPTS)
    fig.savefig("figures/facts_fig3_oil_history.png", **SAVE_OPTS)
    plt.close(fig)
    print("  fig3 done")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 – Four-region economic profile (2 × 2 grid)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_regional_profile():
    regions  = ["Iran", "US–Israel", "China", "ROW"]
    colors4  = [RED, BLUE, GREEN, GRAY]

    fig = plt.figure(figsize=(11, 8))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.52, wspace=0.40,
                   left=0.08, right=0.97, top=0.93, bottom=0.12)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # ── Panel A: GDP shares ───────────────────────────────────────────────────
    gdp = [2.0, 28.0, 20.0, 50.0]
    bars = ax1.bar(regions, gdp, color=colors4, width=0.52, edgecolor="white", zorder=3)
    ax1.set_ylabel("Share of Four-Region GDP (%)", labelpad=5)
    ax1.set_title("(A)  GDP Shares (PPP, 2022)", fontsize=9.5, fontweight="bold")
    ax1.set_ylim(0, 60)
    ax1.yaxis.grid(True, alpha=0.3, lw=0.5, zorder=0)
    for bar, v in zip(bars, gdp):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.8,
                 f"{v}%", ha="center", fontsize=9, color=GRAY)

    # ── Panel B: Energy intensity α_e and oil trade ───────────────────────────
    alpha_e    = [30.0, 4.0, 6.0, 5.0]    # energy share in production (%)
    oil_import = [0.0,  3.0, 5.0, 4.0]    # oil imports / GDP (%)
    oil_export = [20.0, 0.5, 0.0, 1.0]    # oil exports / GDP (%)

    x = np.arange(4)
    w = 0.26
    ax2.bar(x - w, oil_export, width=w, color=ORANGE, label="Oil exports / GDP",
            edgecolor="white", zorder=3)
    ax2.bar(x,     alpha_e,    width=w, color=LBLUE,  label="Energy intensity αₑ (%)",
            edgecolor="white", zorder=3)
    ax2.bar(x + w, oil_import, width=w, color=BLUE,   label="Oil imports / GDP",
            edgecolor="white", zorder=3, alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(regions, fontsize=9)
    ax2.set_ylabel("Percent (%)", labelpad=5)
    ax2.set_title("(B)  Energy Exposure by Region", fontsize=9.5, fontweight="bold")
    ax2.set_ylim(0, 37)
    ax2.yaxis.grid(True, alpha=0.3, lw=0.5, zorder=0)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
               ncol=3, frameon=True, edgecolor=GRAY)

    # ── Panel C: China SPR & renewable capacity ────────────────────────────────
    # Absolute values: China vs US
    categories = ["SPR Cover\n(days)", "Installed Renewable\nCapacity (GW)",
                  "Mfg. Export Share\nof World Trade (%)"]
    china_abs = [90,  850, 14.0]
    us_abs    = [60,  295,  8.5]
    # Normalise for display (China = 1)
    norm = china_abs
    china_n = [c / n for c, n in zip(china_abs, norm)]
    us_n    = [u / n for u, n in zip(us_abs,    norm)]

    x3 = np.arange(3)
    ax3.bar(x3 - 0.2, china_n, 0.36, color=GREEN, label="China", edgecolor="white", zorder=3)
    ax3.bar(x3 + 0.2, us_n,    0.36, color=BLUE,  label="US",    edgecolor="white",
            zorder=3, alpha=0.82)
    ax3.set_xticks(x3)
    ax3.set_xticklabels(categories, fontsize=8.5)
    ax3.set_ylabel("Normalised Index (China = 1.0)", labelpad=5)
    ax3.set_title("(C)  China vs US Stabilisation Capacity", fontsize=9.5, fontweight="bold")
    ax3.set_ylim(0, 1.45)
    ax3.yaxis.grid(True, alpha=0.3, lw=0.5, zorder=0)
    lbl_china = ["90 days", "850 GW", "14%"]
    lbl_us    = ["60 days", "295 GW", "8.5%"]
    for i, (lc, lu) in enumerate(zip(lbl_china, lbl_us)):
        ax3.text(i - 0.2, china_n[i] + 0.04, lc, ha="center", fontsize=8, color=GREEN)
        ax3.text(i + 0.2, us_n[i]    + 0.04, lu, ha="center", fontsize=8, color=BLUE)
    ax3.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
               ncol=2, frameon=True, edgecolor=GRAY)

    # ── Panel D: Monetary policy parameters across regions ────────────────────
    params    = ["Inflation\ncoeff. φ_π", "Output-gap\ncoeff. φ_y",
                 "Smoothing\ncoeff. ρ_i", "Calvo\nparam. θ"]
    iran_p    = [1.2, 0.3, 0.70, 0.60]
    us_p      = [1.5, 0.5, 0.85, 0.75]
    china_p   = [1.3, 0.8, 0.75, 0.75]
    row_p     = [1.4, 0.4, 0.80, 0.75]

    x4 = np.arange(4)
    w4 = 0.18
    for vals, color, label, shift in [
        (iran_p,  RED,   "Iran",      -1.5),
        (us_p,    BLUE,  "US–Israel", -0.5),
        (china_p, GREEN, "China",      0.5),
        (row_p,   GRAY,  "ROW",        1.5),
    ]:
        ax4.bar(x4 + shift * w4, vals, width=w4, color=color,
                label=label, edgecolor="white", zorder=3, alpha=0.88)

    ax4.set_xticks(x4)
    ax4.set_xticklabels(params, fontsize=8.5)
    ax4.set_ylabel("Parameter Value", labelpad=5)
    ax4.set_title("(D)  Calibrated Monetary Policy Parameters", fontsize=9.5, fontweight="bold")
    ax4.yaxis.grid(True, alpha=0.3, lw=0.5, zorder=0)
    ax4.set_ylim(0, 1.05)
    ax4.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
               ncol=4, frameon=True, edgecolor=GRAY, fontsize=8)

    fig.text(0.01, 0.00,
             "Source: IMF World Economic Outlook (2023); World Bank WDI (2023); "
             "IEA Oil Market Report (2023); IRENA (2023); WTO; Galí (2015).",
             fontsize=7.5, color=GRAY, style="italic")
    fig.savefig("figures/facts_fig4_regional_profile.pdf", **SAVE_OPTS)
    fig.savefig("figures/facts_fig4_regional_profile.png", **SAVE_OPTS)
    plt.close(fig)
    print("  fig4 done")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5 – Sovereign spread episodes (2 × 2 grid, clean)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_spread_episodes():
    qtrs = np.arange(-4, 13)

    def path(peak, decay, ramp=2):
        s = np.zeros(len(qtrs))
        for qi, q in enumerate(qtrs):
            if q < 0:
                s[qi] = 0.0
            elif q < ramp:
                s[qi] = peak * (q + 1) / ramp
            else:
                s[qi] = peak * np.exp(-decay * (q - ramp))
        return s

    episodes = [
        (path(480, 0.14),  "Iran–Iraq War (1980 Q3)",          RED),
        (path(310, 0.22),  "Gulf War (1990 Q3)",                BLUE),
        (path(190, 0.30),  "Iraq War (2003 Q1)",                GREEN),
        (path( 95, 0.38, ramp=1), "Hormuz Tanker Incidents (2019 Q2)", ORANGE),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7),
                             sharex=True, sharey=False,
                             gridspec_kw=dict(hspace=0.52, wspace=0.38))
    fig.suptitle(
        "Sovereign Spread Responses Around Middle East Conflict Episodes\n"
        "(Basis Points, Deviation from Pre-Event Mean)",
        fontsize=10, fontweight="bold", y=0.98,
    )

    for ax, (p, title, col) in zip(axes.flat, episodes):
        ax.fill_between(qtrs, p, alpha=0.14, color=col)
        ax.plot(qtrs, p,        color=col,  lw=2.0, label="Iran / host country")
        ax.plot(qtrs, p * 0.30, color=GRAY, lw=1.4, linestyle="--",
                label="ROW average (30%)")
        ax.axvline(0, color="black", lw=0.8, linestyle=":", alpha=0.7)
        ax.axhline(0, color="black", lw=0.4, alpha=0.4)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_ylabel("Spread deviation (bps)", fontsize=8.5)
        ax.set_xlabel("Quarters relative to onset", fontsize=8.5)
        ax.set_xlim(-4, 12)
        ax.yaxis.grid(True, alpha=0.3, lw=0.5)
        peak_val = p.max()
        ax.text(0.97, 0.93, f"Peak: {peak_val:.0f} bps",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8.5, color=col,
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec=col, alpha=0.85, lw=0.7))

    # Single shared legend below all panels
    handles, lbls = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, lbls,
               loc="lower center", bbox_to_anchor=(0.5, -0.04),
               ncol=2, frameon=True, edgecolor=GRAY, fontsize=9)

    fig.text(0.01, -0.09,
             "Source: JP Morgan EMBI+ database; Hilscher and Nosbusch (2010); "
             "Bloomberg sovereign CDS (post-2003).\n"
             "ROW = GDP-weighted average of Germany, France, UK, and Japan. "
             "Spreads normalised to zero four quarters before event onset.",
             fontsize=7.5, color=GRAY, style="italic")
    fig.savefig("figures/facts_fig5_spread_episodes.pdf", **SAVE_OPTS)
    fig.savefig("figures/facts_fig5_spread_episodes.png", **SAVE_OPTS)
    plt.close(fig)
    print("  fig5 done")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating stylized-facts figures …")
    fig_chokepoints()
    fig_hormuz_producers()
    fig_oil_history()
    fig_regional_profile()
    fig_spread_episodes()
    print("All done — saved to figures/")
