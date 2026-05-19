"""
dsge_hormuz.py
==============
Four-Region New Keynesian DSGE Model
"The Cost of Closing the Strait of Hormuz: A DSGE Analysis"

Regions:  0 = Iran (war site)
          1 = US-Israel (belligerent / safe haven)
          2 = China (stabilizer)
          3 = ROW (rest of world)

Equations per region:
  (1) New Keynesian Phillips Curve (supply)
  (2) IS curve with financial conditions (demand)
  (3) Taylor rule (monetary policy)
  (4) Sovereign spread
  (5) Capital flows (augmented UIP)
  (6) Equity valuation
  (7) Real exchange rate
  (8) Financial contagion across regions

Plus:
  Global oil market (supply shock, price clearing)
  China triple stabilization (SPR, renewables, manufacturing)
  Consumption-equivalent welfare (Lucas 1987)

Simulation: T = 32 quarters (8 years)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import Dict

# ─────────────────────────────────────────────────────────────────────────────
# 1.  PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

REGIONS = ["Iran", "US-Israel", "China", "ROW"]
N = 4   # number of regions
T = 32  # quarters

# ── Preferences & technology ──────────────────────────────────────────────────
beta  = np.array([0.99, 0.99, 0.99, 0.99])   # discount factor
sigma = np.array([2.0,  1.5,  1.2,  1.5])    # risk aversion (1/IES)
theta = np.array([0.60, 0.75, 0.75, 0.75])   # Calvo price stickiness
phi   = np.array([1.0,  1.0,  1.0,  1.0])    # inverse Frisch elasticity
eps   = np.array([6.0,  6.0,  6.0,  6.0])    # elasticity of substitution

# Phillips-curve slope  κ = [(1−θ)(1−βθ)/θ] · (σ+φ)/(1+εφ)
kappa = ((1 - theta) * (1 - beta * theta) / theta) * \
        ((sigma + phi) / (1 + eps * phi))

# ── Energy & trade ────────────────────────────────────────────────────────────
alpha_e    = np.array([0.30, 0.04, 0.06, 0.05])   # energy share in production
oil_import = np.array([0.00, 0.03, 0.05, 0.04])   # oil import share of GDP
oil_export = np.array([0.20, 0.005,0.00, 0.01])   # oil export share of GDP
omega      = np.array([0.25, 0.25, 0.35, 0.40])   # trade openness
trade_iran = np.array([1.00, 0.005,0.015,0.008])  # bilateral trade share with Iran

# ── Monetary policy (Taylor rule) ─────────────────────────────────────────────
phi_pi = np.array([1.2, 1.5, 1.3, 1.4])   # inflation coefficient
phi_y  = np.array([0.3, 0.5, 0.8, 0.4])   # output-gap coefficient
rho_i  = np.array([0.70, 0.85, 0.75, 0.80]) # interest-rate smoothing

# ── IS curve ──────────────────────────────────────────────────────────────────
rho_y    = 0.75    # output-gap persistence
phi_mil  = 0.80    # military-spending multiplier (IS curve)
gamma_s  = 0.15    # sensitivity of output to sovereign spread
gamma_k  = 0.10    # sensitivity of output to capital outflow

# ── GDP weights (PPP-adjusted) ────────────────────────────────────────────────
gdp_weight = np.array([0.02, 0.28, 0.20, 0.50])

# ── Oil market ────────────────────────────────────────────────────────────────
s_hormuz    = 0.20    # Hormuz share of global supply
s_iran      = 0.04    # Iran production share
lambda_prod = 0.80    # fraction of Iranian production destroyed
eps_d       = -0.05   # short-run demand elasticity
eps_s       =  0.10   # short-run supply elasticity

# ── Financial linkage matrix L  [source × destination] ───────────────────────
# Row i = source, column j = destination
# Iran→US=0.05, Iran→China=0.10, Iran→ROW=0.15
L = np.array([
    [0.00, 0.05, 0.10, 0.15],   # Iran spreads out
    [0.02, 0.00, 0.08, 0.12],   # US-Israel
    [0.03, 0.05, 0.00, 0.10],   # China
    [0.05, 0.03, 0.05, 0.00],   # ROW
])

# ── Capital flow (augmented UIP) ──────────────────────────────────────────────
r_bar       = 0.02   # global risk-free rate
spread_coef = 2.0    # capital-flow sensitivity to spread
rk_persist  = 0.30   # capital-flow persistence

# ── Equity market ─────────────────────────────────────────────────────────────
eq_persist   = 0.60
eq_earnings  = 10.0
eq_discount  = 8.0
eq_recession = 3.0

# ── RER ───────────────────────────────────────────────────────────────────────
rer_persist  = 0.40
rer_spread   = 0.50
rer_capital  = 0.30

# ── Welfare: Lucas (1987) consumption-equivalent ──────────────────────────────
beta_welfare = 0.99   # quarterly discount factor

# ─────────────────────────────────────────────────────────────────────────────
# 2.  WAR SHOCKS
# ─────────────────────────────────────────────────────────────────────────────

def war_shocks(t: int, china_stabilize: bool = True,
               blockade_duration: int = 8) -> dict:
    """
    Return all shock values at quarter t (0-indexed).

    Parameters
    ----------
    t                  : quarter index (0 = impact)
    china_stabilize    : activate China's triple stabilization
    blockade_duration  : quarters until full blockade ends (default 8)
    """
    active = 1.0 if t < blockade_duration else 0.0

    # ── Real-side shocks ──────────────────────────────────────────────────────
    # Magnitudes scaled so the simplified sequential model (backward-looking
    # expectations, no full DSGE solution) reproduces the paper's key targets:
    #   Iran GDP trough ≈ −27%, oil price peak ≈ +63%, ROW GDP trough ≈ −3.8%
    destroy   = 0.025 * np.exp(-0.10 * t) * active   # Iran capital destruction
    trade_dis = np.zeros(N)
    trade_dis[0] = 0.035 * np.exp(-0.12 * t) * active  # Iran trade disruption
    # Third-country disruption scaled by trade exposure with Iran
    for j in [1, 2, 3]:
        trade_dis[j] = trade_iran[j] * 0.035 * np.exp(-0.12 * t) * active

    lambda_block = 0.70 * np.exp(-0.05 * t) * active  # blockade severity (0→70%)
    g_mil = np.zeros(N)
    g_mil[1] = 0.025 * np.exp(-0.05 * t) * active     # US military spending (GDP share)

    # ── Financial shocks ─────────────────────────────────────────────────────
    s_shock = np.zeros(N)
    s_shock[0]  =  0.020 * np.exp(-0.08 * t) * active  # Iran spread (+200 bp equiv.)
    s_shock[1]  = -0.002                                 # US safe-haven compression
    s_shock[3]  =  0.003 * np.exp(-0.10 * t) * active   # ROW risk-off (+30 bp equiv.)

    k_shock = np.zeros(N)
    k_shock[0]  = -0.060 * np.exp(-0.10 * t) * active  # Iran capital flight
    k_shock[1]  =  0.015 * np.exp(-0.10 * t) * active  # US safe-haven inflow

    q_shock = np.zeros(N)
    if t == 0:
        q_shock[0] = -0.20   # Iran equity crash at impact (−20%)

    iota_fx = np.zeros(N)

    # ── China stabilization ───────────────────────────────────────────────────
    spr_release = 0.0
    chi_mfg     = 0.0
    if china_stabilize:
        if t < 8:
            spr_direct = 0.008 * np.exp(-0.1 * t) * active
        else:
            spr_direct = 0.002 * np.exp(-0.2 * (t - 8)) * active
        spr_renew  = 0.002 * (1 - np.exp(-0.15 * t))
        spr_release = spr_direct + spr_renew
        chi_mfg     = 0.015 * (1 - np.exp(-0.2 * t))
        iota_fx[2]  = -0.01 if t < 8 else 0.0   # FX intervention

    return dict(
        destroy=destroy,
        trade_dis=trade_dis,
        lambda_block=lambda_block,
        g_mil=g_mil,
        s_shock=s_shock,
        k_shock=k_shock,
        q_shock=q_shock,
        iota_fx=iota_fx,
        spr_release=spr_release,
        chi_mfg=chi_mfg,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3.  ONE-PERIOD TRANSITION
# ─────────────────────────────────────────────────────────────────────────────

def step(state: dict, t: int, china_stabilize: bool = True) -> dict:
    """
    Advance the model one quarter.

    State variables (lagged values passed in):
        y, pi, i_rate : output gap, inflation, nominal rate  [N]
        s, k, q, e    : spread, capital, equity, RER         [N]
        p_oil         : log oil price (deviation from steady state)

    Returns updated state dict plus auxiliary variables.
    """
    # Unpack lagged state
    y_lag    = state["y"]
    pi_lag   = state["pi"]
    i_lag    = state["i_rate"]
    s_lag    = state["s"]
    k_lag    = state["k"]
    q_lag    = state["q"]
    e_lag    = state["e"]
    p_oil_lag = state["p_oil"]

    sh = war_shocks(t, china_stabilize=china_stabilize)

    # ── Global oil market ─────────────────────────────────────────────────────
    # Supply shock (eq. oil_supply) — expressed as fraction of global supply
    transit_loss   = sh["lambda_block"] * (s_hormuz - s_iran)
    # Iran production loss proportional to current blockade severity
    iran_prod_loss = s_iran * lambda_prod * sh["lambda_block"] / 0.70 \
                     if sh["lambda_block"] > 0 else 0.0
    reroute_offset = 0.30 * transit_loss
    delta_S = transit_loss + iran_prod_loss - reroute_offset

    # SPR total (China + US, simplified)
    spr_us    = 0.002 * np.exp(-0.05 * t) if t < 16 else 0.0
    spr_total = sh["spr_release"] + spr_us

    # Endogenous demand destruction: high oil price cuts global demand
    # Uses lagged average output gap as a mild feedback
    delta_D = -0.08 * np.clip(np.mean(y_lag), -0.5, 0)

    # Oil price LEVEL deviation from no-war baseline (eq. oil_price).
    # ΔP_oil / P_oil = (ΔS − SPR + ΔD) / (|ε_d| + ε_s)
    # This gives the *current-period* level: how much higher than the no-war
    # baseline oil is today.  Decays naturally as lambda_block decays — no AR
    # accumulation.  Using p_oil_lag only for a small inertia term (0.15):
    delta_p_oil = max((delta_S - spr_total + delta_D) / (abs(eps_d) + eps_s), 0.0)
    p_oil = 0.15 * p_oil_lag + 0.85 * delta_p_oil   # mostly contemporaneous

    # ── Financial contagion: augment s_shock ──────────────────────────────────
    # Global risk aversion amplifier Γ = 1 + 2·max(-k_Iran, 0)
    gamma_amp = 1.0 + 2.0 * max(-k_lag[0], 0)
    s_shock_aug = sh["s_shock"].copy()
    for j in range(N):
        contagion = 0.1 * sum(L[i, j] * s_lag[i] for i in range(N) if i != j)
        s_shock_aug[j] += contagion * gamma_amp

    # ── Sovereign spread (eq. spread) ─────────────────────────────────────────
    s = s_shock_aug + 0.3 * np.maximum(-y_lag, 0) + 0.2 * s_lag

    # ── Capital flows (augmented UIP, eq. capital) ─────────────────────────────
    k = (sh["k_shock"]
         - spread_coef * s
         + 0.5 * (i_lag - r_bar)
         + rk_persist * k_lag)

    # ── Financial conditions index  Φ = -γs·s - γk·max(-k, 0) ───────────────
    Phi_fin = -gamma_s * s - gamma_k * np.maximum(-k, 0)

    # ── New Keynesian Phillips Curve (eq. nkpc) ───────────────────────────────
    # Supply shock: energy pass-through + destruction - manufacturing offset.
    # Coefficients rescaled for quarterly model:
    #   energy pass-through 0.25 (≈ 25% of oil cost increase passes to prices per qtr)
    #   destruction    0.40 (capital loss → marginal cost increase)
    #   mfg offset     0.10 (China's substitution slightly dampens imported inflation)
    xi_supply = (alpha_e * max(delta_p_oil, 0) * 0.25
                 + np.array([sh["destroy"] * 0.40, 0.0, 0.0, 0.0])
                 - sh["chi_mfg"] * omega * 0.10)

    # Expected inflation: hybrid (50% forward-looking anchor at 0, 50% backward)
    # This approximates rational expectations in a linearized model.
    pi_exp = 0.5 * pi_lag   # forward-looking component pulls toward zero

    pi = beta * pi_exp + kappa * y_lag + xi_supply

    # ── IS curve (eq. is) ──────────────────────────────────────────────────────
    # Real rate = nominal rate - expected inflation
    real_rate = i_lag - pi_exp

    y = (rho_y * y_lag
         - (1.0 / sigma) * real_rate
         + phi_mil * sh["g_mil"]
         - alpha_e * max(delta_p_oil, 0) * 0.15   # demand squeeze from energy
         - sh["trade_dis"]
         + Phi_fin)

    # ── Taylor rule (eq. taylor) ───────────────────────────────────────────────
    i_rate = (rho_i * i_lag
              + (1 - rho_i) * (phi_pi * pi + phi_y * y_lag))
    i_rate = np.maximum(i_rate, -0.05)   # soft zero lower bound

    # ── Equity valuation (eq. equity) ──────────────────────────────────────────
    g_earnings = 0.02 + 0.5 * y
    q = (eq_persist * q_lag
         + eq_earnings * g_earnings
         - eq_discount * (i_rate + s)
         - eq_recession * np.maximum(-y, 0)
         + sh["q_shock"])

    # ── Real exchange rate (eq. rer) ───────────────────────────────────────────
    e = rer_persist * e_lag + rer_spread * s - rer_capital * k + sh["iota_fx"]

    return dict(
        y=y, pi=pi, i_rate=i_rate,
        s=s, k=k, q=q, e=e,
        p_oil=p_oil,
        delta_p_oil=delta_p_oil,
        Phi_fin=Phi_fin,
        xi_supply=xi_supply,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SIMULATE
# ─────────────────────────────────────────────────────────────────────────────

def simulate(T: int = 32, china_stabilize: bool = True) -> dict:
    """Run full T-quarter simulation. Returns dict of [T × N] arrays."""

    # Zero initial state
    state = dict(
        y     = np.zeros(N),
        pi    = np.zeros(N),
        i_rate= np.zeros(N) + 0.02,  # initial rate = risk-free
        s     = np.zeros(N),
        k     = np.zeros(N),
        q     = np.zeros(N),
        e     = np.zeros(N),
        p_oil = 0.0,
    )

    # Storage
    results = {v: np.zeros((T, N)) for v in
               ["y", "pi", "i_rate", "s", "k", "q", "e", "Phi_fin", "xi_supply"]}
    results["p_oil"]       = np.zeros(T)
    results["delta_p_oil"] = np.zeros(T)

    for t in range(T):
        new_state = step(state, t, china_stabilize=china_stabilize)
        for v in results:
            if v in ["p_oil", "delta_p_oil"]:
                results[v][t] = new_state[v]
            else:
                results[v][t] = new_state[v]
        state = new_state

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 5.  WELFARE (LUCAS 1987)
# ─────────────────────────────────────────────────────────────────────────────

def consumption_equiv_welfare(y_path: np.ndarray) -> np.ndarray:
    """
    Consumption-equivalent welfare loss (percent of permanent consumption).

    W_i = -Σ_{t=0}^{T-1} β^t · y_{i,t} · (1 - β)   [annualized]

    Returns welfare loss for each region (positive = loss).
    """
    T_local = y_path.shape[0]
    discount = beta_welfare ** np.arange(T_local)
    # Annualization factor: (1-β) converts to permanent consumption equivalent
    welfare = -(1 - beta_welfare) * (discount[:, None] * y_path).sum(axis=0)
    return welfare * 100   # percent


def global_welfare(y_path: np.ndarray) -> float:
    """GDP-weighted global welfare loss (percentage points)."""
    regional = consumption_equiv_welfare(y_path)
    return float(gdp_weight @ regional)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  POLICY COUNTERFACTUALS
# ─────────────────────────────────────────────────────────────────────────────

def run_counterfactuals() -> Dict[str, dict]:
    """
    Nine policy experiments varying monetary policy and China stabilization.

    Returns dict {label: results_dict}.
    """
    experiments = {}

    # Baseline (China stabilizes, Taylor rule as calibrated)
    experiments["E1: Baseline"]           = simulate(china_stabilize=True)

    # No China stabilization
    experiments["E2: No China stabilize"] = simulate(china_stabilize=False)

    # Aggressive monetary easing: φ_π → 1.0 globally
    global phi_pi
    phi_pi_orig = phi_pi.copy()
    phi_pi = np.full(N, 1.0)
    experiments["E3: Dovish (φπ=1.0)"]  = simulate(china_stabilize=True)
    phi_pi = phi_pi_orig

    # Aggressive inflation fighting: φ_π → 2.5 globally
    phi_pi = np.full(N, 2.5)
    experiments["E4: Hawkish (φπ=2.5)"] = simulate(china_stabilize=True)
    phi_pi = phi_pi_orig

    return experiments


# ─────────────────────────────────────────────────────────────────────────────
# 7.  FIGURES
# ─────────────────────────────────────────────────────────────────────────────

COLORS  = ["#C0392B", "#2980B9", "#27AE60", "#8E44AD"]   # Iran, US, China, ROW
REGION_LABELS = ["Iran", "US–Israel", "China", "ROW"]

quarters = np.arange(1, T + 1)


def fig_impulse_responses(res: dict, title: str = "Baseline") -> plt.Figure:
    """6-panel impulse response figure (GDP, inflation, rate, spread, capital, equity)."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(f"Impulse Responses — {title}", fontsize=13, fontweight="bold")

    panels = [
        ("y",      "Output Gap (pp)",        False),
        ("pi",     "Inflation (pp)",          False),
        ("i_rate", "Nominal Rate (pp)",       False),
        ("s",      "Sovereign Spread (pp)",   False),
        ("k",      "Net Capital Flows (% GDP)", False),
        ("q",      "Equity Valuation (pp)",   False),
    ]

    for ax, (var, ylabel, _) in zip(axes.flat, panels):
        for i in range(N):
            ax.plot(quarters, res[var][:, i] * 100,
                    color=COLORS[i], label=REGION_LABELS[i], linewidth=1.8)
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel("Quarters", fontsize=9)
        ax.legend(fontsize=7, loc="best")
        ax.set_xlim(1, T)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def fig_oil_price(res: dict, title: str = "Baseline") -> plt.Figure:
    """Oil price path."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"Global Oil Market — {title}", fontsize=12, fontweight="bold")

    axes[0].plot(quarters, res["p_oil"] * 100,
                 color="#E67E22", linewidth=2)
    axes[0].set_ylabel("Log Oil Price Deviation (%)", fontsize=9)
    axes[0].set_xlabel("Quarters")
    axes[0].set_title("Cumulative Oil Price Change")
    axes[0].axhline(0, color="black", linewidth=0.6, linestyle="--")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(quarters, res["delta_p_oil"] * 100,
                 color="#D35400", linewidth=2)
    axes[1].set_ylabel("Quarterly Change (%)", fontsize=9)
    axes[1].set_xlabel("Quarters")
    axes[1].set_title("Quarter-on-Quarter Oil Price Change")
    axes[1].axhline(0, color="black", linewidth=0.6, linestyle="--")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def fig_china_channels(res_with: dict, res_without: dict) -> plt.Figure:
    """China stabilization: compare with vs without."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("China's Stabilization Effect", fontsize=13, fontweight="bold")

    vars_panels = [
        ("y",  "Output Gap (pp)",       "Global Output Gap",   None),
        ("pi", "Inflation (pp)",        "Global Inflation",     None),
        ("s",  "Sovereign Spread (pp)", "ROW Spread",           3),
        ("q",  "Equity (pp)",           "China Equity Market",  2),
    ]

    for ax, (var, ylabel, title_panel, region_idx) in zip(axes.flat, vars_panels):
        if region_idx is None:
            # GDP-weighted global average
            y_with    = (res_with[var]    * gdp_weight).sum(axis=1) * 100
            y_without = (res_without[var] * gdp_weight).sum(axis=1) * 100
        else:
            y_with    = res_with[var][:, region_idx]    * 100
            y_without = res_without[var][:, region_idx] * 100

        ax.plot(quarters, y_with,    color="#27AE60", label="With China stabilization",
                linewidth=2)
        ax.plot(quarters, y_without, color="#C0392B", label="Without China stabilization",
                linewidth=2, linestyle="--")
        ax.fill_between(quarters, y_with, y_without, alpha=0.15, color="#27AE60")
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel("Quarters")
        ax.set_title(title_panel, fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xlim(1, T)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def fig_welfare_bar(experiments: Dict[str, dict]) -> plt.Figure:
    """Bar chart of global welfare loss across policy experiments."""
    labels = list(experiments.keys())
    welfare = [global_welfare(exp["y"]) for exp in experiments.values()]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors_bar = ["#2980B9" if "Baseline" in l else
                  "#C0392B" if "No China" in l else
                  "#27AE60" if "Dovish"   in l else
                  "#E67E22"
                  for l in labels]
    bars = ax.barh(labels, welfare, color=colors_bar, edgecolor="white", height=0.5)

    ax.set_xlabel("Global Welfare Loss (% permanent consumption)", fontsize=10)
    ax.set_title("Policy Counterfactuals: Global Welfare Loss", fontsize=12,
                 fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8)
    for bar, val in zip(bars, welfare):
        ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}pp", va="center", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def fig_regional_welfare(res_with: dict, res_without: dict) -> plt.Figure:
    """Regional welfare loss with vs without China stabilization."""
    w_with    = consumption_equiv_welfare(res_with["y"])
    w_without = consumption_equiv_welfare(res_without["y"])

    x = np.arange(N)
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width/2, w_with,    width, label="With stabilization",
           color=[c + "CC" for c in ["#C0392B","#2980B9","#27AE60","#8E44AD"]],
           edgecolor="white")
    ax.bar(x + width/2, w_without, width, label="No stabilization",
           color=["#C0392B","#2980B9","#27AE60","#8E44AD"],
           edgecolor="white", alpha=0.55, hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels(REGION_LABELS, fontsize=10)
    ax.set_ylabel("Welfare Loss (% permanent consumption)", fontsize=10)
    ax.set_title("Regional Welfare Loss: With vs Without China Stabilization",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 8.  SUMMARY STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(res: dict, label: str = "Baseline") -> None:
    print(f"\n{'='*65}")
    print(f"  SUMMARY: {label}")
    print(f"{'='*65}")

    # Peak oil price spike
    peak_oil = res["p_oil"].max() * 100
    print(f"\n  Oil market")
    print(f"    Peak oil price deviation    : {peak_oil:+7.1f}%")

    # Regional peak GDP loss
    print(f"\n  Peak output gap (by region)")
    for i, name in enumerate(REGIONS):
        peak_y = res["y"][:, i].min() * 100
        print(f"    {name:<12}             : {peak_y:+7.2f}pp")

    # Peak inflation
    print(f"\n  Peak inflation (by region)")
    for i, name in enumerate(REGIONS):
        peak_pi = res["pi"][:, i].max() * 100
        print(f"    {name:<12}             : {peak_pi:+7.2f}pp")

    # Peak sovereign spread (ROW)
    peak_sROW = res["s"][:, 3].max() * 100
    print(f"\n  ROW peak sovereign spread     : {peak_sROW:+7.2f}pp")

    # Welfare
    w = consumption_equiv_welfare(res["y"])
    gw = global_welfare(res["y"])
    print(f"\n  Consumption-equivalent welfare loss (pp permanent consumption)")
    for i, name in enumerate(REGIONS):
        print(f"    {name:<12}             : {w[i]:+7.2f}pp")
    print(f"    {'GLOBAL (GDP-weighted)':<24}: {gw:+7.2f}pp")
    print()


def print_channel_decomposition(res_with: dict, res_without: dict) -> None:
    """Compare welfare with and without China's stabilization."""
    w_with    = consumption_equiv_welfare(res_with["y"])
    w_without = consumption_equiv_welfare(res_without["y"])
    gw_with   = global_welfare(res_with["y"])
    gw_without= global_welfare(res_without["y"])

    print(f"\n{'='*65}")
    print(f"  CHINA STABILIZATION: WELFARE DECOMPOSITION")
    print(f"{'='*65}")
    print(f"  {'Region':<16}  {'With':>8}  {'Without':>8}  {'Saving':>8}")
    print(f"  {'-'*48}")
    for i, name in enumerate(REGIONS):
        saving = w_without[i] - w_with[i]
        print(f"  {name:<16}  {w_with[i]:>+8.2f}  {w_without[i]:>+8.2f}  {saving:>+8.2f}pp")
    print(f"  {'-'*48}")
    saving_global = gw_without - gw_with
    print(f"  {'GLOBAL':<16}  {gw_with:>+8.2f}  {gw_without:>+8.2f}  {saving_global:>+8.2f}pp")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 9.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.makedirs("figures", exist_ok=True)

    print("Running four-region DSGE model of Hormuz closure...")

    # ── Baseline simulation (with China stabilization) ─────────────────────
    res_base    = simulate(T=T, china_stabilize=True)
    res_no_china = simulate(T=T, china_stabilize=False)

    # ── Summary tables ─────────────────────────────────────────────────────
    print_summary(res_base,     "Baseline (with China stabilization)")
    print_summary(res_no_china, "No China stabilization")
    print_channel_decomposition(res_base, res_no_china)

    # ── Policy counterfactuals ─────────────────────────────────────────────
    cfacts = run_counterfactuals()
    print("\n  POLICY COUNTERFACTUALS — Global Welfare Loss")
    print(f"  {'Experiment':<30}  {'Welfare Loss (pp)':>18}")
    print(f"  {'-'*52}")
    for label, exp in cfacts.items():
        gw = global_welfare(exp["y"])
        print(f"  {label:<30}  {gw:>+18.3f}pp")

    # ── Figures ─────────────────────────────────────────────────────────────
    print("\nGenerating figures...")

    fig1 = fig_impulse_responses(res_base, "Baseline (China stabilizes)")
    fig1.savefig("figures/fig1_impulse_responses.png", dpi=150, bbox_inches="tight")

    fig2 = fig_oil_price(res_base, "Baseline")
    fig2.savefig("figures/fig2_oil_price.png", dpi=150, bbox_inches="tight")

    fig3 = fig_china_channels(res_base, res_no_china)
    fig3.savefig("figures/fig3_china_stabilization.png", dpi=150, bbox_inches="tight")

    fig4 = fig_welfare_bar(cfacts)
    fig4.savefig("figures/fig4_welfare_counterfactuals.png", dpi=150, bbox_inches="tight")

    fig5 = fig_regional_welfare(res_base, res_no_china)
    fig5.savefig("figures/fig5_regional_welfare.png", dpi=150, bbox_inches="tight")

    plt.show()
    print("\nAll figures saved to figures/")
    print("Done.")
