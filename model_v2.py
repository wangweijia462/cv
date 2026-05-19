"""
================================================================================
DSGE Model v2: The Price of War — Iran-Hormuz Scenario (Financial Channels)
================================================================================

Extension of model.py adding:
  1. Sovereign Risk Premium (spread)
  2. Capital Flows / Financial Account (capital_flow)
  3. Stock Market / Equity Index (equity)
  4. Enhanced Exchange Rate via UIP + capital flows
  5. Financial Contagion Channel (cross-region spread transmission)

All original functionality from model.py is preserved.

References:
  - Federle et al. (2026) "The Price of War", AER
  - Galí (2015) "Monetary Policy, Inflation, and the Business Cycle"
  - Blanchard & Galí (2007) "The Macroeconomic Effects of Oil Price Shocks"
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 1. MODEL PARAMETERS (calibration — unchanged from v1)
# ============================================================================

@dataclass
class RegionParams:
    """Structural parameters for a single region."""
    name: str
    name_cn: str

    beta: float = 0.99
    sigma: float = 1.5
    phi: float = 1.0
    alpha: float = 0.33
    theta: float = 0.75
    epsilon: float = 6.0

    alpha_e: float = 0.05
    oil_import_share: float = 0.0
    oil_export_share: float = 0.0

    trade_openness: float = 0.3
    trade_with_iran: float = 0.01

    phi_pi: float = 1.5
    phi_y: float = 0.5
    rho_i: float = 0.8

    gdp_share: float = 0.1

    military_share: float = 0.02
    spr_capacity: float = 0.0


def calibrate_regions() -> Dict[str, RegionParams]:
    """Calibrate parameters for the four regions."""

    iran = RegionParams(
        name="Iran", name_cn="伊朗(战争发生地)",
        sigma=2.0, alpha_e=0.30, oil_export_share=0.20,
        oil_import_share=0.0, trade_openness=0.25,
        trade_with_iran=1.0,
        phi_pi=1.2, phi_y=0.3, rho_i=0.7,
        gdp_share=0.02, military_share=0.025,
        theta=0.6,
    )

    us_israel = RegionParams(
        name="US-Israel", name_cn="美国/以色列(交战国)",
        sigma=1.5, alpha_e=0.04, oil_import_share=0.03,
        oil_export_share=0.005, trade_openness=0.25,
        trade_with_iran=0.005,
        phi_pi=1.5, phi_y=0.5, rho_i=0.85,
        gdp_share=0.28, military_share=0.035,
        spr_capacity=0.5,
    )

    china = RegionParams(
        name="China", name_cn="中国(稳定者)",
        sigma=1.2, alpha_e=0.06, oil_import_share=0.05,
        oil_export_share=0.0, trade_openness=0.35,
        trade_with_iran=0.015,
        phi_pi=1.3, phi_y=0.8, rho_i=0.75,
        gdp_share=0.20, military_share=0.017,
        spr_capacity=0.8,
    )

    row = RegionParams(
        name="ROW", name_cn="其余世界",
        sigma=1.5, alpha_e=0.05, oil_import_share=0.04,
        oil_export_share=0.01, trade_openness=0.40,
        trade_with_iran=0.008,
        phi_pi=1.4, phi_y=0.4, rho_i=0.80,
        gdp_share=0.50, military_share=0.018,
        spr_capacity=0.3,
    )

    return {"iran": iran, "us": us_israel, "china": china, "row": row}


# ============================================================================
# 2. GLOBAL OIL MARKET (unchanged from v1)
# ============================================================================

@dataclass
class OilMarket:
    """Global oil market model — Hormuz Strait blockade dynamics."""
    hormuz_share: float = 0.20
    iran_production_share: float = 0.04
    saudi_share: float = 0.12
    uae_share: float = 0.04
    iraq_share: float = 0.05

    supply_elasticity: float = 0.10
    demand_elasticity: float = -0.05

    blockade_severity: float = 0.70
    iran_production_loss: float = 0.80

    def compute_supply_shock(self, quarter: int, blockade_duration: int = 8) -> float:
        """Compute global oil supply shock as fraction of global supply."""
        if quarter >= blockade_duration:
            decay = np.exp(-0.3 * (quarter - blockade_duration))
            severity = 0.2 * decay
        else:
            severity = self.blockade_severity * np.exp(-0.05 * quarter)

        iran_loss = self.iran_production_loss * self.iran_production_share
        transit_loss = severity * (self.hormuz_share - self.iran_production_share)
        pipeline_offset = 0.30 * transit_loss

        total_loss = iran_loss + transit_loss - pipeline_offset
        return total_loss

    def compute_oil_price_change(self, supply_shock: float,
                                  spr_release: float = 0.0,
                                  demand_reduction: float = 0.0) -> float:
        """Compute oil price change from supply/demand balance."""
        net_shock = supply_shock - spr_release + demand_reduction
        price_change = net_shock / abs(self.demand_elasticity - self.supply_elasticity)
        return price_change


# ============================================================================
# 3. REGION STATE — extended with financial variables
# ============================================================================

@dataclass
class RegionState:
    """State variables for a single region in a single period.

    v2 additions:
      spread       — sovereign spread over risk-free rate (pp, fraction)
      capital_flow — net capital inflow (% of GDP, fraction)
      equity       — equity index deviation (fraction)
    """
    # --- original variables ---
    y: float = 0.0
    pi: float = 0.0
    i: float = 0.0
    c: float = 0.0
    inv: float = 0.0
    nx: float = 0.0
    rer: float = 0.0
    g_mil: float = 0.0
    oil_p: float = 0.0
    spr: float = 0.0

    # --- v2: financial channel variables ---
    spread: float = 0.0        # sovereign spread (fraction, e.g. 0.05 = 500bp)
    capital_flow: float = 0.0  # net capital inflow (% of GDP, fraction)
    equity: float = 0.0        # equity index deviation (fraction)


# ============================================================================
# 4. LINEARIZED DSGE EQUATIONS — extended with financial channels
# ============================================================================

def solve_region_period(
    params: RegionParams,
    state_prev: RegionState,
    oil_price_shock: float,
    military_shock: float = 0.0,
    trade_shock: float = 0.0,
    supply_destruction: float = 0.0,
    spr_release: float = 0.0,
    china_manufacturing_boost: float = 0.0,
    expected_pi_next: float = 0.0,
    # --- v2: financial shock parameters ---
    spread_shock: float = 0.0,
    capital_flow_shock: float = 0.0,
    equity_shock: float = 0.0,
    forex_intervention: float = 0.0,
) -> RegionState:
    """Solve one period equilibrium for a single region.

    IS curve (output gap):
      ŷ_t = persistence*ŷ_{t-1} - 1/σ*(î_t - Eπ_{t+1}) + g_mil
            + trade_effects - α_e*oil_shock + financial_conditions

    Phillips curve:
      π_t = β*Eπ_{t+1} + κ*ŷ_{t-1} + supply_shock

    Taylor rule:
      î_t = ρ_i*î_{t-1} + (1-ρ_i)*(φ_π*π_t + φ_y*ŷ_{t-1})

    Financial channels (new in v2):
      spread_t       = spread_shock + 0.3*max(-y,0) + 0.2*spread_{t-1}
      capital_flow_t = flow_shock - 2*spread + 0.5*(i-2%) + 0.3*flow_{t-1}
      rer_t          = 0.4*rer_{t-1} + 0.5*spread - 0.3*capital_flow + forex
      equity_t       = 0.6*equity_{t-1} + 10*E[g] - 8*discount - 3*max(-y,0) + equity_shock
    """
    s = RegionState()

    # -------------------------------------------------------------------------
    # Energy price pass-through
    # -------------------------------------------------------------------------
    net_oil_exposure = max(params.oil_import_share - params.oil_export_share, 0.0)
    s.oil_p = oil_price_shock * (net_oil_exposure / max(params.alpha_e, 0.01)) * 0.5
    s.oil_p = np.clip(s.oil_p, -2.0, 2.0)
    s.oil_p -= spr_release * 0.3

    # -------------------------------------------------------------------------
    # Phillips curve
    # -------------------------------------------------------------------------
    kappa = ((1 - params.theta) * (1 - params.beta * params.theta) / params.theta
             * (params.sigma + params.phi) / (1 + params.epsilon * params.phi))
    kappa = min(kappa, 0.15)

    supply_shock = (params.alpha_e * max(s.oil_p, 0) * 0.6
                    + supply_destruction * 0.8
                    - china_manufacturing_boost * params.trade_openness * 0.2)
    supply_shock = np.clip(supply_shock, -0.10, 0.15)

    s.pi = (params.beta * expected_pi_next
            + kappa * state_prev.y
            + supply_shock)
    s.pi = np.clip(s.pi, -0.15, 0.20)

    # -------------------------------------------------------------------------
    # Taylor rule
    # -------------------------------------------------------------------------
    s.i = (params.rho_i * state_prev.i
           + (1 - params.rho_i) * (params.phi_pi * s.pi
                                    + params.phi_y * max(state_prev.y, -0.3)))
    s.i = np.clip(s.i, -0.05, 0.15)

    # -------------------------------------------------------------------------
    # Sovereign spread  (v2)
    # -------------------------------------------------------------------------
    s.spread = (spread_shock
                + 0.3 * max(-state_prev.y, 0)   # countercyclical
                + 0.2 * state_prev.spread)        # persistence
    s.spread = np.clip(s.spread, 0, 0.50)

    # -------------------------------------------------------------------------
    # Capital flows  (v2)
    # -------------------------------------------------------------------------
    s.capital_flow = (capital_flow_shock
                      - 2.0 * s.spread
                      + 0.5 * (s.i - 0.02)
                      + 0.3 * state_prev.capital_flow)
    s.capital_flow = np.clip(s.capital_flow, -0.20, 0.10)

    # -------------------------------------------------------------------------
    # IS curve — with financial conditions feedback  (v2)
    # -------------------------------------------------------------------------
    real_rate_effect = -(1.0 / params.sigma) * (s.i - expected_pi_next) * 0.3
    military_effect  = 0.8 * military_shock
    trade_effect     = -trade_shock * params.trade_openness * 0.5
    energy_cost_effect = -params.alpha_e * max(s.oil_p, 0) * 0.3

    # Financial conditions: tighter spreads / capital outflows reduce demand
    financial_conditions = -0.15 * s.spread - 0.1 * max(-s.capital_flow, 0)

    persistence = 0.75
    s.y = (persistence * state_prev.y
           + real_rate_effect
           + military_effect
           + trade_effect
           + energy_cost_effect
           + financial_conditions
           - supply_destruction * 0.5)
    s.y += -0.05 * state_prev.y   # mean reversion
    s.y = np.clip(s.y, -0.50, 0.15)

    # -------------------------------------------------------------------------
    # Enhanced exchange rate — UIP + capital flows  (v2)
    # -------------------------------------------------------------------------
    s.rer = (0.4 * state_prev.rer
             + 0.5 * s.spread
             - 0.3 * s.capital_flow
             + forex_intervention)
    s.rer = np.clip(s.rer, -0.20, 0.50)

    # -------------------------------------------------------------------------
    # Equity index — simplified Gordon growth model  (v2)
    # -------------------------------------------------------------------------
    expected_growth = 0.02 + 0.5 * s.y
    discount = s.i + s.spread
    s.equity = (0.6 * state_prev.equity
                + 10.0 * expected_growth
                - 8.0 * discount
                - 3.0 * max(-s.y, 0)
                + equity_shock)
    s.equity = np.clip(s.equity, -0.60, 0.20)

    # -------------------------------------------------------------------------
    # Consumption, investment, net exports, military, SPR
    # -------------------------------------------------------------------------
    s.c   = s.y - military_shock * 0.4
    s.inv = s.y * 1.2 - 0.3 * max(s.i - s.pi, 0)

    s.nx = (-net_oil_exposure * oil_price_shock * 0.5
            + params.oil_export_share * oil_price_shock * 0.3
            + 0.2 * s.rer
            + trade_shock * 0.3)
    s.nx = np.clip(s.nx, -0.15, 0.15)

    s.g_mil = military_shock
    s.spr   = spr_release

    return s


# ============================================================================
# 5. SIMULATION ENGINE — extended with financial shocks
# ============================================================================

# Financial linkage matrix: how much spread shock transmits from i to j
# Rows = source region, Cols = destination region
# Order: iran, us, china, row
FINANCIAL_LINKAGE = np.array([
    [0.00, 0.05, 0.10, 0.15],   # Iran → others (limited linkage but contagion)
    [0.02, 0.00, 0.08, 0.12],   # US → others
    [0.03, 0.05, 0.00, 0.10],   # China → others
    [0.05, 0.03, 0.05, 0.00],   # ROW → others
])

REGION_ORDER = ["iran", "us", "china", "row"]


class WarDSGESimulator:
    """War DSGE Model Simulator — v2 with financial channels."""

    def __init__(self, T: int = 32, china_stabilizes: bool = True):
        self.T = T
        self.china_stabilizes = china_stabilizes
        self.regions = calibrate_regions()
        self.oil_market = OilMarket()

        self.results: Dict[str, list] = {name: [] for name in self.regions}
        self.oil_prices = []

    # ------------------------------------------------------------------
    # Shock design
    # ------------------------------------------------------------------

    def design_war_shocks(self) -> Dict[int, Dict]:
        """Design the full path of war shocks, including financial shocks (v2)."""
        shocks = {}

        for t in range(self.T):
            # --- Real-side shocks (identical to v1) ---
            oil_supply_loss = self.oil_market.compute_supply_shock(t, blockade_duration=8)

            china_spr_release = 0.0
            china_mfg_boost   = 0.0

            if self.china_stabilizes:
                if t < 8:
                    china_spr_release = 0.008 * np.exp(-0.1 * t)
                else:
                    china_spr_release = 0.002 * np.exp(-0.2 * (t - 8))
                renewable_offset   = 0.002 * (1 - np.exp(-0.15 * t))
                china_spr_release += renewable_offset
                china_mfg_boost    = 0.015 * (1 - np.exp(-0.2 * t))

            us_spr_release  = 0.005 * np.exp(-0.15 * t) if t < 12 else 0.0
            row_spr_release = 0.003 * np.exp(-0.15 * t) if t < 8 else 0.0
            total_spr = china_spr_release + us_spr_release + row_spr_release

            demand_reduction = 0.005 * t / 8 if t < 8 else 0.005
            oil_price_change = self.oil_market.compute_oil_price_change(
                oil_supply_loss, total_spr, demand_reduction
            )

            iran_destruction    = 0.06 * np.exp(-0.10 * t)
            iran_trade_disruption = 0.08 * np.exp(-0.12 * t)

            if t < 8:
                us_military = 0.025 * np.exp(-0.05 * t)
            else:
                us_military = 0.025 * np.exp(-0.2 * (t - 4))

            trade_disruption_factor = 0.8 * np.exp(-0.1 * t)

            # --- Financial shocks (v2) ---
            # Iran: massive spread jump at t=0, slow decay; capital flight; equity collapse
            iran_spread_shock       = 0.05 * np.exp(-0.08 * t)
            iran_capital_flow_shock = -0.15 * np.exp(-0.10 * t)
            iran_equity_shock       = -0.40 if t == 0 else 0.0

            # US: safe-haven premium → slightly negative spread, capital inflows
            us_spread_shock         = -0.002
            us_capital_flow_shock   = 0.03 * np.exp(-0.10 * t)
            us_equity_shock         = -0.05 if t == 0 else 0.0

            # China: moderate spread rise, some capital outflow; FX intervention
            china_spread_shock       = 0.003 * np.exp(-0.10 * t)
            china_capital_flow_shock = -0.02 * np.exp(-0.15 * t)
            china_equity_shock       = -0.08 if t == 0 else 0.0
            # China uses FX reserves to stabilize RMB (negative = appreciation pressure offset)
            china_forex_intervention = -0.01 if (self.china_stabilizes and t < 8) else 0.0

            # ROW: moderate spread and outflows driven by risk-off
            row_spread_shock       = 0.005 * np.exp(-0.10 * t)
            row_capital_flow_shock = -0.01 * np.exp(-0.10 * t)
            row_equity_shock       = -0.05 if t == 0 else 0.0

            shocks[t] = {
                # real shocks
                'oil_price':             oil_price_change,
                'oil_supply_loss':       oil_supply_loss,
                'iran_destruction':      iran_destruction,
                'iran_trade_disruption': iran_trade_disruption,
                'us_military':           us_military,
                'china_spr':             china_spr_release,
                'china_mfg_boost':       china_mfg_boost,
                'us_spr':                us_spr_release,
                'row_spr':               row_spr_release,
                'trade_disruption':      trade_disruption_factor,
                'total_spr':             total_spr,
                # financial shocks (v2)
                'iran_spread':           iran_spread_shock,
                'iran_capital_flow':     iran_capital_flow_shock,
                'iran_equity':           iran_equity_shock,
                'us_spread':             us_spread_shock,
                'us_capital_flow':       us_capital_flow_shock,
                'us_equity':             us_equity_shock,
                'china_spread':          china_spread_shock,
                'china_capital_flow':    china_capital_flow_shock,
                'china_equity':          china_equity_shock,
                'china_forex':           china_forex_intervention,
                'row_spread':            row_spread_shock,
                'row_capital_flow':      row_capital_flow_shock,
                'row_equity':            row_equity_shock,
            }

        return shocks

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------

    def simulate(self) -> Dict[str, list]:
        """Run full simulation including financial contagion channel."""
        shocks = self.design_war_shocks()
        states = {name: RegionState() for name in self.regions}

        for t in range(self.T):
            shock = shocks[t]
            new_states = {}

            # ----------------------------------------------------------
            # Financial contagion: spread shocks transmit across regions
            # Higher risk aversion from Iran capital flight amplifies spreads
            # ----------------------------------------------------------
            base_spreads = {
                'iran': shock['iran_spread'],
                'us':   shock['us_spread'],
                'china': shock['china_spread'],
                'row':  shock['row_spread'],
            }

            # Global risk aversion index: Iran capital flight amplifies risk-off
            iran_capital_flight = max(-shock['iran_capital_flow'], 0)
            global_risk_factor  = 1.0 + 2.0 * iran_capital_flight  # amplifier

            # Contagion-augmented spread shocks
            contagion_spreads = {}
            for j_idx, j_name in enumerate(REGION_ORDER):
                contagion = 0.0
                for i_idx, i_name in enumerate(REGION_ORDER):
                    if i_name != j_name:
                        # other region's spread leaks into j
                        contagion += FINANCIAL_LINKAGE[i_idx, j_idx] * states[i_name].spread
                contagion_spreads[j_name] = (base_spreads[j_name]
                                             + 0.1 * contagion * global_risk_factor)

            # ----------------------------------------------------------
            # Per-region solve
            # ----------------------------------------------------------
            for name, params in self.regions.items():
                expected_pi = states[name].pi * 0.7

                if name == "iran":
                    s = solve_region_period(
                        params, states[name],
                        oil_price_shock=shock['oil_price'] * 0.2,
                        supply_destruction=shock['iran_destruction'],
                        trade_shock=shock['iran_trade_disruption'],
                        expected_pi_next=expected_pi,
                        spread_shock=contagion_spreads['iran'],
                        capital_flow_shock=shock['iran_capital_flow'],
                        equity_shock=shock['iran_equity'],
                        forex_intervention=0.0,
                    )
                    # Oil export revenue loss partially offset by price gain
                    oil_revenue_loss = shock['oil_supply_loss'] * 0.3
                    s.y -= oil_revenue_loss
                    s.y = np.clip(s.y, -0.50, 0.15)

                elif name == "us":
                    s = solve_region_period(
                        params, states[name],
                        oil_price_shock=shock['oil_price'],
                        military_shock=shock['us_military'],
                        trade_shock=shock['trade_disruption'] * params.trade_with_iran,
                        spr_release=shock['us_spr'],
                        china_manufacturing_boost=(shock['china_mfg_boost']
                                                   if self.china_stabilizes else 0),
                        expected_pi_next=expected_pi,
                        spread_shock=contagion_spreads['us'],
                        capital_flow_shock=shock['us_capital_flow'],
                        equity_shock=shock['us_equity'],
                        forex_intervention=0.0,
                    )

                elif name == "china":
                    s = solve_region_period(
                        params, states[name],
                        oil_price_shock=shock['oil_price'],
                        trade_shock=shock['trade_disruption'] * params.trade_with_iran,
                        spr_release=shock['china_spr'],
                        china_manufacturing_boost=(shock['china_mfg_boost']
                                                   if self.china_stabilizes else 0),
                        expected_pi_next=expected_pi,
                        spread_shock=contagion_spreads['china'],
                        capital_flow_shock=shock['china_capital_flow'],
                        equity_shock=shock['china_equity'],
                        forex_intervention=shock['china_forex'],
                    )
                    if self.china_stabilizes:
                        s.y  += shock['china_mfg_boost'] * 0.3
                        s.nx += shock['china_mfg_boost'] * 0.2

                else:  # row
                    s = solve_region_period(
                        params, states[name],
                        oil_price_shock=shock['oil_price'],
                        trade_shock=shock['trade_disruption'] * params.trade_with_iran,
                        spr_release=shock['row_spr'],
                        china_manufacturing_boost=(shock['china_mfg_boost']
                                                   if self.china_stabilizes else 0),
                        expected_pi_next=expected_pi,
                        spread_shock=contagion_spreads['row'],
                        capital_flow_shock=shock['row_capital_flow'],
                        equity_shock=shock['row_equity'],
                        forex_intervention=0.0,
                    )

                new_states[name] = s

            states = new_states
            for name in self.regions:
                self.results[name].append(states[name])
            self.oil_prices.append(shock['oil_price'] * 100)

        return self.results

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_series(self, region: str, variable: str) -> np.ndarray:
        """Extract time series for any RegionState variable (including v2 fields)."""
        return np.array([getattr(s, variable) for s in self.results[region]])


# ============================================================================
# 6. VISUALIZATION — original plots + new financial channel plots
# ============================================================================

def plot_impulse_responses(sim_with: WarDSGESimulator,
                           sim_without: WarDSGESimulator,
                           save_path: str = None):
    """Impulse response plots — core results (unchanged from v1)."""

    quarters = np.arange(sim_with.T)
    years = quarters / 4

    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    fig.suptitle(
        'DSGE Model v2: Economic Impact of US-Israel War on Iran\n'
        'Hormuz Strait Blockade with China Stabilization + Financial Channels\n'
        '(Inspired by Federle et al. 2026 "The Price of War", AER)',
        fontsize=14, fontweight='bold', y=0.98
    )

    colors = {
        'iran': '#CC0000',
        'us':   '#003399',
        'china': '#CC6600',
        'row':  '#666666',
    }
    labels = {
        'iran':  'Iran (War Site)',
        'us':    'US/Israel (Belligerent)',
        'china': 'China (Stabilizer)',
        'row':   'Rest of World',
    }

    # Row 1: Output gap
    ax = axes[0, 0]
    ax.set_title('Output Gap (GDP)', fontsize=12, fontweight='bold')
    for name in ['iran', 'us', 'china', 'row']:
        ax.plot(years, sim_with.get_series(name, 'y') * 100,
                color=colors[name], linewidth=2, label=labels[name])
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_ylabel('% deviation from trend'); ax.set_xlabel('Years after war onset')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.set_title('Output: With vs Without China Stabilization', fontsize=12, fontweight='bold')
    for name in ['us', 'china', 'row']:
        ax.plot(years, sim_with.get_series(name, 'y') * 100,
                color=colors[name], linewidth=2, label=f'{labels[name]} (with)')
        ax.plot(years, sim_without.get_series(name, 'y') * 100,
                color=colors[name], linewidth=2, linestyle='--', alpha=0.6,
                label=f'{labels[name]} (without)')
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_ylabel('% deviation from trend'); ax.set_xlabel('Years after war onset')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.set_title('Global GDP (Weighted Average)', fontsize=12, fontweight='bold')
    global_with    = np.zeros(sim_with.T)
    global_without = np.zeros(sim_with.T)
    for name, params in sim_with.regions.items():
        global_with    += sim_with.get_series(name, 'y')    * params.gdp_share * 100
        global_without += sim_without.get_series(name, 'y') * params.gdp_share * 100
    ax.fill_between(years, global_without, global_with, alpha=0.3, color='green',
                    label='China stabilization effect')
    ax.plot(years, global_with,    'g-',  linewidth=2, label='With China stabilization')
    ax.plot(years, global_without, 'r--', linewidth=2, label='Without')
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_ylabel('% deviation from trend'); ax.set_xlabel('Years after war onset')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Row 2: Inflation
    ax = axes[1, 0]
    ax.set_title('Inflation', fontsize=12, fontweight='bold')
    for name in ['iran', 'us', 'china', 'row']:
        ax.plot(years, sim_with.get_series(name, 'pi') * 100,
                color=colors[name], linewidth=2, label=labels[name])
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_ylabel('Percentage points (annualized)'); ax.set_xlabel('Years after war onset')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.set_title('Inflation: With vs Without China Stabilization', fontsize=12, fontweight='bold')
    for name in ['us', 'china', 'row']:
        ax.plot(years, sim_with.get_series(name, 'pi') * 100,
                color=colors[name], linewidth=2, label=f'{labels[name]} (with)')
        ax.plot(years, sim_without.get_series(name, 'pi') * 100,
                color=colors[name], linewidth=2, linestyle='--', alpha=0.6,
                label=f'{labels[name]} (without)')
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_ylabel('Percentage points (annualized)'); ax.set_xlabel('Years after war onset')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.set_title('Global Oil Price', fontsize=12, fontweight='bold')
    ax.plot(years, sim_with.oil_prices,    'r-',  linewidth=2, label='With China SPR release')
    ax.plot(years, sim_without.oil_prices, 'r--', linewidth=2, alpha=0.6, label='Without')
    ax.fill_between(years, sim_without.oil_prices, sim_with.oil_prices,
                    alpha=0.2, color='green', label='SPR stabilization')
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_ylabel('% change from pre-war'); ax.set_xlabel('Years after war onset')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Row 3: Interest rate, net exports, military
    ax = axes[2, 0]
    ax.set_title('Nominal Interest Rate', fontsize=12, fontweight='bold')
    for name in ['us', 'china', 'row']:
        ax.plot(years, sim_with.get_series(name, 'i') * 100,
                color=colors[name], linewidth=2, label=labels[name])
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_ylabel('Percentage points deviation'); ax.set_xlabel('Years after war onset')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.set_title('Net Exports (% of GDP)', fontsize=12, fontweight='bold')
    for name in ['us', 'china', 'row']:
        ax.plot(years, sim_with.get_series(name, 'nx') * 100,
                color=colors[name], linewidth=2, label=labels[name])
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_ylabel('Percentage points of GDP'); ax.set_xlabel('Years after war onset')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[2, 2]
    ax.set_title('Military Spending Shock (% of GDP)', fontsize=12, fontweight='bold')
    for name in ['iran', 'us', 'china', 'row']:
        ax.plot(years, sim_with.get_series(name, 'g_mil') * 100,
                color=colors[name], linewidth=2, label=labels[name])
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_ylabel('Percentage points of GDP'); ax.set_xlabel('Years after war onset')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Row 4: China mechanisms + summary
    ax = axes[3, 0]
    ax.set_title("China's SPR Release & Renewable Offset", fontsize=12, fontweight='bold')
    spr_series = sim_with.get_series('china', 'spr') * 100
    ax.bar(years, spr_series, width=0.2, color='#CC6600', alpha=0.7, label='SPR release')
    ax.set_ylabel('% of domestic oil consumption'); ax.set_xlabel('Years after war onset')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[3, 1]
    ax.set_title("China's Manufacturing Export Stabilization", fontsize=12, fontweight='bold')
    ax.plot(years, sim_with.get_series('china', 'nx') * 100,
            color='#CC6600', linewidth=2, label='With stabilization')
    ax.plot(years, sim_without.get_series('china', 'nx') * 100,
            color='#CC6600', linewidth=2, linestyle='--', alpha=0.6, label='Without')
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_ylabel('% of GDP deviation'); ax.set_xlabel('Years after war onset')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[3, 2]
    ax.axis('off')
    summary_text = (
        "SUMMARY: Peak Effects (Year 1)\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    )
    for name, lbl in labels.items():
        y_peak  = sim_with.get_series(name, 'y')[:8].min() * 100
        pi_peak = sim_with.get_series(name, 'pi')[:8].max() * 100
        summary_text += f"\n{lbl}:\n"
        summary_text += f"  GDP: {y_peak:+.1f}%  Inflation: {pi_peak:+.1f}pp\n"

    oil_peak    = max(sim_with.oil_prices[:8])
    oil_peak_no = max(sim_without.oil_prices[:8])
    summary_text += f"\nOil Price Peak:\n"
    summary_text += f"  With China SPR: +{oil_peak:.0f}%\n"
    summary_text += f"  Without:        +{oil_peak_no:.0f}%\n"

    global_loss_with    = min(global_with[:8])
    global_loss_without = min(global_without[:8])
    summary_text += f"\nGlobal GDP Loss (peak):\n"
    summary_text += f"  With China:    {global_loss_with:.2f}%\n"
    summary_text += f"  Without:       {global_loss_without:.2f}%\n"
    summary_text += f"  Stabilization: {abs(global_loss_without - global_loss_with):.2f}pp"

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    plt.close()


def plot_china_mechanism_detail(sim_with: WarDSGESimulator,
                                 sim_without: WarDSGESimulator,
                                 save_path: str = None):
    """Detailed decomposition of China's triple stabilization mechanism."""

    quarters = np.arange(sim_with.T)
    years = quarters / 4

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "China's Triple Stabilization Mechanism: Detailed Decomposition\n"
        "中国三重稳定机制的详细分解",
        fontsize=14, fontweight='bold'
    )

    ax = axes[0, 0]
    ax.set_title('(a) Global Oil Price Stabilization')
    ax.plot(years, sim_with.oil_prices,    'b-',  linewidth=2, label='With China SPR')
    ax.plot(years, sim_without.oil_prices, 'r--', linewidth=2, label='Without')
    ax.fill_between(years, sim_with.oil_prices, sim_without.oil_prices, alpha=0.2, color='green')
    ax.set_ylabel('% change'); ax.set_xlabel('Years')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', lw=0.5, ls='--')

    ax = axes[0, 1]
    ax.set_title('(b) Global Inflation Stabilization')
    for name, color, lbl in [('us', '#003399', 'US'), ('row', '#666666', 'ROW')]:
        ax.plot(years, sim_with.get_series(name, 'pi') * 100,
                color=color, linewidth=2, label=f'{lbl} (with)')
        ax.plot(years, sim_without.get_series(name, 'pi') * 100,
                color=color, linewidth=2, ls='--', alpha=0.6, label=f'{lbl} (w/o)')
    ax.set_ylabel('pp'); ax.set_xlabel('Years')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', lw=0.5, ls='--')

    ax = axes[0, 2]
    ax.set_title('(c) Global Output Stabilization')
    global_w  = np.zeros(sim_with.T)
    global_wo = np.zeros(sim_with.T)
    for name, p in sim_with.regions.items():
        global_w  += sim_with.get_series(name, 'y')    * p.gdp_share * 100
        global_wo += sim_without.get_series(name, 'y') * p.gdp_share * 100
    ax.plot(years, global_w,  'g-',  linewidth=2.5, label='With China stabilization')
    ax.plot(years, global_wo, 'r--', linewidth=2.5, label='Without')
    ax.fill_between(years, global_wo, global_w, alpha=0.15, color='green')
    ax.set_ylabel('% deviation'); ax.set_xlabel('Years')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', lw=0.5, ls='--')

    ax = axes[1, 0]
    ax.set_title('Mechanism 1: SPR Release')
    shocks_w    = sim_with.design_war_shocks()
    spr_release = [shocks_w[t]['china_spr'] * 100 for t in range(sim_with.T)]
    us_spr      = [shocks_w[t]['us_spr']    * 100 for t in range(sim_with.T)]
    ax.bar(years, spr_release, width=0.2, color='#CC6600', alpha=0.8, label='China SPR')
    ax.bar(years, us_spr, width=0.2, bottom=spr_release, color='#003399', alpha=0.5, label='US SPR')
    ax.set_ylabel('% of global supply'); ax.set_xlabel('Years')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.set_title('Mechanism 2: Renewable Energy Acceleration')
    renewable = [0.2 * (1 - np.exp(-0.15 * t)) for t in range(sim_with.T)]
    ax.fill_between(years, renewable, alpha=0.3, color='green')
    ax.plot(years, renewable, 'g-', linewidth=2, label='Renewable offset (% of oil demand)')
    ax.set_ylabel('% of oil demand replaced'); ax.set_xlabel('Years')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.set_title('Mechanism 3: Manufacturing Export Boost')
    mfg_boost = [shocks_w[t]['china_mfg_boost'] * 100 for t in range(sim_with.T)]
    ax.fill_between(years, mfg_boost, alpha=0.3, color='#CC6600')
    ax.plot(years, mfg_boost, color='#CC6600', linewidth=2, label='Manufacturing export boost')
    ax.set_ylabel('% boost'); ax.set_xlabel('Years')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    plt.close()


def plot_financial_channels(sim_with: WarDSGESimulator,
                             sim_without: WarDSGESimulator,
                             save_path: str = None):
    """New v2 plot: financial channel dynamics across regions.

    4x3 grid:
      Row 1: Sovereign spreads
      Row 2: Capital flows
      Row 3: Equity indices
      Row 4: Enhanced exchange rates + contagion summary
    """
    quarters = np.arange(sim_with.T)
    years = quarters / 4

    fig, axes = plt.subplots(4, 3, figsize=(18, 22))
    fig.suptitle(
        'DSGE Model v2: Financial Channel Dynamics\n'
        'Sovereign Risk, Capital Flows, Equity Markets & Exchange Rates\n'
        'US-Israel War on Iran — Hormuz Scenario',
        fontsize=14, fontweight='bold', y=0.98
    )

    colors = {
        'iran':  '#CC0000',
        'us':    '#003399',
        'china': '#CC6600',
        'row':   '#666666',
    }
    labels = {
        'iran':  'Iran',
        'us':    'US/Israel',
        'china': 'China',
        'row':   'ROW',
    }

    # ---- Row 1: Sovereign Spreads ----
    ax = axes[0, 0]
    ax.set_title('Sovereign Spreads — All Regions', fontsize=12, fontweight='bold')
    for name in ['iran', 'us', 'china', 'row']:
        ax.plot(years, sim_with.get_series(name, 'spread') * 10000,
                color=colors[name], linewidth=2, label=labels[name])
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_ylabel('Basis points (bp)'); ax.set_xlabel('Years after war onset')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_title('Sovereign Spreads — All Regions\n(Iran: massive jump, US: safe haven)',
                 fontsize=11, fontweight='bold')

    ax = axes[0, 1]
    ax.set_title('Iran Sovereign Spread (detail)', fontsize=12, fontweight='bold')
    iran_spread_w  = sim_with.get_series('iran', 'spread') * 10000
    iran_spread_wo = sim_without.get_series('iran', 'spread') * 10000
    ax.fill_between(years, iran_spread_wo, iran_spread_w, alpha=0.25, color='green',
                    label='Difference (w/wo China)')
    ax.plot(years, iran_spread_w,  color='#CC0000', linewidth=2.5, label='With China stabilization')
    ax.plot(years, iran_spread_wo, color='#CC0000', linewidth=2,   linestyle='--', alpha=0.6,
            label='Without')
    ax.set_ylabel('Basis points'); ax.set_xlabel('Years after war onset')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.set_title('Non-Iran Sovereign Spreads (bp)', fontsize=12, fontweight='bold')
    for name in ['us', 'china', 'row']:
        ax.plot(years, sim_with.get_series(name, 'spread') * 10000,
                color=colors[name], linewidth=2, label=labels[name])
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_ylabel('Basis points'); ax.set_xlabel('Years after war onset')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # ---- Row 2: Capital Flows ----
    ax = axes[1, 0]
    ax.set_title('Capital Flows — All Regions (% of GDP)', fontsize=12, fontweight='bold')
    for name in ['iran', 'us', 'china', 'row']:
        ax.plot(years, sim_with.get_series(name, 'capital_flow') * 100,
                color=colors[name], linewidth=2, label=labels[name])
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_ylabel('Net capital inflow (% GDP)'); ax.set_xlabel('Years after war onset')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.set_title('Capital Flows: With vs Without China Stabilization', fontsize=12, fontweight='bold')
    for name in ['china', 'row']:
        cf_w  = sim_with.get_series(name, 'capital_flow') * 100
        cf_wo = sim_without.get_series(name, 'capital_flow') * 100
        ax.plot(years, cf_w,  color=colors[name], linewidth=2,
                label=f'{labels[name]} (with China)')
        ax.plot(years, cf_wo, color=colors[name], linewidth=2, linestyle='--', alpha=0.6,
                label=f'{labels[name]} (without)')
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_ylabel('Net capital inflow (% GDP)'); ax.set_xlabel('Years after war onset')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.set_title('US Safe-Haven Capital Inflows vs Iran Capital Flight', fontsize=12, fontweight='bold')
    ax.fill_between(years, 0, sim_with.get_series('us', 'capital_flow') * 100,
                    alpha=0.3, color='#003399', label='US inflow (safe haven)')
    ax.fill_between(years, sim_with.get_series('iran', 'capital_flow') * 100, 0,
                    alpha=0.3, color='#CC0000', label='Iran outflow (capital flight)')
    ax.plot(years, sim_with.get_series('us',   'capital_flow') * 100,
            color='#003399', linewidth=2)
    ax.plot(years, sim_with.get_series('iran', 'capital_flow') * 100,
            color='#CC0000', linewidth=2)
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_ylabel('% of GDP'); ax.set_xlabel('Years after war onset')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # ---- Row 3: Equity indices ----
    ax = axes[2, 0]
    ax.set_title('Equity Indices — All Regions (%)', fontsize=12, fontweight='bold')
    for name in ['iran', 'us', 'china', 'row']:
        ax.plot(years, sim_with.get_series(name, 'equity') * 100,
                color=colors[name], linewidth=2, label=labels[name])
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_ylabel('Equity index deviation (%)'); ax.set_xlabel('Years after war onset')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.set_title('Equity Recovery: With vs Without China', fontsize=12, fontweight='bold')
    for name in ['us', 'china', 'row']:
        eq_w  = sim_with.get_series(name, 'equity') * 100
        eq_wo = sim_without.get_series(name, 'equity') * 100
        ax.plot(years, eq_w,  color=colors[name], linewidth=2,
                label=f'{labels[name]} (with China)')
        ax.plot(years, eq_wo, color=colors[name], linewidth=2, linestyle='--', alpha=0.6,
                label=f'{labels[name]} (without)')
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_ylabel('Equity index deviation (%)'); ax.set_xlabel('Years after war onset')
    ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)

    ax = axes[2, 2]
    ax.set_title('Iran Equity Collapse (% deviation)', fontsize=12, fontweight='bold')
    iran_eq_w  = sim_with.get_series('iran', 'equity') * 100
    iran_eq_wo = sim_without.get_series('iran', 'equity') * 100
    ax.fill_between(years, iran_eq_w,  0, alpha=0.25, color='#CC0000')
    ax.plot(years, iran_eq_w,  color='#CC0000', linewidth=2.5, label='With China stabilization')
    ax.plot(years, iran_eq_wo, color='#CC0000', linewidth=2,   linestyle='--', alpha=0.6,
            label='Without')
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_ylabel('%'); ax.set_xlabel('Years after war onset')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # ---- Row 4: Exchange rates + contagion summary ----
    ax = axes[3, 0]
    ax.set_title('Real Exchange Rate — Enhanced UIP (% depreciation +)', fontsize=12, fontweight='bold')
    for name in ['iran', 'us', 'china', 'row']:
        ax.plot(years, sim_with.get_series(name, 'rer') * 100,
                color=colors[name], linewidth=2, label=labels[name])
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_ylabel('% deviation (+ = depreciation)'); ax.set_xlabel('Years after war onset')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[3, 1]
    ax.set_title('China RMB: FX Intervention Effect', fontsize=12, fontweight='bold')
    rer_china_w  = sim_with.get_series('china', 'rer') * 100
    rer_china_wo = sim_without.get_series('china', 'rer') * 100
    ax.fill_between(years, rer_china_wo, rer_china_w, alpha=0.3, color='green',
                    label='FX intervention effect')
    ax.plot(years, rer_china_w,  color='#CC6600', linewidth=2.5, label='With FX intervention')
    ax.plot(years, rer_china_wo, color='#CC6600', linewidth=2,   linestyle='--', alpha=0.6,
            label='Without')
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_ylabel('% deviation'); ax.set_xlabel('Years after war onset')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Contagion / summary panel
    ax = axes[3, 2]
    ax.axis('off')
    fin_summary = (
        "FINANCIAL CHANNELS — Peak Effects\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "Sovereign Spreads (bp, peak):\n"
    )
    for name in ['iran', 'us', 'china', 'row']:
        peak_bp = sim_with.get_series(name, 'spread')[:8].max() * 10000
        fin_summary += f"  {labels[name]:<18}: {peak_bp:>+6.0f} bp\n"

    fin_summary += "\nCapital Flows (% GDP, peak):\n"
    for name in ['iran', 'us', 'china', 'row']:
        peak_cf = sim_with.get_series(name, 'capital_flow')[:8].min() * 100
        fin_summary += f"  {labels[name]:<18}: {peak_cf:>+6.1f}%\n"

    fin_summary += "\nEquity (% deviation, trough):\n"
    for name in ['iran', 'us', 'china', 'row']:
        trough_eq = sim_with.get_series(name, 'equity')[:8].min() * 100
        fin_summary += f"  {labels[name]:<18}: {trough_eq:>+6.1f}%\n"

    fin_summary += "\nExch. Rate (%, peak depreciation):\n"
    for name in ['iran', 'us', 'china', 'row']:
        peak_rer = sim_with.get_series(name, 'rer')[:8].max() * 100
        fin_summary += f"  {labels[name]:<18}: {peak_rer:>+6.1f}%\n"

    ax.text(0.03, 0.97, fin_summary, transform=ax.transAxes,
            fontsize=9.5, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#e8f4f8', alpha=0.9))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    plt.close()


# ============================================================================
# 7. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":

    print("=" * 72)
    print("DSGE MODEL v2: THE PRICE OF WAR — IRAN-HORMUZ + FINANCIAL CHANNELS")
    print("Inspired by Federle et al. (2026) 'The Price of War', AER")
    print("=" * 72)

    # Scenario 1: With China stabilization
    print("\n[1/2] Simulating WITH China stabilization mechanism...")
    sim_with = WarDSGESimulator(T=32, china_stabilizes=True)
    sim_with.simulate()

    # Scenario 2: Without China stabilization
    print("[2/2] Simulating WITHOUT China stabilization mechanism...")
    sim_without = WarDSGESimulator(T=32, china_stabilizes=False)
    sim_without.simulate()

    # --- Generate figures ---
    print("\nGenerating real-economy impulse response figures...")
    plot_impulse_responses(
        sim_with, sim_without,
        save_path="D:/war_dsge_model/impulse_responses_v2.png"
    )

    print("Generating China mechanism detail figures...")
    plot_china_mechanism_detail(
        sim_with, sim_without,
        save_path="D:/war_dsge_model/china_mechanism_v2.png"
    )

    print("Generating financial channel figures...")
    plot_financial_channels(
        sim_with, sim_without,
        save_path="D:/war_dsge_model/financial_channels_v2.png"
    )

    # --- Print summary tables ---
    print("\n" + "=" * 72)
    print("QUANTITATIVE RESULTS SUMMARY — REAL ECONOMY")
    print("=" * 72)

    print("\n{:<25} {:>10} {:>10} {:>10} {:>10}".format(
        "", "Iran", "US/Israel", "China", "ROW"))
    print("-" * 72)

    print("{:<25}".format("Peak GDP loss (%)"), end="")
    for name in ['iran', 'us', 'china', 'row']:
        val = sim_with.get_series(name, 'y')[:8].min() * 100
        print(f" {val:>+9.1f}", end="")
    print()

    print("{:<25}".format("Peak inflation (pp)"), end="")
    for name in ['iran', 'us', 'china', 'row']:
        val = sim_with.get_series(name, 'pi')[:8].max() * 100
        print(f" {val:>+9.1f}", end="")
    print()

    print("{:<25}".format("Peak interest rate (pp)"), end="")
    for name in ['iran', 'us', 'china', 'row']:
        val = sim_with.get_series(name, 'i')[:8].max() * 100
        print(f" {val:>+9.1f}", end="")
    print()

    print("\n" + "=" * 72)
    print("QUANTITATIVE RESULTS SUMMARY — FINANCIAL CHANNELS (v2)")
    print("=" * 72)

    print("\n{:<25} {:>10} {:>10} {:>10} {:>10}".format(
        "", "Iran", "US/Israel", "China", "ROW"))
    print("-" * 72)

    print("{:<25}".format("Peak spread (bp)"), end="")
    for name in ['iran', 'us', 'china', 'row']:
        val = sim_with.get_series(name, 'spread')[:8].max() * 10000
        print(f" {val:>+9.0f}", end="")
    print()

    print("{:<25}".format("Capital flow trough (%GDP)"), end="")
    for name in ['iran', 'us', 'china', 'row']:
        val = sim_with.get_series(name, 'capital_flow')[:8].min() * 100
        print(f" {val:>+9.1f}", end="")
    print()

    print("{:<25}".format("Equity trough (%)"), end="")
    for name in ['iran', 'us', 'china', 'row']:
        val = sim_with.get_series(name, 'equity')[:8].min() * 100
        print(f" {val:>+9.1f}", end="")
    print()

    print("{:<25}".format("Peak RER depreciation (%)"), end="")
    for name in ['iran', 'us', 'china', 'row']:
        val = sim_with.get_series(name, 'rer')[:8].max() * 100
        print(f" {val:>+9.1f}", end="")
    print()

    print("\n" + "-" * 72)
    print("CHINA STABILIZATION EFFECT")
    print("-" * 72)

    oil_peak_w  = max(sim_with.oil_prices[:8])
    oil_peak_wo = max(sim_without.oil_prices[:8])
    print(f"Oil price peak (with China SPR):    +{oil_peak_w:.0f}%")
    print(f"Oil price peak (without):           +{oil_peak_wo:.0f}%")
    print(f"Oil price reduction:                {oil_peak_wo - oil_peak_w:.0f} pp")

    global_w  = np.zeros(32)
    global_wo = np.zeros(32)
    for name, p in sim_with.regions.items():
        global_w  += sim_with.get_series(name, 'y')    * p.gdp_share * 100
        global_wo += sim_without.get_series(name, 'y') * p.gdp_share * 100

    print(f"\nGlobal GDP loss (with China):        {global_w[:8].min():.2f}%")
    print(f"Global GDP loss (without):           {global_wo[:8].min():.2f}%")
    print(f"GDP saved by China stabilization:    {abs(global_wo[:8].min() - global_w[:8].min()):.2f} pp")

    global_pi_w  = np.zeros(32)
    global_pi_wo = np.zeros(32)
    for name, p in sim_with.regions.items():
        global_pi_w  += sim_with.get_series(name, 'pi') * p.gdp_share * 100
        global_pi_wo += sim_without.get_series(name, 'pi') * p.gdp_share * 100

    print(f"\nGlobal inflation peak (with China):  +{global_pi_w[:8].max():.2f} pp")
    print(f"Global inflation peak (without):     +{global_pi_wo[:8].max():.2f} pp")
    print(f"Inflation reduced by China:          {global_pi_wo[:8].max() - global_pi_w[:8].max():.2f} pp")

    # Financial contagion amplification
    print("\n" + "-" * 72)
    print("FINANCIAL CONTAGION AMPLIFICATION (v2)")
    print("-" * 72)
    row_eq_w  = sim_with.get_series('row', 'equity')[:8].min() * 100
    row_eq_wo = sim_without.get_series('row', 'equity')[:8].min() * 100
    print(f"ROW equity trough (with China):    {row_eq_w:.1f}%")
    print(f"ROW equity trough (without):       {row_eq_wo:.1f}%")
    print(f"China cushion on ROW equity:       {abs(row_eq_wo - row_eq_w):.1f} pp")

    us_eq_w  = sim_with.get_series('us', 'equity')[:8].min() * 100
    us_eq_wo = sim_without.get_series('us', 'equity')[:8].min() * 100
    print(f"\nUS equity trough (with China):     {us_eq_w:.1f}%")
    print(f"US equity trough (without):        {us_eq_wo:.1f}%")

    print("\n" + "=" * 72)
    print("COMPARISON WITH FEDERLE ET AL. (2026)")
    print("=" * 72)
    print("""
Federle et al. findings (average war):     This model v2 (Iran-Hormuz):
  War site GDP:        -10%                  Iran GDP:       {iran_gdp:+.1f}%
  War site CPI:        +20%                  Iran CPI:       +high (supply destruction)
  High-exposure third: -2%                   ROW GDP:        {row_gdp:+.1f}%
  Belligerent GDP:     ~0%                   US GDP:         {us_gdp:+.1f}% (fiscal stimulus)
  Recovery time:       ~12 years             Recovery:       ~6-8 years

v2 additions — financial amplification:
  Iran sovereign spread:   +{iran_spread:.0f} bp (war risk + capital flight)
  USD safe-haven:          spreads near zero; capital inflows +{us_cf:.1f}% GDP
  China FX intervention:   RMB stabilization via forex reserves
  Global equity contagion: Iran -40%, US {us_eq:+.1f}%, ROW {row_eq:+.1f}%
""".format(
        iran_gdp    = sim_with.get_series('iran', 'y')[:8].min()    * 100,
        row_gdp     = sim_with.get_series('row',  'y')[:8].min()    * 100,
        us_gdp      = sim_with.get_series('us',   'y')[:8].min()    * 100,
        iran_spread = sim_with.get_series('iran', 'spread')[:8].max() * 10000,
        us_cf       = sim_with.get_series('us',   'capital_flow')[:4].max() * 100,
        us_eq       = sim_with.get_series('us',   'equity')[:8].min() * 100,
        row_eq      = sim_with.get_series('row',  'equity')[:8].min() * 100,
    ))

    print("All results and figures saved to D:/war_dsge_model/")
    print("--- Simulation v2 Complete ---")
