"""
================================================================================
DSGE Model v3: The Price of War — Iran-Hormuz Scenario
                (Higher-Order Perturbation + Endogenous Stabilizer)
================================================================================

Extension of model_v2.py addressing supervisor critiques:

  CRITIQUE 1: Linearization is inappropriate for rare-disaster shocks (27% GDP).
              => Add 2nd-order perturbation correction terms (state-dependent
                 coefficients capturing nonlinearities far from steady state).

  CRITIQUE 2: China's SPR release and manufacturing expansion were exogenous
              deterministic paths.
              => Replace with Taylor-rule-style reaction functions:
                   spr_t = rho_spr * spr_{t-1}
                          + phi_oil * (oil_p_t - target)
                          + phi_yglobal * max(-y_global_{t-1}, 0)
                   mfg_t = rho_mfg * mfg_{t-1}
                          + phi_demand * max(-y_global_{t-1}, 0)
                          + phi_trade * trade_disruption_t

References:
  - Schmitt-Grohe & Uribe (2004) "Solving DSGE Models w/ Higher-Order Perturbation"
  - Fernandez-Villaverde et al. (2016) "Solution & Estimation Methods for DSGE"
  - Gourio (2012) "Disaster Risk and Business Cycles" AER
================================================================================
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Reuse v2 components
from model_v2 import (
    RegionParams, RegionState, OilMarket,
    calibrate_regions, FINANCIAL_LINKAGE, REGION_ORDER,
)


# ============================================================================
# 1. NONLINEAR (2nd-ORDER) REGION SOLVER
# ============================================================================

def solve_region_period_nonlinear(
    params: RegionParams,
    state_prev: RegionState,
    oil_price_shock: float,
    military_shock: float = 0.0,
    trade_shock: float = 0.0,
    supply_destruction: float = 0.0,
    spr_release: float = 0.0,
    china_manufacturing_boost: float = 0.0,
    expected_pi_next: float = 0.0,
    spread_shock: float = 0.0,
    capital_flow_shock: float = 0.0,
    equity_shock: float = 0.0,
    forex_intervention: float = 0.0,
    order: int = 2,
) -> RegionState:
    """Solve one period equilibrium with optional 2nd-order perturbation.

    For rare-disaster shocks the standard 1st-order linearization understates
    the nonlinear interaction between (a) supply destruction, (b) the
    monetary-policy reaction, and (c) financial frictions. The 2nd-order
    correction adds quadratic state/shock cross terms that vanish near the
    steady state but become quantitatively important when |y| > 0.10.

    Equations (departures from v2 in CAPS):

      Phillips curve (NL):
        pi_t = beta*Epi + kappa*y_{t-1} + supply_shock
               + ETA_PI * supply_shock^2          [convex pass-through]
               + GAMMA_PI * max(-y_{t-1},0)*pi_{t-1}  [downward rigidity break]

      IS curve (NL):
        y_t = persistence*y_{t-1} + ...
              + ZETA_Y * y_{t-1}^2               [precautionary saving]
              + XI_Y * supply_destruction^2      [physical-capital nonlin.]
              + LAMBDA_Y * spread_t * max(-y_{t-1},0)  [financial accelerator]

      Sovereign spread (NL):
        spread_t = base + DELTA_S * max(-y_{t-1},0)^2  [convex risk premium]

    With order=1 we recover the v2 baseline; order=2 activates the
    quadratic terms.
    """
    s = RegionState()

    # ---- Energy price pass-through (unchanged) ----
    net_oil_exposure = max(params.oil_import_share - params.oil_export_share, 0.0)
    s.oil_p = oil_price_shock * (net_oil_exposure / max(params.alpha_e, 0.01)) * 0.5
    s.oil_p = np.clip(s.oil_p, -2.0, 2.0)
    s.oil_p -= spr_release * 0.3

    # ---- Phillips curve with 2nd-order correction ----
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

    if order >= 2:
        # Convex pass-through: marginal cost rises faster as supply shock grows
        eta_pi = 0.25
        s.pi += eta_pi * supply_shock * abs(supply_shock)
        # Downward nominal rigidity break: deep recessions see persistent
        # disinflation pressure interact with prior inflation
        gamma_pi = 0.15
        s.pi += -gamma_pi * max(-state_prev.y, 0) * max(state_prev.pi, 0)

    s.pi = np.clip(s.pi, -0.15, 0.30)

    # ---- Taylor rule with ZLB nonlinearity ----
    s.i = (params.rho_i * state_prev.i
           + (1 - params.rho_i) * (params.phi_pi * s.pi
                                    + params.phi_y * max(state_prev.y, -0.3)))
    if order >= 2:
        # ZLB friction: as i hits zero, effective rate is bounded => weaker
        # transmission. Capture as quadratic dampening.
        zlb_dampen = 0.15
        s.i += -zlb_dampen * max(-s.i, 0) ** 2
    s.i = np.clip(s.i, -0.05, 0.15)

    # ---- Sovereign spread with 2nd-order convexity ----
    s.spread = (spread_shock
                + 0.3 * max(-state_prev.y, 0)
                + 0.2 * state_prev.spread)

    if order >= 2:
        # Convex sovereign risk: deep recessions sharply elevate default risk
        # (we cap the recession depth used here at 20% to prevent runaway)
        recession_depth = min(max(-state_prev.y, 0), 0.20)
        delta_s = 0.20
        s.spread += delta_s * recession_depth ** 2
        # Doom-loop: high spread + capital flight feedback
        doom_loop = 0.10
        s.spread += doom_loop * max(-state_prev.capital_flow, 0) * state_prev.spread

    s.spread = np.clip(s.spread, 0, 0.40)

    # ---- Capital flows with quadratic flight-to-safety ----
    s.capital_flow = (capital_flow_shock
                      - 2.0 * s.spread
                      + 0.5 * (s.i - 0.02)
                      + 0.3 * state_prev.capital_flow)
    if order >= 2:
        # Sudden-stop nonlinearity: when spread > 5%, flight accelerates
        sudden_stop = 0.8
        s.capital_flow += -sudden_stop * max(s.spread - 0.05, 0) ** 2
    s.capital_flow = np.clip(s.capital_flow, -0.25, 0.10)

    # ---- IS curve with 2nd-order corrections ----
    real_rate_effect = -(1.0 / params.sigma) * (s.i - expected_pi_next) * 0.3
    military_effect  = 0.8 * military_shock
    trade_effect     = -trade_shock * params.trade_openness * 0.5
    energy_cost_effect = -params.alpha_e * max(s.oil_p, 0) * 0.3
    financial_conditions = -0.15 * s.spread - 0.1 * max(-s.capital_flow, 0)

    persistence = 0.75
    s.y = (persistence * state_prev.y
           + real_rate_effect + military_effect + trade_effect
           + energy_cost_effect + financial_conditions
           - supply_destruction * 0.5)
    s.y += -0.05 * state_prev.y

    if order >= 2:
        # Precautionary saving: deep recessions reduce consumption
        # super-linearly (Carroll 1997). Cap the depth used to avoid runaway.
        recession_depth = min(max(-state_prev.y, 0), 0.20)
        zeta_y = 0.15
        s.y += -zeta_y * recession_depth ** 2
        # Capital destruction nonlinearity: physical capital loss compounds
        xi_y = 0.20
        s.y += -xi_y * supply_destruction ** 2
        # Financial accelerator: tight conditions in a downturn amplify.
        # Use capped recession depth so this doesn't explode.
        lambda_y = 0.30
        s.y += -lambda_y * s.spread * recession_depth

    s.y = np.clip(s.y, -0.40, 0.15)

    # ---- Exchange rate ----
    s.rer = (0.4 * state_prev.rer
             + 0.5 * s.spread
             - 0.3 * s.capital_flow
             + forex_intervention)
    if order >= 2:
        # Currency crisis nonlinearity: abrupt depreciations
        crisis_jump = 0.30
        s.rer += crisis_jump * max(s.spread - 0.10, 0)
    s.rer = np.clip(s.rer, -0.20, 0.80)

    # ---- Equity ----
    expected_growth = 0.02 + 0.5 * s.y
    discount = s.i + s.spread
    s.equity = (0.6 * state_prev.equity
                + 10.0 * expected_growth
                - 8.0 * discount
                - 3.0 * max(-s.y, 0)
                + equity_shock)
    if order >= 2:
        # Disaster risk premium: equity collapses convexly with recession depth
        disaster_eq = 3.0
        s.equity += -disaster_eq * max(-s.y, 0) ** 2
    s.equity = np.clip(s.equity, -0.70, 0.20)

    # ---- Standard accounting ----
    s.c   = s.y - military_shock * 0.4
    s.inv = s.y * 1.2 - 0.3 * max(s.i - s.pi, 0)
    s.nx = (-net_oil_exposure * oil_price_shock * 0.5
            + params.oil_export_share * oil_price_shock * 0.3
            + 0.2 * s.rer
            + trade_shock * 0.3)
    s.nx = np.clip(s.nx, -0.15, 0.15)
    s.g_mil = military_shock
    s.spr = spr_release

    return s


# ============================================================================
# 2. ENDOGENOUS CHINA STABILIZER RULES
# ============================================================================

@dataclass
class ChinaStabilizerRule:
    """Taylor-rule-style reaction functions for China's stabilization tools.

    Replaces the exogenous deterministic paths in v2:
        china_spr = 0.008 * exp(-0.1*t)
        china_mfg = 0.015 * (1 - exp(-0.2*t))

    With endogenous reaction functions:
        spr_t = rho_spr*spr_{t-1}
              + phi_oil * max(oil_p_t - oil_target, 0)
              + phi_yglob * max(-y_global_{t-1}, 0)
              + spr_max_cap

        mfg_t = rho_mfg*mfg_{t-1}
              + phi_dem  * max(-y_global_{t-1}, 0)
              + phi_trd  * trade_disruption_t

    The rules are activated whenever oil-price spillovers or global-output
    shortfalls cross policy thresholds — i.e., they react TO the state of the
    world economy rather than following a calendar.
    """

    # SPR rule
    rho_spr:    float = 0.85
    phi_oil:    float = 0.020   # 1pp oil price shock => 2bp SPR release
    phi_yglob:  float = 0.025   # 1pp global GDP loss => 2.5bp SPR release
    oil_target: float = 0.10    # SPR activates when oil > +10%
    spr_max:    float = 0.020   # cap SPR release at 2% of global supply

    # Renewable substitution rule
    rho_renew:  float = 0.95    # high persistence (capacity buildup)
    phi_renew:  float = 0.005   # ramps up when oil price elevated

    # Manufacturing rule
    rho_mfg:    float = 0.80
    phi_dem:    float = 0.30    # 1pp global GDP loss => 30bp mfg expansion
    phi_trd:    float = 0.020   # responds to trade disruption
    mfg_max:    float = 0.030   # cap mfg boost at 3%

    def reaction(self, oil_p: float, y_global_prev: float,
                 spr_prev: float, mfg_prev: float, renew_prev: float,
                 trade_disruption: float) -> Tuple[float, float, float]:
        """Compute current-period SPR, manufacturing, renewable responses."""
        spr = (self.rho_spr * spr_prev
               + self.phi_oil  * max(oil_p - self.oil_target, 0)
               + self.phi_yglob * max(-y_global_prev, 0))
        spr = min(max(spr, 0), self.spr_max)

        renew = (self.rho_renew * renew_prev
                 + self.phi_renew * max(oil_p - 0.05, 0))
        renew = min(max(renew, 0), 0.05)

        mfg = (self.rho_mfg * mfg_prev
               + self.phi_dem * max(-y_global_prev, 0)
               + self.phi_trd * trade_disruption)
        mfg = min(max(mfg, 0), self.mfg_max)

        return spr, renew, mfg


# ============================================================================
# 3. SIMULATOR v3 — endogenous China + nonlinear option
# ============================================================================

class WarDSGESimulatorV3:
    """v3 simulator with optional 2nd-order solution and endogenous stabilizer."""

    def __init__(self, T: int = 32,
                 china_stabilizes: bool = True,
                 order: int = 2,
                 stabilizer_rule: ChinaStabilizerRule = None,
                 financial_linkage: np.ndarray = None,
                 phi_pi_override: Dict[str, float] = None,
                 phi_y_override: Dict[str, float] = None,
                 disable_financial_channels: bool = False,
                 disable_contagion: bool = False):
        self.T = T
        self.china_stabilizes = china_stabilizes
        self.order = order
        self.regions = calibrate_regions()
        self.oil_market = OilMarket()
        self.stabilizer_rule = stabilizer_rule or ChinaStabilizerRule()
        self.financial_linkage = (financial_linkage
                                  if financial_linkage is not None
                                  else FINANCIAL_LINKAGE.copy())
        self.disable_financial_channels = disable_financial_channels
        self.disable_contagion = disable_contagion

        # Apply Taylor-rule overrides for counterfactuals
        if phi_pi_override:
            for r, v in phi_pi_override.items():
                self.regions[r].phi_pi = v
        if phi_y_override:
            for r, v in phi_y_override.items():
                self.regions[r].phi_y = v

        self.results: Dict[str, list] = {name: [] for name in self.regions}
        self.oil_prices = []
        self.china_spr_path  = []
        self.china_mfg_path  = []
        self.china_renew_path = []

    # ------------------------------------------------------------------
    def simulate(self) -> Dict[str, list]:
        states = {name: RegionState() for name in self.regions}

        # Endogenous stabilizer state
        spr_prev_china = 0.0
        mfg_prev_china = 0.0
        renew_prev = 0.0
        y_global_prev = 0.0

        for t in range(self.T):
            # ---- Compute oil supply shock from blockade ----
            oil_supply_loss = self.oil_market.compute_supply_shock(t, blockade_duration=8)

            # First-pass oil price WITHOUT China stabilization
            demand_reduction = 0.005 * t / 8 if t < 8 else 0.005
            oil_price_pre = self.oil_market.compute_oil_price_change(
                oil_supply_loss, 0.0, demand_reduction
            )

            # ---- Endogenous China reaction (NEW in v3) ----
            if self.china_stabilizes:
                china_spr, renew, china_mfg = self.stabilizer_rule.reaction(
                    oil_p=oil_price_pre,
                    y_global_prev=y_global_prev,
                    spr_prev=spr_prev_china,
                    mfg_prev=mfg_prev_china,
                    renew_prev=renew_prev,
                    trade_disruption=0.8 * np.exp(-0.1 * t),
                )
            else:
                china_spr, renew, china_mfg = 0.0, 0.0, 0.0

            # US/ROW SPR are still calendar-driven (allies)
            us_spr  = 0.005 * np.exp(-0.15 * t) if t < 12 else 0.0
            row_spr = 0.003 * np.exp(-0.15 * t) if t < 8 else 0.0
            total_spr = china_spr + us_spr + row_spr + renew

            # Recompute oil price with stabilizer
            oil_price_change = self.oil_market.compute_oil_price_change(
                oil_supply_loss, total_spr, demand_reduction
            )

            iran_destruction = 0.06 * np.exp(-0.10 * t)
            iran_trade_disruption = 0.08 * np.exp(-0.12 * t)
            us_military = (0.025 * np.exp(-0.05 * t) if t < 8
                           else 0.025 * np.exp(-0.2 * (t - 4)))
            trade_disruption_factor = 0.8 * np.exp(-0.1 * t)

            # Financial shocks (same as v2)
            iran_spread_shock       = 0.05 * np.exp(-0.08 * t)
            iran_capital_flow_shock = -0.15 * np.exp(-0.10 * t)
            iran_equity_shock       = -0.40 if t == 0 else 0.0

            us_spread_shock       = -0.002
            us_capital_flow_shock = 0.03 * np.exp(-0.10 * t)
            us_equity_shock       = -0.05 if t == 0 else 0.0

            china_spread_shock       = 0.003 * np.exp(-0.10 * t)
            china_capital_flow_shock = -0.02 * np.exp(-0.15 * t)
            china_equity_shock       = -0.08 if t == 0 else 0.0
            china_forex_intervention = -0.01 if (self.china_stabilizes and t < 8) else 0.0

            row_spread_shock       = 0.005 * np.exp(-0.10 * t)
            row_capital_flow_shock = -0.01 * np.exp(-0.10 * t)
            row_equity_shock       = -0.05 if t == 0 else 0.0

            # Disable financial channels for counterfactual
            if self.disable_financial_channels:
                iran_spread_shock = us_spread_shock = china_spread_shock = row_spread_shock = 0.0
                iran_capital_flow_shock = us_capital_flow_shock = 0.0
                china_capital_flow_shock = row_capital_flow_shock = 0.0
                iran_equity_shock = us_equity_shock = 0.0
                china_equity_shock = row_equity_shock = 0.0

            base_spreads = {
                'iran': iran_spread_shock,
                'us':   us_spread_shock,
                'china': china_spread_shock,
                'row':  row_spread_shock,
            }

            iran_capital_flight = max(-iran_capital_flow_shock, 0)
            global_risk_factor  = 1.0 + 2.0 * iran_capital_flight

            contagion_spreads = {}
            if self.disable_contagion:
                contagion_spreads = base_spreads.copy()
            else:
                for j_idx, j_name in enumerate(REGION_ORDER):
                    contagion = 0.0
                    for i_idx, i_name in enumerate(REGION_ORDER):
                        if i_name != j_name:
                            contagion += self.financial_linkage[i_idx, j_idx] * states[i_name].spread
                    contagion_spreads[j_name] = (base_spreads[j_name]
                                                 + 0.1 * contagion * global_risk_factor)

            # ---- Per-region solve ----
            new_states = {}
            for name, params in self.regions.items():
                expected_pi = states[name].pi * 0.7

                if name == "iran":
                    s = solve_region_period_nonlinear(
                        params, states[name],
                        oil_price_shock=oil_price_change * 0.2,
                        supply_destruction=iran_destruction,
                        trade_shock=iran_trade_disruption,
                        expected_pi_next=expected_pi,
                        spread_shock=contagion_spreads['iran'],
                        capital_flow_shock=iran_capital_flow_shock,
                        equity_shock=iran_equity_shock,
                        order=self.order,
                    )
                    s.y -= oil_supply_loss * 0.3
                    s.y = np.clip(s.y, -0.55, 0.15)
                elif name == "us":
                    s = solve_region_period_nonlinear(
                        params, states[name],
                        oil_price_shock=oil_price_change,
                        military_shock=us_military,
                        trade_shock=trade_disruption_factor * params.trade_with_iran,
                        spr_release=us_spr,
                        china_manufacturing_boost=china_mfg if self.china_stabilizes else 0,
                        expected_pi_next=expected_pi,
                        spread_shock=contagion_spreads['us'],
                        capital_flow_shock=us_capital_flow_shock,
                        equity_shock=us_equity_shock,
                        order=self.order,
                    )
                elif name == "china":
                    s = solve_region_period_nonlinear(
                        params, states[name],
                        oil_price_shock=oil_price_change,
                        trade_shock=trade_disruption_factor * params.trade_with_iran,
                        spr_release=china_spr,
                        china_manufacturing_boost=china_mfg if self.china_stabilizes else 0,
                        expected_pi_next=expected_pi,
                        spread_shock=contagion_spreads['china'],
                        capital_flow_shock=china_capital_flow_shock,
                        equity_shock=china_equity_shock,
                        forex_intervention=china_forex_intervention,
                        order=self.order,
                    )
                    if self.china_stabilizes:
                        s.y  += china_mfg * 0.3
                        s.nx += china_mfg * 0.2
                else:  # row
                    s = solve_region_period_nonlinear(
                        params, states[name],
                        oil_price_shock=oil_price_change,
                        trade_shock=trade_disruption_factor * params.trade_with_iran,
                        spr_release=row_spr,
                        china_manufacturing_boost=china_mfg if self.china_stabilizes else 0,
                        expected_pi_next=expected_pi,
                        spread_shock=contagion_spreads['row'],
                        capital_flow_shock=row_capital_flow_shock,
                        equity_shock=row_equity_shock,
                        order=self.order,
                    )
                new_states[name] = s

            states = new_states
            for name in self.regions:
                self.results[name].append(states[name])
            self.oil_prices.append(oil_price_change * 100)
            self.china_spr_path.append(china_spr)
            self.china_mfg_path.append(china_mfg)
            self.china_renew_path.append(renew)

            # Update endogenous stabilizer state
            spr_prev_china = china_spr
            mfg_prev_china = china_mfg
            renew_prev = renew

            # Update lagged global GDP
            y_global = sum(states[r].y * self.regions[r].gdp_share for r in self.regions)
            y_global_prev = y_global

        return self.results

    def get_series(self, region: str, variable: str) -> np.ndarray:
        return np.array([getattr(s, variable) for s in self.results[region]])

    def global_gdp(self) -> np.ndarray:
        out = np.zeros(self.T)
        for name, p in self.regions.items():
            out += self.get_series(name, 'y') * p.gdp_share
        return out

    def global_inflation(self) -> np.ndarray:
        out = np.zeros(self.T)
        for name, p in self.regions.items():
            out += self.get_series(name, 'pi') * p.gdp_share
        return out


# ============================================================================
# 4. WELFARE COMPUTATION
# ============================================================================

def consumption_equivalent_welfare(
    sim: WarDSGESimulatorV3,
    discount: float = 0.99,
    risk_aversion: float = 1.5,
) -> Dict[str, float]:
    """Compute consumption-equivalent welfare loss for each region (Lucas 1987).

    Welfare = sum_t beta^t * u(c_t),  u(c) = c^(1-sigma)/(1-sigma)
    Then find lambda such that  W(c*(1-lambda)) = W_war.
    Returns lambda as % of permanent consumption.
    """
    out = {}
    for name in sim.regions:
        c = sim.get_series(name, 'c')          # log-deviation
        # Map to consumption level: C_t = (1+c_t) * C_ss, normalized C_ss=1
        C = np.maximum(1 + c, 0.01)
        weights = discount ** np.arange(sim.T)
        if abs(risk_aversion - 1.0) < 1e-6:
            U_war = (weights * np.log(C)).sum()
            U_ss  = (weights * np.log(np.ones_like(C))).sum()
        else:
            U_war = (weights * (C ** (1 - risk_aversion) - 1) / (1 - risk_aversion)).sum()
            U_ss  = (weights * (np.ones_like(C) ** (1 - risk_aversion) - 1) /
                     (1 - risk_aversion)).sum()
        # Solve U_war = U(c*(1-lambda)*ones) for lambda
        T = sim.T
        if abs(risk_aversion - 1.0) < 1e-6:
            sum_w = weights.sum()
            log_lambda_arg = (U_ss - U_war) / sum_w
            lam = 1.0 - np.exp(-log_lambda_arg)
        else:
            sum_w = weights.sum()
            denom = (sum_w / (1 - risk_aversion))
            ratio = (U_war + denom) / denom
            ratio = max(ratio, 1e-6)
            lam = 1.0 - ratio ** (1.0 / (1 - risk_aversion))
        out[name] = lam * 100  # percent
    return out


# ============================================================================
# 5. MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("DSGE MODEL v3 — Higher-Order Perturbation + Endogenous Stabilizer")
    print("=" * 72)

    # ---- Run linear vs nonlinear comparison ----
    print("\n[1/4] Linear (1st-order)  + endogenous China rule...")
    sim_linear = WarDSGESimulatorV3(T=32, china_stabilizes=True, order=1)
    sim_linear.simulate()

    print("[2/4] Nonlinear (2nd-order) + endogenous China rule...")
    sim_nonlin = WarDSGESimulatorV3(T=32, china_stabilizes=True, order=2)
    sim_nonlin.simulate()

    print("[3/4] Nonlinear, no China stabilization...")
    sim_nonlin_no = WarDSGESimulatorV3(T=32, china_stabilizes=False, order=2)
    sim_nonlin_no.simulate()

    print("[4/4] Welfare computation...")
    w_linear = consumption_equivalent_welfare(sim_linear)
    w_nonlin = consumption_equivalent_welfare(sim_nonlin)
    w_no_china = consumption_equivalent_welfare(sim_nonlin_no)

    print("\n" + "=" * 72)
    print("LINEAR vs NONLINEAR COMPARISON")
    print("=" * 72)
    print(f"\n{'Region':<12} {'Linear Peak':>15} {'Nonlinear Peak':>17} "
          f"{'Diff (pp)':>12}")
    print("-" * 72)
    for name in ['iran', 'us', 'china', 'row']:
        lin_peak = sim_linear.get_series(name, 'y')[:8].min() * 100
        nl_peak  = sim_nonlin.get_series(name, 'y')[:8].min() * 100
        diff     = nl_peak - lin_peak
        print(f"{name:<12} {lin_peak:>+14.2f}% {nl_peak:>+16.2f}% {diff:>+11.2f}")

    print("\n" + "=" * 72)
    print("ENDOGENOUS CHINA STABILIZER PATHS (vs v2 deterministic)")
    print("=" * 72)
    print(f"\n{'Quarter':<10} {'SPR (%)':>12} {'Mfg (%)':>12} {'Renew (%)':>12}")
    print("-" * 50)
    for t in [0, 1, 2, 4, 6, 8, 12, 16, 24]:
        print(f"{t:<10} {sim_nonlin.china_spr_path[t]*100:>+11.3f} "
              f"{sim_nonlin.china_mfg_path[t]*100:>+11.3f} "
              f"{sim_nonlin.china_renew_path[t]*100:>+11.3f}")

    print("\n" + "=" * 72)
    print("WELFARE LOSSES (consumption-equivalent, %)")
    print("=" * 72)
    print(f"\n{'Region':<12} {'Linear':>12} {'Nonlinear':>12} {'No China':>12}")
    print("-" * 60)
    for name in ['iran', 'us', 'china', 'row']:
        print(f"{name:<12} {w_linear[name]:>11.3f}% "
              f"{w_nonlin[name]:>11.3f}% {w_no_china[name]:>11.3f}%")
