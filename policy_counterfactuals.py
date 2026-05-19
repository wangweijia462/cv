"""
================================================================================
Policy Counterfactuals — Welfare Analysis
================================================================================

ADDRESSES SUPERVISOR CRITIQUE 4:
  "The welfare analysis would be much richer if you ran policy
   counterfactuals: vary the Taylor rules, switch financial channels on/off,
   and quantify the welfare consequences of each policy regime."

EXPERIMENTS:
  E1. Baseline                            (v3 calibration, China stabilizes)
  E2. No China stabilization              (rule turned off)
  E3. Aggressive Fed                      (US phi_pi = 2.5 vs 1.5)
  E4. Dovish Fed                          (US phi_pi = 1.0)
  E5. Coordinated easing                  (all phi_y *= 2)
  E6. No financial channels               (spreads/equity/cap flows = 0)
  E7. No financial contagion              (L matrix zero off-diagonal)
  E8. Strong China stabilizer             (phi_yglob doubled)
  E9. Empirical L matrix                  (VAR-estimated linkage)

For each experiment, report:
  - Peak GDP loss per region
  - Consumption-equivalent welfare loss (Lucas 1987)
  - Differential vs baseline ("welfare gain from policy X")
================================================================================
"""

import numpy as np
import os
from model_v3 import (
    WarDSGESimulatorV3, ChinaStabilizerRule, consumption_equivalent_welfare
)


# Try to load empirically estimated L matrix
L_VAR_PATH = "D:/war_dsge_model/L_var_estimated.npy"
if os.path.exists(L_VAR_PATH):
    L_EMPIRICAL = np.load(L_VAR_PATH)
else:
    L_EMPIRICAL = None


# ============================================================================
# Experiment definitions
# ============================================================================

def run_experiment(name: str, **kwargs) -> dict:
    """Run one counterfactual and return summary statistics."""
    sim = WarDSGESimulatorV3(T=32, order=2, **kwargs)
    sim.simulate()

    welfare = consumption_equivalent_welfare(sim)
    peak_y = {r: sim.get_series(r, 'y')[:8].min() * 100 for r in sim.regions}
    peak_pi = {r: sim.get_series(r, 'pi')[:8].max() * 100 for r in sim.regions}
    global_y = sim.global_gdp()
    global_loss = global_y[:8].min() * 100

    return {
        "name":        name,
        "welfare":     welfare,
        "peak_y":      peak_y,
        "peak_pi":     peak_pi,
        "global_loss": global_loss,
        "sim":         sim,
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("POLICY COUNTERFACTUAL ANALYSIS — DSGE v3")
    print("=" * 80)

    experiments = []

    # E1. Baseline
    experiments.append(run_experiment(
        "E1. Baseline (China stabilizes, all channels on)",
        china_stabilizes=True,
    ))

    # E2. No China
    experiments.append(run_experiment(
        "E2. No China stabilization",
        china_stabilizes=False,
    ))

    # E3. Aggressive Fed
    experiments.append(run_experiment(
        "E3. Aggressive Fed (phi_pi US = 2.5)",
        china_stabilizes=True,
        phi_pi_override={"us": 2.5},
    ))

    # E4. Dovish Fed
    experiments.append(run_experiment(
        "E4. Dovish Fed (phi_pi US = 1.0)",
        china_stabilizes=True,
        phi_pi_override={"us": 1.0},
    ))

    # E5. Coordinated easing
    experiments.append(run_experiment(
        "E5. Coordinated easing (phi_y x 2)",
        china_stabilizes=True,
        phi_y_override={"us": 1.0, "china": 1.6, "row": 0.8},
    ))

    # E6. No financial channels
    experiments.append(run_experiment(
        "E6. No financial channels",
        china_stabilizes=True,
        disable_financial_channels=True,
    ))

    # E7. No contagion
    experiments.append(run_experiment(
        "E7. No financial contagion (L = 0)",
        china_stabilizes=True,
        disable_contagion=True,
    ))

    # E8. Strong China stabilizer
    strong_rule = ChinaStabilizerRule(
        phi_oil=0.040, phi_yglob=0.050, spr_max=0.030,
        phi_dem=0.60, mfg_max=0.050,
    )
    experiments.append(run_experiment(
        "E8. Strong China stabilizer (2x rule coefficients)",
        china_stabilizes=True,
        stabilizer_rule=strong_rule,
    ))

    # E9. Empirical L matrix
    if L_EMPIRICAL is not None:
        # Cap empirical L at 0.20 for stability (some non-Iran rows are noisy)
        L_capped = np.clip(L_EMPIRICAL, 0, 0.20)
        experiments.append(run_experiment(
            "E9. VAR-estimated L matrix (capped at 0.20)",
            china_stabilizes=True,
            financial_linkage=L_capped,
        ))

    # ---- Print summary table ----
    print("\n" + "=" * 80)
    print("COUNTERFACTUAL RESULTS — Peak GDP losses (%)")
    print("=" * 80)
    print(f"\n{'Experiment':<48} {'Iran':>8} {'US':>8} {'China':>8} "
          f"{'ROW':>8} {'World':>8}")
    print("-" * 80)
    for exp in experiments:
        name = exp["name"][:47]
        py = exp["peak_y"]
        gw = exp["global_loss"]
        print(f"{name:<48} {py['iran']:>+7.1f} {py['us']:>+7.1f} "
              f"{py['china']:>+7.1f} {py['row']:>+7.1f} {gw:>+7.2f}")

    print("\n" + "=" * 80)
    print("CONSUMPTION-EQUIVALENT WELFARE LOSSES (% of permanent consumption)")
    print("=" * 80)
    print(f"\n{'Experiment':<48} {'Iran':>8} {'US':>8} {'China':>8} {'ROW':>8}")
    print("-" * 80)
    for exp in experiments:
        name = exp["name"][:47]
        w = exp["welfare"]
        print(f"{name:<48} {w['iran']:>+7.2f} {w['us']:>+7.2f} "
              f"{w['china']:>+7.2f} {w['row']:>+7.2f}")

    # ---- Welfare differentials vs baseline ----
    baseline = experiments[0]
    print("\n" + "=" * 80)
    print("WELFARE GAIN vs BASELINE (positive = better than baseline, %)")
    print("=" * 80)
    print(f"\n{'Experiment':<48} {'Iran':>8} {'US':>8} {'China':>8} {'ROW':>8}")
    print("-" * 80)
    for exp in experiments[1:]:
        name = exp["name"][:47]
        gain = {r: baseline["welfare"][r] - exp["welfare"][r] for r in exp["welfare"]}
        print(f"{name:<48} {gain['iran']:>+7.2f} {gain['us']:>+7.2f} "
              f"{gain['china']:>+7.2f} {gain['row']:>+7.2f}")

    # ---- Save results ----
    np.savez("D:/war_dsge_model/counterfactual_results.npz",
             names=[e["name"] for e in experiments],
             peak_y=[list(e["peak_y"].values()) for e in experiments],
             welfare=[list(e["welfare"].values()) for e in experiments],
             global_loss=[e["global_loss"] for e in experiments],
             regions=["iran", "us", "china", "row"])

    print("\nResults saved to D:/war_dsge_model/counterfactual_results.npz")
    print("=" * 80)
