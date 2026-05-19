"""
================================================================================
Sensitivity Analysis: War Severity Scenarios + China Stabilization Decomposition
================================================================================

Three war scenarios:
  - Mild:     Limited strikes, partial Hormuz disruption (30% blockade)
  - Baseline: Full strikes, substantial Hormuz blockade (70% blockade)
  - Severe:   Prolonged war, near-total Hormuz shutdown (90% blockade)

China stabilization decomposition:
  - No stabilization (counterfactual)
  - SPR only
  - SPR + Renewables
  - SPR + Renewables + Manufacturing (full)
"""

import numpy as np
import matplotlib.pyplot as plt
from model import (WarDSGESimulator, OilMarket, calibrate_regions,
                   RegionParams, RegionState, solve_region_period,
                   plot_impulse_responses)
from copy import deepcopy


# ============================================================================
# 1. WAR SEVERITY SCENARIOS
# ============================================================================

class ScenarioSimulator(WarDSGESimulator):
    """支持不同战争烈度情景的模拟器"""

    def __init__(self, T=32, china_stabilizes=True,
                 blockade_severity=0.70,
                 iran_destruction_scale=1.0,
                 us_military_scale=1.0,
                 blockade_duration=8):
        super().__init__(T, china_stabilizes)
        self.oil_market.blockade_severity = blockade_severity
        self.iran_destruction_scale = iran_destruction_scale
        self.us_military_scale = us_military_scale
        self.blockade_duration = blockade_duration

    def design_war_shocks(self):
        """Override with scaled shocks"""
        shocks = super().design_war_shocks()
        for t in shocks:
            # Recalculate oil supply loss with new blockade severity
            shocks[t]['oil_supply_loss'] = self.oil_market.compute_supply_shock(
                t, self.blockade_duration)
            # Scale destruction and military
            shocks[t]['iran_destruction'] *= self.iran_destruction_scale
            shocks[t]['iran_trade_disruption'] *= self.iran_destruction_scale
            shocks[t]['us_military'] *= self.us_military_scale
            # Recalculate oil price
            total_spr = shocks[t].get('total_spr', 0)
            demand_red = 0.005 * min(t / 8, 1.0)
            shocks[t]['oil_price'] = self.oil_market.compute_oil_price_change(
                shocks[t]['oil_supply_loss'], total_spr, demand_red)
        return shocks


def run_severity_scenarios():
    """运行三种战争烈度情景"""

    scenarios = {
        'mild': {
            'label': 'Mild (Limited Strikes)',
            'label_cn': '温和情景 (有限打击)',
            'blockade_severity': 0.30,
            'iran_destruction_scale': 0.5,
            'us_military_scale': 0.6,
            'blockade_duration': 4,
            'color': '#4CAF50',
        },
        'baseline': {
            'label': 'Baseline (Full Strikes)',
            'label_cn': '基线情景 (全面打击)',
            'blockade_severity': 0.70,
            'iran_destruction_scale': 1.0,
            'us_military_scale': 1.0,
            'blockade_duration': 8,
            'color': '#FF9800',
        },
        'severe': {
            'label': 'Severe (Prolonged War)',
            'label_cn': '严重情景 (持久战)',
            'blockade_severity': 0.90,
            'iran_destruction_scale': 1.8,
            'us_military_scale': 1.5,
            'blockade_duration': 12,
            'color': '#F44336',
        },
    }

    results = {}
    for name, config in scenarios.items():
        sim = ScenarioSimulator(
            T=32, china_stabilizes=True,
            blockade_severity=config['blockade_severity'],
            iran_destruction_scale=config['iran_destruction_scale'],
            us_military_scale=config['us_military_scale'],
            blockade_duration=config['blockade_duration'],
        )
        sim.simulate()
        results[name] = sim

    return results, scenarios


def plot_severity_scenarios(results, scenarios, save_path=None):
    """绘制三种烈度情景对比"""

    T = 32
    years = np.arange(T) / 4

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(
        'War Severity Scenario Analysis\n'
        '战争烈度情景分析: 温和 / 基线 / 严重',
        fontsize=14, fontweight='bold'
    )

    regions_plot = [
        ('iran', 'Iran (War Site)'),
        ('china', 'China'),
        ('row', 'Rest of World'),
    ]

    variables = [
        ('y', 'Output Gap (%)', 100),
        ('pi', 'Inflation (pp)', 100),
        ('i', 'Interest Rate (pp)', 100),
    ]

    for col, (region, region_label) in enumerate(regions_plot):
        for row, (var, var_label, scale) in enumerate(variables):
            ax = axes[row, col]
            for sname, config in scenarios.items():
                sim = results[sname]
                series = sim.get_series(region, var) * scale
                ax.plot(years, series, color=config['color'], linewidth=2,
                        label=config['label'])

            ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
            ax.set_ylabel(var_label, fontsize=10)
            ax.set_xlabel('Years', fontsize=9)
            if row == 0:
                ax.set_title(region_label, fontsize=12, fontweight='bold')
            if col == 0 and row == 0:
                ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


# ============================================================================
# 2. CHINA STABILIZATION DECOMPOSITION
# ============================================================================

class DecompositionSimulator(WarDSGESimulator):
    """支持中国稳定机制分解的模拟器"""

    def __init__(self, T=32, enable_spr=True, enable_renewable=True,
                 enable_manufacturing=True):
        super().__init__(T, china_stabilizes=True)
        self.enable_spr = enable_spr
        self.enable_renewable = enable_renewable
        self.enable_manufacturing = enable_manufacturing

    def design_war_shocks(self):
        shocks = super().design_war_shocks()
        for t in shocks:
            if not self.enable_spr:
                shocks[t]['china_spr'] = 0.0
                # Recalculate total SPR
                shocks[t]['total_spr'] = shocks[t]['us_spr'] + shocks[t]['row_spr']
            if not self.enable_renewable:
                # Remove renewable component from china_spr
                renewable = 0.002 * (1 - np.exp(-0.15 * t))
                if self.enable_spr:
                    shocks[t]['china_spr'] = max(0, shocks[t]['china_spr'] - renewable)
                    shocks[t]['total_spr'] = (shocks[t]['china_spr']
                                               + shocks[t]['us_spr']
                                               + shocks[t]['row_spr'])
            if not self.enable_manufacturing:
                shocks[t]['china_mfg_boost'] = 0.0

            # Recalculate oil price
            demand_red = 0.005 * min(t / 8, 1.0)
            shocks[t]['oil_price'] = self.oil_market.compute_oil_price_change(
                shocks[t]['oil_supply_loss'], shocks[t]['total_spr'], demand_red)
        return shocks


def run_decomposition():
    """运行中国稳定机制分解"""

    configs = {
        'none': {
            'label': 'No China Stabilization',
            'label_cn': '无中国稳定',
            'spr': False, 'renewable': False, 'mfg': False,
            'color': '#F44336', 'ls': '--',
        },
        'spr_only': {
            'label': '+ SPR Release Only',
            'label_cn': '+ 仅SPR释放',
            'spr': True, 'renewable': False, 'mfg': False,
            'color': '#FF9800', 'ls': '-.',
        },
        'spr_renewable': {
            'label': '+ SPR + Renewables',
            'label_cn': '+ SPR + 新能源',
            'spr': True, 'renewable': True, 'mfg': False,
            'color': '#2196F3', 'ls': ':',
        },
        'full': {
            'label': '+ SPR + Renewables + Manufacturing',
            'label_cn': '+ SPR + 新能源 + 制造业 (完整)',
            'spr': True, 'renewable': True, 'mfg': True,
            'color': '#4CAF50', 'ls': '-',
        },
    }

    results = {}
    for name, config in configs.items():
        if name == 'none':
            sim = WarDSGESimulator(T=32, china_stabilizes=False)
        else:
            sim = DecompositionSimulator(
                T=32,
                enable_spr=config['spr'],
                enable_renewable=config['renewable'],
                enable_manufacturing=config['mfg'],
            )
        sim.simulate()
        results[name] = sim

    return results, configs


def plot_decomposition(results, configs, save_path=None):
    """绘制中国稳定机制的逐步分解"""

    T = 32
    years = np.arange(T) / 4

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "Decomposition of China's Stabilization Mechanism\n"
        "中国稳定机制的逐步分解: SPR → 新能源 → 制造业出口",
        fontsize=14, fontweight='bold'
    )

    plot_specs = [
        (0, 0, 'Oil Price', '全球油价变化 (%)', 'oil_prices', None, 1.0),
        (0, 1, 'Global GDP', '全球GDP偏离趋势 (%)', 'global_gdp', None, 1.0),
        (0, 2, 'Global Inflation', '全球通胀 (pp)', 'global_pi', None, 1.0),
        (1, 0, 'China GDP', '中国GDP偏离趋势 (%)', 'y', 'china', 100),
        (1, 1, 'US GDP', '美国GDP偏离趋势 (%)', 'y', 'us', 100),
        (1, 2, 'ROW GDP', '其余世界GDP偏离趋势 (%)', 'y', 'row', 100),
    ]

    for row, col, title_en, title_cn, var, region, scale in plot_specs:
        ax = axes[row, col]
        ax.set_title(f'{title_en}\n{title_cn}', fontsize=11)

        for cname, config in configs.items():
            sim = results[cname]
            if var == 'oil_prices':
                series = np.array(sim.oil_prices)
            elif var == 'global_gdp':
                series = np.zeros(T)
                for rname, p in sim.regions.items():
                    series += sim.get_series(rname, 'y') * p.gdp_share * 100
            elif var == 'global_pi':
                series = np.zeros(T)
                for rname, p in sim.regions.items():
                    series += sim.get_series(rname, 'pi') * p.gdp_share * 100
            else:
                series = sim.get_series(region, var) * scale

            ax.plot(years, series, color=config['color'], linewidth=2,
                    linestyle=config['ls'], label=config['label'])

        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.set_xlabel('Years', fontsize=9)
        ax.grid(True, alpha=0.3)
        if row == 0 and col == 0:
            ax.legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


# ============================================================================
# 3. WELFARE ANALYSIS
# ============================================================================

def compute_welfare_costs(sim: WarDSGESimulator, discount=0.99) -> Dict:
    """计算各区域的福利损失 (消费等价变动)

    基于Lucas (1987)的消费等价度量:
    λ = Σ β^t * [c_t(war) - c_t(no war)] / Σ β^t
    """
    from typing import Dict
    T = sim.T
    welfare = {}

    for name, params in sim.regions.items():
        c_series = sim.get_series(name, 'c')
        discounts = np.array([discount**t for t in range(T)])

        # 消费等价损失 (% of steady-state consumption)
        welfare_loss = np.sum(discounts * c_series) / np.sum(discounts) * 100

        # GDP损失 (累计)
        y_series = sim.get_series(name, 'y')
        cumulative_gdp_loss = np.sum(y_series) * 100 / 4  # 年化

        welfare[name] = {
            'consumption_equivalent': welfare_loss,
            'cumulative_gdp_loss': cumulative_gdp_loss,
            'peak_gdp_loss': y_series.min() * 100,
            'peak_inflation': sim.get_series(name, 'pi').max() * 100,
        }

    return welfare


def print_welfare_table(sim_with, sim_without):
    """打印福利分析表"""
    from typing import Dict

    w_with = compute_welfare_costs(sim_with)
    w_without = compute_welfare_costs(sim_without)

    print("\n" + "=" * 80)
    print("WELFARE ANALYSIS: Consumption-Equivalent Loss")
    print("福利分析: 消费等价损失")
    print("=" * 80)

    header = f"{'Region':<20} {'Peak GDP':>10} {'Cum GDP':>10} {'Cons Eq':>10} │ {'w/o China':>10} {'Saved':>10}"
    print(header)
    print("-" * 80)

    for name in ['iran', 'us', 'china', 'row']:
        w = w_with[name]
        wo = w_without[name]
        saved = wo['consumption_equivalent'] - w['consumption_equivalent']
        label = sim_with.regions[name].name_cn
        print(f"{label:<20} {w['peak_gdp_loss']:>+9.1f}% {w['cumulative_gdp_loss']:>+9.1f}% "
              f"{w['consumption_equivalent']:>+9.2f}% │ "
              f"{wo['consumption_equivalent']:>+9.2f}% {saved:>+9.2f}pp")

    # Global
    global_w = sum(w_with[n]['consumption_equivalent'] * sim_with.regions[n].gdp_share
                   for n in sim_with.regions)
    global_wo = sum(w_without[n]['consumption_equivalent'] * sim_without.regions[n].gdp_share
                    for n in sim_without.regions)
    global_saved = global_wo - global_w
    print("-" * 80)
    print(f"{'Global (weighted)':<20} {'':>10} {'':>10} "
          f"{global_w:>+9.2f}% │ "
          f"{global_wo:>+9.2f}% {global_saved:>+9.2f}pp")

    # Convert to dollar terms (rough)
    world_gdp_2025 = 110  # trillion USD
    print(f"\n{'Approximate dollar values (2025 world GDP = $110T):'}")
    for name in ['iran', 'us', 'china', 'row']:
        loss = abs(w_with[name]['cumulative_gdp_loss']) / 100 * world_gdp_2025 * sim_with.regions[name].gdp_share
        print(f"  {sim_with.regions[name].name_cn:<20}: ~${loss:.1f}T cumulative GDP loss")

    global_loss_w = abs(sum(w_with[n]['cumulative_gdp_loss'] * sim_with.regions[n].gdp_share
                           for n in sim_with.regions)) / 100 * world_gdp_2025
    global_loss_wo = abs(sum(w_without[n]['cumulative_gdp_loss'] * sim_without.regions[n].gdp_share
                            for n in sim_without.regions)) / 100 * world_gdp_2025
    print(f"\n  Global GDP loss (with China):     ~${global_loss_w:.1f}T")
    print(f"  Global GDP loss (without China):  ~${global_loss_wo:.1f}T")
    print(f"  China stabilization saves:        ~${global_loss_wo - global_loss_w:.1f}T")


# ============================================================================
# 4. MAIN
# ============================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("SENSITIVITY & DECOMPOSITION ANALYSIS")
    print("=" * 70)

    # --- Severity Scenarios ---
    print("\n[1/3] Running war severity scenarios...")
    sev_results, sev_configs = run_severity_scenarios()
    plot_severity_scenarios(sev_results, sev_configs,
                            save_path="D:/war_dsge_model/severity_scenarios.png")

    # Print severity summary
    print("\nSeverity Scenario Summary:")
    print(f"{'Scenario':<15} {'Iran GDP':>10} {'Oil Price':>10} {'Global GDP':>12} {'ROW GDP':>10}")
    print("-" * 60)
    for sname in ['mild', 'baseline', 'severe']:
        sim = sev_results[sname]
        iran_gdp = sim.get_series('iran', 'y')[:8].min() * 100
        oil_peak = max(sim.oil_prices[:8])
        row_gdp = sim.get_series('row', 'y')[:8].min() * 100
        global_gdp = sum(sim.get_series(n, 'y')[:8].min() * sim.regions[n].gdp_share * 100
                         for n in sim.regions)
        print(f"{sev_configs[sname]['label']:<15} {iran_gdp:>+9.1f}% {oil_peak:>+9.0f}% "
              f"{global_gdp:>+11.2f}% {row_gdp:>+9.1f}%")

    # --- Decomposition ---
    print("\n[2/3] Running China stabilization decomposition...")
    dec_results, dec_configs = run_decomposition()
    plot_decomposition(dec_results, dec_configs,
                        save_path="D:/war_dsge_model/china_decomposition.png")

    # Print decomposition summary
    print("\nChina Stabilization Decomposition:")
    print(f"{'Mechanism':<40} {'Global GDP':>12} {'Global CPI':>12} {'Oil Price':>10}")
    print("-" * 76)
    for dname in ['none', 'spr_only', 'spr_renewable', 'full']:
        sim = dec_results[dname]
        global_gdp = sum(sim.get_series(n, 'y')[:8].min() * sim.regions[n].gdp_share * 100
                         for n in sim.regions)
        global_pi = sum(sim.get_series(n, 'pi')[:8].max() * sim.regions[n].gdp_share * 100
                        for n in sim.regions)
        oil_peak = max(sim.oil_prices[:8])
        print(f"{dec_configs[dname]['label']:<40} {global_gdp:>+11.2f}% "
              f"{global_pi:>+11.2f}pp {oil_peak:>+9.0f}%")

    # --- Welfare Analysis ---
    print("\n[3/3] Computing welfare costs...")
    sim_with = WarDSGESimulator(T=32, china_stabilizes=True)
    sim_with.simulate()
    sim_without = WarDSGESimulator(T=32, china_stabilizes=False)
    sim_without.simulate()
    print_welfare_table(sim_with, sim_without)

    print("\n" + "=" * 70)
    print("All sensitivity analysis results saved to D:/war_dsge_model/")
    print("=" * 70)
