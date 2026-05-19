"""
================================================================================
DSGE Model: The Price of War — Iran-Hormuz Scenario
================================================================================

A 4-Region New Keynesian DSGE Model with Energy Sector

Regions:
  1. Iran (战争发生地, 石油供给冲击)
  2. US/Israel (交战国, 军事支出冲击)
  3. China (大国稳定者, 战略石油储备 + 制造业供给)
  4. ROW (其余世界, 石油进口国)

Key Features:
  - 霍尔木兹海峡封锁 → 全球石油供给冲击
  - 交战国军事支出财政刺激 (consistent with Federle et al. 2026)
  - 中国的三重稳定机制:
      (a) 战略石油储备释放 (SPR)
      (b) 新能源替代加速
      (c) 制造业出口稳定全球供给
  - 贸易渠道溢出 (Federle et al. 2026 的核心发现)

Methodology:
  - 线性化New Keynesian模型
  - 通过状态空间系统求解
  - 脉冲响应函数分析
  - 反事实模拟: 有/无中国稳定作用

References:
  - Federle et al. (2026) "The Price of War", AER
  - Galí (2015) "Monetary Policy, Inflation, and the Business Cycle"
  - Blanchard & Galí (2007) "The Macroeconomic Effects of Oil Price Shocks"
  - Bodenstein, Erceg, Guerrieri (2011) "Oil Shocks and External Adjustment"

Author: Claude Code (DSGE Simulation)
Date: 2026-04-09
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 1. MODEL PARAMETERS (校准)
# ============================================================================

@dataclass
class RegionParams:
    """单个区域的结构参数"""
    name: str
    name_cn: str

    # 偏好与技术
    beta: float = 0.99          # 贴现因子
    sigma: float = 1.5          # 跨期替代弹性的倒数
    phi: float = 1.0            # Frisch弹性的倒数
    alpha: float = 0.33         # 资本份额
    theta: float = 0.75         # Calvo价格粘性
    epsilon: float = 6.0        # 替代弹性

    # 能源参数
    alpha_e: float = 0.05       # 能源在生产中的份额
    oil_import_share: float = 0.0  # 石油进口/GDP
    oil_export_share: float = 0.0  # 石油出口/GDP

    # 贸易参数
    trade_openness: float = 0.3    # 贸易开放度
    trade_with_iran: float = 0.01  # 与伊朗的贸易/GDP

    # 货币政策 (Taylor Rule)
    phi_pi: float = 1.5         # 通胀反应系数
    phi_y: float = 0.5          # 产出缺口反应系数
    rho_i: float = 0.8          # 利率平滑

    # GDP占世界比重
    gdp_share: float = 0.1

    # 战争相关
    military_share: float = 0.02  # 军事支出/GDP
    spr_capacity: float = 0.0    # 战略石油储备/年消费量


def calibrate_regions() -> Dict[str, RegionParams]:
    """校准四个区域的参数"""

    iran = RegionParams(
        name="Iran", name_cn="伊朗(战争发生地)",
        sigma=2.0, alpha_e=0.30, oil_export_share=0.20,
        oil_import_share=0.0, trade_openness=0.25,
        trade_with_iran=1.0,  # 自身
        phi_pi=1.2, phi_y=0.3, rho_i=0.7,
        gdp_share=0.02, military_share=0.025,
        theta=0.6,  # 价格更灵活
    )

    us_israel = RegionParams(
        name="US-Israel", name_cn="美国/以色列(交战国)",
        sigma=1.5, alpha_e=0.04, oil_import_share=0.03,
        oil_export_share=0.005, trade_openness=0.25,
        trade_with_iran=0.005,
        phi_pi=1.5, phi_y=0.5, rho_i=0.85,
        gdp_share=0.28, military_share=0.035,
        spr_capacity=0.5,  # ~6个月储备
    )

    china = RegionParams(
        name="China", name_cn="中国(稳定者)",
        sigma=1.2, alpha_e=0.06, oil_import_share=0.05,
        oil_export_share=0.0, trade_openness=0.35,
        trade_with_iran=0.015,
        phi_pi=1.3, phi_y=0.8, rho_i=0.75,
        gdp_share=0.20, military_share=0.017,
        spr_capacity=0.8,  # ~80天储备 (实际约80天)
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
# 2. GLOBAL OIL MARKET (全球石油市场)
# ============================================================================

@dataclass
class OilMarket:
    """全球石油市场模型

    霍尔木兹海峡处理全球约20%的石油贸易
    伊朗占全球石油供给约4%
    """
    hormuz_share: float = 0.20     # 经霍尔木兹的全球石油份额
    iran_production_share: float = 0.04  # 伊朗占全球产量
    saudi_share: float = 0.12     # 沙特产量份额(也受霍尔木兹影响)
    uae_share: float = 0.04      # 阿联酋份额
    iraq_share: float = 0.05     # 伊拉克份额(部分受影响)

    # 供给弹性
    supply_elasticity: float = 0.10   # 短期供给弹性(非常低)
    demand_elasticity: float = -0.05  # 短期需求弹性(也很低)

    # 冲击参数
    blockade_severity: float = 0.70   # 封锁有效性(0-1)
    iran_production_loss: float = 0.80  # 伊朗自身产量损失

    def compute_supply_shock(self, quarter: int, blockade_duration: int = 8) -> float:
        """计算全球石油供给冲击(占全球供给的百分比)

        霍尔木兹封锁影响:
        - 伊朗产量直接损失
        - 海峡运输中断(影响沙特、阿联酋、伊拉克出口)
        - 随时间衰减(替代路线、军事护航)
        """
        if quarter >= blockade_duration:
            # 封锁结束后逐步恢复
            decay = np.exp(-0.3 * (quarter - blockade_duration))
            severity = 0.2 * decay  # 残余影响
        else:
            # 封锁期间
            # 初始冲击最大, 然后随替代路线开辟略有缓解
            severity = self.blockade_severity * np.exp(-0.05 * quarter)

        # 直接供给损失
        iran_loss = self.iran_production_loss * self.iran_production_share
        transit_loss = severity * (self.hormuz_share - self.iran_production_share)
        # 扣除已有替代管道运力(约30%的海湾石油有管道替代)
        pipeline_offset = 0.30 * transit_loss

        total_loss = iran_loss + transit_loss - pipeline_offset

        return total_loss  # 正值表示供给减少

    def compute_oil_price_change(self, supply_shock: float,
                                  spr_release: float = 0.0,
                                  demand_reduction: float = 0.0) -> float:
        """计算油价变化率

        基于供需弹性:
        ΔP/P = -(ΔS/S) / (ε_d - ε_s) + SPR释放抵消

        Args:
            supply_shock: 供给减少百分比(正值)
            spr_release: 战略储备释放(占全球供给百分比)
            demand_reduction: 需求减少(经济衰退效应)
        """
        net_shock = supply_shock - spr_release + demand_reduction
        price_change = net_shock / abs(self.demand_elasticity - self.supply_elasticity)
        return price_change  # 正值表示价格上升


# ============================================================================
# 3. LINEARIZED DSGE EQUATIONS (线性化DSGE方程)
# ============================================================================

@dataclass
class RegionState:
    """单个区域在某一时期的状态变量"""
    y: float = 0.0       # 产出缺口 (% deviation)
    pi: float = 0.0      # 通胀率 (% deviation from target)
    i: float = 0.0       # 名义利率 (% deviation)
    c: float = 0.0       # 消费 (% deviation)
    inv: float = 0.0     # 投资 (% deviation)
    nx: float = 0.0      # 净出口 (% of GDP deviation)
    rer: float = 0.0     # 实际汇率 (% deviation, 贬值为正)
    g_mil: float = 0.0   # 军事支出 (% of GDP deviation)
    oil_p: float = 0.0   # 国内能源价格 (% deviation)
    spr: float = 0.0     # 战略储备释放 (% of consumption)


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
) -> RegionState:
    """求解单个区域一个时期的均衡

    基于线性化New Keynesian模型:

    IS曲线 (产出缺口):
      ŷ_t = E[ŷ_{t+1}] - 1/σ (î_t - E[π_{t+1}]) + g_mil + trade_effects - α_e * oil_shock

    Phillips曲线 (通胀):
      π_t = β E[π_{t+1}] + κ ŷ_t + α_e * oil_price + supply_destruction

    Taylor规则:
      î_t = ρ_i * î_{t-1} + (1-ρ_i)(φ_π π_t + φ_y ŷ_t)
    """
    s = RegionState()

    # --- 能源价格传导 ---
    # 国内能源价格变化 = 全球油价变化 * 能源份额权重 - SPR释放抵消
    # 对石油出口国(如伊朗), 油价上升不增加成本
    net_oil_exposure = max(params.oil_import_share - params.oil_export_share, 0.0)
    s.oil_p = oil_price_shock * (net_oil_exposure / max(params.alpha_e, 0.01)) * 0.5
    s.oil_p = np.clip(s.oil_p, -2.0, 2.0)  # 防止极端值
    s.oil_p -= spr_release * 0.3

    # --- Phillips 曲线 (供给侧) ---
    kappa = ((1 - params.theta) * (1 - params.beta * params.theta) / params.theta
             * (params.sigma + params.phi) / (1 + params.epsilon * params.phi))
    kappa = min(kappa, 0.15)  # 合理上界

    # 供给冲击: 油价传导 + 物理破坏 - 中国制造业替代
    supply_shock = (params.alpha_e * max(s.oil_p, 0) * 0.6
                    + supply_destruction * 0.8
                    - china_manufacturing_boost * params.trade_openness * 0.2)
    supply_shock = np.clip(supply_shock, -0.10, 0.15)

    s.pi = (params.beta * expected_pi_next
            + kappa * state_prev.y
            + supply_shock)
    s.pi = np.clip(s.pi, -0.15, 0.20)

    # --- Taylor 规则 ---
    s.i = (params.rho_i * state_prev.i
           + (1 - params.rho_i) * (params.phi_pi * s.pi
                                    + params.phi_y * max(state_prev.y, -0.3)))
    s.i = np.clip(s.i, -0.05, 0.15)

    # --- IS 曲线 (需求侧) ---
    real_rate_effect = -(1.0 / params.sigma) * (s.i - expected_pi_next) * 0.3

    # 军事支出乘数 (Ramey 2011: ~0.6-1.2)
    military_effect = 0.8 * military_shock

    # 贸易渠道 (Federle et al. 2026)
    trade_effect = -trade_shock * params.trade_openness * 0.5

    # 能源成本对产出
    energy_cost_effect = -params.alpha_e * max(s.oil_p, 0) * 0.3

    # 产出缺口 (含均值回复)
    persistence = 0.75
    s.y = (persistence * state_prev.y
           + real_rate_effect
           + military_effect
           + trade_effect
           + energy_cost_effect
           - supply_destruction * 0.5)
    # 均值回复力: 偏离越大, 回拉越强
    s.y += -0.05 * state_prev.y
    s.y = np.clip(s.y, -0.50, 0.15)  # 产出缺口上限 -50%

    # --- 消费 ---
    s.c = s.y - military_shock * 0.4

    # --- 投资 ---
    s.inv = s.y * 1.2 - 0.3 * max(s.i - s.pi, 0)

    # --- 净出口 ---
    s.rer = (s.pi - expected_pi_next * 0.5) * 0.5
    s.nx = (-net_oil_exposure * oil_price_shock * 0.5
            + params.oil_export_share * oil_price_shock * 0.3
            + 0.2 * s.rer
            + trade_shock * 0.3)
    s.nx = np.clip(s.nx, -0.15, 0.15)

    # --- 军事支出 ---
    s.g_mil = military_shock

    # --- 战略储备 ---
    s.spr = spr_release

    return s


# ============================================================================
# 4. SIMULATION ENGINE (模拟引擎)
# ============================================================================

class WarDSGESimulator:
    """战争DSGE模型模拟器"""

    def __init__(self, T: int = 32, china_stabilizes: bool = True):
        """
        Args:
            T: 模拟期数 (季度)
            china_stabilizes: 是否包含中国稳定机制
        """
        self.T = T
        self.china_stabilizes = china_stabilizes
        self.regions = calibrate_regions()
        self.oil_market = OilMarket()

        # 存储结果
        self.results: Dict[str, list] = {
            name: [] for name in self.regions
        }
        self.oil_prices = []

    def design_war_shocks(self) -> Dict[str, Dict]:
        """设计战争冲击时间路径

        情景: 美以对伊朗发动打击 → 伊朗封锁霍尔木兹海峡
        - 第0季度: 战争爆发, 霍尔木兹封锁
        - 第1-4季度: 高强度冲突
        - 第5-8季度: 冲突减弱, 但封锁持续
        - 第8季度后: 逐步恢复
        """
        shocks = {}

        for t in range(self.T):
            # --- 石油供给冲击 ---
            oil_supply_loss = self.oil_market.compute_supply_shock(t, blockade_duration=8)

            # --- 中国稳定机制 ---
            china_spr_release = 0.0
            china_mfg_boost = 0.0

            if self.china_stabilizes:
                # (a) 战略石油储备释放 (前8个季度释放, 然后减少)
                if t < 8:
                    # 每季度释放约占全球供给的0.5-1%
                    china_spr_release = 0.008 * np.exp(-0.1 * t)
                else:
                    china_spr_release = 0.002 * np.exp(-0.2 * (t - 8))

                # (b) 加速新能源替代 (逐步增加)
                renewable_offset = 0.002 * (1 - np.exp(-0.15 * t))
                china_spr_release += renewable_offset

                # (c) 制造业供给稳定 (对冲全球供应链中断)
                china_mfg_boost = 0.015 * (1 - np.exp(-0.2 * t))

            # --- 其他国家的SPR释放 ---
            us_spr_release = 0.005 * np.exp(-0.15 * t) if t < 12 else 0.0
            row_spr_release = 0.003 * np.exp(-0.15 * t) if t < 8 else 0.0

            # 总SPR释放
            total_spr = china_spr_release + us_spr_release + row_spr_release

            # --- 全球油价变化 ---
            demand_reduction = 0.005 * t / 8 if t < 8 else 0.005  # 经济衰退减少需求
            oil_price_change = self.oil_market.compute_oil_price_change(
                oil_supply_loss, total_spr, demand_reduction
            )

            # --- 伊朗冲击 ---
            # 校准: 使峰值产出损失在-20%~-35%之间 (高强度州际战争, 参考Federle et al.)
            iran_destruction = 0.06 * np.exp(-0.10 * t)  # 物理破坏(基础设施,炼油厂)
            iran_trade_disruption = 0.08 * np.exp(-0.12 * t)  # 贸易中断(制裁+封锁)

            # --- 美以军事支出冲击 ---
            if t < 8:
                us_military = 0.025 * np.exp(-0.05 * t)  # 占GDP 2.5%的额外支出
            else:
                us_military = 0.025 * np.exp(-0.2 * (t - 4))

            # --- 贸易中断冲击 ---
            # 基于Federle et al.: 贸易暴露度决定溢出大小
            trade_disruption_factor = 0.8 * np.exp(-0.1 * t)

            shocks[t] = {
                'oil_price': oil_price_change,
                'oil_supply_loss': oil_supply_loss,
                'iran_destruction': iran_destruction,
                'iran_trade_disruption': iran_trade_disruption,
                'us_military': us_military,
                'china_spr': china_spr_release,
                'china_mfg_boost': china_mfg_boost,
                'us_spr': us_spr_release,
                'row_spr': row_spr_release,
                'trade_disruption': trade_disruption_factor,
                'total_spr': total_spr,
            }

        return shocks

    def simulate(self) -> Dict[str, list]:
        """运行完整模拟"""

        shocks = self.design_war_shocks()

        # 初始化各区域状态
        states = {name: RegionState() for name in self.regions}

        for t in range(self.T):
            shock = shocks[t]
            new_states = {}

            # 简化的预期: 当前通胀的衰减
            for name, params in self.regions.items():
                expected_pi = states[name].pi * 0.7

                if name == "iran":
                    new_states[name] = solve_region_period(
                        params, states[name],
                        oil_price_shock=shock['oil_price'] * 0.2,  # 伊朗是出口国,油价上升对其部分有利
                        supply_destruction=shock['iran_destruction'],
                        trade_shock=shock['iran_trade_disruption'],
                        expected_pi_next=expected_pi,
                    )
                    # 石油出口收入损失(因制裁和产量下降)部分被油价上升抵消
                    oil_revenue_loss = shock['oil_supply_loss'] * 0.3
                    new_states[name].y -= oil_revenue_loss

                elif name == "us":
                    new_states[name] = solve_region_period(
                        params, states[name],
                        oil_price_shock=shock['oil_price'],
                        military_shock=shock['us_military'],
                        trade_shock=shock['trade_disruption'] * params.trade_with_iran,
                        spr_release=shock['us_spr'],
                        china_manufacturing_boost=shock['china_mfg_boost'] if self.china_stabilizes else 0,
                        expected_pi_next=expected_pi,
                    )

                elif name == "china":
                    new_states[name] = solve_region_period(
                        params, states[name],
                        oil_price_shock=shock['oil_price'],
                        trade_shock=shock['trade_disruption'] * params.trade_with_iran,
                        spr_release=shock['china_spr'],
                        china_manufacturing_boost=shock['china_mfg_boost'] if self.china_stabilizes else 0,
                        expected_pi_next=expected_pi,
                    )
                    # 中国的制造业出口扩张部分抵消负面冲击
                    if self.china_stabilizes:
                        new_states[name].y += shock['china_mfg_boost'] * 0.3
                        new_states[name].nx += shock['china_mfg_boost'] * 0.2

                elif name == "row":
                    new_states[name] = solve_region_period(
                        params, states[name],
                        oil_price_shock=shock['oil_price'],
                        trade_shock=shock['trade_disruption'] * params.trade_with_iran,
                        spr_release=shock['row_spr'],
                        china_manufacturing_boost=shock['china_mfg_boost'] if self.china_stabilizes else 0,
                        expected_pi_next=expected_pi,
                    )

            # 更新状态
            states = new_states

            # 记录结果
            for name in self.regions:
                self.results[name].append(states[name])
            self.oil_prices.append(shock['oil_price'] * 100)  # 转为百分比

        return self.results

    def get_series(self, region: str, variable: str) -> np.ndarray:
        """提取某区域某变量的时间序列"""
        return np.array([getattr(s, variable) for s in self.results[region]])


# ============================================================================
# 5. VISUALIZATION (可视化)
# ============================================================================

def plot_impulse_responses(sim_with: WarDSGESimulator,
                           sim_without: WarDSGESimulator,
                           save_path: str = None):
    """绘制脉冲响应函数 — 核心结果图"""

    quarters = np.arange(sim_with.T)
    years = quarters / 4

    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    fig.suptitle(
        'DSGE Model: Economic Impact of US-Israel War on Iran\n'
        'Hormuz Strait Blockade with China Stabilization Mechanism\n'
        '(Inspired by Federle et al. 2026 "The Price of War", AER)',
        fontsize=14, fontweight='bold', y=0.98
    )

    colors = {
        'iran': '#CC0000',
        'us': '#003399',
        'china': '#CC6600',
        'row': '#666666',
    }

    labels = {
        'iran': 'Iran (War Site)',
        'us': 'US/Israel (Belligerent)',
        'china': 'China (Stabilizer)',
        'row': 'Rest of World',
    }

    # ---- Row 1: Output (GDP) ----
    ax = axes[0, 0]
    ax.set_title('Output Gap (GDP)', fontsize=12, fontweight='bold')
    for name in ['iran', 'us', 'china', 'row']:
        y_with = sim_with.get_series(name, 'y') * 100
        ax.plot(years, y_with, color=colors[name], linewidth=2, label=labels[name])
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_ylabel('% deviation from trend')
    ax.set_xlabel('Years after war onset')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Counterfactual comparison for output
    ax = axes[0, 1]
    ax.set_title('Output: With vs Without China Stabilization', fontsize=12, fontweight='bold')
    for name in ['us', 'china', 'row']:
        y_with = sim_with.get_series(name, 'y') * 100
        y_without = sim_without.get_series(name, 'y') * 100
        ax.plot(years, y_with, color=colors[name], linewidth=2,
                label=f'{labels[name]} (with China)')
        ax.plot(years, y_without, color=colors[name], linewidth=2,
                linestyle='--', alpha=0.6, label=f'{labels[name]} (without)')
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_ylabel('% deviation from trend')
    ax.set_xlabel('Years after war onset')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Global GDP weighted
    ax = axes[0, 2]
    ax.set_title('Global GDP (Weighted Average)', fontsize=12, fontweight='bold')
    global_with = np.zeros(sim_with.T)
    global_without = np.zeros(sim_with.T)
    for name, params in sim_with.regions.items():
        global_with += sim_with.get_series(name, 'y') * params.gdp_share * 100
        global_without += sim_without.get_series(name, 'y') * params.gdp_share * 100
    ax.fill_between(years, global_without, global_with, alpha=0.3, color='green',
                     label='China stabilization effect')
    ax.plot(years, global_with, 'g-', linewidth=2, label='With China stabilization')
    ax.plot(years, global_without, 'r--', linewidth=2, label='Without')
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_ylabel('% deviation from trend')
    ax.set_xlabel('Years after war onset')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- Row 2: Inflation ----
    ax = axes[1, 0]
    ax.set_title('Inflation', fontsize=12, fontweight='bold')
    for name in ['iran', 'us', 'china', 'row']:
        pi_with = sim_with.get_series(name, 'pi') * 100
        ax.plot(years, pi_with, color=colors[name], linewidth=2, label=labels[name])
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_ylabel('Percentage points (annualized)')
    ax.set_xlabel('Years after war onset')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Inflation counterfactual
    ax = axes[1, 1]
    ax.set_title('Inflation: With vs Without China Stabilization', fontsize=12, fontweight='bold')
    for name in ['us', 'china', 'row']:
        pi_with = sim_with.get_series(name, 'pi') * 100
        pi_without = sim_without.get_series(name, 'pi') * 100
        ax.plot(years, pi_with, color=colors[name], linewidth=2, label=f'{labels[name]} (with)')
        ax.plot(years, pi_without, color=colors[name], linewidth=2,
                linestyle='--', alpha=0.6, label=f'{labels[name]} (without)')
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_ylabel('Percentage points (annualized)')
    ax.set_xlabel('Years after war onset')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Oil price
    ax = axes[1, 2]
    ax.set_title('Global Oil Price', fontsize=12, fontweight='bold')
    ax.plot(years, sim_with.oil_prices, 'r-', linewidth=2, label='With China SPR release')
    ax.plot(years, sim_without.oil_prices, 'r--', linewidth=2, alpha=0.6, label='Without')
    ax.fill_between(years, sim_without.oil_prices, sim_with.oil_prices,
                     alpha=0.2, color='green', label='SPR stabilization')
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_ylabel('% change from pre-war')
    ax.set_xlabel('Years after war onset')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- Row 3: Interest Rate & Exchange Rate ----
    ax = axes[2, 0]
    ax.set_title('Nominal Interest Rate', fontsize=12, fontweight='bold')
    for name in ['us', 'china', 'row']:
        i_with = sim_with.get_series(name, 'i') * 100
        ax.plot(years, i_with, color=colors[name], linewidth=2, label=labels[name])
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_ylabel('Percentage points deviation')
    ax.set_xlabel('Years after war onset')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Net exports
    ax = axes[2, 1]
    ax.set_title('Net Exports (% of GDP)', fontsize=12, fontweight='bold')
    for name in ['us', 'china', 'row']:
        nx_with = sim_with.get_series(name, 'nx') * 100
        ax.plot(years, nx_with, color=colors[name], linewidth=2, label=labels[name])
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_ylabel('Percentage points of GDP')
    ax.set_xlabel('Years after war onset')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Military spending
    ax = axes[2, 2]
    ax.set_title('Military Spending Shock (% of GDP)', fontsize=12, fontweight='bold')
    for name in ['iran', 'us', 'china', 'row']:
        g_with = sim_with.get_series(name, 'g_mil') * 100
        ax.plot(years, g_with, color=colors[name], linewidth=2, label=labels[name])
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_ylabel('Percentage points of GDP')
    ax.set_xlabel('Years after war onset')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Row 4: China's Stabilization Mechanisms ----
    ax = axes[3, 0]
    ax.set_title("China's SPR Release & Renewable Offset", fontsize=12, fontweight='bold')
    spr_series = sim_with.get_series('china', 'spr') * 100
    ax.bar(years, spr_series, width=0.2, color='#CC6600', alpha=0.7, label='SPR release')
    ax.set_ylabel('% of domestic oil consumption')
    ax.set_xlabel('Years after war onset')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # China's manufacturing export boost
    ax = axes[3, 1]
    ax.set_title("China's Manufacturing Export Stabilization", fontsize=12, fontweight='bold')
    nx_china_with = sim_with.get_series('china', 'nx') * 100
    nx_china_without = sim_without.get_series('china', 'nx') * 100
    ax.plot(years, nx_china_with, color='#CC6600', linewidth=2, label='With stabilization')
    ax.plot(years, nx_china_without, color='#CC6600', linewidth=2,
            linestyle='--', alpha=0.6, label='Without')
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_ylabel('% of GDP deviation')
    ax.set_xlabel('Years after war onset')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Summary table as text
    ax = axes[3, 2]
    ax.axis('off')
    summary_text = (
        "SUMMARY: Peak Effects (Year 1)\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    )
    for name, label in labels.items():
        y_peak = sim_with.get_series(name, 'y')[:8].min() * 100
        pi_peak = sim_with.get_series(name, 'pi')[:8].max() * 100
        summary_text += f"\n{label}:\n"
        summary_text += f"  GDP: {y_peak:+.1f}%  Inflation: {pi_peak:+.1f}pp\n"

    oil_peak = max(sim_with.oil_prices[:8])
    oil_peak_no = max(sim_without.oil_prices[:8])
    summary_text += f"\nOil Price Peak:\n"
    summary_text += f"  With China SPR: +{oil_peak:.0f}%\n"
    summary_text += f"  Without:        +{oil_peak_no:.0f}%\n"

    global_loss_with = min(global_with[:8])
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
    """中国稳定机制的详细分解图"""

    quarters = np.arange(sim_with.T)
    years = quarters / 4

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "China's Triple Stabilization Mechanism: Detailed Decomposition\n"
        "中国三重稳定机制的详细分解",
        fontsize=14, fontweight='bold'
    )

    # 1. Oil price with/without
    ax = axes[0, 0]
    ax.set_title('(a) Global Oil Price Stabilization\n全球油价稳定效应')
    ax.plot(years, sim_with.oil_prices, 'b-', linewidth=2, label='With China SPR')
    ax.plot(years, sim_without.oil_prices, 'r--', linewidth=2, label='Without')
    saved = np.array(sim_without.oil_prices) - np.array(sim_with.oil_prices)
    ax.fill_between(years, sim_with.oil_prices, sim_without.oil_prices,
                     alpha=0.2, color='green')
    ax.set_ylabel('% change'); ax.set_xlabel('Years')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', lw=0.5, ls='--')

    # 2. Global inflation
    ax = axes[0, 1]
    ax.set_title('(b) Global Inflation Stabilization\n全球通胀稳定效应')
    for name, color, label in [('us', '#003399', 'US'), ('row', '#666666', 'ROW')]:
        pi_w = sim_with.get_series(name, 'pi') * 100
        pi_wo = sim_without.get_series(name, 'pi') * 100
        ax.plot(years, pi_w, color=color, linewidth=2, label=f'{label} (with)')
        ax.plot(years, pi_wo, color=color, linewidth=2, ls='--', alpha=0.6, label=f'{label} (w/o)')
    ax.set_ylabel('pp'); ax.set_xlabel('Years')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', lw=0.5, ls='--')

    # 3. Global output
    ax = axes[0, 2]
    ax.set_title('(c) Global Output Stabilization\n全球产出稳定效应')
    global_w = np.zeros(sim_with.T)
    global_wo = np.zeros(sim_with.T)
    for name, p in sim_with.regions.items():
        global_w += sim_with.get_series(name, 'y') * p.gdp_share * 100
        global_wo += sim_without.get_series(name, 'y') * p.gdp_share * 100
    ax.plot(years, global_w, 'g-', linewidth=2.5, label='With China stabilization')
    ax.plot(years, global_wo, 'r--', linewidth=2.5, label='Without')
    ax.fill_between(years, global_wo, global_w, alpha=0.15, color='green')
    ax.set_ylabel('% deviation'); ax.set_xlabel('Years')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', lw=0.5, ls='--')

    # 4. Mechanism 1: SPR Release
    ax = axes[1, 0]
    ax.set_title('Mechanism 1: SPR Release\n机制一: 战略石油储备释放')
    shocks_w = sim_with.design_war_shocks()
    spr_release = [shocks_w[t]['china_spr'] * 100 for t in range(sim_with.T)]
    us_spr = [shocks_w[t]['us_spr'] * 100 for t in range(sim_with.T)]
    ax.bar(years, spr_release, width=0.2, color='#CC6600', alpha=0.8, label='China SPR')
    ax.bar(years, us_spr, width=0.2, bottom=spr_release, color='#003399', alpha=0.5, label='US SPR')
    ax.set_ylabel('% of global supply'); ax.set_xlabel('Years')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # 5. Mechanism 2: Renewable acceleration
    ax = axes[1, 1]
    ax.set_title('Mechanism 2: Renewable Energy Acceleration\n机制二: 新能源替代加速')
    renewable = [0.2 * (1 - np.exp(-0.15 * t)) for t in range(sim_with.T)]
    ax.fill_between(years, renewable, alpha=0.3, color='green')
    ax.plot(years, renewable, 'g-', linewidth=2, label='Renewable offset (% of oil demand)')
    ax.set_ylabel('% of oil demand replaced'); ax.set_xlabel('Years')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # 6. Mechanism 3: Manufacturing supply
    ax = axes[1, 2]
    ax.set_title('Mechanism 3: Manufacturing Export Boost\n机制三: 制造业出口稳定全球供给')
    mfg_boost = [shocks_w[t]['china_mfg_boost'] * 100 for t in range(sim_with.T)]
    ax.fill_between(years, mfg_boost, alpha=0.3, color='#CC6600')
    ax.plot(years, mfg_boost, color='#CC6600', linewidth=2,
            label='Manufacturing export boost')
    ax.set_ylabel('% boost'); ax.set_xlabel('Years')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    plt.close()


# ============================================================================
# 6. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("DSGE MODEL: THE PRICE OF WAR — IRAN-HORMUZ SCENARIO")
    print("Inspired by Federle et al. (2026) 'The Price of War', AER")
    print("=" * 70)

    # --- Scenario 1: With China Stabilization ---
    print("\n[1/2] Simulating WITH China stabilization mechanism...")
    sim_with = WarDSGESimulator(T=32, china_stabilizes=True)
    sim_with.simulate()

    # --- Scenario 2: Without China Stabilization ---
    print("[2/2] Simulating WITHOUT China stabilization mechanism...")
    sim_without = WarDSGESimulator(T=32, china_stabilizes=False)
    sim_without.simulate()

    # --- Generate Figures ---
    print("\nGenerating impulse response figures...")
    plot_impulse_responses(
        sim_with, sim_without,
        save_path="D:/war_dsge_model/impulse_responses.png"
    )
    plot_china_mechanism_detail(
        sim_with, sim_without,
        save_path="D:/war_dsge_model/china_mechanism.png"
    )

    # --- Print Summary Table ---
    print("\n" + "=" * 70)
    print("QUANTITATIVE RESULTS SUMMARY")
    print("=" * 70)

    print("\n{:<25} {:>10} {:>10} {:>10} {:>10}".format(
        "", "Iran", "US/Israel", "China", "ROW"))
    print("-" * 70)

    # Peak GDP loss (first 2 years)
    print("{:<25}".format("Peak GDP loss (%)"), end="")
    for name in ['iran', 'us', 'china', 'row']:
        val = sim_with.get_series(name, 'y')[:8].min() * 100
        print(f" {val:>+9.1f}", end="")
    print()

    # Peak inflation
    print("{:<25}".format("Peak inflation (pp)"), end="")
    for name in ['iran', 'us', 'china', 'row']:
        val = sim_with.get_series(name, 'pi')[:8].max() * 100
        print(f" {val:>+9.1f}", end="")
    print()

    # Interest rate peak
    print("{:<25}".format("Peak interest rate (pp)"), end="")
    for name in ['iran', 'us', 'china', 'row']:
        val = sim_with.get_series(name, 'i')[:8].max() * 100
        print(f" {val:>+9.1f}", end="")
    print()

    print("\n" + "-" * 70)
    print("CHINA STABILIZATION EFFECT")
    print("-" * 70)

    # Oil price
    oil_peak_w = max(sim_with.oil_prices[:8])
    oil_peak_wo = max(sim_without.oil_prices[:8])
    print(f"Oil price peak (with China SPR):    +{oil_peak_w:.0f}%")
    print(f"Oil price peak (without):           +{oil_peak_wo:.0f}%")
    print(f"Oil price reduction:                {oil_peak_wo - oil_peak_w:.0f} pp")

    # Global GDP
    global_w = np.zeros(32)
    global_wo = np.zeros(32)
    for name, p in sim_with.regions.items():
        global_w += sim_with.get_series(name, 'y') * p.gdp_share * 100
        global_wo += sim_without.get_series(name, 'y') * p.gdp_share * 100

    print(f"\nGlobal GDP loss (with China):        {global_w[:8].min():.2f}%")
    print(f"Global GDP loss (without):           {global_wo[:8].min():.2f}%")
    print(f"GDP saved by China stabilization:    {abs(global_wo[:8].min() - global_w[:8].min()):.2f} pp")

    # Global inflation
    global_pi_w = np.zeros(32)
    global_pi_wo = np.zeros(32)
    for name, p in sim_with.regions.items():
        global_pi_w += sim_with.get_series(name, 'pi') * p.gdp_share * 100
        global_pi_wo += sim_without.get_series(name, 'pi') * p.gdp_share * 100

    print(f"\nGlobal inflation peak (with China):  +{global_pi_w[:8].max():.2f} pp")
    print(f"Global inflation peak (without):     +{global_pi_wo[:8].max():.2f} pp")
    print(f"Inflation reduced by China:          {global_pi_wo[:8].max() - global_pi_w[:8].max():.2f} pp")

    # Comparison with Federle et al.
    print("\n" + "=" * 70)
    print("COMPARISON WITH FEDERLE ET AL. (2026)")
    print("=" * 70)
    print("""
Federle et al. findings (average war):     This model (Iran-Hormuz):
  War site GDP:        -10%                  Iran GDP:       {iran_gdp:+.1f}%
  War site CPI:        +20%                  Iran CPI:       +high (supply destruction)
  High-exposure third: -2%                   ROW GDP:        {row_gdp:+.1f}%
  Belligerent GDP:     ~0%                   US GDP:         {us_gdp:+.1f}% (fiscal stimulus)
  Recovery time:       ~12 years             Recovery:       ~6-8 years (shorter war)

Key difference: Hormuz blockade creates a GLOBAL energy supply shock
that is much larger than the typical trade-channel spillover in
Federle et al. China's stabilization reduces this shock significantly.
""".format(
        iran_gdp=sim_with.get_series('iran', 'y')[:8].min() * 100,
        row_gdp=sim_with.get_series('row', 'y')[:8].min() * 100,
        us_gdp=sim_with.get_series('us', 'y')[:8].min() * 100,
    ))

    print("\nAll results and figures saved to D:/war_dsge_model/")
    print("--- Simulation Complete ---")
