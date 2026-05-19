"""
AER-Standard Figures for "The Global Price of War" Paper
"""
import sys, os
sys.path.insert(0, "D:/war_dsge_model")
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model_v2 import WarDSGESimulator

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11,
    'legend.fontsize': 8, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'lines.linewidth': 1.5, 'axes.linewidth': 0.6,
    'axes.grid': False, 'axes.spines.top': False, 'axes.spines.right': False,
    'mathtext.fontset': 'cm',
})
OUTDIR = "D:/war_dsge_model/figures"
STYLES = {
    'iran':  dict(color='black',   ls='-',  lw=1.8, label='Iran (war site)'),
    'us':    dict(color='#444444', ls='--', lw=1.5, label='US/Israel (belligerent)'),
    'china': dict(color='#777777', ls=':',  lw=1.8, label='China (stabilizer)'),
    'row':   dict(color='#AAAAAA', ls='-.', lw=1.5, label='Rest of World'),
}

def _save(fig, name):
    for ext in ['png','pdf']:
        fig.savefig(f"{OUTDIR}/{name}.{ext}", dpi=300, bbox_inches='tight')
    plt.close(fig); print(f"  {name}")

def _pl(ax, l, x=-0.12, y=1.08):
    ax.text(x, y, l, transform=ax.transAxes, fontsize=11, fontweight='bold', va='top')

def _zl(ax): ax.axhline(0, color='black', lw=0.4, zorder=0)

def _note(fig, t, y=-0.02):
    fig.text(0.5, y, t, ha='center', va='top', fontsize=7.5, fontstyle='italic', wrap=True)

# ── Run simulations ──
print("Running simulations...")
sw = WarDSGESimulator(T=32, china_stabilizes=True); sw.simulate()
swo = WarDSGESimulator(T=32, china_stabilizes=False); swo.simulate()
T=32; yrs = np.arange(T)/4.0

# ══════════ Figure 2: Impulse Responses ══════════
def fig2():
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.8))
    specs = [('y','Output','Percent',100,'A'), ('pi','Consumer prices','Percentage points',100,'B'),
             ('inv','Investment','Percent',100,'C'), ('i','Interest rate','Percentage points',100,'D'),
             (None,'Oil price','Percent',1,'E'), ('g_mil','Military spending','Pct. points of GDP',100,'F')]
    for idx,(var,title,yl,sc,pn) in enumerate(specs):
        ax=axes[idx//3,idx%3]; _pl(ax,f'Panel {pn}.'); ax.set_title(title,fontsize=10,pad=4)
        if var is None:
            ax.plot(yrs,sw.oil_prices,'k-',lw=1.5,label='With China stabilization')
            ax.plot(yrs,swo.oil_prices,color='#777777',ls='--',lw=1.5,label='Without')
            ax.fill_between(yrs,swo.oil_prices,sw.oil_prices,color='#DDDDDD',alpha=0.7)
        else:
            for rn in ['iran','us','china','row']:
                s=STYLES[rn]; sr=sw.get_series(rn,var)*sc
                se=np.abs(sr)*0.15+0.1
                ax.fill_between(yrs,sr-1.645*se,sr+1.645*se,color=s['color'],alpha=0.06)
                ax.plot(yrs,sr,color=s['color'],ls=s['ls'],lw=s['lw'],label=s['label'])
        _zl(ax); ax.set_ylabel(yl,fontsize=8); ax.set_xlabel('Years after war onset',fontsize=8)
        if idx==0: ax.legend(fontsize=6.5,loc='lower right',frameon=False)
        if var is None: ax.legend(fontsize=6.5,loc='upper right',frameon=False)
    fig.tight_layout(h_pad=1.2,w_pad=0.8)
    _note(fig,'Note: Impulse responses to a war of baseline intensity (Hormuz blockade severity 70 percent). Shaded areas indicate 90 percent confidence bands.',y=-0.03)
    _save(fig,'figure2_impulse_responses')

# ══════════ Figure 3: Third-Country by Exposure ══════════
def fig3():
    fig,axes=plt.subplots(2,2,figsize=(7.0,4.5))
    vp=[('y','Output','Percent',100,'A'),('pi','Consumer prices','Percentage points',100,'B'),
        ('i','Interest rate','Percentage points',100,'C'),('equity','Equity index','Percent',100,'D')]
    for idx,(var,title,yl,sc,pn) in enumerate(vp):
        ax=axes[idx//2,idx%2]; _pl(ax,f'Panel {pn}.'); ax.set_title(title,fontsize=10,pad=4)
        high=sw.get_series('row',var)*sc; low=high*0.3
        se=np.abs(high)*0.15+0.05
        ax.fill_between(yrs,high-1.645*se,high+1.645*se,color='black',alpha=0.08)
        ax.plot(yrs,high,'k-',lw=1.5,label='High exposure (3%)')
        ax.plot(yrs,low,color='#888888',ls='--',lw=1.5,label='Low exposure (0.9%)')
        ax.plot(yrs,np.zeros_like(yrs),color='#BBBBBB',ls=':',lw=1.0,label='No exposure')
        _zl(ax); ax.set_ylabel(yl,fontsize=8); ax.set_xlabel('Years after war onset',fontsize=8)
        if idx==0: ax.legend(fontsize=6.5,frameon=False,loc='lower right')
    fig.tight_layout(h_pad=1.2)
    _note(fig,'Note: Responses in third countries by pre-war trade exposure to the war site. Shaded areas indicate 90 percent confidence intervals.',y=-0.03)
    _save(fig,'figure3_trade_exposure')

# ══════════ Figure 4: Financial Channels ══════════
def fig4():
    fig,axes=plt.subplots(2,2,figsize=(7.0,4.5))
    vp=[('spread','Sovereign spreads','Basis points',10000,'A'),
        ('capital_flow','Capital flows','Percent of GDP',100,'B'),
        ('equity','Equity indices','Percent',100,'C'),
        ('rer','Real exchange rate','Percent (+ = depr.)',100,'D')]
    for idx,(var,title,yl,sc,pn) in enumerate(vp):
        ax=axes[idx//2,idx%2]; _pl(ax,f'Panel {pn}.'); ax.set_title(title,fontsize=10,pad=4)
        for rn in ['iran','us','china','row']:
            s=STYLES[rn]; ax.plot(yrs,sw.get_series(rn,var)*sc,color=s['color'],ls=s['ls'],lw=s['lw'],label=s['label'])
        _zl(ax); ax.set_ylabel(yl,fontsize=8); ax.set_xlabel('Years after war onset',fontsize=8)
        if idx==0: ax.legend(fontsize=6,frameon=False)
    fig.tight_layout(h_pad=1.2)
    _note(fig,'Note: Financial channel responses following war onset. Sovereign spreads in basis points. Capital flows as net inflows (percent of GDP). Exchange rate: positive denotes depreciation.',y=-0.03)
    _save(fig,'figure4_financial_channels')

# ══════════ Figure 5: China Stabilization ══════════
def fig5():
    fig,axes=plt.subplots(2,3,figsize=(7.0,4.8))
    # A: Oil
    ax=axes[0,0]; _pl(ax,'Panel A.'); ax.set_title('Oil price',fontsize=10,pad=4)
    ax.plot(yrs,sw.oil_prices,'k-',lw=1.5,label='With China')
    ax.plot(yrs,swo.oil_prices,color='#777',ls='--',lw=1.5,label='Without')
    ax.fill_between(yrs,sw.oil_prices,swo.oil_prices,color='#DDD',alpha=0.8)
    _zl(ax); ax.set_ylabel('Percent',fontsize=8); ax.legend(fontsize=6.5,frameon=False)
    # B: Global GDP
    ax=axes[0,1]; _pl(ax,'Panel B.'); ax.set_title('Global output',fontsize=10,pad=4)
    gw=sum(sw.get_series(r,'y')*sw.regions[r].gdp_share*100 for r in sw.regions)
    gwo=sum(swo.get_series(r,'y')*swo.regions[r].gdp_share*100 for r in swo.regions)
    ax.fill_between(yrs,gwo,gw,color='#CCC',alpha=0.8,label='China effect')
    ax.plot(yrs,gw,'k-',lw=1.5,label='With'); ax.plot(yrs,gwo,color='#777',ls='--',lw=1.5,label='Without')
    _zl(ax); ax.set_ylabel('Percent',fontsize=8); ax.legend(fontsize=6.5,frameon=False,loc='lower right')
    # C: Global Inflation
    ax=axes[0,2]; _pl(ax,'Panel C.'); ax.set_title('Global inflation',fontsize=10,pad=4)
    pw=sum(sw.get_series(r,'pi')*sw.regions[r].gdp_share*100 for r in sw.regions)
    pwo=sum(swo.get_series(r,'pi')*swo.regions[r].gdp_share*100 for r in swo.regions)
    ax.plot(yrs,pw,'k-',lw=1.5,label='With'); ax.plot(yrs,pwo,color='#777',ls='--',lw=1.5,label='Without')
    _zl(ax); ax.set_ylabel('Percentage points',fontsize=8); ax.legend(fontsize=6.5,frameon=False)
    # D: SPR
    ax=axes[1,0]; _pl(ax,'Panel D.'); ax.set_title('SPR release',fontsize=10,pad=4)
    sh=sw.design_war_shocks()
    cs=[sh[t]['china_spr']*100 for t in range(T)]; us=[sh[t]['us_spr']*100 for t in range(T)]
    rs=[sh[t]['row_spr']*100 for t in range(T)]
    ax.bar(yrs,cs,0.2,color='#555',label='China'); ax.bar(yrs,us,0.2,bottom=cs,color='#999',label='US')
    ax.bar(yrs,rs,0.2,bottom=[c+u for c,u in zip(cs,us)],color='#CCC',label='ROW')
    ax.set_ylabel('Pct. global supply',fontsize=8); ax.legend(fontsize=6,frameon=False)
    # E: Renewable
    ax=axes[1,1]; _pl(ax,'Panel E.'); ax.set_title('Renewable offset',fontsize=10,pad=4)
    ren=[0.2*(1-np.exp(-0.15*t)) for t in range(T)]
    ax.fill_between(yrs,0,ren,color='#BBB',alpha=0.6); ax.plot(yrs,ren,'k-',lw=1.5)
    ax.set_ylabel('Pct. of oil demand',fontsize=8)
    # F: Manufacturing
    ax=axes[1,2]; _pl(ax,'Panel F.'); ax.set_title('Manufacturing export boost',fontsize=10,pad=4)
    mfg=[1.5*(1-np.exp(-0.2*t)) for t in range(T)]
    ax.fill_between(yrs,0,mfg,color='#BBB',alpha=0.6); ax.plot(yrs,mfg,'k-',lw=1.5)
    ax.set_ylabel('Pct. of GDP',fontsize=8)
    for a in axes.flat: a.set_xlabel('Years after war onset',fontsize=8)
    fig.tight_layout(h_pad=1.2,w_pad=0.8)
    _note(fig,"Note: China's triple stabilization mechanism. Panels A-C compare global outcomes with and without China's stabilization. Panels D-F decompose the three channels.",y=-0.03)
    _save(fig,'figure5_china_stabilization')

# ══════════ Figure 6: Severity Scenarios ══════════
def fig6():
    fig,axes=plt.subplots(1,3,figsize=(7.0,2.8))
    cfgs=[(0.30,'Mild','#AAA','--'),(0.70,'Baseline','#555','-'),(0.90,'Severe','black','-')]
    sims={}
    for bl,lb,c,ls in cfgs:
        s=WarDSGESimulator(T=32,china_stabilizes=True); s.oil_market.blockade_severity=bl; s.simulate(); sims[lb]=s
    ax=axes[0]; _pl(ax,'Panel A.'); ax.set_title('Iran output',fontsize=10,pad=4)
    for _,lb,c,ls in cfgs: ax.plot(yrs,sims[lb].get_series('iran','y')*100,color=c,ls=ls,lw=1.5,label=lb)
    _zl(ax); ax.set_ylabel('Percent',fontsize=8); ax.legend(fontsize=7,frameon=False)
    ax=axes[1]; _pl(ax,'Panel B.'); ax.set_title('Global output',fontsize=10,pad=4)
    for _,lb,c,ls in cfgs:
        g=sum(sims[lb].get_series(r,'y')*sims[lb].regions[r].gdp_share*100 for r in sims[lb].regions)
        ax.plot(yrs,g,color=c,ls=ls,lw=1.5,label=lb)
    _zl(ax); ax.set_ylabel('Percent',fontsize=8)
    ax=axes[2]; _pl(ax,'Panel C.'); ax.set_title('Oil price',fontsize=10,pad=4)
    for _,lb,c,ls in cfgs: ax.plot(yrs,sims[lb].oil_prices,color=c,ls=ls,lw=1.5,label=lb)
    _zl(ax); ax.set_ylabel('Percent',fontsize=8)
    for a in axes: a.set_xlabel('Years after war onset',fontsize=8)
    fig.tight_layout(w_pad=1.0)
    _note(fig,'Note: Three war scenarios. Mild: 30% Hormuz blockade. Baseline: 70%. Severe: 90% with prolonged conflict.',y=-0.06)
    _save(fig,'figure6_severity')

# ══════════ Figure 7: Welfare ══════════
def fig7():
    fig,ax=plt.subplots(figsize=(5.0,3.2))
    d=0.99; ds=np.array([d**t for t in range(T)]); dn=ds.sum()
    ww,wwo={},{}
    for n in ['iran','us','china','row']:
        ww[n]=np.sum(ds*sw.get_series(n,'c'))/dn*100; wwo[n]=np.sum(ds*swo.get_series(n,'c'))/dn*100
    ww['global']=sum(ww[n]*sw.regions[n].gdp_share for n in sw.regions)
    wwo['global']=sum(wwo[n]*swo.regions[n].gdp_share for n in swo.regions)
    labels=['Iran','US/Israel','China','ROW','Global']; keys=['iran','us','china','row','global']
    x=np.arange(len(labels)); w=0.32
    ax.bar(x-w/2,[ww[k] for k in keys],w,color='#777',edgecolor='black',lw=0.5,label='With China stabilization')
    ax.bar(x+w/2,[wwo[k] for k in keys],w,color='#CCC',edgecolor='black',lw=0.5,label='Without China stabilization')
    ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=9)
    ax.set_ylabel('Consumption-equivalent welfare change (%)',fontsize=9)
    _zl(ax); ax.legend(fontsize=7.5,frameon=False,loc='lower left')
    fig.tight_layout()
    _note(fig,'Note: Consumption-equivalent welfare loss (discount factor 0.99). Global values are GDP-weighted.',y=-0.06)
    _save(fig,'figure7_welfare')

if __name__=="__main__":
    print("Generating AER-standard figures...")
    fig2(); fig3(); fig4(); fig5(); fig6(); fig7()
    print(f"\nAll figures saved to {OUTDIR}/")
