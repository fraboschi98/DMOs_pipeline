# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 14:38:18 2026

@author: francesca.boschi
"""

# -*- coding: utf-8 -*-
"""
@author: francesca.boschi

Research question
    In atypical parkinsonism, do changes in clinical scales and digital mobility
    outcomes (DMOs) correspond to how patients and clinicians perceive change across
    supervised and home-based physiotherapy (PT)?

Groups at each visit (binary GIC: Impr = 1-3, Not Impr = 4-7, 0/NaN = missing):
    BI = both improved (true improvers)   PGIC Impr,     CGIC Impr       green
    CO = clinician only                   PGIC Not Impr, CGIC Impr       blue
    PO = patient only                     PGIC Impr,     CGIC Not Impr   orange
    BN = both not improved                PGIC Not Impr, CGIC Not Impr   grey

MODEL (per outcome, per scheme)
    Linear mixed model, random intercept per patient (REML):
        y ~ 0 + group:phase  [+ diagnosis + arm]
    y = Δ / SD (SD of all APD rows), signed so that + = better; divided by the SD only
    (not centred), so 0 = no change from baseline (V2).
    adjusted   : + diagnosis (PSP vs MSA) and arm (SPT vs GPT), centred 0/1 covariates,
                 so group means are adjusted means at the sample mix.
    unadjusted : without covariates.

ANALYSIS 1  Change from baseline in each group (H0: mean change = 0)
            BI, CO, PO, BN per phase and outcome.
ANALYSIS 2  Discordant groups vs true improvers
            PO − BI, CO − BI   (does measured change in each discordant group
                                differ from agreed improvement?)
            PO − CO            (direct comparison of the two discordant groups)

FDR (Benjamini-Hochberg) within domain (DMOs: 6 outcomes; clinical scales: 3),
per quantity and phase.

Schemes
    unpooled (main) : Lab | At Home wk 3-4 | At Home wk 5-7
    pooled          : Lab | At Home wk 3-7  (V4 and V5 rows both kept)
All changes and ratings are CUMULATIVE from baseline: At Home includes the Lab phase.

Output: GIC_Analysis/04_correspondence/two_analyses/{unpooled,pooled}/{adjusted,unadjusted}/
    analysis1_change_from_baseline_<scheme>.csv / .png
    analysis2_vs_true_improvers_<scheme>.csv / .png
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import statsmodels.formula.api as smf
from scipy import stats
plt.close('all')
warnings.filterwarnings('ignore')


#%% Settings ================================================================

PROJECT = r"C:\Users\francesca.boschi\OneDrive - University of Luxembourg (1)\T_LCSB Digital Medicine - Francesca B-Gaetan Project"
IN       = os.path.join(PROJECT, "dataset_included_patients.csv")
OUT_ROOT = os.path.join(PROJECT, "GIC_Analysis", "04_correspondence", "two_analyses")

DOMAINS = {
    'DMOs (gait)': [
        'delta_gait_velocity_2x10m_pref_median',
        'delta_stride_length_2x10m_pref_median',
        'delta_stride_time_2x10m_pref_median',
        'delta_stance_percentage_2x10m_pref_median',
        'delta_cadence_steps_min_2min_mean',
        'delta_gait_velocity_modulation',
    ],
    'Clinical scales': [
        'delta_BBS',
        'delta_MDS_UPDRS_III',
        'delta_MDS_UPDRS_PIGD',
    ],
}

LABELS = {
    'delta_gait_velocity_2x10m_pref_median':     'Gait velocity',
    'delta_stride_length_2x10m_pref_median':     'Stride length',
    'delta_stride_time_2x10m_pref_median':       'Stride time',
    'delta_stance_percentage_2x10m_pref_median': 'Stance %',
    'delta_cadence_steps_min_2min_mean':         'Cadence (2-min walk)',
    'delta_gait_velocity_modulation':            'Velocity modulation (fast − slow)',
    'delta_BBS':                                 'BBS',
    'delta_MDS_UPDRS_III':                       'MDS-UPDRS III',
    'delta_MDS_UPDRS_PIGD':                      'MDS-UPDRS PIGD',
}

UNITS = {
    'delta_gait_velocity_2x10m_pref_median':     'm/s',
    'delta_stride_length_2x10m_pref_median':     'm',
    'delta_stride_time_2x10m_pref_median':       's',
    'delta_stance_percentage_2x10m_pref_median': '%',
    'delta_cadence_steps_min_2min_mean':         'steps/min',
    'delta_gait_velocity_modulation':            'm/s',
    'delta_BBS':                                 'pts',
    'delta_MDS_UPDRS_III':                       'pts',
    'delta_MDS_UPDRS_PIGD':                      'pts',
}
DECIMALS = {'m/s': 3, 'm': 3, 's': 3, '%': 2, 'steps/min': 1, 'pts': 1}

# lower = better -> sign flipped so that positive = improvement
FLIP = [
    'delta_MDS_UPDRS_III',
    'delta_MDS_UPDRS_PIGD',
    'delta_stance_percentage_2x10m_pref_median',
    'delta_stride_time_2x10m_pref_median',
]

DIAG_CODES = {1: 'MSA', 2: 'PSP'}      # 3 = PD, excluded. CHECK your codes
ARM_CODES  = {1: 'GPT', 2: 'SPT'}

LAB  = 'Lab (wk 1-2)'
H1   = 'At Home (wk 3-4)'
H2   = 'At Home (wk 5-7)'
HOME = 'At Home (wk 3-7)'
SCHEMES = {
    'unpooled': {'Visit3-Visit2': LAB, 'Visit4-Visit2': H1, 'Visit5-Visit2': H2},
    'pooled':   {'Visit3-Visit2': LAB, 'Visit4-Visit2': HOME, 'Visit5-Visit2': HOME},
}
MODELS = {'adjusted': ['diag', 'arm'], 'unadjusted': []}

GROUPS = ['BI', 'CO', 'PO', 'BN']
GROUP_NAME = {'BI': 'Both improved (true improvers)', 'CO': 'Clinician only',
              'PO': 'Patient only', 'BN': 'Both not improved'}
GROUP_DEF = {'BI': 'PGIC Impr, CGIC Impr', 'CO': 'PGIC Not Impr, CGIC Impr',
             'PO': 'PGIC Impr, CGIC Not Impr', 'BN': 'PGIC Not Impr, CGIC Not Impr'}

# each analysis: quantity -> (weights on group means within a phase, label, colour, offset)
ANALYSES = {
    'analysis1_change_from_baseline': {
        'title': 'Analysis 1 – change from baseline in each agreement group',
        'note':  'Each dot = mean change from V2 in that group (0 = no change).',
        'quant': {
            'BI': ({'BI': 1}, f"{GROUP_NAME['BI']} ({GROUP_DEF['BI']})", '#31A354', -0.30),
            'CO': ({'CO': 1}, f"{GROUP_NAME['CO']} ({GROUP_DEF['CO']})", '#3182BD', -0.10),
            'PO': ({'PO': 1}, f"{GROUP_NAME['PO']} ({GROUP_DEF['PO']})", '#E6550D',  0.10),
            'BN': ({'BN': 1}, f"{GROUP_NAME['BN']} ({GROUP_DEF['BN']})", '#7F7F7F',  0.30),
        },
        'n_of': {'BI': ['BI'], 'CO': ['CO'], 'PO': ['PO'], 'BN': ['BN']},
    },
    'analysis2_vs_true_improvers': {
        'title': 'Analysis 2 – discordant groups compared with true improvers',
        'note':  'Each dot = difference in change between groups (0 = same change).',
        'quant': {
            'CO−BI': ({'CO': 1, 'BI': -1}, 'Clinician only − true improvers', '#3182BD', -0.22),
            'PO−BI': ({'PO': 1, 'BI': -1}, 'Patient only − true improvers',   '#E6550D',  0.00),
            'PO−CO': ({'PO': 1, 'CO': -1}, 'Patient only − clinician only',   '#7B3294',  0.22),
        },
        'n_of': {'CO−BI': ['CO', 'BI'], 'PO−BI': ['PO', 'BI'], 'PO−CO': ['PO', 'CO']},
    },
}

ALPHA = 0.05      # filled dot / bold / * = FDR p < ALPHA
MIN_N = 5         # min rows in a group x phase cell to report quantities using it


#%% Loading =================================================================

FEATURES = [f for fs in DOMAINS.values() for f in fs]
DOMAIN_OF = {f: dom for dom, fs in DOMAINS.items() for f in fs}

df = pd.read_csv(IN, index_col=0)
df['delta_gait_velocity_modulation'] = (df['delta_gait_velocity_2x10m_fast_median']
                                        - df['delta_gait_velocity_2x10m_slow_median'])

missing_cols = [f for f in FEATURES if f not in df.columns]
if missing_cols:
    raise KeyError(f'Not in dataset: {missing_cols}')

df = df[df['diagnosis'] != 3].copy()                       # APD only
df['diag'] = df['diagnosis'].map(DIAG_CODES)
df['arm']  = df['Group'].map(ARM_CODES)

cg, pg = df['CGI-C'], df['PGI-C']
c_imp, c_not = cg.between(1, 3), cg.between(4, 7)
p_imp, p_not = pg.between(1, 3), pg.between(4, 7)
df['grp'] = np.select([p_imp & c_imp, p_not & c_imp, p_imp & c_not, p_not & c_not],
                      ['BI', 'CO', 'PO', 'BN'], default='')

d0 = df[df['delta_Visit'].isin(SCHEMES['unpooled']) & (df['grp'] != '')
        & df['diag'].notna() & df['arm'].notna()].copy()
for c in FLIP:
    if c in d0.columns:
        d0[c] = -d0[c]

# SD of each change over all APD rows (scale only, not centred)
SD = {f: d0[f].std() for f in FEATURES}


#%% Helpers =================================================================

def bh_fdr(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if ok.sum() == 0:
        return out
    pv = p[ok]; m = len(pv)
    order = np.argsort(pv)
    adj = pv[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    res = np.empty(m); res[order] = np.minimum(adj, 1)
    out[ok] = res
    return out


def mixed(s, used):
    """
    y ~ 0 + C(cell) [+ centred covariates], random intercept per patient (REML).
    Several optimisers; the fit with the highest REML likelihood and sensible standard
    errors is kept (one optimiser alone can stop at a wrong solution).
    """
    formula = 'y ~ 0 + C(cell)' + ''.join(f' + {c}' for c in used)
    best = None
    for method in ['bfgs', 'powell', 'nm', 'lbfgs']:
        try:
            fit = smf.mixedlm(formula, s, groups=s['PATID']).fit(reml=True, method=method)
        except Exception:
            continue
        if not (np.all(np.isfinite(fit.bse_fe)) and fit.bse_fe.max() < 10):
            continue
        if best is None or fit.llf > best.llf + 1e-6:
            best = fit
    return best


def lincomb(fit, w):
    """Estimate, SE, p of sum(weight * fixed effect). w = {param_name: weight}."""
    fe = fit.fe_params
    L = np.zeros(len(fe))
    for t, wt in w.items():
        L[fe.index.get_loc(t)] = wt
    V = fit.cov_params().loc[fe.index, fe.index].values
    est = float(L @ fe.values)
    se = float(np.sqrt(L @ V @ L))
    p = 2 * stats.norm.sf(abs(est / se)) if se > 0 else np.nan
    return est, se, p


#%% Model per outcome ========================================================

def fit_feature(g, feat, phases, covs):
    """Fit one model; return rows for both analyses."""
    s = g[['PATID', 'grp', 'phase', 'diag', 'arm', feat]].dropna()
    s = s.rename(columns={feat: 'y_raw'}).reset_index(drop=True)
    sd = SD[feat]
    s['y'] = s['y_raw'] / sd
    s['cell'] = s['grp'] + '|' + s['phase']
    used = []
    if 'diag' in covs and s['diag'].nunique() > 1:
        s['diag_c'] = (s['diag'] == 'PSP').astype(float); s['diag_c'] -= s['diag_c'].mean()
        used.append('diag_c')
    if 'arm' in covs and s['arm'].nunique() > 1:
        s['arm_c'] = (s['arm'] == 'SPT').astype(float); s['arm_c'] -= s['arm_c'].mean()
        used.append('arm_c')
    n = s.groupby('cell').size()
    n_pat = s.groupby('cell')['PATID'].nunique()

    fit = mixed(s, used)
    if fit is None:
        print(f'  model failed: {LABELS[feat]}')
        return []

    rows = []
    for ana, A in ANALYSES.items():
        for ph in phases:
            for code, (w, name, _, _) in A['quant'].items():
                cells = {f'{grp}|{ph}': wt for grp, wt in w.items()}
                if any(n.get(c, 0) < MIN_N for c in cells):
                    continue
                est, se, p = lincomb(fit, {f'C(cell)[{c}]': wt for c, wt in cells.items()})
                lo, hi = est - 1.96 * se, est + 1.96 * se
                rows.append(dict(
                    analysis=ana, phase=ph, domain=DOMAIN_OF[feat], feature=feat,
                    label=LABELS[feat], unit=UNITS[feat], quantity=code, quantity_name=name,
                    **{f'n_{gr}': int(n.get(f'{gr}|{ph}', 0)) for gr in GROUPS},
                    **{f'npat_{gr}': int(n_pat.get(f'{gr}|{ph}', 0)) for gr in GROUPS},
                    est_SD=est, low_SD=lo, high_SD=hi, p=p, SD_raw=sd,
                    est_raw=est * sd, low_raw=lo * sd, high_raw=hi * sd,
                    covariates=', '.join(used) or 'none'))
    return rows


#%% Plot ====================================================================

def plot_analysis(res, A, phases, title, fname):
    """
    Panels = phases.  Rows = outcomes grouped by domain.
    One dot ± 95% CI per quantity (SD units). Filled = FDR p < ALPHA.
    Right of each panel: values in real units (+ = better), n, * = FDR p < ALPHA.
    """
    Q = A['quant']
    ypos, ticks, labels, bold = {}, [], [], []
    y = 0
    for dom, feats in DOMAINS.items():
        ticks.append(y); labels.append(dom); bold.append(True); y += 1
        for f in feats:
            ypos[f] = y
            ticks.append(y); labels.append(LABELS[f]); bold.append(False); y += 1
        y += 0.5
    n_rows = y

    xmin = min(-1.0, np.nanmin(res['low_SD']) - 0.1)
    xmax = max(1.0, np.nanmax(res['high_SD']) + 0.2)

    n_ph = len(phases)
    fig = plt.figure(figsize=(6.2 * n_ph + 2, 0.72 * n_rows + 2.4))
    gs = GridSpec(1, 2 * n_ph, figure=fig, width_ratios=[3.0, 2.3] * n_ph, wspace=0.05)
    ax0 = None

    for i, ph in enumerate(phases):
        ax = fig.add_subplot(gs[0, 2 * i], sharey=ax0)
        ax_t = fig.add_subplot(gs[0, 2 * i + 1], sharey=ax)
        ax0 = ax0 or ax
        r_ph = res[res['phase'] == ph]
        ax.axvline(0, color='black', lw=0.8, zorder=1)

        for f, yy in ypos.items():
            rr = r_ph[r_ph['feature'] == f]
            u, dec = UNITS[f], DECIMALS[UNITS[f]]
            for code, (_, _, col, off) in Q.items():
                rc = rr[rr['quantity'] == code]
                if not len(rc):
                    continue
                row = rc.iloc[0]
                y_c = yy + off
                sig = row['p_fdr'] < ALPHA
                ax.errorbar(row['est_SD'], y_c,
                            xerr=[[row['est_SD'] - row['low_SD']],
                                  [row['high_SD'] - row['est_SD']]],
                            fmt='o', color=col, mfc=col if sig else 'white',
                            ms=4.5, capsize=2.5, lw=1.3, zorder=3)
                n_txt = ' '.join(f"n{gr}={row['n_' + gr]}" for gr in A['n_of'][code])
                ax_t.text(0.02, y_c,
                          f"{code} {row['est_raw']:+.{dec}f} "
                          f"[{row['low_raw']:+.{dec}f}, {row['high_raw']:+.{dec}f}] {u}"
                          f"  {n_txt}{' *' if sig else ''}",
                          color=col, va='center', fontsize=6.8,
                          fontweight='bold' if sig else 'normal')

        ax_t.text(0.02, -1.0, 'real units [95% CI], + = better', fontsize=7.5,
                  fontweight='bold', va='center')
        ax_t.set_xlim(0, 1)
        ax_t.axis('off')

        ax.set_xlim(xmin, xmax)
        ax.set_title(ph, fontsize=11, fontweight='bold')
        ax.set_xlabel('SD units (+ = better)', fontsize=8)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0)
        if i == 0:
            ax.set_yticks(ticks, labels, fontsize=9)
            for t, b in zip(ax.get_yticklabels(), bold):
                if b:
                    t.set_fontweight('bold'); t.set_fontsize(10)
        else:
            ax.tick_params(axis='y', labelleft=False)

    ax0.set_ylim(n_rows - 0.5, -1.6)

    handles = [plt.Line2D([], [], marker='o', color=col, ls='-', label=f'{code}: {name}')
               for code, (_, name, col, _) in Q.items()]
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(title, fontsize=10.5, fontweight='bold')
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(fname, dpi=200, bbox_inches='tight')
    plt.show()


#%% Main ====================================================================

for scheme, pmap in SCHEMES.items():
    phases = list(dict.fromkeys(pmap.values()))
    g = d0.copy()
    g['phase'] = g['delta_Visit'].map(pmap)
    print(f'\n===== {scheme}: rows per group and phase =====')
    print(pd.crosstab(g['grp'], g['phase']).reindex(index=GROUPS, columns=phases).to_string())

    for model, covs in MODELS.items():
        out = os.path.join(OUT_ROOT, scheme, model)
        os.makedirs(out, exist_ok=True)

        rows = []
        for f in FEATURES:
            rows += fit_feature(g, f, phases, covs)
        R = pd.DataFrame(rows)
        # FDR within domain, per analysis x quantity x phase
        R['p_fdr'] = np.nan
        for _, idx in R.groupby(['analysis', 'quantity', 'phase', 'domain']).groups.items():
            R.loc[idx, 'p_fdr'] = bh_fdr(R.loc[idx, 'p'])
        R.insert(0, 'model', model); R.insert(0, 'scheme', scheme)

        adj_txt = ('adjusted for diagnosis and arm' if model == 'adjusted' else 'unadjusted')
        for ana, A in ANALYSES.items():
            Ra = R[R['analysis'] == ana]
            Ra.to_csv(os.path.join(out, f'{ana}_{scheme}.csv'), index=False,
                      encoding='utf-8-sig')
            title = (f"{A['title']}  –  APD, {scheme}, {adj_txt}\n"
                     f"{A['note']}  Linear mixed model, random intercept per patient. "
                     f"Filled dot / bold / * = FDR p < {ALPHA} (within domain, per quantity "
                     f"and phase).\nChanges are cumulative from baseline (V2).")
            plot_analysis(Ra, A, phases, title, os.path.join(out, f'{ana}_{scheme}.png'))

            print(f'\n===== {scheme} | {model} | {ana}: FDR p < {ALPHA} =====')
            sig = Ra[Ra['p_fdr'] < ALPHA]
            print(sig[['phase', 'label', 'quantity', 'est_raw', 'low_raw', 'high_raw', 'unit',
                       'p', 'p_fdr']].round(3).to_string(index=False) if len(sig) else 'none')

print(f'\nSaved to: {OUT_ROOT}')