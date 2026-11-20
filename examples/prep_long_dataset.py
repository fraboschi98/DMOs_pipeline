

# -*- coding: utf-8 -*-
"""
@author: francesca.boschi

Analysis dataset: one row per patient per visit (V2 baseline, V3, V4, V5).
For every outcome X: X (raw value), X_baseline (V2 value), delta_X (visit − V2).
"""

import os
import numpy as np
import pandas as pd

PROJECT = r"C:\Users\francesca.boschi\OneDrive - University of Luxembourg (1)\T_LCSB Digital Medicine - Francesca B-Gaetan Project"
OUT = os.path.join(PROJECT, "GIC_Analysis", "00_data")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(os.path.join(PROJECT, "dataset_included_patients.csv"), index_col=0)
lab_tests = pd.read_csv(os.path.join(PROJECT, "gait_lab_cleaned.csv"))
XLS = os.path.join(PROJECT, "RCT_Clinical_Data.xlsx")

VISITS = ['Visit2', 'Visit3', 'Visit4', 'Visit5']          # Visit2 = baseline
DIAG = {1: 'MSA', 2: 'PSP', 3: 'PD'}                       # CHECK your codes
ARM  = {1: 'GPT', 2: 'SPT'}

# clinical scales from the Excel sheets: new name -> Excel column
CLINICAL = {'BBS': 'BBS', 'UPDRS_III': 'MDS_UPDRS_III', 'PIGD': 'MDS_PIGD',
            'FOG_Q': 'FOG-Q', 'PDQ8': 'PDQ-8', 'PGI_S': 'PGI-S', 'CGI_S': 'CGI-S'}
# gait from lab_tests (median over trials): new name -> test
GAIT = {'gait_velocity': '2x10m_pref', 'gait_velocity_fast': '2x10m_fast',
        'gait_velocity_slow': '2x10m_slow'}
PATIENT = ['age', 'Sex', 'disease_duration', 'HY', 'MoCA', 'FAB']


# 1. clinical scales: one sheet per visit
clin = []
for v in VISITS:
    s = pd.read_excel(XLS, sheet_name=v)
    c = pd.DataFrame({'PATID': s['PATID'], 'visit': v})
    for new, col in CLINICAL.items():
        c[new] = pd.to_numeric(s[col], errors='coerce') if col in s else np.nan
    clin.append(c)
clin = pd.concat(clin)

# 2. gait velocity from lab_tests
g = lab_tests[lab_tests['Tests'].isin(GAIT.values()) & (lab_tests['Aggregated_Method'] == 'median')]
gait = (g.pivot_table(index=['PATID', 'Visit'], columns='Tests', values='gait_velocity')
          .rename(columns={t: n for n, t in GAIT.items()}).reset_index()
          .rename(columns={'Visit': 'visit'}))

# 3. patient info and PGI-C / CGI-C from df
pat = df.groupby('PATID', as_index=False)[['diagnosis', 'Group'] + PATIENT].first()
pat['diagnosis'] = pat['diagnosis'].map(DIAG)
pat['arm'] = pat.pop('Group').map(ARM)

gic = df.assign(visit=df['delta_Visit'].str.split('-').str[0])[['PATID', 'visit', 'PGI-C', 'CGI-C']]
for c in ['PGI-C', 'CGI-C']:
    gic[c] = gic[c].where(gic[c].between(1, 7))           # 0 / NaN -> missing

# 4. long table
L = pd.MultiIndex.from_product([pat['PATID'], VISITS], names=['PATID', 'visit']).to_frame(index=False)
L = (L.merge(pat, on='PATID', how='left')
      .merge(gic, on=['PATID', 'visit'], how='left')
      .merge(gait, on=['PATID', 'visit'], how='left')
      .merge(clin, on=['PATID', 'visit'], how='left'))
L['delta_Visit'] = np.where(L['visit'] == 'Visit2', 'baseline', L['visit'] + '-Visit2')
OUTCOMES = list(GAIT) + list(CLINICAL)
for o in OUTCOMES:
    base = L[L['visit'] == 'Visit2'].set_index('PATID')[o]
    L[f'{o}_baseline'] = L['PATID'].map(base)
    L[f'delta_{o}'] = np.where(L['visit'] == 'Visit2', np.nan, L[o] - L[f'{o}_baseline'])

# 5. check: BBS change from Excel must equal delta_BBS in df
chk = L.merge(df.assign(visit=df['delta_Visit'].str.split('-').str[0])[['PATID', 'visit', 'delta_BBS']]
              .rename(columns={'delta_BBS': 'delta_BBS_df'}), on=['PATID', 'visit'])
same = ((chk['delta_BBS'] - chk['delta_BBS_df']).abs() < 1e-6)
print(f'Check BBS: {same.sum()} of {chk["delta_BBS_df"].notna().sum()} rows identical')
print('\nValues per visit (APD):')
print(L[L['diagnosis'].isin(['MSA', 'PSP'])].groupby('visit')[['PGI-C', 'CGI-C'] + OUTCOMES].count())

# 6. save

cols = ['PATID', 'visit', 'delta_Visit', 'diagnosis', 'arm'] + PATIENT + ['PGI-C', 'CGI-C']
for o in OUTCOMES:
    cols += [o, f'{o}_baseline', f'delta_{o}']
L[cols].sort_values(['PATID', 'visit']).to_csv(
    os.path.join(OUT, 'analysis_dataset_long.csv'), index=False, encoding='utf-8-sig')
print(f'\nSaved: {OUT}')