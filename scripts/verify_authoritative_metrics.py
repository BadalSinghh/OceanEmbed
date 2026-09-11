# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

df = pd.read_csv('results/argo_validation/argo_matched_observations.csv')
print("=== ARGO VALIDATION METRICS (CSV COMPUTATION) ===")
print(f"Total observations: {len(df)}")
print(f"Unique profiles: {df['profile_id'].nunique()}")
print(f"Unique dates: {df['date'].nunique()}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print()

models = [
    ('GLORYS12', 'glorys_temp'),
    ('CNN Baseline', 'cnn_temp'),
    ('CBAM-CNN', 'cbam_temp'),
    ('OceanEmbed', 'oe_temp')
]

y_obs = df['obs_temp'].values
obs_mean = np.mean(y_obs)
ss_tot = np.sum((y_obs - obs_mean)**2)

print(f"{'Model':22s} | {'RMSE (C)':10s} | {'MAE (C)':10s} | {'Bias (C)':10s} | {'Pearson R':10s} | {'R2':10s}")
print('-' * 80)
for name, col in models:
    y_pred = df[col].values
    err = y_pred - y_obs
    rmse = np.sqrt(np.mean(err**2))
    mae = np.mean(np.abs(err))
    bias = np.mean(err)
    r = np.corrcoef(y_pred, y_obs)[0, 1]
    r2 = 1.0 - np.sum(err**2) / ss_tot
    print(f"{name:22s} | {rmse:10.4f} | {mae:10.4f} | {bias:+10.4f} | {r:10.4f} | {r2:10.4f}")

print('\n=== PER-DEPTH RMSE (C) ===')
print(f"{'Depth':6s} | {'N':5s} | {'GLORYS':8s} | {'CNN':8s} | {'CBAM':8s} | {'OceanEmbed':10s} | {'Top Model':10s}")
print('-' * 70)
for d in sorted(df['depth_m'].unique()):
    sub = df[df['depth_m'] == d]
    n = len(sub)
    g_rmse = np.sqrt(np.mean((sub['glorys_temp'] - sub['obs_temp'])**2))
    c_rmse = np.sqrt(np.mean((sub['cnn_temp'] - sub['obs_temp'])**2))
    b_rmse = np.sqrt(np.mean((sub['cbam_temp'] - sub['obs_temp'])**2))
    o_rmse = np.sqrt(np.mean((sub['oe_temp'] - sub['obs_temp'])**2))
    top = ['CNN', 'CBAM', 'OceanEmbed'][np.argmin([c_rmse, b_rmse, o_rmse])]
    print(f"{int(d):5d}m | {n:5d} | {g_rmse:8.4f} | {c_rmse:8.4f} | {b_rmse:8.4f} | {o_rmse:10.4f} | {top:10s}")

print('\n=== DEPTH ZONE RMSE (C) ===')
for z_name, depths in [('Surface (0-30m)', [0, 5, 10, 20, 30]), ('Thermocline (50-200m)', [50, 75, 100, 125, 150, 200]), ('Deep (300-1000m)', [300, 500, 700, 1000])]:
    sub = df[df['depth_m'].isin(depths)]
    n = len(sub)
    c_rmse = np.sqrt(np.mean((sub['cnn_temp'] - sub['obs_temp'])**2))
    b_rmse = np.sqrt(np.mean((sub['cbam_temp'] - sub['obs_temp'])**2))
    o_rmse = np.sqrt(np.mean((sub['oe_temp'] - sub['obs_temp'])**2))
    print(f"{z_name:25s} (N={n:4d}) | CNN: {c_rmse:.4f} C | CBAM: {b_rmse:.4f} C | OceanEmbed: {o_rmse:.4f} C")
