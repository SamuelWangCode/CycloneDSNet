import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error

# =========================
# 0. Global style
# =========================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "axes.titlesize": 20,
    "axes.labelsize": 17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.dpi": 220,
    "savefig.dpi": 450,
    "axes.linewidth": 1.1,
    "mathtext.default": "regular"
})

# =========================
# 1. Paths
# =========================
base_path = '/bigdata3/WangGuanSong/Weaformer/all_models/weaformer_v2.0/typhoon_intensity_bc/results'
df_24h = pd.read_csv(f'{base_path}/Full_Test_24h/statistics_24h.csv')
df_48h = pd.read_csv(f'{base_path}/Full_Test_48h/statistics_48h.csv')
df_72h = pd.read_csv(f'{base_path}/Full_Test_72h/statistics_72h.csv')
df_96h = pd.read_csv(f'{base_path}/Full_Test_96h/statistics_96h.csv')

# =========================
# 2. Naming and style
# =========================
MODEL_NAME_MAP = {
    'TX': 'TianXing',
    'WRF': 'SHTM',
    'DSNet': 'CycloneDSNet-Deterministic',
    'Diff_Mean': 'CycloneDSNet-Diffusion'
}

COLORS = {
    'TianXing': '#4C78A8',
    'SHTM': '#54A24B',
    'CycloneBCNet': '#9C755F',
    'CycloneDSNet-Deterministic': '#F58518',
    'CycloneDSNet-Diffusion': '#B279A2'
}

MARKERS = {
    'TianXing': 'o',
    'SHTM': 's',
    'CycloneBCNet': 'X',
    'CycloneDSNet-Deterministic': '^',
    'CycloneDSNet-Diffusion': 'D'
}

LINESTYLES = {
    'TianXing': '-',
    'SHTM': '-',
    'CycloneBCNet': '--',
    'CycloneDSNet-Deterministic': '-',
    'CycloneDSNet-Diffusion': '-'
}

# =========================
# 3. CycloneBCNet metrics
# =========================
bcnet_metrics = pd.DataFrame([
    [24, 'Wind', 'CycloneBCNet', 5.34, 7.73],
    [24, 'Pres', 'CycloneBCNet', 6.94, 10.9],
    [48, 'Wind', 'CycloneBCNet', 5.24, 7.20],
    [48, 'Pres', 'CycloneBCNet', 7.00, 10.4],
    [72, 'Wind', 'CycloneBCNet', 5.77, 8.18],
    [72, 'Pres', 'CycloneBCNet', 7.69, 11.5],
    [96, 'Wind', 'CycloneBCNet', 6.69, 9.46],
    [96, 'Pres', 'CycloneBCNet', 9.36, 13.8],
], columns=['Hour', 'Variable', 'Model', 'MAE', 'RMSE'])


# =========================
# 4. Bootstrap helper
# =========================
def bootstrap_ci(y_true, y_pred, metric_func, n_boot=1000, ci=95):
    rng = np.random.RandomState(42)
    indices = np.arange(len(y_true))
    scores = []

    for _ in range(n_boot):
        sample_idx = rng.choice(indices, len(indices), replace=True)
        score = metric_func(y_true.iloc[sample_idx], y_pred.iloc[sample_idx])
        scores.append(score)

    lower = np.percentile(scores, (100 - ci) / 2)
    upper = np.percentile(scores, 100 - (100 - ci) / 2)
    return lower, upper


def rmse_func(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae_func(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


# =========================
# 5. Metric calculation
# =========================
def calculate_metrics_with_ci(df, hour):
    metrics = []
    std_models = ['TX', 'WRF', 'DSNet', 'Diff_Mean']

    for var in ['Wind', 'Pres']:
        target = f'CMA_{var}'
        for model in std_models:
            pred_col = f'{model}_{var}'
            valid_mask = df[target].notna() & df[pred_col].notna()
            y_true = df.loc[valid_mask, target]
            y_pred = df.loc[valid_mask, pred_col]

            rmse = rmse_func(y_true, y_pred)
            mae = mae_func(y_true, y_pred)

            rmse_low, rmse_up = np.nan, np.nan
            mae_low, mae_up = np.nan, np.nan

            if model == 'Diff_Mean':
                rmse_low, rmse_up = bootstrap_ci(y_true, y_pred, rmse_func)
                mae_low, mae_up = bootstrap_ci(y_true, y_pred, mae_func)

            metrics.append({
                'Hour': hour,
                'Variable': var,
                'Model': MODEL_NAME_MAP[model],
                'RMSE': rmse,
                'MAE': mae,
                'RMSE_Lower': rmse_low,
                'RMSE_Upper': rmse_up,
                'MAE_Lower': mae_low,
                'MAE_Upper': mae_up
            })
    return metrics


# =========================
# 6. Aggregate current metrics
# =========================
all_metrics = []
for h, df in zip([24, 48, 72, 96], [df_24h, df_48h, df_72h, df_96h]):
    all_metrics.extend(calculate_metrics_with_ci(df, h))
metrics_df = pd.DataFrame(all_metrics)

# =========================
# 7. Figure 4: trends
# =========================
metrics_plot_df = pd.concat([metrics_df, bcnet_metrics], ignore_index=True)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
plot_cfg = [
    ('Wind', 'MAE', 0, 0),
    ('Wind', 'RMSE', 0, 1),
    ('Pres', 'MAE', 1, 0),
    ('Pres', 'RMSE', 1, 1)
]

plot_order = [
    'TianXing',
    'SHTM',
    'CycloneBCNet',
    'CycloneDSNet-Deterministic',
    'CycloneDSNet-Diffusion'
]

for var, metric, r, c in plot_cfg:
    ax = axes[r, c]
    subset = metrics_plot_df[metrics_plot_df['Variable'] == var]

    for model in plot_order:
        model_data = subset[subset['Model'] == model].sort_values('Hour')
        if model_data.empty:
            continue

        ax.plot(
            model_data['Hour'],
            model_data[metric],
            label=model,
            color=COLORS[model],
            marker=MARKERS[model],
            linestyle=LINESTYLES[model],
            linewidth=2.8,
            markersize=8
        )

        if model == 'CycloneDSNet-Diffusion':
            ax.fill_between(
                model_data['Hour'],
                model_data[f'{metric}_Lower'],
                model_data[f'{metric}_Upper'],
                color=COLORS[model],
                alpha=0.20,
                linewidth=0
            )

    title_str = 'Vmax' if var == 'Wind' else 'Pmin'
    unit = r'm s$^{-1}$' if var == 'Wind' else 'hPa'

    ax.set_title(f'{title_str} - {metric}', pad=8, fontweight='bold')
    ax.set_xticks([24, 48, 72, 96])
    ax.set_xlabel('Forecast Hour (h)')
    ax.set_ylabel(f'{metric} ({unit})')
    ax.grid(True, alpha=0.30)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc='lower center',
    bbox_to_anchor=(0.5, 0.01),
    ncol=5,
    frameon=True
)
plt.tight_layout()
plt.subplots_adjust(bottom=0.11)
plt.savefig('Figure_intensity_skill_trends.png', bbox_inches='tight')
plt.close()

# =========================
# 8. Figure 5: scatter (4 rows × 2 cols)
# =========================
combined_df = pd.concat([df_24h, df_48h, df_72h, df_96h], ignore_index=True)
models_raw = ['TX', 'WRF', 'DSNet', 'Diff_Mean']
model_titles = [MODEL_NAME_MAP[m] for m in models_raw]

fig, axes = plt.subplots(4, 2, figsize=(14, 22))
vars_info = [('Wind', r'm s$^{-1}$', [0, 85]), ('Pres', 'hPa', [890, 1020])]

for col_idx, (var, unit, lims) in enumerate(vars_info):
    target_col = f'CMA_{var}'
    for row_idx, (model_raw, title) in enumerate(zip(models_raw, model_titles)):
        pred_col = f'{model_raw}_{var}'
        ax = axes[row_idx, col_idx]

        valid = combined_df[[target_col, pred_col]].dropna()
        rmse = np.sqrt(mean_squared_error(valid[target_col], valid[pred_col]))
        mae = mean_absolute_error(valid[target_col], valid[pred_col])
        corr = np.corrcoef(valid[target_col], valid[pred_col])[0, 1]

        ax.scatter(
            valid[target_col],
            valid[pred_col],
            alpha=0.38,
            s=22,
            c=COLORS[title],
            edgecolors='none'
        )
        ax.plot(lims, lims, 'k--', linewidth=1.2, alpha=0.7)

        txt = f'RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nCorr: {corr:.3f}'
        ax.text(
            0.05, 0.95, txt,
            transform=ax.transAxes,
            va='top',
            fontsize=13,
            bbox=dict(
                boxstyle='round,pad=0.28',
                facecolor='white',
                alpha=0.88,
                edgecolor='0.7'
            )
        )

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.25)

        # 行标题：模型
        ax.set_title(title.replace('CycloneDSNet-', 'CycloneDSNet-\n'), fontweight='bold', pad=6)

        # 列坐标轴标签
        if col_idx == 0:
            ax.set_xlabel(r'Observed Vmax (m s$^{-1}$)')
            ax.set_ylabel(r'Predicted Vmax (m s$^{-1}$)')
        else:
            ax.set_xlabel('Observed Pmin (hPa)')
            ax.set_ylabel('Predicted Pmin (hPa)')

plt.tight_layout()
plt.savefig('Figure_scatter.png', bbox_inches='tight')
plt.close()

# =========================
# 9. Figure 6: boxplots
# =========================
dfs = []
for h, df in zip([24, 48, 72, 96], [df_24h, df_48h, df_72h, df_96h]):
    t = df.copy()
    t['Forecast Hour'] = h
    dfs.append(t)
full_df = pd.concat(dfs)

plot_data = []
models_comp = ['WRF', 'DSNet', 'Diff_Mean']
model_labels_comp = {m: MODEL_NAME_MAP[m] for m in models_comp}

for var in ['Wind', 'Pres']:
    target = f'CMA_{var}'
    for model in models_comp:
        abs_err = (full_df[f'{model}_{var}'] - full_df[target]).abs()
        plot_data.append(pd.DataFrame({
            'Forecast Hour': full_df['Forecast Hour'],
            'Absolute Error': abs_err,
            'Model': model_labels_comp[model],
            'Variable': var
        }))
plot_df = pd.concat(plot_data)

fig, axes = plt.subplots(1, 2, figsize=(19, 8.5))

box_order = ['SHTM', 'CycloneDSNet-Deterministic', 'CycloneDSNet-Diffusion']
palette_box = [COLORS[m] for m in box_order]

for i, var in enumerate(['Wind', 'Pres']):
    ax = axes[i]
    sub = plot_df[plot_df['Variable'] == var]

    sns.boxplot(
        data=sub,
        x='Forecast Hour',
        y='Absolute Error',
        hue='Model',
        hue_order=box_order,
        palette=palette_box,
        ax=ax,
        width=0.65,
        showfliers=False,
        linewidth=1.2
    )

    if var == 'Wind':
        ax.set_title('(a) Absolute Vmax error distribution', fontsize=22, fontweight='bold', pad=10)
        ax.set_ylabel(r'Absolute Vmax error (m s$^{-1}$)', fontsize=20)
    else:
        ax.set_title('(b) Absolute Pmin error distribution', fontsize=22, fontweight='bold', pad=10)
        ax.set_ylabel('Absolute Pmin error (hPa)', fontsize=20)

    ax.set_xlabel('Forecast hour (h)', fontsize=20)
    ax.tick_params(axis='both', labelsize=18)
    ax.grid(True, alpha=0.25)

    leg = ax.legend(
        loc='upper left',
        frameon=True,
        title=None,
        fontsize=18,
        handlelength=1.6,
        borderpad=0.5,
        labelspacing=0.4
    )

plt.tight_layout()
plt.savefig('Figure_boxplots.png', bbox_inches='tight')
plt.close()

print("Saved:")
print("  Figure_intensity_skill_trends.png")
print("  Figure_scatter.png")
print("  Figure_boxplots.png")
