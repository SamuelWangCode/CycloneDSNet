import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

# =========================
# 0. Global style
# =========================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "axes.titlesize": 20,
    "axes.labelsize": 17,
    "xtick.labelsize": 12,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "figure.dpi": 400,
    "savefig.dpi": 450,
    "axes.linewidth": 1.1,
    "mathtext.default": "regular"
})

# =========================
# 1. Config
# =========================
RI_THRESHOLD = 15.433  # 30 kt in m s^-1
YEAR_FILTER = 2024

STATS_FILE_PRIMARY  = '/bigdata3/WangGuanSong/Weaformer/all_models/weaformer_v2.0/typhoon_intensity_bc/results/Full_Test_24h/statistics_24h.csv'
STATS_FILE_ADDITIONAL = '/bigdata3/WangGuanSong/Weaformer/all_models/weaformer_v2.0/typhoon_intensity_bc/results/Full_Test_24h/statistics_24h_additional.csv'
TYPHOONS_CSV = './data_file/typhoons.csv'

OUT_FIG = 'Figure7_RI_2024.png'
OUT_TABLE = 'Table3_RI_metrics_2024.csv'

# =========================
# 2. Unified naming / colors
# =========================
MODEL_NAME_MAP = {
    'TX': 'TianXing',
    'WRF': 'SHTM',
    'DSNet': 'CycloneDSNet-Deterministic',
    'Diff_Mean': 'CycloneDSNet-Diffusion'
}

COLORS = {
    'TX': '#4C78A8',  # TianXing blue
    'WRF': '#54A24B',  # SHTM green
    'DSNet': '#F58518',  # Deterministic orange
    'Diff_Mean': '#B279A2'  # Diffusion purple
}

MARKERS = {
    'TX': 'x',
    'WRF': '^',
    'DSNet': 's',
    'Diff_Mean': 'o'
}

SIZES = {
    'TX': 70,
    'WRF': 70,
    'DSNet': 70,
    'Diff_Mean': 95
}


# =========================
# 3. Helpers
# =========================
def parse_date(x):
    return datetime.strptime(str(x), '%Y%m%d%H')


def load_and_merge():
    dfs = []

    if os.path.exists(STATS_FILE_PRIMARY):
        dfs.append(pd.read_csv(STATS_FILE_PRIMARY))

    if os.path.exists(STATS_FILE_ADDITIONAL):
        df_additional = pd.read_csv(STATS_FILE_ADDITIONAL)
        if 'Start Date' in df_miss.columns:
            df_miss = df_miss.rename(columns={'Start Date': 'Time'})
        dfs.append(df_miss)

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    df['ID'] = df['ID'].apply(lambda x: str(int(x)).zfill(4))
    df['Init_Time'] = df['Time'].apply(parse_date)

    # Remove duplicate forecast entries after merging multiple result files
    df = df.drop_duplicates(subset=['ID', 'Init_Time'], keep='last')
    return df


def compute_ri_metrics(df_all, models):
    df_ri_obs = df_all[df_all['Obs_RI']].copy()

    records = []
    records.append({
        'Model': 'Obs',
        'Mean Delta (m s^-1)': round(df_ri_obs['Obs_Delta'].mean(), 2),
        'MAE (m s^-1)': np.nan,
        'POD': 1.0,
        'FAR': 0.0,
        'CSI': 1.0
    })

    for m_key, m_name in models.items():
        valid_ri_deltas = df_ri_obs[f"{m_key}_Delta"].dropna()

        if len(valid_ri_deltas) > 0:
            mean_d = valid_ri_deltas.mean()
            mae = (valid_ri_deltas - df_ri_obs.loc[valid_ri_deltas.index, 'Obs_Delta']).abs().mean()
        else:
            mean_d = np.nan
            mae = np.nan

        tp = ((df_all['Obs_RI']) & (df_all[f'{m_key}_Hit_RI'])).sum()
        fn = ((df_all['Obs_RI']) & (~df_all[f'{m_key}_Hit_RI'])).sum()
        fp = ((~df_all['Obs_RI']) & (df_all[f'{m_key}_Hit_RI'])).sum()

        pod = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        far = fp / (tp + fp) if (tp + fp) > 0 else 0.0
        csi = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0.0

        records.append({
            'Model': m_name,
            'Mean Delta (m s^-1)': round(mean_d, 2) if not np.isnan(mean_d) else np.nan,
            'MAE (m s^-1)': round(mae, 2) if not np.isnan(mae) else np.nan,
            'POD': round(pod, 3),
            'FAR': round(far, 3),
            'CSI': round(csi, 3)
        })

    return pd.DataFrame(records)


def plot_ri_scatter(df, models):
    df_plot = df[df['Obs_RI']].copy()
    if df_plot.empty:
        print("No observed RI cases found.")
        return

    df_plot = df_plot.sort_values('Obs_Delta').reset_index(drop=True)
    x = np.arange(len(df_plot))

    fig, ax = plt.subplots(figsize=(16, 9))

    # Observed truth
    ax.plot(
        x, df_plot['Obs_Delta'],
        linestyle='None',
        marker='*',
        markersize=15,
        color='black',
        label='CMA best-track records',
        zorder=10
    )

    # Model predictions
    for m_key, m_name in models.items():
        delta_vals = df_plot[f'{m_key}_Delta'].values
        mask = ~np.isnan(delta_vals)

        ax.scatter(
            x[mask],
            delta_vals[mask],
            label=m_name,
            color=COLORS[m_key],
            marker=MARKERS[m_key],
            s=SIZES[m_key],
            alpha=0.80,
            edgecolors='none'
        )

    # RI threshold
    ax.axhline(
        y=RI_THRESHOLD,
        color='red',
        linestyle='--',
        linewidth=2.2,
        label=fr'RI threshold ({RI_THRESHOLD:.1f} m s$^{{-1}}$)'
    )

    ax.set_title(
        'Observed 2024 RI cases and model-predicted 24-h Vmax changes',
        pad=10,
        fontweight='bold'
    )
    ax.set_ylabel(r'24-h $\Delta$ Vmax from observed initial intensity (m s$^{-1}$)')
    ax.set_xlabel(r'Observed RI case rank (ordered by observed 24-h $\Delta$ Vmax)')
    ax.set_xticks([])

    ax.grid(True, linestyle='--', alpha=0.35)
    ax.tick_params(axis='both', labelsize=15)
    # legend
    ax.legend(
        loc='lower right',
        frameon=True,
        ncol=2,
        fontsize=15,
        markerscale=1.15,
        handlelength=1.8,
        borderpad=0.6,
        labelspacing=0.5,
        columnspacing=1.2
    )

    plt.tight_layout()
    plt.savefig(OUT_FIG, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {OUT_FIG}")


# =========================
# 4. Main
# =========================
def main():
    print("Running RI verification (POD / FAR / CSI) ...")

    # 1) Observations
    df_obs = pd.read_csv(TYPHOONS_CSV)
    df_obs.columns = [c.strip() for c in df_obs.columns]
    df_obs['Parsed_Date'] = df_obs['Date'].apply(parse_date)
    df_obs['ID'] = df_obs['ID'].apply(lambda x: str(int(x)).zfill(4))

    obs_dict = dict(zip(zip(df_obs['ID'], df_obs['Parsed_Date']), df_obs['Wind Speed']))

    # 2) Predictions
    df_pred = load_and_merge()
    df_pred = df_pred[df_pred['Init_Time'].dt.year == YEAR_FILTER].copy()
    print(f"Total forecast samples in {YEAR_FILTER}: {len(df_pred)}")

    models = {
        'TX': 'TianXing',
        'WRF': 'SHTM',
        'DSNet': 'CycloneDSNet-Deterministic',
        'Diff_Mean': 'CycloneDSNet-Diffusion'
    }

    # 3) Match full samples
    results = []

    for _, row in df_pred.iterrows():
        tid, t0 = row['ID'], row['Init_Time']
        t24 = t0 + timedelta(hours=24)

        if (tid, t0) not in obs_dict or (tid, t24) not in obs_dict:
            continue

        obs_t0 = obs_dict[(tid, t0)]
        obs_t24 = obs_dict[(tid, t24)]
        delta_obs = obs_t24 - obs_t0
        is_obs_ri = delta_obs >= RI_THRESHOLD

        entry = {
            'ID': tid,
            'Time': t0,
            'Obs_Delta': delta_obs,
            'Obs_RI': is_obs_ri
        }

        for m_key in models:
            val = row[f"{m_key}_Wind"]

            if val <= 0.1 or val == -999:
                entry[f"{m_key}_Delta"] = np.nan
                entry[f"{m_key}_Hit_RI"] = False
            else:
                d_pred = val - obs_t0
                entry[f"{m_key}_Delta"] = d_pred
                entry[f"{m_key}_Hit_RI"] = (d_pred >= RI_THRESHOLD)

        results.append(entry)

    df_all = pd.DataFrame(results)

    if len(df_all) == 0:
        print("No matched samples found.")
        return

    df_ri_obs = df_all[df_all['Obs_RI']].copy()
    print(f"Observed RI cases: {len(df_ri_obs)}")

    # 4) Metrics table
    metrics_df = compute_ri_metrics(df_all, models)
    metrics_df.to_csv(OUT_TABLE, index=False)
    print(f"Saved table: {OUT_TABLE}")

    print("\nRI metrics summary:")
    print(metrics_df.to_string(index=False))

    # 5) Plot
    plot_ri_scatter(df_all, models)


if __name__ == '__main__':
    main()
