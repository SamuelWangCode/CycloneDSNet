import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Global style
# =========================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.titlesize": 17,
    "axes.labelsize": 15,
    "xtick.labelsize": 11,
    "ytick.labelsize": 12,
    "legend.fontsize": 10.5,
    "figure.dpi": 220,
    "savefig.dpi": 400,
    "axes.linewidth": 1.0,
})

# =========================
# Paths and constants
# =========================
CASE_DIR = "./results/case_cache_full_24h"
CSV_PATH = os.path.join(CASE_DIR, "case_summary.csv")
OUT_DIR = CASE_DIR
RI_THRESHOLD = 15.433  # m s^-1

COLORS = {
    'CMA': 'black',
    'TianXing': '#4C78A8',
    'Reference': '#54A24B',
    'Deterministic': '#F58518',
    'Diffusion': '#B279A2',
    'RI': '#C44E52',
    'Bar': '#A8A8A8'
}


# =========================
# Helpers
# =========================
def sparse_xticks(ax, x, labels, max_ticks=7):
    n = len(labels)
    if n <= max_ticks:
        tick_idx = np.arange(n)
    else:
        step = int(np.ceil(n / max_ticks))
        tick_idx = np.arange(0, n, step)
        if tick_idx[-1] != n - 1:
            tick_idx = np.append(tick_idx, n - 1)

    # 如果最后两个 tick 太近，就删掉倒数第二个
    if len(tick_idx) >= 2 and (tick_idx[-1] - tick_idx[-2] <= 1):
        tick_idx = np.delete(tick_idx, -2)

    ax.set_xticks(tick_idx)
    ax.set_xticklabels([labels[i] for i in tick_idx], rotation=0, ha='center')


def load_case_summary(case_id='2411', hour=24):
    df = pd.read_csv(CSV_PATH, dtype={'ID': str, 'Init_Time': str})
    df['ID'] = df['ID'].str.zfill(4)
    df['Init_Time'] = pd.to_datetime(df['Init_Time'], format='%Y%m%d%H')
    df = df[(df['ID'] == case_id) & (df['Hour'] == hour)].copy()
    df = df.sort_values('Init_Time').reset_index(drop=True)
    return df


def load_npz_stats(case_id='2411', hour=24):
    pattern = os.path.join(CASE_DIR, f"{case_id}_*_{hour}h.npz")
    files = sorted(glob.glob(pattern))

    if len(files) == 0:
        raise FileNotFoundError(f"No npz files found with pattern: {pattern}")

    rows = []
    for fp in files:
        data = np.load(fp, allow_pickle=True)

        base = os.path.basename(fp)
        parts = base.replace(".npz", "").split("_")
        init_str = parts[1]

        vmax_members = data['vmax_members'].astype(float)

        rows.append({
            'Init_Time': pd.to_datetime(init_str, format='%Y%m%d%H'),
            'Diff_Median_Vmax': np.median(vmax_members),
            'Diff_P05_Vmax': np.percentile(vmax_members, 5),
            'Diff_P95_Vmax': np.percentile(vmax_members, 95),
        })

    return pd.DataFrame(rows).sort_values('Init_Time').reset_index(drop=True)


def build_case_dataframe():
    df_csv = load_case_summary(case_id='2411', hour=24)
    df_npz = load_npz_stats(case_id='2411', hour=24)
    sub = pd.merge(df_csv, df_npz, on='Init_Time', how='inner')
    sub = sub.sort_values('Init_Time').reset_index(drop=True)
    return sub


def plot_case_evolution(sub, mode='mean'):
    """
    mode='mean'   -> Diffusion mean + P10-P90
    mode='median' -> Diffusion median + P05-P95
    """
    assert mode in ['mean', 'median']

    x = np.arange(len(sub))
    labels = [t.strftime('%m-%d\n%H') for t in sub['Init_Time']]

    if mode == 'mean':
        diff_line = sub['Diff_Mean_Vmax'].values
        band_low = sub['Diff_P10_Vmax'].values
        band_high = sub['Diff_P90_Vmax'].values
        diff_label = 'CycloneDSNet-Diffusion mean intensity'
        band_label = 'CycloneDSNet-Diffusion 10th–90th percentile range'
        out_name = 'Figure_case_Yagi_mean_p10p90.png'
    else:
        diff_line = sub['Diff_Median_Vmax'].values
        band_low = sub['Diff_P05_Vmax'].values
        band_high = sub['Diff_P95_Vmax'].values
        diff_label = 'CycloneDSNet-Diffusion median'
        band_label = '5–95% range'
        out_name = 'Figure_case_Yagi_median_p05p95.png'

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10.8, 8.8),
        gridspec_kw={'height_ratios': [1.10, 0.95]}
    )

    fig.subplots_adjust(
        top=0.93,
        bottom=0.10,
        left=0.10,
        right=0.90,
        hspace=0.42
    )

    # =========================
    # Top panel: Vmax evolution
    # =========================
    l1, = ax1.plot(x, sub['CMA_Vmax_valid'].values, '-o', color=COLORS['CMA'],
                   linewidth=2.3, markersize=5.0, label='CMA truth', zorder=5)

    l2, = ax1.plot(x, sub['TX_Vmax'].values, '-o', color=COLORS['TianXing'],
                   linewidth=2.0, markersize=4.6, label='TianXing', zorder=4)

    l3, = ax1.plot(x, sub['REF_Vmax'].values, '-s', color=COLORS['Reference'],
                   linewidth=2.0, markersize=4.8, label='SHTM reference', zorder=4)

    l4, = ax1.plot(x, sub['DSNet_Vmax'].values, '-^', color=COLORS['Deterministic'],
                   linewidth=2.1, markersize=5.0, label='CycloneDSNet-Deterministic', zorder=4)

    l5, = ax1.plot(x, diff_line, '-D', color=COLORS['Diffusion'],
                   linewidth=2.3, markersize=4.8, label=diff_label, zorder=5)

    band = ax1.fill_between(x, band_low, band_high,
                            color=COLORS['Diffusion'], alpha=0.15, linewidth=0,
                            label=band_label, zorder=2)

    ax1.set_title('Yagi: Vmax evolution for 24 h lead time forecasts', fontweight='bold', pad=8)
    ax1.set_ylabel(r'Vmax (m s$^{-1}$)')
    ax1.set_xlim(-0.5, len(sub) - 0.5)
    ax1.grid(True, alpha=0.22)

    # 顶部图不显示 x tick label
    ax1.tick_params(axis='x', which='both', labelbottom=False)

    # =========================
    # Bottom panel: RI tendency + RI probability
    # =========================
    ax2b = ax2.twinx()

    b1 = ax2.bar(x, sub['Obs_Delta'].values, color=COLORS['Bar'],
                 alpha=0.50, width=0.66, label=r'Observed $\Delta V_{max}$', zorder=2)

    l6 = ax2.axhline(RI_THRESHOLD, color=COLORS['RI'], linestyle='--',
                     linewidth=2.0, label=r'RI threshold (15.4 m s$^{-1}$)', zorder=3)

    l7, = ax2b.plot(x, sub['RI_Prob'].values, '-o', color=COLORS['Diffusion'],
                    linewidth=2.4, markersize=5.0, label='CycloneDSNet-Diffusion RI probability', zorder=4)

    ax2.set_title('Yagi: observed 24-h intensification and diffusion-based RI probability',
                  fontweight='bold', pad=8)
    ax2.set_ylabel(r'Observed 24-h $\Delta$ Vmax (m s$^{-1}$)')
    ax2b.set_ylabel('RI probability')
    ax2.set_xlabel('Forecast initialization time')

    ymin = min(-45, float(np.nanmin(sub['Obs_Delta'].values)) - 2)
    ymax = max(30, float(np.nanmax(sub['Obs_Delta'].values)) + 2)
    ax2.set_ylim(ymin, ymax)
    ax2b.set_ylim(0, 1.0)

    ax2.grid(True, alpha=0.22)
    sparse_xticks(ax2, x, labels, max_ticks=7)

    # 下排图例也放外面
    ax2.legend(
        handles=[l6, b1, l7],
        labels=[r'RI threshold (15.4 m s$^{-1}$)', r'Observed $\Delta V_{max}$', 'Diffusion RI probability'],
        loc='upper left',
        bbox_to_anchor=(0.01, 0.98),
        ncol=1,
        frameon=True
    )

    fig.legend(
        handles=[l1, l2, l3, l4, l5, band],
        labels=['CMA truth', 'TianXing', 'SHTM reference',
                'CycloneDSNet-Deterministic', diff_label, band_label],
        loc='center',
        bbox_to_anchor=(0.5, 0.53),
        ncol=3,
        frameon=True,
        columnspacing=1.4,
        handlelength=2.0,
        handletextpad=0.5
    )

    out_path = os.path.join(OUT_DIR, out_name)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def main():
    sub = build_case_dataframe()
    plot_case_evolution(sub, mode='mean')
    plot_case_evolution(sub, mode='median')


if __name__ == '__main__':
    main()
