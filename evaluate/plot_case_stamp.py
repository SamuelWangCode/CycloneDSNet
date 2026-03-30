import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from scipy.stats import gaussian_kde

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14.0,
    "axes.titlesize": 14.5,
    "axes.labelsize": 14.0,
    "xtick.labelsize": 11.8,
    "ytick.labelsize": 11.8,
    "legend.fontsize": 12.2,
    "figure.dpi": 220,
    "savefig.dpi": 400,
})

CASE_NPZ = "./results/case_cache_full_24h/2411_2024090306_24h.npz"
OUT_FIG = "./results/case_cache_full_24h/Figure_case_postage_2411_2024090306.png"


def calc_rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def calc_ssim(ref, pred):
    data_range = float(max(ref.max(), pred.max()) - min(ref.min(), pred.min()))
    if data_range <= 0:
        data_range = 1.0
    return float(ssim(ref, pred, data_range=data_range))


def vmax_field(x):
    return float(np.max(x))


def delta_v(v, cma_vmax):
    return abs(float(v) - float(cma_vmax))


def add_textbox(ax, text, color='black'):
    ax.text(
        0.02, 0.98, text,
        transform=ax.transAxes,
        ha='left', va='top',
        fontsize=15.2, color=color,
        linespacing=1.2,
        bbox=dict(boxstyle='round,pad=0.26',
                  facecolor='white', alpha=0.86, edgecolor='0.7')
    )


def nearest_member_by_quantile(member_values, q):
    target = np.percentile(member_values, q)
    idx = int(np.argmin(np.abs(member_values - target)))
    return idx


def fmt_metrics(ref, img, cma_vmax, include_rmse=True, include_ssim=True, include_delta=True):
    lines = [f"Vmax={vmax_field(img):.1f}"]
    if include_ssim:
        lines.append(f"SSIM={calc_ssim(ref, img):.3f}")
    if include_rmse:
        lines.append(f"RMSE={calc_rmse(ref, img):.2f}")
    if include_delta:
        lines.append(f"|ΔV|={delta_v(vmax_field(img), cma_vmax):.2f}")
    return "\n".join(lines)


def main():
    data = np.load(CASE_NPZ, allow_pickle=True)

    storm_id = str(data["case_id"])
    storm_name = str(data["case_name"])
    init_time = str(data["init_time"])
    valid_time = str(data["valid_time"]) if "valid_time" in data.files else "unknown"
    lead = int(data["forecast_hour"])
    cma_vmax = float(data["cma_wind_valid"])
    ri_prob = float(data["ri_prob"])

    tx = data["tx_ws"]
    ref = data["ref_ws"]
    det = data["dsnet_ws"]
    ens = data["samples_ws"]  # [N, H, W]
    vmax_members = data["vmax_members"]

    diff_mean_field = np.mean(ens, axis=0)
    diff_mean_intensity = float(np.mean(vmax_members))

    # ---- member diagnostics ----
    ssim_scores = np.array([calc_ssim(ref, ens[i]) for i in range(ens.shape[0])])
    best_ssim_idx = int(np.argmax(ssim_scores))
    worst_ssim_idx = int(np.argmin(ssim_scores))

    int_diff = np.abs(vmax_members - cma_vmax)
    best_int_idx = int(np.argmin(int_diff))
    worst_int_idx = int(np.argmax(int_diff))

    q10_idx = nearest_member_by_quantile(vmax_members, 10)
    q30_idx = nearest_member_by_quantile(vmax_members, 30)
    q70_idx = nearest_member_by_quantile(vmax_members, 70)
    q90_idx = nearest_member_by_quantile(vmax_members, 90)

    best_ssim_member = ens[best_ssim_idx]
    worst_ssim_member = ens[worst_ssim_idx]
    best_int_member = ens[best_int_idx]
    worst_int_member = ens[worst_int_idx]
    q10_member = ens[q10_idx]
    q30_member = ens[q30_idx]
    q70_member = ens[q70_idx]
    q90_member = ens[q90_idx]

    # ---- plotting range ----
    all_main = np.concatenate([
        tx.ravel(), ref.ravel(), det.ravel(), diff_mean_field.ravel(),
        best_ssim_member.ravel(), best_int_member.ravel(),
        worst_ssim_member.ravel(), worst_int_member.ravel(),
        q10_member.ravel(), q30_member.ravel(), q70_member.ravel(), q90_member.ravel()
    ])
    vmin = 0.0
    vmax = max(np.percentile(all_main, 99.8), cma_vmax)

    fig = plt.figure(figsize=(23.5, 14.6))
    gs = fig.add_gridspec(
        5, 6,
        height_ratios=[1.08, 1.08, 0.08, 0.10, 0.92],
        hspace=0.16, wspace=0.06
    )
    fig.subplots_adjust(left=0.035, right=0.987, top=0.925, bottom=0.075)

    # ---------- first two rows ----------
    panels = [
        (tx, "TianXing input"),
        (ref, "SHTM reference"),
        (det, "CycloneDSNet-\nDeterministic"),
        (diff_mean_field, "Diffusion ensemble\nmean field"),
        (best_ssim_member, "Best SSIM member"),
        (best_int_member, "Best intensity member"),

        (worst_ssim_member, "Worst SSIM member"),
        (worst_int_member, "Worst intensity member"),
        (q10_member, "P10 member"),
        (q30_member, "P30 member"),
        (q70_member, "P70 member"),
        (q90_member, "P90 member"),
    ]

    axes = []
    ims = []
    for i, (img, title) in enumerate(panels):
        r = 0 if i < 6 else 1
        c = i % 6
        ax = fig.add_subplot(gs[r, c])
        im = ax.imshow(img, cmap="turbo", origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=16.0, fontweight='bold', pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        axes.append(ax)
        ims.append(im)

    # ---------- metric boxes ----------
    add_textbox(axes[0], fmt_metrics(ref, tx, cma_vmax, include_ssim=False, include_rmse=False))
    add_textbox(axes[1], fmt_metrics(ref, ref, cma_vmax, include_ssim=False, include_rmse=False))
    add_textbox(axes[2], fmt_metrics(ref, det, cma_vmax))

    add_textbox(
        axes[3],
        f"Vmax(field)={vmax_field(diff_mean_field):.1f}\n"
        f"mean Vmax(ens)={diff_mean_intensity:.1f}\n"
        f"SSIM={calc_ssim(ref, diff_mean_field):.3f}\n"
        f"RMSE={calc_rmse(ref, diff_mean_field):.2f}\n"
        f"|ΔV|={delta_v(diff_mean_intensity, cma_vmax):.2f}"
    )

    add_textbox(axes[4], fmt_metrics(ref, best_ssim_member, cma_vmax))
    add_textbox(axes[5], fmt_metrics(ref, best_int_member, cma_vmax), color='darkred')

    add_textbox(axes[6], fmt_metrics(ref, worst_ssim_member, cma_vmax))
    add_textbox(axes[7], fmt_metrics(ref, worst_int_member, cma_vmax))
    add_textbox(axes[8], fmt_metrics(ref, q10_member, cma_vmax))
    add_textbox(axes[9], fmt_metrics(ref, q30_member, cma_vmax))
    add_textbox(axes[10], fmt_metrics(ref, q70_member, cma_vmax))
    add_textbox(axes[11], fmt_metrics(ref, q90_member, cma_vmax))

    # ---------- dedicated colorbar row ----------
    cax = fig.add_subplot(gs[2, 1:5])
    cbar = fig.colorbar(ims[0], cax=cax, orientation='horizontal')
    cbar.set_label(r"10-m wind speed (m s$^{-1}$)", fontsize=15, labelpad=6)  # 不写label，避免继续打架
    cbar.ax.tick_params(labelsize=14, pad=1.5)
    spacer_ax = fig.add_subplot(gs[3, :])
    spacer_ax.axis("off")
    # ---------- bottom PDF panel ----------
    ax_pdf = fig.add_subplot(gs[4, :])

    x_min = min(np.min(vmax_members), vmax_field(tx), vmax_field(ref), vmax_field(det),
                diff_mean_intensity, cma_vmax) - 3
    x_max = max(np.max(vmax_members), vmax_field(tx), vmax_field(ref), vmax_field(det),
                diff_mean_intensity, cma_vmax) + 3
    xs = np.linspace(x_min, x_max, 400)

    kde = gaussian_kde(vmax_members)
    ys = kde(xs)

    ax_pdf.fill_between(xs, ys, color="#C084C1", alpha=0.35,
                        label="CycloneDSNet-Diffusion ensemble-member Vmax distribution")
    ax_pdf.plot(xs, ys, color="#B279A2", linewidth=2.0)

    ax_pdf.axvline(cma_vmax, color='red', linestyle='-', linewidth=2.2,
                   label=f"CMA truth ({cma_vmax:.1f})")
    ax_pdf.axvline(vmax_field(ref), color='#54A24B', linestyle='--', linewidth=2.0,
                   label=f"SHTM reference ({vmax_field(ref):.1f})")
    ax_pdf.axvline(vmax_field(tx), color='#4C78A8', linestyle=':', linewidth=2.0,
                   label=f"TianXing ({vmax_field(tx):.1f})")
    ax_pdf.axvline(vmax_field(det), color='#F58518', linestyle='-.', linewidth=2.0,
                   label=f"CycloneDSNet-Deterministic ({vmax_field(det):.1f})")
    ax_pdf.axvline(diff_mean_intensity, color='#B279A2', linestyle='-.', linewidth=2.0,
                   label=f"CycloneDSNet-Diffusion ensemble-mean intensity ({diff_mean_intensity:.1f})")

    ax_pdf.set_title("Distribution of diffusion-ensemble-member Vmax",
                     fontsize=17, fontweight='bold', pad=7)
    ax_pdf.set_xlabel(r"Vmax (m s$^{-1}$)", fontsize=15.5)
    ax_pdf.set_ylabel("Density", fontsize=15.5)
    ax_pdf.tick_params(axis='both', labelsize=14)
    ax_pdf.grid(True, alpha=0.25)
    ax_pdf.legend(
        loc='upper left',
        ncol=1,
        frameon=True,
        fontsize=14.5,
        handlelength=2.5,
        borderpad=0.45,
        labelspacing=0.40
    )

    fig.suptitle(
        f"{storm_name} ({storm_id}), init={init_time}, valid={valid_time}, "
        f"lead time= {lead} h, CMA Vmax={cma_vmax:.1f} m s$^{{-1}}$, "
        f"RI probability={ri_prob:.2f}",
        fontsize=19.5, fontweight='bold', y=0.989
    )

    plt.savefig(OUT_FIG, bbox_inches='tight', dpi=600)
    plt.close()
    print(f"Saved: {OUT_FIG}")


if __name__ == "__main__":
    main()
