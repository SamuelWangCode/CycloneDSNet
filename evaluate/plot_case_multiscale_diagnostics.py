import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 11.5,
    "ytick.labelsize": 11.5,
    "legend.fontsize": 10.8,
    "figure.dpi": 220,
    "savefig.dpi": 400,
})

# =========================
# USER CONFIG
# =========================
CASE_NPZ = "./results/case_cache_full_24h_uv_t2m/2411_2024090306_24h.npz"
OUT_FIG = "./results/case_cache_full_24h_uv_t2m/Figure_case_multiscale_diagnostics.png"

DX_KM = 11.1
RADIAL_BIN_KM = 25.0
PDF_BINS = np.linspace(0, 55, 56)


def resize_to_shape(arr, out_shape):
    x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    y = F.interpolate(x, size=out_shape, mode="bilinear", align_corners=False)
    return y[0, 0].cpu().numpy()


def wind_speed(u, v):
    return np.sqrt(u ** 2 + v ** 2)


def find_center_from_mslp(mslp):
    return np.unravel_index(np.argmin(mslp), mslp.shape)


def radial_distance_km(shape, center, dx_km=11.1):
    h, w = shape
    cy, cx = center
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) * dx_km
    return rr


def radial_profile(field, center, dx_km=11.1, bin_km=25.0, max_km=None):
    rr = radial_distance_km(field.shape, center, dx_km=dx_km)
    if max_km is None:
        max_km = rr.max()

    bins = np.arange(0, max_km + bin_km, bin_km)
    centers = 0.5 * (bins[:-1] + bins[1:])
    prof = np.full(len(centers), np.nan)

    for i in range(len(centers)):
        mask = (rr >= bins[i]) & (rr < bins[i + 1])
        if np.any(mask):
            prof[i] = np.mean(field[mask])

    return centers, prof


def histogram_pdf(values, bins):
    hist, edges = np.histogram(values.ravel(), bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist = np.maximum(hist, 1e-6)
    return centers, hist


def isotropic_spectrum_2d(field, dx_km=11.1):
    f = field - np.mean(field)
    F2 = np.fft.fft2(f)
    P2 = np.abs(F2) ** 2

    ny, nx = field.shape
    ky = np.fft.fftfreq(ny, d=dx_km)
    kx = np.fft.fftfreq(nx, d=dx_km)
    KX, KY = np.meshgrid(kx, ky)
    KR = np.sqrt(KX ** 2 + KY ** 2)

    kmax = KR.max()
    nbins = min(nx, ny) // 2
    kbins = np.linspace(0, kmax, nbins + 1)
    kcent = 0.5 * (kbins[:-1] + kbins[1:])
    spec = np.full(nbins, np.nan)

    for i in range(nbins):
        mask = (KR >= kbins[i]) & (KR < kbins[i + 1])
        if np.any(mask):
            spec[i] = np.mean(P2[mask])

    valid = np.isfinite(spec) & (kcent > 0)
    return kcent[valid], spec[valid]


def kinetic_energy_spectrum(u, v, dx_km=11.1):
    ku, Eu = isotropic_spectrum_2d(u, dx_km=dx_km)
    kv, Ev = isotropic_spectrum_2d(v, dx_km=dx_km)
    n = min(len(ku), len(kv))
    k = ku[:n]
    E = 0.5 * (Eu[:n] + Ev[:n])
    return k, E


def member_center_curve_stats(curves):
    arr = np.stack(curves, axis=0)
    median = np.nanmedian(arr, axis=0)
    p10 = np.nanpercentile(arr, 10, axis=0)
    p90 = np.nanpercentile(arr, 90, axis=0)
    mean = np.nanmean(arr, axis=0)
    return mean, median, p10, p90


def closest_member_idx(vmax_members, cma_vmax):
    return int(np.argmin(np.abs(vmax_members - cma_vmax)))


def main():
    data = np.load(CASE_NPZ, allow_pickle=True)

    required = [
        "tx_u10", "tx_v10", "tx_t2m", "tx_mslp",
        "ref_u10", "ref_v10", "ref_t2m", "ref_mslp",
        "dsnet_u10", "dsnet_v10", "dsnet_t2m", "dsnet_mslp",
        "samples_u10", "samples_v10", "samples_t2m", "samples_mslp",
        "cma_wind_valid", "case_id", "case_name", "init_time", "valid_time",
        "forecast_hour", "ri_prob"
    ]
    for k in required:
        if k not in data.files:
            raise KeyError(f"Missing key: {k}")

    storm_id = str(data["case_id"])
    storm_name = str(data["case_name"])
    init_time = str(data["init_time"])
    valid_time = str(data["valid_time"])
    lead = int(data["forecast_hour"])
    cma_vmax = float(data["cma_wind_valid"])
    ri_prob = float(data["ri_prob"])

    ref_u = data["ref_u10"]
    ref_v = data["ref_v10"]
    ref_t2m = data["ref_t2m"]
    ref_mslp = data["ref_mslp"]
    out_shape = ref_u.shape

    # interpolate TianXing to 0.1° grid
    tx_u = resize_to_shape(data["tx_u10"], out_shape)
    tx_v = resize_to_shape(data["tx_v10"], out_shape)
    tx_t2m = resize_to_shape(data["tx_t2m"], out_shape)
    tx_mslp = resize_to_shape(data["tx_mslp"], out_shape)

    ds_u = data["dsnet_u10"]
    ds_v = data["dsnet_v10"]
    ds_t2m = data["dsnet_t2m"]
    ds_mslp = data["dsnet_mslp"]

    smp_u = data["samples_u10"]
    smp_v = data["samples_v10"]
    smp_t2m = data["samples_t2m"]
    smp_mslp = data["samples_mslp"]

    tx_ws = wind_speed(tx_u, tx_v)
    ref_ws = wind_speed(ref_u, ref_v)
    ds_ws = wind_speed(ds_u, ds_v)
    smp_ws = wind_speed(smp_u, smp_v)

    vmax_members = np.max(smp_ws, axis=(1, 2))
    best_idx = closest_member_idx(vmax_members, cma_vmax)

    best_u = smp_u[best_idx]
    best_v = smp_v[best_idx]
    best_t2m = smp_t2m[best_idx]
    best_mslp = smp_mslp[best_idx]
    best_ws = smp_ws[best_idx]

    ctr_tx = find_center_from_mslp(tx_mslp)
    ctr_ref = find_center_from_mslp(ref_mslp)
    ctr_ds = find_center_from_mslp(ds_mslp)
    ctr_best = find_center_from_mslp(best_mslp)
    ctr_members = [find_center_from_mslp(smp_mslp[i]) for i in range(smp_mslp.shape[0])]

    # radial profiles
    r_ref, prof_ref = radial_profile(ref_ws, ctr_ref, dx_km=DX_KM, bin_km=RADIAL_BIN_KM)
    r_tx, prof_tx = radial_profile(tx_ws, ctr_tx, dx_km=DX_KM, bin_km=RADIAL_BIN_KM,
                                   max_km=r_ref[-1] + RADIAL_BIN_KM / 2)
    r_ds, prof_ds = radial_profile(ds_ws, ctr_ds, dx_km=DX_KM, bin_km=RADIAL_BIN_KM,
                                   max_km=r_ref[-1] + RADIAL_BIN_KM / 2)
    r_best, prof_best = radial_profile(best_ws, ctr_best, dx_km=DX_KM, bin_km=RADIAL_BIN_KM,
                                       max_km=r_ref[-1] + RADIAL_BIN_KM / 2)

    member_profiles = []
    for i in range(smp_ws.shape[0]):
        _, p = radial_profile(
            smp_ws[i], ctr_members[i],
            dx_km=DX_KM, bin_km=RADIAL_BIN_KM,
            max_km=r_ref[-1] + RADIAL_BIN_KM / 2
        )
        member_profiles.append(p)
    prof_mean, prof_median, prof_p10, prof_p90 = member_center_curve_stats(member_profiles)

    # pdf
    x_pdf_tx, pdf_tx = histogram_pdf(tx_ws, PDF_BINS)
    x_pdf_ref, pdf_ref = histogram_pdf(ref_ws, PDF_BINS)
    x_pdf_ds, pdf_ds = histogram_pdf(ds_ws, PDF_BINS)
    x_pdf_best, pdf_best = histogram_pdf(best_ws, PDF_BINS)

    member_pdfs = []
    for i in range(smp_ws.shape[0]):
        _, pdf_i = histogram_pdf(smp_ws[i], PDF_BINS)
        member_pdfs.append(pdf_i)
    pdf_mean, pdf_median, pdf_p10, pdf_p90 = member_center_curve_stats(member_pdfs)

    # KE spectrum
    k_tx, ke_tx = kinetic_energy_spectrum(tx_u, tx_v, dx_km=DX_KM)
    k_ref, ke_ref = kinetic_energy_spectrum(ref_u, ref_v, dx_km=DX_KM)
    k_ds, ke_ds = kinetic_energy_spectrum(ds_u, ds_v, dx_km=DX_KM)
    k_best, ke_best = kinetic_energy_spectrum(best_u, best_v, dx_km=DX_KM)

    member_ke = []
    for i in range(smp_u.shape[0]):
        _, ke_i = kinetic_energy_spectrum(smp_u[i], smp_v[i], dx_km=DX_KM)
        member_ke.append(ke_i)
    min_len_ke = min(len(x) for x in member_ke + [ke_tx, ke_ref, ke_ds, ke_best])
    k_common_ke = k_ref[:min_len_ke]
    member_ke = [x[:min_len_ke] for x in member_ke]
    ke_mean, ke_median, ke_p10, ke_p90 = member_center_curve_stats(member_ke)
    ke_tx = ke_tx[:min_len_ke]
    ke_ref = ke_ref[:min_len_ke]
    ke_ds = ke_ds[:min_len_ke]
    ke_best = ke_best[:min_len_ke]

    # t2m spectrum
    k_tx_t, te_tx = isotropic_spectrum_2d(tx_t2m, dx_km=DX_KM)
    k_ref_t, te_ref = isotropic_spectrum_2d(ref_t2m, dx_km=DX_KM)
    k_ds_t, te_ds = isotropic_spectrum_2d(ds_t2m, dx_km=DX_KM)
    k_best_t, te_best = isotropic_spectrum_2d(best_t2m, dx_km=DX_KM)

    member_te = []
    for i in range(smp_t2m.shape[0]):
        _, te_i = isotropic_spectrum_2d(smp_t2m[i], dx_km=DX_KM)
        member_te.append(te_i)
    min_len_te = min(len(x) for x in member_te + [te_tx, te_ref, te_ds, te_best])
    k_common_te = k_ref_t[:min_len_te]
    member_te = [x[:min_len_te] for x in member_te]
    te_mean, te_median, te_p10, te_p90 = member_center_curve_stats(member_te)
    te_tx = te_tx[:min_len_te]
    te_ref = te_ref[:min_len_te]
    te_ds = te_ds[:min_len_te]
    te_best = te_best[:min_len_te]

    colors = {
        "tx": "#4C78A8",
        "ref": "#54A24B",
        "ds": "#F58518",
        "best": "#E45756",
        "ens": "#B279A2",
    }

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 10.2))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.90, bottom=0.08, wspace=0.26, hspace=0.28)

    # (a) radial profile: median + 10–90 band
    ax = axes[0, 0]
    ax.plot(r_tx, prof_tx, color=colors["tx"], linewidth=2.0, label="TianXing (interpolated)")
    ax.plot(r_ref, prof_ref, color=colors["ref"], linewidth=2.0, label="SHTM")
    ax.plot(r_ds, prof_ds, color=colors["ds"], linewidth=2.0, label="CycloneDSNet-Deterministic")
    ax.plot(r_best, prof_best, color=colors["best"], linewidth=2.0, label="Best-intensity member")
    ax.plot(r_ref, prof_median, color=colors["ens"], linewidth=2.2, label="CycloneDSNet-Diffusion ensemble mean")
    ax.fill_between(r_ref, prof_p10, prof_p90, color=colors["ens"], alpha=0.18, label="Diffusion 10th–90th percentile range")
    ax.set_title("(a) Radial profile of 10-m wind speed", fontweight='bold')
    ax.set_xlabel("Distance from TC center (km)")
    ax.set_ylabel(r"Wind speed (m s$^{-1}$)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, fontsize=10)

    # (b) PDF: median + 10–90 band
    ax = axes[0, 1]
    ax.plot(x_pdf_tx, pdf_tx, color=colors["tx"], linewidth=2.0, label="TianXing (interpolated)")
    ax.plot(x_pdf_ref, pdf_ref, color=colors["ref"], linewidth=2.0, label="SHTM")
    ax.plot(x_pdf_ds, pdf_ds, color=colors["ds"], linewidth=2.0, label="CycloneDSNet-Deterministic")
    ax.plot(x_pdf_best, pdf_best, color=colors["best"], linewidth=2.0, label="Best-intensity member")
    ax.plot(x_pdf_ref, pdf_median, color=colors["ens"], linewidth=2.2, label="CycloneDSNet-Diffusion ensemble mean")
    ax.fill_between(x_pdf_ref, pdf_p10, pdf_p90, color=colors["ens"], alpha=0.18, label="Diffusion 10th–90th percentile range")
    ax.set_yscale("log")
    ax.set_title("(b) Probability density of 10-m wind speed", fontweight='bold')
    ax.set_xlabel(r"10-m wind speed (m s$^{-1}$)")
    ax.set_ylabel("PDF")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, fontsize=10, loc="lower left")

    # (c) KE spectrum: no band
    ax = axes[1, 0]
    ax.plot(k_tx, ke_tx, color=colors["tx"], linewidth=2.0, label="TianXing (interpolated)")
    ax.plot(k_common_ke, ke_ref, color=colors["ref"], linewidth=2.0, label="SHTM")
    ax.plot(k_common_ke, ke_ds, color=colors["ds"], linewidth=2.0, label="CycloneDSNet-Deterministic")
    ax.plot(k_common_ke, ke_best, color=colors["best"], linewidth=2.0, label="Best-intensity member")
    ax.plot(k_common_ke, ke_mean, color=colors["ens"], linewidth=2.2, label="CycloneDSNet-Diffusion member mean")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("(c) Kinetic energy spectrum", fontweight='bold')
    ax.set_xlabel(r"Wavenumber (cycles km$^{-1}$)")
    ax.set_ylabel("Kinetic energy")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, fontsize=10, loc="lower left")

    # (d) t2m spectrum: no band
    ax = axes[1, 1]
    ax.plot(k_tx_t, te_tx, color=colors["tx"], linewidth=2.0, label="TianXing (interpolated)")
    ax.plot(k_common_te, te_ref, color=colors["ref"], linewidth=2.0, label="SHTM")
    ax.plot(k_common_te, te_ds, color=colors["ds"], linewidth=2.0, label="CycloneDSNet-Deterministic")
    ax.plot(k_common_te, te_best, color=colors["best"], linewidth=2.0, label="Best-intensity member")
    ax.plot(k_common_te, te_mean, color=colors["ens"], linewidth=2.2, label="CycloneDSNet-Diffusion member mean")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("(d) 2-m temperature power spectrum", fontweight='bold')
    ax.set_xlabel(r"Wavenumber (cycles km$^{-1}$)")
    ax.set_ylabel("Temperature power")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, fontsize=10, loc="lower left")

    fig.suptitle(
        f"{storm_name} ({storm_id}) | init={init_time} | valid={valid_time} | lead time = {lead} h | "
        f"CMA Vmax={cma_vmax:.1f} m s$^{{-1}}$ | RI probability={ri_prob:.2f}",
        fontsize=16.5, fontweight='bold', y=0.97
    )

    plt.savefig(OUT_FIG, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUT_FIG}")


if __name__ == "__main__":
    main()
