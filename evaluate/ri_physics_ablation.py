import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import tqdm

from typhoon_intensity_bc.model.dsnet import DSNet
from typhoon_intensity_bc.model.diffusion_unet import DenoiseUNet
from typhoon_intensity_bc.project.construct_dataset_ds import get_dataloader_ds, DIM_CONFIG
from typhoon_intensity_bc.project.construct_dataset import FieldNormalizer, MultiChannelNormalizer

# =========================
# Config
# =========================
GPU_ID = 0
HOUR = 24
N_MEMBERS = 20
MAX_RI_CASES = 30
MAX_NONRI_CASES = 30
RI_THRESHOLD = 15.433

TEST_CSV = "./data_file/forecast_24h_test_set.csv"
TYPHOONS_CSV = "./data_file/typhoons.csv"
SAVE_DIR = "./results/ri_physics_ablation"
os.makedirs(SAVE_DIR, exist_ok=True)

ROOT_FIELD = '/bigdata4/wxz_data/typhoon_intensity_bc/field_data_extraction'
ROOT_VALUE = '/data4/wxz_data/typhoon_intensity_bc/value_data_extraction'
ROOT_TRACK = '/data4/wxz_data/typhoon_intensity_bc/track_forecast_data'

DSNET_CKPT = "./data_file/DSNet/typhoon_downscale_24h/dsnet-24h-ft-epoch=65-val_loss=1.6689.ckpt"
DIFF_CKPT = "./data_file/Diffusion/typhoon_diff_24h/diff-24h-epoch=822-val_loss=0.0058.ckpt"

# =========================
# Channel names
# =========================
LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
SFC_NAMES = ["u10", "v10", "u100", "v100", "t2m", "sp", "mslp", "tcwv"]
PL_VARS = ["u", "v", "z", "t", "r"]

CHANNEL_NAMES = SFC_NAMES.copy()
for var in PL_VARS:
    for lev in LEVELS:
        CHANNEL_NAMES.append(f"{var}{lev}")

name_to_idx = {n: i for i, n in enumerate(CHANNEL_NAMES)}

CHANNEL_GROUPS = {
    "thermodynamic_support": ["t2m", "tcwv", "r700", "r850", "r925"],
    "low_level_dynamics": ["u850", "v850", "u925", "v925", "mslp", "sp"],
    "upper_level_outflow": ["u200", "v200", "u250", "v250", "z200", "z250"],
    "midlevel_structure": ["z500", "t500", "r500", "z700", "t700", "r700"],
}

TC_VECTOR_GROUPS = {
    "track_history": [0, 1],  # lat, lon
    "intensity_history": [2, 3],  # vmax, pmin
}


def parse_date(x):
    return datetime.strptime(str(x), "%Y%m%d%H")


def build_obs_lookup():
    df = pd.read_csv(TYPHOONS_CSV)
    df.columns = [c.strip() for c in df.columns]
    df["ID"] = df["ID"].apply(lambda x: str(int(x)).zfill(4))
    df["Parsed_Date"] = df["Date"].apply(parse_date)
    lookup = {}
    for _, row in df.iterrows():
        lookup[(row["ID"], row["Parsed_Date"])] = float(row["Wind Speed"])
    return lookup


def extract_norm_stats(gpu_id):
    stats = np.load('./data_file/stats/hr_minmax_stats.npz')
    vmin = torch.tensor(stats['vmin'], dtype=torch.float32).view(1, -1, 1, 1).cuda(gpu_id)
    vmax = torch.tensor(stats['vmax'], dtype=torch.float32).view(1, -1, 1, 1).cuda(gpu_id)
    return {'mean': vmin, 'std': vmax - vmin}


def denormalize(tensor, stats):
    return tensor * stats['std'] + stats['mean']


def vmax_from_field(field):
    ws = torch.sqrt(field[:, 0] ** 2 + field[:, 1] ** 2)
    return ws.view(ws.shape[0], -1).max(dim=1).values


class DDIMSampler:
    def __init__(self, model, device='cuda', timesteps=1000):
        self.model = model
        self.device = device
        self.num_train_timesteps = timesteps
        self.beta = torch.linspace(1e-4, 0.02, self.num_train_timesteps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.model.eval()

    @torch.no_grad()
    def sample(self, cond, shape, ddim_steps=50, eta=0.0, seed=0):
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)

        batch_size = shape[0]
        times = torch.linspace(0, self.num_train_timesteps - 1, steps=ddim_steps + 1).long().to(self.device)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        img = torch.randn(shape, device=self.device, generator=g)

        for t, t_prev in time_pairs:
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            model_input = torch.cat([img, cond], dim=1)
            eps = self.model(model_input, t_tensor)

            alpha_bar_t = self.alpha_bar[t]
            alpha_bar_prev = self.alpha_bar[t_prev] if t_prev >= 0 else torch.tensor(1.0).to(self.device)

            pred_x0 = (img - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
            sigma_t = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev))
            dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma_t ** 2) * eps
            noise = 0.0

            img = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt + sigma_t * noise

        return img


def build_loader():
    field_norm = FieldNormalizer(num_channels=73)
    field_norm.load('./data_file/stats', f'field_{HOUR}')

    intensity_norm = MultiChannelNormalizer(2)
    position_norm = MultiChannelNormalizer(2)
    intensity_norm.load('./data_file/stats', 'intensity')
    position_norm.load('./data_file/stats', 'position')

    stats_hr = np.load('./data_file/stats/hr_minmax_stats.npz')
    cfg = DIM_CONFIG[HOUR]

    loader = get_dataloader_ds(
        TEST_CSV, batch_size=1,
        root_dir_new_dataset=ROOT_FIELD,
        root_dir_value=ROOT_VALUE,
        root_dir_track=ROOT_TRACK,
        typhoons_csv_path=TYPHOONS_CSV,
        out_channels=4,
        hr_size=(cfg['hr_size'], cfg['hr_size']),
        shuffle=False, augment=False,
        field_normalizer=field_norm,
        intensity_normalizer=intensity_norm,
        position_normalizer=position_norm,
        hr_min=torch.tensor(stats_hr['vmin'], dtype=torch.float32).view(4, 1, 1),
        hr_max=torch.tensor(stats_hr['vmax'], dtype=torch.float32).view(4, 1, 1)
    )
    return loader


def build_models(norm_stats):
    cfg = DIM_CONFIG[HOUR]
    ds_params = {'model_config': {
        'in_shape1': (cfg['T'], 73, cfg['input_size'], cfg['input_size']),
        'in_shape2': (cfg['T'], 4),
        'hid_S': 16, 'hid_T': 256, 'N_S': 4, 'N_T': 4,
        'out_channels': 4, 'model_type': 'gSTA', 'mlp_ratio': 8.,
        'drop': 0.0, 'num_upsample': 2, 'scale_factor': 2,
        'hr_size': (cfg['hr_size'], cfg['hr_size'])
    }}
    dsnet = DSNet.load_from_checkpoint(
        DSNET_CKPT, map_location='cpu', strict=True,
        normalizer_stats=norm_stats, **ds_params
    ).cuda(GPU_ID).eval()

    checkpoint = torch.load(DIFF_CKPT, map_location='cpu', weights_only=False)
    unet = DenoiseUNet(in_channels=8, out_channels=4, init_dim=64).cuda(GPU_ID)
    unet.load_state_dict(
        {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('model.')}
    )
    unet.eval()

    return dsnet, DDIMSampler(unet, device=f'cuda:{GPU_ID}')


def ablate_group(x1, x2, field_group=None, vector_group=None):
    xa = x1.clone()
    xb = x2.clone()

    if field_group is not None:
        idxs = [name_to_idx[n] for n in CHANNEL_GROUPS[field_group] if n in name_to_idx]
        if len(idxs) > 0:
            xa[:, :, idxs, :, :] = 0.0

    if vector_group is not None:
        idxs = TC_VECTOR_GROUPS[vector_group]
        xb[:, :, idxs] = 0.0

    return xa, xb


def evaluate_case(dsnet, sampler, x1, x2, cma_init, norm_stats):
    with torch.no_grad():
        det = dsnet(x1, x2)
        det_real = denormalize(det, norm_stats)
        det_vmax = vmax_from_field(det_real).item()

        vmax_members = []
        for m in range(N_MEMBERS):
            sample = sampler.sample(det, shape=det.shape, ddim_steps=50, eta=0.0, seed=1000 + m)
            sample_real = denormalize(sample, norm_stats)
            vmax_members.append(vmax_from_field(sample_real).item())

        vmax_members = np.array(vmax_members)
        ens_mean = float(np.mean(vmax_members))
        ri_prob = float(np.mean((vmax_members - cma_init) >= RI_THRESHOLD))

    return det_vmax, ens_mean, ri_prob


def main():
    obs_lookup = build_obs_lookup()
    loader = build_loader()
    norm_stats = extract_norm_stats(GPU_ID)
    dsnet, sampler = build_models(norm_stats)

    results = []
    ri_count = 0
    nonri_count = 0

    for i in tqdm(range(len(loader.dataset))):
        info = loader.dataset.data_info.iloc[i]
        tid = str(int(info['ID'])).zfill(4)
        init_time = parse_date(info['Start Date'])
        valid_time = init_time + timedelta(hours=HOUR)

        if (tid, init_time) not in obs_lookup or (tid, valid_time) not in obs_lookup:
            continue

        cma_init = obs_lookup[(tid, init_time)]
        cma_valid = obs_lookup[(tid, valid_time)]
        obs_ri = int((cma_valid - cma_init) >= RI_THRESHOLD)

        if obs_ri == 1 and ri_count >= MAX_RI_CASES:
            continue
        if obs_ri == 0 and nonri_count >= MAX_NONRI_CASES:
            continue

        batch = loader.dataset[i]
        x1 = batch[0].unsqueeze(0).cuda(GPU_ID)
        x2 = batch[1].unsqueeze(0).cuda(GPU_ID)

        base_det, base_ens_mean, base_ri_prob = evaluate_case(dsnet, sampler, x1, x2, cma_init, norm_stats)

        if obs_ri == 1:
            ri_count += 1
        else:
            nonri_count += 1

        for g in CHANNEL_GROUPS:
            xa, xb = ablate_group(x1, x2, field_group=g)
            det_v, ens_v, ri_p = evaluate_case(dsnet, sampler, xa, xb, cma_init, norm_stats)

            results.append({
                "Obs_RI": obs_ri,
                "Group": g,
                "Type": "field",
                "Base_Det_Vmax": base_det,
                "Ablated_Det_Vmax": det_v,
                "Drop_Det_Vmax": base_det - det_v,
                "Base_EnsMean_Vmax": base_ens_mean,
                "Ablated_EnsMean_Vmax": ens_v,
                "Drop_EnsMean_Vmax": base_ens_mean - ens_v,
                "Base_RI_Prob": base_ri_prob,
                "Ablated_RI_Prob": ri_p,
                "Drop_RI_Prob": base_ri_prob - ri_p
            })

        for g in TC_VECTOR_GROUPS:
            xa, xb = ablate_group(x1, x2, vector_group=g)
            det_v, ens_v, ri_p = evaluate_case(dsnet, sampler, xa, xb, cma_init, norm_stats)

            results.append({
                "Obs_RI": obs_ri,
                "Group": g,
                "Type": "vector",
                "Base_Det_Vmax": base_det,
                "Ablated_Det_Vmax": det_v,
                "Drop_Det_Vmax": base_det - det_v,
                "Base_EnsMean_Vmax": base_ens_mean,
                "Ablated_EnsMean_Vmax": ens_v,
                "Drop_EnsMean_Vmax": base_ens_mean - ens_v,
                "Base_RI_Prob": base_ri_prob,
                "Ablated_RI_Prob": ri_p,
                "Drop_RI_Prob": base_ri_prob - ri_p
            })

        if ri_count >= MAX_RI_CASES and nonri_count >= MAX_NONRI_CASES:
            break

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(SAVE_DIR, "ri_physics_ablation_raw.csv"), index=False)

    summary = df.groupby(["Obs_RI", "Type", "Group"])[
        ["Drop_Det_Vmax", "Drop_EnsMean_Vmax", "Drop_RI_Prob"]].mean().reset_index()
    summary.to_csv(os.path.join(SAVE_DIR, "ri_physics_ablation_summary.csv"), index=False)

    # Plot RI-case drop in RI probability
    ri_summary = summary[summary["Obs_RI"] == 1].copy()
    plt.figure(figsize=(10, 5))
    plt.bar(ri_summary["Group"], ri_summary["Drop_RI_Prob"])
    plt.xticks(rotation=30, ha='right')
    plt.ylabel("Mean drop in RI probability")
    plt.title("RI-case sensitivity to physics-guided predictor groups")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "ri_prob_drop_RIcases.png"), dpi=300)
    plt.close()

    print("Saved results to:", SAVE_DIR)


if __name__ == "__main__":
    main()
