import argparse
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import math

# === Import Models ===
from typhoon_intensity_bc.model.dsnet import DSNet
from typhoon_intensity_bc.model.diffusion_unet import DenoiseUNet
from typhoon_intensity_bc.project.construct_dataset_ds import get_dataloader_ds, DIM_CONFIG
from typhoon_intensity_bc.project.construct_dataset import FieldNormalizer, MultiChannelNormalizer

TIMESTEPS = 1000


# === DDIM Sampler (Fast) ===
class DiffusionSampler:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.num_train_timesteps = TIMESTEPS

        # Beta Schedule
        self.beta = torch.linspace(1e-4, 0.02, self.num_train_timesteps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        self.model.eval()

    @torch.no_grad()
    def sample(self, cond, shape, ddim_steps=50, eta=0.0):
        batch_size = shape[0]
        # Generate time steps: [T-1, ..., 0]
        times = torch.linspace(0, self.num_train_timesteps - 1, steps=ddim_steps + 1).long().to(self.device)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        img = torch.randn(shape, device=self.device)

        for t, t_prev in time_pairs:
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            model_input = torch.cat([img, cond], dim=1)
            eps = self.model(model_input, t_tensor)

            alpha_bar_t = self.alpha_bar[t]
            alpha_bar_prev = self.alpha_bar[t_prev] if t_prev >= 0 else torch.tensor(1.0).to(self.device)

            pred_x0 = (img - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
            sigma_t = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev))
            dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma_t ** 2) * eps
            noise = torch.randn_like(img) if sigma_t > 0 else 0.

            img = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt + sigma_t * noise

        return img


def extract_norm_stats(gpu_id):
    stats = np.load('./data_file/stats/hr_minmax_stats.npz')
    vmin = torch.tensor(stats['vmin'], dtype=torch.float32).view(1, -1, 1, 1).cuda(gpu_id)
    vmax = torch.tensor(stats['vmax'], dtype=torch.float32).view(1, -1, 1, 1).cuda(gpu_id)
    return {'mean': vmin, 'std': vmax - vmin}


def denormalize(tensor, stats):
    return tensor * stats['std'] + stats['mean']


def load_typhoon_names(csv_path):
    try:
        df = pd.read_csv(csv_path)
        return {str(int(row['ID'])).zfill(4): str(row['Name']).strip() for _, row in df.iterrows()}
    except:
        return {}


def main(args):
    hour, gpu_id = args.hour, args.gpu
    part_idx = args.part_idx
    total_parts = args.total_parts

    # === 1. Prepare Data Subset ===
    test_csv = f'./data_file/forecast_{hour}h_cases.csv'

    print(f"[GPU {gpu_id}] Loading TEST dataset: {test_csv} ...")
    if not os.path.exists(test_csv):
        print(f"[GPU {gpu_id}] Error: Dataset file not found: {test_csv}")
        return

    full_df = pd.read_csv(test_csv)

    # Calculate Slice
    total_samples = len(full_df)
    chunk_size = math.ceil(total_samples / total_parts)
    start_idx = part_idx * chunk_size
    end_idx = min((part_idx + 1) * chunk_size, total_samples)

    subset_df = full_df.iloc[start_idx:end_idx]

    if subset_df.empty:
        print(f"[GPU {gpu_id}] No samples assigned to this worker. Exiting.")
        return

    # Save temporary CSV for DataLoader
    temp_csv = f"./temp_input_{hour}h_part{part_idx}.csv"
    subset_df.to_csv(temp_csv, index=False)

    print(
        f"[GPU {gpu_id}] Processing Part {part_idx + 1}/{total_parts}: Samples {start_idx} to {end_idx} (Count: {len(subset_df)})")

    # === 2. Build DataLoader ===
    field_norm = FieldNormalizer(num_channels=73)
    field_norm.load('./data_file/stats', f'field_{hour}')

    ROOT_NEW = '/bigdata4/wxz_data/typhoon_intensity_bc/field_data_extraction_final_v13'
    ROOT_VALUE = '/data4/wxz_data/typhoon_intensity_bc/value_data_extraction'
    ROOT_TRACK = '/data4/wxz_data/typhoon_intensity_bc/track_forecast_data'
    TYPHOONS_CSV = './data_file/typhoons.csv'
    id_name_map = load_typhoon_names(TYPHOONS_CSV)

    norm_stats = extract_norm_stats(gpu_id)
    stats_hr = np.load('./data_file/stats/wrf_hr_minmax_new.npz')
    cfg = DIM_CONFIG[hour]

    intensity_norm = MultiChannelNormalizer(2)
    position_norm = MultiChannelNormalizer(2)
    intensity_norm.load('./data_file/stats', 'intensity')
    position_norm.load('./data_file/stats', 'position')

    loader = get_dataloader_ds(
        temp_csv, batch_size=1, root_dir_new_dataset=ROOT_NEW,
        root_dir_value=ROOT_VALUE, root_dir_track=ROOT_TRACK, typhoons_csv_path=TYPHOONS_CSV,
        out_channels=4, hr_size=(cfg['hr_size'], cfg['hr_size']),
        shuffle=False, augment=False, field_normalizer=field_norm,
        intensity_normalizer=intensity_norm, position_normalizer=position_norm,
        hr_min=torch.tensor(stats_hr['vmin'], dtype=torch.float32).view(4, 1, 1),
        hr_max=torch.tensor(stats_hr['vmax'], dtype=torch.float32).view(4, 1, 1)
    )

    # === 3. Load Models ===
    ds_params = {'model_config': {'in_shape1': (cfg['T'], 73, cfg['input_size'], cfg['input_size']),
                                  'in_shape2': (cfg['T'], 4), 'hid_S': 16, 'hid_T': 256, 'N_S': 4, 'N_T': 4,
                                  'out_channels': 4, 'model_type': 'gSTA', 'mlp_ratio': 8., 'drop': 0.0,
                                  'num_upsample': 2, 'scale_factor': 2, 'hr_size': (cfg['hr_size'], cfg['hr_size'])}}

    conditioner = DSNet.load_from_checkpoint(args.dsnet_ckpt, map_location='cpu', strict=True,
                                             normalizer_stats=norm_stats, **ds_params).cuda(gpu_id).eval()

    checkpoint = torch.load(args.diff_ckpt, map_location='cpu', weights_only=False)
    unet = DenoiseUNet(in_channels=8, out_channels=4, init_dim=64).cuda(gpu_id)
    unet.load_state_dict(
        {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('model.')})
    unet.eval()

    # Use DDIM Sampler
    sampler = DiffusionSampler(unet, device=f'cuda:{gpu_id}')

    # === 4. Inference ===
    results = []

    batch_gen = 50

    for i, batch in enumerate(tqdm(loader, position=gpu_id, desc=f"GPU {gpu_id}")):
        x1, x2, y_true, cma = batch
        x1, x2 = x1.cuda(gpu_id), x2.cuda(gpu_id)

        info = loader.dataset.data_info.iloc[i]

        # Condition
        with torch.no_grad():
            cond_pred = conditioner(x1, x2)

        # Diffusion Samples (DDIM 50 Steps)
        all_samples = []
        for _ in range(math.ceil(args.samples / batch_gen)):
            curr = min(batch_gen, args.samples - len(all_samples))
            s = sampler.sample(cond_pred.repeat(curr, 1, 1, 1), (curr, *cond_pred.shape[1:]), ddim_steps=50)
            all_samples.append(s)
        samples = torch.cat(all_samples, dim=0)

        # Post-process
        x1_np = x1[0, -1].unsqueeze(0).cpu().numpy()
        tx_real = field_norm.inverse_transform(x1_np)[0]
        tx_ws = np.sqrt(tx_real[0] ** 2 + tx_real[1] ** 2).max()
        tx_p = (tx_real[6] / 100.0).min()

        y_real = denormalize(y_true.cuda(gpu_id), norm_stats)
        wrf_ws = torch.sqrt(y_real[0, 0] ** 2 + y_real[0, 1] ** 2).max().item()
        wrf_p = (y_real[0, 3] / 100.0).min().item()

        ds_real = denormalize(cond_pred, norm_stats)
        ds_ws = torch.sqrt(ds_real[0, 0] ** 2 + ds_real[0, 1] ** 2).max().item()
        ds_p = (ds_real[0, 3] / 100.0).min().item()

        s_real = denormalize(samples, norm_stats)
        s_ws = torch.sqrt(s_real[:, 0] ** 2 + s_real[:, 1] ** 2).view(args.samples, -1).max(dim=1).values.cpu().numpy()
        s_p = (s_real[:, 3] / 100.0).view(args.samples, -1).min(dim=1).values.cpu().numpy()

        results.append({
            'ID': str(int(info['ID'])).zfill(4),
            'Name': id_name_map.get(str(int(info['ID'])).zfill(4), "Unknown"),
            'Time': str(int(info['Start Date'])),
            'Hour': int(info['Forecast Hour']),
            'CMA_Wind': cma[1].item(), 'CMA_Pres': cma[0].item() / 100.0,
            'WRF_Wind': wrf_ws, 'WRF_Pres': wrf_p,
            'TX_Wind': tx_ws, 'TX_Pres': tx_p,
            'DSNet_Wind': ds_ws, 'DSNet_Pres': ds_p,
            'Diff_Mean_Wind': np.mean(s_ws), 'Diff_Std_Wind': np.std(s_ws),
            'Diff_Min_Wind': np.min(s_ws), 'Diff_Max_Wind': np.max(s_ws),
            'Diff_Distribution_Wind': s_ws.tolist(),
            'Diff_Mean_Pres': np.mean(s_p), 'Diff_Std_Pres': np.std(s_p),
            'Diff_Min_Pres': np.min(s_p), 'Diff_Max_Pres': np.max(s_p),
            'Diff_Distribution_Pres': s_p.tolist()
        })

    # === 5. Save Partial Results ===
    out_file = f"temp_result_{hour}h_part{part_idx}.csv"
    pd.DataFrame(results).to_csv(out_file, index=False)

    # Cleanup temp input file
    if os.path.exists(temp_csv):
        os.remove(temp_csv)

    print(f"[GPU {gpu_id}] Part {part_idx + 1} Done! Saved to {out_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hour', type=int, required=True)
    parser.add_argument('--dsnet_ckpt', type=str, required=True)
    parser.add_argument('--diff_ckpt', type=str, required=True)
    parser.add_argument('--samples', type=int, default=50)
    parser.add_argument('--gpu', type=int, required=True)

    # Parallel processing args
    parser.add_argument('--part_idx', type=int, required=True)
    parser.add_argument('--total_parts', type=int, required=True)

    main(parser.parse_args())
