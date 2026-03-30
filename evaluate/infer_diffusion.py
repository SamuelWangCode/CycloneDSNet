import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import math
import random
import pandas as pd
# 新增 SSIM 库
from skimage.metrics import structural_similarity as ssim

# === 导入你的模型 ===
from typhoon_intensity_bc.model.dsnet import DSNet
from typhoon_intensity_bc.model.diffusion_unet import DenoiseUNet
from typhoon_intensity_bc.project.construct_dataset_ds import get_dataloader_ds, DIM_CONFIG
from typhoon_intensity_bc.project.construct_dataset import FieldNormalizer, MultiChannelNormalizer

# === 配置 ===
GPU_ID = 1
TIMESTEPS = 1000


class DiffusionSampler:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.timesteps = TIMESTEPS
        self.beta = torch.linspace(1e-4, 0.02, self.timesteps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1. - self.alpha_bar)
        self.sqrt_recip_alpha = torch.sqrt(1.0 / self.alpha)
        self.posterior_variance = self.beta * (
                1. - torch.cat([torch.tensor([1.0]).to(device), self.alpha_bar[:-1]])) / (1. - self.alpha_bar)
        self.model.eval()

    @torch.no_grad()
    def p_sample(self, x, t, cond):
        t_tensor = torch.full((x.shape[0],), t, device=self.device, dtype=torch.long)
        model_input = torch.cat([x, cond], dim=1)
        eps_theta = self.model(model_input, t_tensor)

        beta_t = self.beta[t]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t]
        sqrt_recip_alpha_t = self.sqrt_recip_alpha[t]

        mean = sqrt_recip_alpha_t * (x - (beta_t / sqrt_one_minus_alpha_bar_t) * eps_theta)

        if t == 0:
            return mean
        else:
            variance = self.beta[t]
            noise = torch.randn_like(x)
            return mean + torch.sqrt(variance) * noise

    @torch.no_grad()
    def sample(self, cond, shape):
        img = torch.randn(shape, device=self.device)
        for t in reversed(range(0, self.timesteps)):
            img = self.p_sample(img, t, cond)
        return img


def extract_norm_stats():
    stats = np.load('./data_file/stats/wrf_hr_minmax_new.npz')
    vmin, vmax = stats['vmin'], stats['vmax']
    mean = torch.tensor(vmin, dtype=torch.float32).view(1, -1, 1, 1).cuda(GPU_ID)
    std = torch.tensor(vmax - vmin, dtype=torch.float32).view(1, -1, 1, 1).cuda(GPU_ID)
    return {'mean': mean, 'std': std}


def denormalize(tensor, stats):
    return tensor * stats['std'] + stats['mean']


def calc_wind(data):
    return torch.sqrt(data[:, 0] ** 2 + data[:, 1] ** 2)


def calc_mslp(data):
    return data[:, 3] / 100.0


# === SSIM 计算辅助函数 ===
def calculate_ssim_batch(target, samples, data_range):
    """
    target: (H, W)
    samples: (N, H, W)
    data_range: float (max - min of the data)
    """
    scores = []
    for i in range(samples.shape[0]):
        # SSIM 需要输入为 numpy
        score = ssim(target, samples[i], data_range=data_range)
        scores.append(score)
    return np.array(scores)


# === plotting function ===
def visualize_variable(
        var_name, save_path,
        tx_img, wrf_img, dsnet_img,
        samples,
        cma_val,
        best_ssim_idx, worst_ssim_idx,
        best_cma_idx, worst_cma_idx,
        random_indices,
        typhoon_info
):
    ensemble_mean = np.mean(samples, axis=0)

    # === 1. 动态计算 Color Range (基于数据本身) ===

    # 将所有涉及到的像素点聚合在一起 (不包括 NaN)
    # 注意：samples 包含了所有生成样本，这能保证随机选出来的样本也在这个范围内
    # 我们只关心主要的几个图和 samples 的统计特性
    all_pixels = np.concatenate([
        tx_img.flatten(),
        wrf_img.flatten(),
        dsnet_img.flatten(),
        samples.flatten()
    ])

    # 过滤掉可能的 inf 或 nan
    all_pixels = all_pixels[np.isfinite(all_pixels)]

    if var_name == 'Wind':
        unit = "m/s"
        get_intensity = np.max
        cma_str = f"CMA Max: {cma_val:.1f}"
        cmap = 'jet'

        # [风速策略]
        # vmin: 风速最小肯定是 0，不需要动态算，保持 0 可以看清静风区
        vmin = 0
        # vmax: 取数据的 99.9% 分位数，防止个别 200m/s 的噪点毁了整张图
        # 同时要确保 CMA 真值在范围内，否则真值线画不出来
        data_max = np.percentile(all_pixels, 99.95)
        vmax = max(data_max, cma_val) + 2.0  # 稍微留一点顶空

    else:  # MSLP
        unit = "hPa"
        get_intensity = np.min
        cma_str = f"CMA Min: {cma_val:.1f}"
        cmap = 'jet_r'  # 气压越低越红

        # [气压策略]
        # 气压可能出现极低值，也取 0.05% 分位数防噪
        data_min = np.percentile(all_pixels, 0.05)
        data_max = np.percentile(all_pixels, 99.95)

        # vmin: 确保覆盖 CMA 最低压
        vmin = min(data_min, cma_val) - 2.0
        # vmax: 确保覆盖环境气压 (通常 1010-1015)
        vmax = max(data_max, 1015.0)

    # === 2. 设置画布 (保持不变) ===
    fig = plt.figure(figsize=(32, 16), dpi=150)
    gs = fig.add_gridspec(4, 8, height_ratios=[1, 1, 0.1, 0.8], hspace=0.2, wspace=0.05)

    super_title = (f"Typhoon: {typhoon_info['name']} ({typhoon_info['id']}) | "
                   f"Init: {typhoon_info['time']} | Forecast: +{typhoon_info['hour']}h | "
                   f"{cma_str} {unit}")
    fig.suptitle(super_title, fontsize=24, fontweight='bold', y=0.95)

    def plot_subplot(ax, img, title, title_color='black'):
        # 这里的 vmin, vmax 已经是动态计算好的了
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=11, color=title_color, fontweight='bold')
        ax.axis('off')
        return im

    # === Row 1 ===
    plot_subplot(fig.add_subplot(gs[0, 0]), tx_img, f"TianXing Input\n{get_intensity(tx_img):.1f}")
    plot_subplot(fig.add_subplot(gs[0, 1]), wrf_img, f"WRF Target\n{get_intensity(wrf_img):.1f}")
    plot_subplot(fig.add_subplot(gs[0, 2]), dsnet_img, f"DSNet Output\n{get_intensity(dsnet_img):.1f}")

    # Metrics
    best_ssim_img = samples[best_ssim_idx]
    plot_subplot(fig.add_subplot(gs[0, 3]), best_ssim_img, f"Best Struct (SSIM)\n{get_intensity(best_ssim_img):.1f}")

    worst_ssim_img = samples[worst_ssim_idx]
    plot_subplot(fig.add_subplot(gs[0, 4]), worst_ssim_img, f"Worst Struct (SSIM)\n{get_intensity(worst_ssim_img):.1f}")

    best_cma_img = samples[best_cma_idx]
    plot_subplot(fig.add_subplot(gs[0, 5]), best_cma_img, f"Best Intensity (CMA)\n{get_intensity(best_cma_img):.1f}",
                 'red')

    worst_cma_img = samples[worst_cma_idx]
    plot_subplot(fig.add_subplot(gs[0, 6]), worst_cma_img, f"Worst Intensity (CMA)\n{get_intensity(worst_cma_img):.1f}")

    # Ensemble Mean (用于生成 Colorbar)
    im_ref = plot_subplot(fig.add_subplot(gs[0, 7]), ensemble_mean,
                          f"Ensemble Mean\n{get_intensity(ensemble_mean):.1f}")

    # === Row 2 ===
    for i, idx in enumerate(random_indices[:8]):
        ax = fig.add_subplot(gs[1, i])
        s_img = samples[idx]
        plot_subplot(ax, s_img, f"Sample #{idx}\n{get_intensity(s_img):.1f}")

    # === Unified Colorbar ===
    cax = fig.add_subplot(gs[2, 2:6])
    cb = plt.colorbar(im_ref, cax=cax, orientation='horizontal')
    # 动态显示 Range 信息
    cb.set_label(f"{var_name} ({unit}) | Range: [{vmin:.1f}, {vmax:.1f}]", fontsize=14)
    cb.ax.tick_params(labelsize=12)

    # === Row 3: Distribution ===
    ax_dist = fig.add_subplot(gs[3, :])

    intensities = [get_intensity(s) for s in samples]

    sns.kdeplot(intensities, ax=ax_dist, fill=True, color='purple', alpha=0.3, label='Diffusion Ensemble')

    lw = 3
    ax_dist.axvline(cma_val, color='red', linestyle='-', linewidth=lw, label=f'CMA Truth ({cma_val:.1f})')
    ax_dist.axvline(get_intensity(wrf_img), color='green', linestyle='--', linewidth=lw,
                    label=f'WRF Target ({get_intensity(wrf_img):.1f})')
    ax_dist.axvline(get_intensity(tx_img), color='blue', linestyle=':', linewidth=lw,
                    label=f'TianXing ({get_intensity(tx_img):.1f})')
    ax_dist.axvline(get_intensity(dsnet_img), color='orange', linestyle='-.', linewidth=lw,
                    label=f'DSNet ({get_intensity(dsnet_img):.1f})')

    diff_mean_val = np.mean(intensities)
    ax_dist.axvline(diff_mean_val, color='purple', linestyle='-.', linewidth=lw,
                    label=f'Diff Mean ({diff_mean_val:.1f})')

    ax_dist.set_title(f"{var_name} Intensity Probability Distribution", fontsize=16)
    ax_dist.set_xlabel(f"{var_name} ({unit})", fontsize=14)
    # 强制设置 X 轴范围，与上面的 Colorbar 范围对齐，方便上下对照
    # ax_dist.set_xlim(vmin, vmax)
    ax_dist.legend(loc='upper right', fontsize=12)
    ax_dist.grid(True, alpha=0.3)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"🖼️ Saved {var_name} analysis to {save_path}")


def load_typhoon_names(csv_path):
    try:
        df = pd.read_csv(csv_path)
        id_to_name = {}
        for _, row in df.iterrows():
            tid = str(int(row['ID'])).zfill(4)
            name = str(row['Name']).strip()
            id_to_name[tid] = name
        return id_to_name
    except:
        return {}


def main(args):
    hour = args.hour
    num_samples = args.samples

    save_dir = f"./results/diffusion_ensemble_{hour}h"
    os.makedirs(save_dir, exist_ok=True)

    norm_stats = extract_norm_stats()

    print("🛠️ Constructing Test Loader manually...")
    field_norm = FieldNormalizer(num_channels=73)
    field_norm.load('./data_file/stats', f'field_{hour}')

    ROOT_NEW = '/bigdata4/wxz_data/typhoon_intensity_bc/field_data_extraction_final_v13'
    ROOT_VALUE = '/data4/wxz_data/typhoon_intensity_bc/value_data_extraction'
    ROOT_TRACK = '/data4/wxz_data/typhoon_intensity_bc/track_forecast_data'
    TYPHOONS_CSV = './data_file/typhoons.csv'
    TEST_CSV = f'./data_file/forecast_{hour}h_new_test_set.csv'

    id_name_map = load_typhoon_names(TYPHOONS_CSV)
    stats_hr = np.load('./data_file/stats/hr_minmax_stats.npz')
    hr_vmin, hr_vmax = stats_hr['vmin'], stats_hr['vmax']
    cfg = DIM_CONFIG[hour]

    intensity_norm = MultiChannelNormalizer(2)
    position_norm = MultiChannelNormalizer(2)
    intensity_norm.load('./data_file/stats', 'intensity')
    position_norm.load('./data_file/stats', 'position')

    test_loader = get_dataloader_ds(
        TEST_CSV, batch_size=1, root_dir_new_dataset=ROOT_NEW, root_dir_value=ROOT_VALUE,
        root_dir_track=ROOT_TRACK, typhoons_csv_path=TYPHOONS_CSV,
        out_channels=4, hr_size=(cfg['hr_size'], cfg['hr_size']),
        shuffle=False, augment=False, field_normalizer=field_norm,
        intensity_normalizer=intensity_norm, position_normalizer=position_norm,
        hr_min=torch.tensor(hr_vmin, dtype=torch.float32).view(4, 1, 1),
        hr_max=torch.tensor(hr_vmax, dtype=torch.float32).view(4, 1, 1)
    )

    print(f"Loading DSNet from {args.dsnet_ckpt}...")
    ds_params_manual = {
        'model_config': {
            'in_shape1': (cfg['T'], 73, cfg['input_size'], cfg['input_size']),
            'in_shape2': (cfg['T'], 4),
            'hid_S': 16, 'hid_T': 256, 'N_S': 4, 'N_T': 4, 'out_channels': 4,
            'model_type': 'gSTA', 'mlp_ratio': 8., 'drop': 0.0,
            'num_upsample': 2, 'scale_factor': 2, 'hr_size': (cfg['hr_size'], cfg['hr_size'])
        }
    }
    conditioner = DSNet.load_from_checkpoint(args.dsnet_ckpt, map_location='cpu', strict=True,
                                             normalizer_stats=norm_stats, **ds_params_manual).cuda(GPU_ID).eval()

    print(f"Loading Diffusion from {args.diff_ckpt}...")
    checkpoint = torch.load(args.diff_ckpt, map_location='cpu', weights_only=False)
    state_dict = checkpoint['state_dict']
    unet_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
    unet = DenoiseUNet(in_channels=8, out_channels=4, init_dim=64).cuda(GPU_ID)
    unet.load_state_dict(unet_state_dict)
    unet.eval()
    sampler = DiffusionSampler(unet, device=f'cuda:{GPU_ID}')

    print(f"Start Ensemble Inference (N={num_samples})...")
    count = 0

    for i, batch in enumerate(tqdm(test_loader)):
        x1, x2, y_true, cma = batch
        x1, x2, y_true = x1.cuda(GPU_ID), x2.cuda(GPU_ID), y_true.cuda(GPU_ID)

        cma_wind = cma[1].item()
        cma_pres = cma[0].item() / 100.0  # hPa

        if cma_wind < 45.0: continue

        info_row = test_loader.dataset.data_info.iloc[i]
        typhoon_id = str(int(info_row['ID'])).zfill(4)
        start_date = str(int(info_row['Start Date']))
        forecast_hour = int(info_row['Forecast Hour'])
        typhoon_name = id_name_map.get(typhoon_id, "Unknown")

        typhoon_info = {
            'id': typhoon_id, 'name': typhoon_name, 'time': start_date, 'hour': forecast_hour
        }

        # === A. 获取 TianXing Input (LR) ===
        x1_last_np = x1[0, -1].unsqueeze(0).cpu().numpy()
        tx_real_np = field_norm.inverse_transform(x1_last_np)[0]

        tx_u, tx_v = tx_real_np[0], tx_real_np[1]
        tx_p = tx_real_np[6] / 100.0
        tx_ws = np.sqrt(tx_u ** 2 + tx_v ** 2)
        tx_mslp = tx_p

        # === B. 生成 Condition & Diffusion ===
        with torch.no_grad():
            cond_pred = conditioner(x1, x2)

        batch_gen = 10
        all_samples = []
        for _ in range(math.ceil(num_samples / batch_gen)):
            curr_batch = min(batch_gen, num_samples - len(all_samples))
            cond_exp = cond_pred.repeat(curr_batch, 1, 1, 1)
            s = sampler.sample(cond_exp, shape=cond_exp.shape)
            all_samples.append(s)
        samples_tensor = torch.cat(all_samples, dim=0)

        # === C. 反归一化 & 计算物理量 ===
        true_real = denormalize(y_true, norm_stats)
        true_ws = calc_wind(true_real)[0].cpu().numpy()
        true_mslp = calc_mslp(true_real)[0].cpu().numpy()

        dsnet_real = denormalize(cond_pred, norm_stats)
        dsnet_ws = calc_wind(dsnet_real)[0].cpu().numpy()
        dsnet_mslp = calc_mslp(dsnet_real)[0].cpu().numpy()

        samples_real = denormalize(samples_tensor, norm_stats)
        samples_ws = calc_wind(samples_real).cpu().numpy()
        samples_mslp = calc_mslp(samples_real).cpu().numpy()

        # === D. 筛选逻辑 (使用 SSIM) ===
        # 1. 计算 SSIM (vs WRF)
        # 风速 Range: max - min
        ws_range = max(true_ws.max(), samples_ws.max()) - min(true_ws.min(), samples_ws.min())
        ssim_ws = calculate_ssim_batch(true_ws, samples_ws, ws_range)
        best_ssim_w = np.argmax(ssim_ws)
        worst_ssim_w = np.argmin(ssim_ws)

        # 气压 Range
        mslp_range = max(true_mslp.max(), samples_mslp.max()) - min(true_mslp.min(), samples_mslp.min())
        ssim_p = calculate_ssim_batch(true_mslp, samples_mslp, mslp_range)
        best_ssim_p = np.argmax(ssim_p)
        worst_ssim_p = np.argmin(ssim_p)

        # 2. 计算强度差异 (vs CMA)
        # 风速找最大值，气压找最小值
        max_winds = np.max(samples_ws, axis=(1, 2))
        min_press = np.min(samples_mslp, axis=(1, 2))

        # Best: 差值最小; Worst: 差值最大
        diff_wind = np.abs(max_winds - cma_wind)
        best_cma_w = np.argmin(diff_wind)
        worst_cma_w = np.argmax(diff_wind)

        diff_press = np.abs(min_press - cma_pres)
        best_cma_p = np.argmin(diff_press)
        worst_cma_p = np.argmax(diff_press)

        # 3. 随机 8 个
        rand_indices = random.sample(range(num_samples), 8)

        # === E. 绘图 ===
        visualize_variable(
            "MSLP", f"{save_dir}/sample_{count}_mslp_{cma_pres:.0f}.png",
            tx_mslp, true_mslp, dsnet_mslp,
            samples_mslp, cma_pres,
            best_ssim_p, worst_ssim_p, best_cma_p, worst_cma_p,
            rand_indices, typhoon_info
        )

        count += 1
        if count >= 5: break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hour', type=int, required=True)
    parser.add_argument('--dsnet_ckpt', type=str, required=True)
    parser.add_argument('--diff_ckpt', type=str, required=True)
    parser.add_argument('--samples', type=int, default=50)
    args = parser.parse_args()
    main(args)
