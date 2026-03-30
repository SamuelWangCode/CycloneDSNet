import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from typhoon_intensity_bc.model.dsnet import DSNet, PhysicalIntensityLoss

from typhoon_intensity_bc.model.dsnet import DSNet
from typhoon_intensity_bc.model.diffusion_unet import DenoiseUNet
from typhoon_intensity_bc.project.construct_dataset_ds import (
    train_loader_ds_24, val_loader_ds_24, test_loader_ds_24, model_params_ds_24,
    train_loader_ds_48, val_loader_ds_48, test_loader_ds_48, model_params_ds_48,
    train_loader_ds_72, val_loader_ds_72, test_loader_ds_72, model_params_ds_72,
    train_loader_ds_96, val_loader_ds_96, test_loader_ds_96, model_params_ds_96,
    DIM_CONFIG
)


# === 工具函数 ===
def extract_norm_stats(hour):
    """加载统计量 (MinMax)"""
    stats = np.load('./data_file/stats/hr_minmax_stats.npz')
    norm_min = torch.tensor(stats['vmin'], dtype=torch.float32).view(1, -1, 1, 1)
    norm_range = torch.tensor(stats['vmax'] - stats['vmin'], dtype=torch.float32).view(1, -1, 1, 1)
    return {'mean': norm_min, 'std': norm_range}  # 这里的 key 叫 mean/std 只是为了兼容接口，实际是 min/range


class DiffusionModule(pl.LightningModule):
    def __init__(self, dsnet_ckpt, ds_params, norm_stats, timesteps=1000, lr=1e-4):
        super().__init__()
        self.save_hyperparameters(ignore=['norm_stats'])
        self.lr = lr

        print(f"[Diff] Loading frozen DSNet conditioner from {dsnet_ckpt}...")

        self.conditioner = DSNet.load_from_checkpoint(
            dsnet_ckpt,
            map_location='cpu',
            strict=True,  # <--- 改为 True
            normalizer_stats=norm_stats,
            **ds_params
        )
        self.conditioner.freeze()
        self.conditioner.eval()

        self.model = DenoiseUNet(in_channels=8, out_channels=4, init_dim=64)

        self.timesteps = timesteps
        self.register_buffer('beta', torch.linspace(1e-4, 0.02, timesteps))
        alpha = 1. - self.beta
        self.register_buffer('alpha_bar', torch.cumprod(alpha, dim=0))
        self.loss_fn = nn.MSELoss()
        current_img_size = ds_params['model_config']['hr_size'][0]
        self.phy_loss = PhysicalIntensityLoss(
            normalizer_mean=norm_stats['mean'],
            normalizer_std=norm_stats['std'],
            mslp_idx=3, u_idx=0, v_idx=1,
            weight=1.0,
            img_size=current_img_size
        )

    def forward(self, x, t, cond):
        model_input = torch.cat([x, cond], dim=1)
        return self.model(model_input, t)

    def extract(self, a, t, x_shape):
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def predict_x0_from_noise(self, x_t, t, noise):
        """
        根据公式：x_0 = (x_t - sqrt(1 - alpha_bar) * noise) / sqrt(alpha_bar)
        从当前噪声图和预测噪声反推 x_0
        """
        sqrt_recip_alpha_bar = 1.0 / torch.sqrt(self.alpha_bar)
        sqrt_recip_m1_alpha_bar = torch.sqrt(1.0 / self.alpha_bar - 1.0)

        sqrt_recip_ab_t = self.extract(sqrt_recip_alpha_bar, t, x_t.shape)
        sqrt_recip_m1_ab_t = self.extract(sqrt_recip_m1_alpha_bar, t, x_t.shape)

        return sqrt_recip_ab_t * x_t - sqrt_recip_m1_ab_t * noise

    def training_step(self, batch, batch_idx):
        x1, x2, y_true, cma = batch
        cma_pres, cma_wind = cma[0], cma[1]

        # 1. Condition & Noise (保持不变)
        with torch.no_grad():
            cond_pred = self.conditioner(x1, x2)
            cond_pred = torch.clamp(cond_pred, -10, 10)

        B = y_true.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=self.device).long()
        noise = torch.randn_like(y_true)
        alpha_bar_t = self.extract(self.alpha_bar, t, y_true.shape)
        x_t = torch.sqrt(alpha_bar_t) * y_true + torch.sqrt(1. - alpha_bar_t) * noise

        # 4. Predict
        pred_noise = self(x_t, t, cond_pred)

        # 5. MSE Loss
        loss_mse = self.loss_fn(pred_noise, noise)

        # 6. Physical Loss
        pred_x0 = self.predict_x0_from_noise(x_t, t, pred_noise)

        # 计算物理 Loss
        loss_phy = self.phy_loss(pred_x0, y_true, cma_pres, cma_wind)

        # 7. Total Loss (修正权重!)
        total_loss = loss_mse + 5e-5 * loss_phy

        # Log
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('mse_loss', loss_mse, prog_bar=True)
        self.log('phy_loss', loss_phy, prog_bar=True)  # 观察这个原始值

        return total_loss

    def validation_step(self, batch, batch_idx):
        x1, x2, y_true, cma = batch
        cma_pres, cma_wind = cma[0], cma[1]

        with torch.no_grad():
            cond_pred = self.conditioner(x1, x2)
            cond_pred = torch.clamp(cond_pred, -10, 10)

            B = y_true.shape[0]
            t = torch.randint(0, self.timesteps, (B,), device=self.device).long()
            noise = torch.randn_like(y_true)

            alpha_bar_t = self.extract(self.alpha_bar, t, y_true.shape)
            x_t = torch.sqrt(alpha_bar_t) * y_true + torch.sqrt(1. - alpha_bar_t) * noise

            pred_noise = self(x_t, t, cond_pred)

            # 1. 计算 MSE
            loss_mse = self.loss_fn(pred_noise, noise)

            # 2. 计算 Phy Loss
            pred_x0 = self.predict_x0_from_noise(x_t, t, pred_noise)
            loss_phy = self.phy_loss(pred_x0, y_true, cma_pres, cma_wind)

            # 3. 计算 Total Val Loss
            val_total_loss = loss_mse + 5e-5 * loss_phy

        self.log('val_loss', val_total_loss, prog_bar=True, sync_dist=True)
        self.log('val_mse', loss_mse, prog_bar=False, sync_dist=True)
        self.log('val_phy', loss_phy, prog_bar=False, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)


def main(args):
    hour = args.hour
    dsnet_ckpt = args.dsnet_ckpt

    print(f"\n==========================================")
    print(f"Diffusion Training Session: {hour}h Forecast")
    print(f"Condition Model: Frozen DSNet from {dsnet_ckpt}")
    print(f"==========================================\n")

    config_map = {
        24: (train_loader_ds_24, val_loader_ds_24),
        48: (train_loader_ds_48, val_loader_ds_48),
        72: (train_loader_ds_72, val_loader_ds_72),
        96: (train_loader_ds_96, val_loader_ds_96),
    }
    train_loader, val_loader = config_map[hour]

    norm_stats = extract_norm_stats(hour)

    cfg = DIM_CONFIG[hour]
    ds_params_manual = {
        'model_config': {
            'in_shape1': (cfg['T'], 73, cfg['input_size'], cfg['input_size']),
            'in_shape2': (cfg['T'], 4),
            'hid_S': 16, 'hid_T': 256, 'N_S': 4, 'N_T': 4, 'out_channels': 4,
            'model_type': 'gSTA', 'mlp_ratio': 8., 'drop': 0.0,
            'num_upsample': 2, 'scale_factor': 2, 'hr_size': (cfg['hr_size'], cfg['hr_size'])
        }
    }

    # 4. 初始化模型
    model = DiffusionModule(
        dsnet_ckpt=dsnet_ckpt,
        ds_params=ds_params_manual,  # <--- 使用手动确认的参数
        norm_stats=norm_stats,
        timesteps=1000,
        lr=1e-4
    )

    # Checkpoint 回调
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'./data_file/Diffusion/typhoon_diff_{hour}h/',
        save_top_k=3,
        verbose=True,
        monitor='val_loss',
        mode='min',
        filename=f'diff-{hour}h-{{epoch:02d}}-{{val_loss:.4f}}'
    )

    trainer = pl.Trainer(
        max_epochs=1000,
        strategy=DDPStrategy(find_unused_parameters=False),
        accelerator="gpu",
        devices=[0],
        log_every_n_steps=10,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hour', type=int, required=True, help='Forecast hour (24/48/72/96)')
    parser.add_argument('--dsnet_ckpt', type=str, required=True, help="Path to pretrained DSNet .ckpt")
    args = parser.parse_args()
    main(args)
