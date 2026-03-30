# train_DS_model.py

import argparse
import os
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy

# 1. 导入模型定义
from typhoon_intensity_bc.model.bcnet import BCNet
from typhoon_intensity_bc.model.dsnet import DSNet

# 2. 导入 Loader 和 Params
from typhoon_intensity_bc.project.construct_dataset_ds import (
    train_loader_ds_24, val_loader_ds_24, test_loader_ds_24, model_params_ds_24,
    train_loader_ds_48, val_loader_ds_48, test_loader_ds_48, model_params_ds_48,
    train_loader_ds_72, val_loader_ds_72, test_loader_ds_72, model_params_ds_72,
    train_loader_ds_96, val_loader_ds_96, test_loader_ds_96, model_params_ds_96,
)

# 3. 导入原始参数
from typhoon_intensity_bc.project.construct_dataset import (
    model_params_24, model_params_48, model_params_72, model_params_96
)


def extract_norm_stats(hour):
    """加载统计量 (Pa单位)"""
    print(f"[Init] Loading HR Stats from hr_minmax_stats.npz ...")
    stats_path = './data_file/stats/hr_minmax_stats.npz'
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Missing stats file: {stats_path}")

    stats = np.load(stats_path)
    vmin = stats['vmin']
    vmax = stats['vmax']
    data_range = vmax - vmin

    # 转换为 Tensor 形状 [1, C, 1, 1]
    norm_min_tensor = torch.tensor(vmin, dtype=torch.float32).view(1, -1, 1, 1)
    norm_range_tensor = torch.tensor(data_range, dtype=torch.float32).view(1, -1, 1, 1)
    return {'mean': norm_min_tensor, 'std': norm_range_tensor}


def main(args):
    hour = args.hour
    resume_path = args.resume_ckpt

    print(f"\n==========================================")
    print(f"Training Session: {hour}h Forecast")
    if resume_path:
        print(f"Mode: Fine-tuning from {resume_path}")
    else:
        print(f"Mode: Training from scratch (Initializing from BCNet)")
    print(f"==========================================\n")

    # === 配置字典 ===
    config_map = {
        24: {
            'loader': (train_loader_ds_24, val_loader_ds_24, test_loader_ds_24),
            'ds_params': model_params_ds_24,
            'bc_params': model_params_24,
            'bc_ckpt': "/bigdata3/WangGuanSong/Weaformer/all_models/weaformer_v2.0/typhoon_intensity_bc/data_file/BCNet/typhoon_intensity_24h/epoch=1028-step=2058.ckpt"
        },
        48: {
            'loader': (train_loader_ds_48, val_loader_ds_48, test_loader_ds_48),
            'ds_params': model_params_ds_48,
            'bc_params': model_params_48,
            'bc_ckpt': "/bigdata3/WangGuanSong/Weaformer/all_models/weaformer_v2.0/typhoon_intensity_bc/data_file/BCNet/typhoon_intensity_48h/epoch=854-step=1710.ckpt"
        },
        72: {
            'loader': (train_loader_ds_72, val_loader_ds_72, test_loader_ds_72),
            'ds_params': model_params_ds_72,
            'bc_params': model_params_72,
            'bc_ckpt': "/bigdata3/WangGuanSong/Weaformer/all_models/weaformer_v2.0/typhoon_intensity_bc/data_file/BCNet/typhoon_intensity_72h/epoch=2716-step=2717.ckpt"
        },
        96: {
            'loader': (train_loader_ds_96, val_loader_ds_96, test_loader_ds_96),
            'ds_params': model_params_ds_96,
            'bc_params': model_params_96,
            'bc_ckpt': "/bigdata3/WangGuanSong/Weaformer/all_models/weaformer_v2.0/typhoon_intensity_bc/data_file/BCNet/typhoon_intensity_96h/epoch=1953-step=1954.ckpt"
        },
    }

    cfg = config_map[hour]
    train_loader, val_loader, test_loader = cfg['loader']
    norm_stats = extract_norm_stats(hour)

    # 回调配置
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'./data_file/BCNet/typhoon_downscale_{hour}h/',
        save_top_k=3,
        verbose=True,
        monitor='val_loss',
        mode='min',
        filename=f'dsnet-{hour}h-ft-{{epoch:02d}}-{{val_loss:.4f}}'  # 加了 ft 标识 fine-tune
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss', patience=100, verbose=True, mode='min'
    )

    trainer = pl.Trainer(
        max_epochs=1000,
        strategy=DDPStrategy(find_unused_parameters=True),
        accelerator="gpu",
        devices=[0],  # 你的显卡ID
        log_every_n_steps=1,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback, early_stop_callback],
    )

    if resume_path:
        print(f"[Init] Loading weights from {resume_path}...")
        ds_model = DSNet.load_from_checkpoint(
            resume_path,
            map_location='cpu',
            strict=False,  # <--- [新增] 允许忽略缺失的 perc_loss 权重
            normalizer_stats=norm_stats,
            **cfg['ds_params']
        )

    else:
        print(f"[Init] Initializing from BCNet Teacher...")
        intensity_model = BCNet.load_from_checkpoint(
            cfg['bc_ckpt'],
            map_location='cpu',
            **cfg['bc_params']
        )

        ds_model = DSNet(
            normalizer_stats=norm_stats,
            **cfg['ds_params']
        )

        # 迁移权重
        src_state = intensity_model.model.state_dict()
        dst_state = ds_model.model.state_dict()
        copied = 0
        transfer_prefixes = ["encoder1.", "spatial_attention_weight.", "mid_layer.", "value_encoder."]
        for name, param in src_state.items():
            if any(name.startswith(prefix) for prefix in transfer_prefixes):
                if name in dst_state and dst_state[name].shape == param.shape:
                    dst_state[name] = param
                    copied += 1
        ds_model.model.load_state_dict(dst_state)
        print(f"[Info] Copied {copied} params from BCNet.")

    if ds_model.int_loss is not None:
        ds_model.int_loss.u_idx = 0
        ds_model.int_loss.v_idx = 1
        ds_model.int_loss.mslp_idx = 3

    print("[Action] Starting Training...")
    trainer.fit(ds_model, train_loader, val_loader)
    trainer.test(ds_model, dataloaders=test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hour', type=int, required=True, choices=[24, 48, 72, 96], help='Forecast hour')
    parser.add_argument('--resume_ckpt', type=str, default=None, help='Path to existing DSNet ckpt to fine-tune')
    args = parser.parse_args()
    main(args)
