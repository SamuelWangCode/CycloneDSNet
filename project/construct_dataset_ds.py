# typhoon_intensity_bc/project/construct_dataset_ds.py

import math
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import torch
import os

from torch.utils.data import DataLoader, Dataset
import torch.serialization as ts

ts.add_safe_globals([MinMaxScaler])

########################################
# 0. 尺寸配置字典
########################################
DIM_CONFIG = {
    24: {'expansion': 26, 'input_size': 53, 'hr_size': 131, 'T': 5, 'batch_size': 16},
    48: {'expansion': 28, 'input_size': 57, 'hr_size': 141, 'T': 9, 'batch_size': 16},
    72: {'expansion': 32, 'input_size': 65, 'hr_size': 161, 'T': 13, 'batch_size': 16},
    96: {'expansion': 36, 'input_size': 73, 'hr_size': 181, 'T': 17, 'batch_size': 16},
}


########################################
# 1. Downscale 任务的 model_params
########################################
def get_model_params(hour):
    cfg = DIM_CONFIG[hour]
    input_size = cfg['input_size']
    hr_size = cfg['hr_size']
    T = cfg['T']
    bs = cfg['batch_size']
    # 估算步数
    est_samples = {24: 1900, 48: 1600, 72: 1350, 96: 1060}
    steps = math.ceil(est_samples.get(hour, 1400) // bs)

    return {
        'save_dir': f'./data_file/BCNet/typhoon_downscale_{hour}h',
        'lr': 0.0002,
        'opt': 'adamw',
        'weight_decay': 0.05,
        'filter_bias_and_bn': True,
        'pre_seq_length': T,
        'aft_seq_length': T,
        'test_mean': None,
        'test_std': None,
        'model_config': {
            'in_shape1': (T, 73, input_size, input_size),
            'in_shape2': (T, 4),
            'hid_S': 16,
            'hid_T': 256,
            'N_S': 4,
            'N_T': 4,
            'out_channels': 4,
            'model_type': 'gSTA',
            'mlp_ratio': 8.,
            'drop': 0.3,
            'drop_path': 0.0,
            'spatio_kernel_enc': 3,
            'act_inplace': True,
            'center_weight': 2.0,
            'N': 13,
            'num_upsample': 2,
            'scale_factor': 2,
            'hr_size': (hr_size, hr_size),
        },
        'metrics': ['mse', 'mae', 'rmse'],
        'epoch': 10000,
        'batch_size': bs,
        'steps_per_epoch': steps,
        'sched': 'cosine',
        'min_lr': 0.000001,
        'warmup_lr': 0.0005,
        'warmup_epoch': 2,
    }


model_params_ds_24 = get_model_params(24)
model_params_ds_48 = get_model_params(48)
model_params_ds_72 = get_model_params(72)
model_params_ds_96 = get_model_params(96)


########################################
# 2. Normalizer (兼容 3D/4D)
########################################
class MultiChannelNormalizer:
    def __init__(self, num_channels):
        self.scalers = {i: MinMaxScaler() for i in range(num_channels)}

    def fit(self, data):
        for i in range(data.shape[1]):
            self.scalers[i].fit(data[:, i].reshape(-1, 1))

    def transform(self, data):
        transformed_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            transformed_data[:, i] = self.scalers[i].transform(
                data[:, i].reshape(-1, 1)
            ).flatten()
        return transformed_data

    def inverse_transform(self, data):
        inv_transformed_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            inv_transformed_data[:, i] = self.scalers[i].inverse_transform(
                data[:, i].reshape(-1, 1)
            ).flatten()
        return inv_transformed_data

    def save(self, path, name):
        torch.save(self.scalers, os.path.join(path, f'{name}.pt'))

    def load(self, path, name):
        self.scalers = torch.load(os.path.join(path, f'{name}.pt'), weights_only=False)


class FieldNormalizer:
    def __init__(self, num_channels):
        self.num_channels = num_channels
        self.scalers = {i: MinMaxScaler() for i in range(num_channels)}

    def fit(self, data):
        for c in range(self.num_channels):
            channel_data = data[:, c, :, :].reshape(-1, 1)
            self.scalers[c].fit(channel_data)

    def transform(self, data):
        transformed = np.empty_like(data)
        # 兼容 3D (Single) 和 4D (Batch/Sequence)
        if data.ndim == 3:
            for c in range(self.num_channels):
                original_shape = data[c, :, :].shape
                channel_data = data[c, :, :].reshape(-1, 1)
                transformed[c, :, :] = self.scalers[c].transform(channel_data).reshape(original_shape)
        elif data.ndim == 4:
            for c in range(self.num_channels):
                original_shape = data[:, c, :, :].shape
                channel_data = data[:, c, :, :].reshape(-1, 1)
                transformed[:, c, :, :] = self.scalers[c].transform(channel_data).reshape(original_shape)
        return transformed

    def inverse_transform(self, data):
        inv_transformed = np.empty_like(data)
        if data.ndim == 3:
            for c in range(self.num_channels):
                original_shape = data[c, :, :].shape
                channel_data = data[c, :, :].reshape(-1, 1)
                inv_transformed[c, :, :] = self.scalers[c].inverse_transform(channel_data).reshape(original_shape)
        elif data.ndim == 4:
            for c in range(self.num_channels):
                original_shape = data[:, c, :, :].shape
                channel_data = data[:, c, :, :].reshape(-1, 1)
                inv_transformed[c, :, :] = self.scalers[c].inverse_transform(channel_data).reshape(original_shape)
        return inv_transformed

    def save(self, path, name):
        torch.save(self.scalers, os.path.join(path, f'{name}.pt'))

    def load(self, path, name):
        self.scalers = torch.load(os.path.join(path, f'{name}.pt'), weights_only=False)


########################################
# 3. Dataset (Target 单位自动纠正)
########################################
class TyphoonDownscaleDataset(Dataset):
    def __init__(self,
                 csv_file,
                 root_dir_new_dataset,
                 root_dir_value,
                 root_dir_track,
                 typhoons_csv_path,
                 out_channels=4,
                 hr_size=(131, 131),
                 sequence_length=1,
                 augment=False,
                 field_normalizer=None,
                 intensity_normalizer=None,
                 position_normalizer=None,
                 hr_min=None,
                 hr_max=None
                 ):
        self.data_info = pd.read_csv(csv_file)
        self.root_dir_new_dataset = root_dir_new_dataset
        self.root_dir_value = root_dir_value
        self.root_dir_track = root_dir_track

        self.sequence_length = sequence_length

        self.augment = augment
        self.field_normalizer = field_normalizer
        self.intensity_normalizer = intensity_normalizer
        self.position_normalizer = position_normalizer

        self.out_channels = out_channels
        self.hr_size = hr_size
        self.hr_min = hr_min
        self.hr_max = hr_max

        self.typhoons_df = None
        if typhoons_csv_path and os.path.exists(typhoons_csv_path):
            print(f"[Dataset] Loading CMA Truth from {typhoons_csv_path}...")
            df = pd.read_csv(typhoons_csv_path)
            df['ID'] = df['ID'].astype(str).str.zfill(4)
            df['Date'] = df['Date'].astype(str)
            df = df.drop_duplicates(subset=['ID', 'Date'])
            self.typhoons_df = df.set_index(['ID', 'Date'])

        def _check_files_exist(row):
            base_name = f"{row['ID']}_{row['Start Date']}_{row['Forecast Hour']}"
            input_pt = os.path.join(self.root_dir_new_dataset, f"{base_name}_input.pt")
            target_pt = os.path.join(self.root_dir_new_dataset, f"{base_name}_target.pt")
            return os.path.exists(input_pt) and os.path.exists(target_pt)

        mask = self.data_info.apply(_check_files_exist, axis=1)
        before = len(self.data_info)
        self.data_info = self.data_info[mask].reset_index(drop=True)
        print(f"[DS] {os.path.basename(csv_file)}: {len(self.data_info)} / {before} samples ready.")

    def __getitem__(self, idx):
        row = self.data_info.iloc[idx]
        base_name = f"{row['ID']}_{row['Start Date']}_{row['Forecast Hour']}"

        # === 1. 加载 Input Field ===
        input_pt_path = os.path.join(self.root_dir_new_dataset, f"{base_name}_input.pt")
        field_obj = torch.load(input_pt_path, weights_only=False)
        field_np = field_obj.numpy() if hasattr(field_obj, 'numpy') else field_obj

        # 归一化 (兼容 4D 序列)
        field_norm = self.field_normalizer.transform(field_np)
        field_input = torch.tensor(field_norm, dtype=torch.float32)

        if field_input.ndim == 3:
            field_input = field_input.unsqueeze(0).repeat(self.sequence_length, 1, 1, 1)

        # === 2. 加载 Target HR ===
        target_pt_path = os.path.join(self.root_dir_new_dataset, f"{base_name}_target.pt")
        target_obj = torch.load(target_pt_path, weights_only=False)
        target_np = target_obj.numpy() if hasattr(target_obj, 'numpy') else target_obj
        target_hr = torch.tensor(target_np, dtype=torch.float32)
        if (self.hr_min is not None) and (self.hr_max is not None):
            target_hr = (target_hr - self.hr_min) / (self.hr_max - self.hr_min + 1e-6)

        # === 3. 加载 Scalars ===
        value_path = os.path.join(self.root_dir_value, f"{base_name}_input.pt")
        track_path = os.path.join(self.root_dir_track, f"{base_name}.pt")

        track_obj = torch.load(track_path, weights_only=False)
        track_np = track_obj.numpy() if hasattr(track_obj, 'numpy') else track_obj
        track_input = torch.tensor(self.position_normalizer.transform(track_np), dtype=torch.float32)

        value_obj = torch.load(value_path, weights_only=False)
        value_raw = value_obj.numpy() if hasattr(value_obj, 'numpy') else value_obj
        intensity_norm = self.intensity_normalizer.transform(value_raw[:, 2:])
        intensity_input = torch.tensor(intensity_norm, dtype=torch.float32)

        if self.augment:
            intensity_input += np.random.uniform(-0.1, 0.1)

        value_input = torch.cat((track_input, intensity_input), dim=1)

        # === 4. CMA Truth ===
        start_date_str = str(int(row['Start Date']))
        start_dt = datetime.strptime(start_date_str, "%Y%m%d%H")
        valid_dt = start_dt + timedelta(hours=int(row['Forecast Hour']))
        valid_date_str = valid_dt.strftime("%Y%m%d%H")
        tid_str = str(int(row['ID'])).zfill(4)

        cma_pres = -999.0
        cma_wind = -999.0
        if self.typhoons_df is not None:
            try:
                if (tid_str, valid_date_str) in self.typhoons_df.index:
                    record = self.typhoons_df.loc[(tid_str, valid_date_str)]
                    if isinstance(record, pd.DataFrame): record = record.iloc[0]
                    cma_pres = float(record['Pressure']) * 100.0
                    cma_wind = float(record['Wind Speed'])
            except:
                pass

        return field_input, value_input, target_hr, (np.float32(cma_pres), np.float32(cma_wind))

    def __len__(self):
        return len(self.data_info)


########################################
# 4. Loader Creator
########################################
def get_dataloader_ds(csv_file, batch_size,
                      root_dir_new_dataset,
                      root_dir_value, root_dir_track,
                      typhoons_csv_path,
                      out_channels=4, hr_size=(131, 131), sequence_length=1,
                      shuffle=True, augment=False,
                      field_normalizer=None, intensity_normalizer=None, position_normalizer=None,
                      hr_min=None, hr_max=None):
    dataset = TyphoonDownscaleDataset(
        csv_file=csv_file,
        root_dir_new_dataset=root_dir_new_dataset,
        root_dir_value=root_dir_value,
        root_dir_track=root_dir_track,
        typhoons_csv_path=typhoons_csv_path,
        out_channels=out_channels,
        hr_size=hr_size,
        sequence_length=sequence_length,
        augment=augment,
        field_normalizer=field_normalizer,
        intensity_normalizer=intensity_normalizer,
        position_normalizer=position_normalizer,
        hr_min=hr_min,
        hr_max=hr_max,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8, pin_memory=True)


########################################
# 5. Initialization
########################################

DATA_ROOT = '/data4/wxz_data/typhoon_intensity_bc'
ROOT_FILED_DATASET = '/bigdata4/wxz_data/typhoon_intensity_bc/field_data_extraction'
ROOT_VALUE = os.path.join(DATA_ROOT, 'value_data_extraction')
ROOT_TRACK = os.path.join(DATA_ROOT, 'track_forecast_data')
TYPHOONS_CSV = '/bigdata3/WangGuanSong/Weaformer/all_models/weaformer_v2.0/typhoon_intensity_bc/data_file/typhoons.csv'

position_normalizer = MultiChannelNormalizer(num_channels=2)
position_normalizer.load('./data_file/stats', 'position')
intensity_normalizer = MultiChannelNormalizer(num_channels=2)
intensity_normalizer.load('./data_file/stats', 'intensity')

print("[Init] Loading Stats (Assuming Pa)...")
wrf_stats = np.load('./data_file/stats/hr_minmax_stats.npz')
hr_min_tensor = torch.tensor(wrf_stats['vmin'], dtype=torch.float32).view(4, 1, 1)
hr_max_tensor = torch.tensor(wrf_stats['vmax'], dtype=torch.float32).view(4, 1, 1)


def create_loaders_for_hour(hour):
    train_csv = f'./data_file/forecast_{hour}h_train_set.csv'
    valid_csv = f'./data_file/forecast_{hour}h_valid_set.csv'
    test_csv = f'./data_file/forecast_{hour}h_test_set.csv'

    field_norm = FieldNormalizer(num_channels=73)
    field_norm.load('./data_file/stats', f'field_{hour}')

    params = get_model_params(hour)
    target_hr_size = params['model_config']['hr_size']
    seq_len = params['pre_seq_length']

    print(f"Creating Loaders for {hour}h | New Dataset: {ROOT_FILED_DATASET} | SeqLen: {seq_len}")

    loader_args = dict(
        batch_size=params['batch_size'],
        root_dir_new_dataset=ROOT_FILED_DATASET,
        root_dir_value=ROOT_VALUE,
        root_dir_track=ROOT_TRACK,
        typhoons_csv_path=TYPHOONS_CSV,
        hr_size=target_hr_size,
        sequence_length=seq_len,
        field_normalizer=field_norm,
        intensity_normalizer=intensity_normalizer,
        position_normalizer=position_normalizer,
        hr_min=hr_min_tensor,
        hr_max=hr_max_tensor
    )

    train_loader = get_dataloader_ds(train_csv, shuffle=True, augment=False, **loader_args)
    val_loader = get_dataloader_ds(valid_csv, shuffle=False, augment=False, **loader_args)
    test_loader = get_dataloader_ds(test_csv, shuffle=False, augment=False, **loader_args)

    return train_loader, val_loader, test_loader


# 创建所有时效的 Loader
train_loader_ds_24, val_loader_ds_24, test_loader_ds_24 = create_loaders_for_hour(24)
train_loader_ds_48, val_loader_ds_48, test_loader_ds_48 = create_loaders_for_hour(48)
train_loader_ds_72, val_loader_ds_72, test_loader_ds_72 = create_loaders_for_hour(72)
train_loader_ds_96, val_loader_ds_96, test_loader_ds_96 = create_loaders_for_hour(96)
