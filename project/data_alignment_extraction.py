import numpy as np
import xarray as xr
import torch
import pandas as pd
import json
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta

# --- 0. 尺寸配置 ---
# T: 时间步长 (例如 24h = 0,6,12,18,24 共5个时刻)
GRID_CONFIG = {
    24: {'in_radius': 26, 'out_radius': 65, 'T': 5},  # WRF出 131
    48: {'in_radius': 28, 'out_radius': 70, 'T': 9},  # WRF出 141
    72: {'in_radius': 32, 'out_radius': 80, 'T': 13},  # WRF出 161
    96: {'in_radius': 36, 'out_radius': 90, 'T': 17}  # WRF出 181
}


# --- 1. 基础工具 ---
def get_valid_time_str(start_date_int, hour):
    start_dt = datetime.strptime(str(start_date_int), "%Y%m%d%H")
    valid_dt = start_dt + timedelta(hours=int(hour))
    return valid_dt.strftime("%Y%m%d%H")


# --- 2. 找眼 (返回 索引 + 经纬度 + 变量名) ---
def find_eye_index(ds, center_lat, center_lon, radius_deg=8.0, tag=""):
    possible_names = ['mslp', 'MSLP', 'msl', 'MSL', 'slp', 'SLP', 'pressure', 'var151', 'PSFC', 'psfc']
    var_name = None
    for v in possible_names:
        if v in ds: var_name = v; break

    if var_name is None:
        for v in ds.data_vars:
            if 'msl' in v.lower() or 'slp' in v.lower() or 'pres' in v.lower(): var_name = v; break

    if var_name is None:
        print(f"  [{tag}] ❌ No Pressure Var Found!");
        return None, None, None, None

    # 1. 找到 center_lat/lon 对应的全局索引
    try:
        center_loc = ds.sel(lat=center_lat, lon=center_lon, method='nearest')
        c_y = int(np.where(ds.lat.values == center_loc.lat.values)[0][0])
        c_x = int(np.where(ds.lon.values == center_loc.lon.values)[0][0])
    except:
        return None, None, None, None

    # 2. 估算 radius 对应的格点数
    d_lat = abs(float(ds.lat[1] - ds.lat[0]))
    d_lon = abs(float(ds.lon[1] - ds.lon[0]))
    r_grid = int(radius_deg / d_lat)

    # 3. 确定搜索范围 (Index)
    y_min = max(0, c_y - r_grid)
    y_max = min(len(ds.lat), c_y + r_grid)
    x_min = max(0, c_x - r_grid)
    x_max = min(len(ds.lon), c_x + r_grid)

    # 4. 取出局部数据找最小值
    val = ds[var_name].isel(lat=slice(y_min, y_max), lon=slice(x_min, x_max)).values
    while val.ndim > 2: val = val[0]  # 确保是 2D

    min_val = np.nanmin(val)
    if np.isnan(min_val): return None, None, None, None

    locs = np.where(val == min_val)
    local_y, local_x = locs[0][0], locs[1][0]

    # 5. 换算回全局索引
    global_y = y_min + local_y
    global_x = x_min + local_x

    # 6. 获取真实经纬度
    real_lat = float(ds.lat[global_y].values)
    real_lon = float(ds.lon[global_x].values)

    return global_y, global_x, real_lat, real_lon, var_name


# --- 3. 受限截取 (Clamped Crop - 核心逻辑) ---
def get_clamped_slice(center_idx, radius, max_size):
    target_size = radius * 2 + 1
    start = center_idx - radius
    if start < 0: start = 0
    if start + target_size > max_size: start = max_size - target_size
    end = start + target_size
    return slice(start, end)


def crop_by_slice(ds, slice_y, slice_x):
    """
    WRF 截取 (Target): 保持 3D 输出 (C, H, W)
    """
    cropped = ds.isel(lat=slice_y, lon=slice_x)
    vars_list = list(cropped.data_vars)
    data_list = []

    for v in vars_list:
        arr = cropped[v].values
        while arr.ndim > 2: arr = arr[0]
        data_list.append(arr)

    final_np = np.stack(data_list, axis=0)  # (C, H, W)

    mid_y_idx = (slice_y.start + slice_y.stop) // 2
    mid_x_idx = (slice_x.start + slice_x.stop) // 2
    mid_y_idx = min(mid_y_idx, len(ds.lat) - 1)
    mid_x_idx = min(mid_x_idx, len(ds.lon) - 1)

    anchor_lat = float(ds.lat[mid_y_idx].values)
    anchor_lon = float(ds.lon[mid_x_idx].values)

    return final_np, vars_list, anchor_lat, anchor_lon


# --- 4. 预报场截取 (修正版：保留时间维 T) ---
def crop_forecast_seq_aligned(ds_seq, target_center_lat, target_center_lon, radius):
    """
    预报场截取 (Input): 输出 4D 张量 (T, C, H, W)
    ds_seq: 包含 T 个时间步的 Dataset
    """
    try:
        # 1. 使用序列中的最后一帧(即当前时刻)来确定 Lat/Lon 索引
        ds_last = ds_seq.isel(time=-1)

        center_loc = ds_last.sel(lat=target_center_lat, lon=target_center_lon, method='nearest')
        c_y = int(np.where(ds_last.lat.values == center_loc.lat.values)[0][0])
        c_x = int(np.where(ds_last.lon.values == center_loc.lon.values)[0][0])

        # 2. 确定切片范围
        y1, y2 = c_y - radius, c_y + radius + 1
        x1, x2 = c_x - radius, c_x + radius + 1

        # 3. 对整个序列进行截取
        # cropped 形状: (Time, Lat, Lon)
        cropped = ds_seq.isel(lat=slice(y1, y2), lon=slice(x1, x2))

        vars_list = list(cropped.data_vars)
        channel_data = []

        # 4. 堆叠通道
        for v in vars_list:
            # val shape: (T, H, W) 或者 (T, Lev, H, W)
            val = cropped[v].values

            if val.ndim == 3:  # (T, H, W)
                val = val[:, np.newaxis, :, :]  # -> (T, 1, H, W)
            elif val.ndim == 4:  # (T, Lev, H, W) -> 假设是多层变量，保持原样
                pass

            channel_data.append(val)

        # 5. 在 Channel 维度(axis=1)拼接
        # result: (T, C_total, H, W)
        final_np = np.concatenate(channel_data, axis=1)

        return final_np, vars_list

    except Exception as e:
        print(f"Forecast Crop Error: {e}")
        return None, None


# --- 5. 画图 (只画最后一帧 T) ---
def plot_check_final(fcst_np, fcst_vars, fcst_target_var, wrf_np, wrf_vars, wrf_target_var, tid, hour, save_dir):
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    # fcst_np shape: (T, C, H, W) -> 取最后一帧 [-1]
    img_f_all = fcst_np[-1]

    try:
        f_idx = fcst_vars.index(fcst_target_var)
    except:
        f_idx = 0
    try:
        w_idx = wrf_vars.index(wrf_target_var)
    except:
        w_idx = 0

    img_f = img_f_all[f_idx].copy()
    img_w = wrf_np[w_idx].copy()  # WRF 是 (C, H, W)

    if np.max(img_f) > 50000: img_f /= 100.0
    if np.max(img_w) > 50000: img_w /= 100.0

    f_min, f_max = np.min(img_f), np.max(img_f)
    w_min, w_max = np.min(img_w), np.max(img_w)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    im1 = axes[0].imshow(img_f, cmap='jet', origin='lower')
    axes[0].set_title(f"Forecast (Last Frame)\nVar: {fcst_target_var}\nRange: [{f_min:.1f}, {f_max:.1f}]")
    axes[0].axhline(img_f.shape[0] // 2, c='w', ls='--')
    axes[0].axvline(img_f.shape[1] // 2, c='w', ls='--')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)

    im2 = axes[1].imshow(img_w, cmap='jet', origin='lower')
    axes[1].set_title(f"WRF (Target)\nVar: {wrf_target_var}\nRange: [{w_min:.1f}, {w_max:.1f}]")
    axes[1].axhline(img_w.shape[0] // 2, c='w', ls='--')
    axes[1].axvline(img_w.shape[1] // 2, c='w', ls='--')
    plt.colorbar(im2, ax=axes[1], fraction=0.046)

    plt.suptitle(f"ID: {tid} | +{hour}h | Input Shape: {fcst_np.shape}")
    plt.savefig(f"{save_dir}/{tid}_{hour}h.png")
    plt.close()


# --- 6. 主流程 ---
def process_case(row, tracking_df, save_dir_nc, debug_dir):
    tid = row['ID']
    sdate = row['Start Date']
    hour = row['Forecast Hour']
    cfg = GRID_CONFIG[hour]

    # 目标时间索引 (例如 24h -> index 4)
    target_idx = hour // 6

    valid_time = get_valid_time_str(sdate, hour)
    pred_path = f'/bigdata1/TianXing_forecast/prediction_{str(sdate)[:-2]}T{str(sdate)[-2:]}.nc'
    wrf_path = f'/bigdata4/wxz_data/typhoon_intensity_bc/hr_field_data_wrf/{tid}_{valid_time}_hr.nc'

    if not os.path.exists(wrf_path): return

    try:
        ds_wrf = xr.open_dataset(wrf_path)
        if 'time' in ds_wrf.dims: ds_wrf = ds_wrf.isel(time=0)

        ds_fcst_full = xr.open_dataset(pred_path)
        ds_fcst_full = ds_fcst_full.sortby('lat')  # 必做: 修正南北反转

        # 截取从 0 到 target_idx 的时间段 (共 T 帧)
        ds_fcst_seq = ds_fcst_full.isel(time=slice(0, target_idx + 1))

        ds_fcst_last = ds_fcst_seq.isel(time=-1)

    except Exception as e:
        print(f"Read Error: {e}")
        return

    rec = tracking_df[(tracking_df['typhoon_id'] == tid) & (tracking_df['date'] == sdate)]
    if rec.empty: return
    try:
        track = json.loads(rec.iloc[0]['track_info'].replace("'", '"'))
        cma_lat = track['lats'][hour // 6]
        cma_lon = track['lons'][hour // 6]
    except:
        return

    # === Step A: WRF (Master) ===
    # 1. 找 WRF 里的真眼
    wy, wx, w_lat, w_lon, w_var = find_eye_index(ds_wrf, cma_lat, cma_lon, 8.0, tag="WRF")
    if wy is None: return

    # 2. WRF 切片 (单帧)
    slice_y = get_clamped_slice(wy, cfg['out_radius'], ds_wrf.sizes['lat'])
    slice_x = get_clamped_slice(wx, cfg['out_radius'], ds_wrf.sizes['lon'])
    wrf_final, wrf_vars, anchor_lat, anchor_lon = crop_by_slice(ds_wrf, slice_y, slice_x)

    # === Step B: Forecast (Slave) ===
    # 1. 在 Forecast 最后一帧里找眼
    fy, fx, f_lat, f_lon, f_var = find_eye_index(ds_fcst_last, cma_lat, cma_lon, 8.0, tag="Fcst")
    if fy is None: return

    # 2. 计算物理 Offset
    offset_lat = w_lat - anchor_lat
    offset_lon = w_lon - anchor_lon

    # 3. 计算 Forecast 应该截取的中心 (基于最后一帧的位置)
    target_f_lat = f_lat - offset_lat
    target_f_lon = f_lon - offset_lon

    # 4. 截取 Forecast 序列 (核心修正 2: 传入整个序列)
    # 这会返回 (T, C, H, W)
    fcst_final, fcst_vars = crop_forecast_seq_aligned(ds_fcst_seq, target_f_lat, target_f_lon, cfg['in_radius'])
    if fcst_final is None: return

    # === 保存 & 画图 ===
    base_name = f"{save_dir_nc}/{tid}_{sdate}_{hour}"

    # 保存时已经是 4D 张量 [T, C, H, W]
    torch.save(torch.from_numpy(fcst_final).float(), f"{base_name}_input.pt")
    torch.save(torch.from_numpy(wrf_final).float(), f"{base_name}_target.pt")

    plot_check_final(fcst_final, fcst_vars, f_var, wrf_final, wrf_vars, w_var, tid, hour, debug_dir)
    print(f"[OK] {tid} {hour}h Saved. Shape: {fcst_final.shape}")


def main():
    df = pd.read_csv('./data_file/forecast_instances.csv')
    trk_main = pd.read_csv('./data_file/typhoon_tracking_results.csv')
    trk_2004 = pd.read_csv('./data_file/typhoon_tracking_results_2024.csv')

    trk = pd.concat([trk_main, trk_2004], ignore_index=True)

    save_dir = '/bigdata4/wxz_data/typhoon_intensity_bc/field_data_extraction'
    debug_dir = './debug'

    if not os.path.exists(save_dir): os.makedirs(save_dir)
    if not os.path.exists(debug_dir): os.makedirs(debug_dir)

    print(f"Processing (With Time Dimension T)...")
    print(f"Output Field Dir: {save_dir}")
    for idx, row in df.iterrows():
        process_case(row, trk, save_dir, debug_dir)


if __name__ == '__main__':
    main()
