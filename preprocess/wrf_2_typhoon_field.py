import pandas as pd
import xarray as xr
import numpy as np
import os
from datetime import datetime

# ================= 配置区域 =================

# 台风路径 CSV 文件
csv_path = "/data3/WangGuanSong/Weaformer/all_models/weaformer_v2.0/typhoon_intensity_bc/data_file/typhoons.csv"

# WRF 数据根目录 pattern
wrf_base_pattern = "/bigdata4/wxz_data/forwang/{year}{month}/"
# WRF 文件名 pattern
wrf_filename_pattern = "wrfout_d01_{year}-{month}-{day}_{hour}_00_00.nc"

# 输出目录
output_dir = "/bigdata4/wxz_data/typhoon_intensity_bc/hr_field_data_wrf/"

# 网格设置 (以台风为中心，向四面各扩展 9 度)
# 半径 9度，对应直径 18度
EXTENT_DEG = 9.0
GRID_SIZE = 181
# lat: 5度 ~ 57度
PHYS_LAT_MIN, PHYS_LAT_MAX = 5.0, 57.0
# lon: 80度 ~ 152度
PHYS_LON_MIN, PHYS_LON_MAX = 80.0, 152.0

# 变量对应关系
LEVEL_VAR_MAP = {
    0: 't2m',
    1: 'u10',
    2: 'v10',
    3: 'mslp'
}


# ===========================================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def process_typhoon_data():
    ensure_dir(output_dir)

    print(f"正在读取台风记录: {csv_path}")
    df = pd.read_csv(csv_path, dtype={'ID': str, 'Date': str})

    count_success = 0
    count_missing = 0
    count_error = 0

    total_samples = len(df)
    print(f"总计样本数: {total_samples}")

    for index, row in df.iterrows():
        if index % 100 == 0:
            print(f"进度: {index}/{total_samples} ...")

        try:
            # 1. 解析信息
            typhoon_id = str(row['ID'])
            date_str = row['Date'].strip()

            center_lat = float(row['Latitude'])
            center_lon = float(row['Longitude'])

            dt = datetime.strptime(date_str, "%Y%m%d%H")
            year_str = dt.strftime("%Y")
            month_str = dt.strftime("%m")
            day_str = dt.strftime("%d")
            hour_str = dt.strftime("%H")

            # 2. 构建路径
            current_wrf_dir = wrf_base_pattern.format(year=year_str, month=month_str)
            current_wrf_file = wrf_filename_pattern.format(year=year_str, month=month_str, day=day_str, hour=hour_str)
            full_wrf_path = os.path.join(current_wrf_dir, current_wrf_file)

            if not os.path.exists(full_wrf_path):
                count_missing += 1
                continue

            # 3. 处理数据
            with xr.open_dataset(full_wrf_path) as ds:
                # 赋予坐标
                n_lat = ds.sizes['lat']
                n_lon = ds.sizes['lon']
                phys_lat_coords = np.linspace(PHYS_LAT_MIN, PHYS_LAT_MAX, n_lat)
                phys_lon_coords = np.linspace(PHYS_LON_MIN, PHYS_LON_MAX, n_lon)
                ds = ds.assign_coords(lat=phys_lat_coords, lon=phys_lon_coords)

                if 'my_output' not in ds.variables:
                    count_error += 1
                    continue

                data_var = ds['my_output']

                # 计算目标 Box (不做判定，直接算)
                target_lat_min = center_lat - EXTENT_DEG
                target_lat_max = center_lat + EXTENT_DEG
                target_lon_min = center_lon - EXTENT_DEG
                target_lon_max = center_lon + EXTENT_DEG

                target_lat = np.linspace(target_lat_min, target_lat_max, GRID_SIZE)
                target_lon = np.linspace(target_lon_min, target_lon_max, GRID_SIZE)

                ds_out = xr.Dataset()

                for level_idx, var_name in LEVEL_VAR_MAP.items():
                    sub_data = data_var.isel(level=level_idx)

                    interpolated = sub_data.interp(
                        lat=target_lat,
                        lon=target_lon,
                        method='linear',
                        kwargs={"fill_value": np.nan}
                    )

                    # 检查 NaN 并填充
                    if np.isnan(interpolated).any():
                        interpolated = interpolated.fillna(0.0)

                    ds_out[var_name] = interpolated

                # 4. 保存
                if 'time' not in ds_out.dims:
                    ds_out = ds_out.expand_dims('time')
                    ds_out.coords['time'] = [np.datetime64(dt)]

                ds_out.attrs['typhoon_id'] = typhoon_id
                ds_out.attrs['init_time'] = date_str
                ds_out.attrs['center_lat'] = center_lat
                ds_out.attrs['center_lon'] = center_lon

                out_filename = f"{typhoon_id}_{date_str}_hr.nc"
                out_file_path = os.path.join(output_dir, out_filename)

                encoding = {v: {'zlib': True, 'complevel': 4} for v in LEVEL_VAR_MAP.values()}
                ds_out.to_netcdf(out_file_path, encoding=encoding)

                count_success += 1
                print(f"[OK] {typhoon_id} {date_str}")

        except Exception as e:
            print(f"[ERROR] 处理 {typhoon_id}_{date_str} 失败: {e}")
            count_error += 1

    print("\n" + "=" * 30)
    print(f"处理完成 Summary:")
    print(f"  成功生成: {count_success}")
    print(f"  文件缺失: {count_missing}")
    print(f"  其他错误: {count_error}")
    print("=" * 30)


if __name__ == "__main__":
    process_typhoon_data()
