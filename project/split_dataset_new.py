import pandas as pd
import numpy as np
import os


def save_file(df, filename):
    if not os.path.exists('./data_file'):
        os.makedirs('./data_file')
    df.to_csv(f'./data_file/{filename}', index=False)
    print(f"Saved {filename}: {df.shape}")
    if not df.empty and 'Class' in df.columns:
        dist = df['Class'].value_counts(normalize=True).sort_index().to_dict()
        dist_str = ", ".join([f"{k}: {v:.2%}" for k, v in dist.items()])
        print(f"  -> Class Dist: {dist_str}")


def strict_stratified_split(df, class_col='Class', time_col='Start Date', train_ratio=0.8):
    if df.empty:
        return df, df

    train_buckets = []
    valid_buckets = []

    # 获取所有出现的类别
    unique_classes = df[class_col].unique()

    for cls in unique_classes:
        df_cls = df[df[class_col] == cls].copy()

        df_cls = df_cls.sort_values(by=time_col)

        n_total = len(df_cls)
        n_train = int(n_total * train_ratio)

        if n_train == 0 and n_total > 0:
            n_train = 1

        curr_train = df_cls.iloc[:n_train]
        curr_valid = df_cls.iloc[n_train:]

        train_buckets.append(curr_train)
        valid_buckets.append(curr_valid)

    final_train = pd.concat(train_buckets).sort_values(by=time_col).reset_index(drop=True)
    final_valid = pd.concat(valid_buckets).sort_values(by=time_col).reset_index(drop=True)

    return final_train, final_valid


def process_pipeline():
    data_path = './data_file/forecast_instances.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    data = pd.read_csv(data_path)
    data['parsed_date'] = pd.to_datetime(data['Start Date'].astype(str), format='%Y%m%d%H')
    data['year'] = data['parsed_date'].dt.year

    forecast_hours = [24, 48, 72, 96]

    for hours in forecast_hours:
        print(f"\n{'=' * 20} Processing {hours}h forecast {'=' * 20}")
        df_hour = data[data['Forecast Hour'] == hours].copy()

        # === 阶段一：WRF 预训练 (2020-2022) ===
        df_wrf = df_hour[df_hour['year'].isin([2020, 2021, 2022])].copy()

        pre_train, pre_valid = strict_stratified_split(
            df_wrf, class_col='Class', time_col='Start Date', train_ratio=0.8
        )

        # === 阶段二：SHTM 微调 (2023) ===
        df_shtm_2023 = df_hour[df_hour['year'] == 2023].copy()

        ft_train, ft_valid = strict_stratified_split(
            df_shtm_2023, class_col='Class', time_col='Start Date', train_ratio=0.7
        )

        # === 阶段三：SHTM 测试 (2024) ===
        test_set = df_hour[df_hour['year'] == 2024].copy()

        cols_to_drop = ['parsed_date', 'year']

        save_file(pre_train.drop(columns=cols_to_drop), f'forecast_{hours}h_pretrain_train.csv')
        save_file(pre_valid.drop(columns=cols_to_drop), f'forecast_{hours}h_pretrain_valid.csv')

        save_file(ft_train.drop(columns=cols_to_drop), f'forecast_{hours}h_finetune_train.csv')
        save_file(ft_valid.drop(columns=cols_to_drop), f'forecast_{hours}h_finetune_valid.csv')

        save_file(test_set.drop(columns=cols_to_drop), f'forecast_{hours}h_test_set.csv')


if __name__ == '__main__':
    process_pipeline()