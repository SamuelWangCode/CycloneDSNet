import json

import pandas as pd
import torch

forecast_df = pd.read_csv('./data_file/forecast_instances.csv')
tracking_df = pd.read_csv('./data_file/typhoon_tracking_results_2024.csv')
typhoons_df = pd.read_csv('./data_file/typhoons.csv')

tracking_df.rename(columns={'typhoon_id': 'ID', 'date': 'Start Date'}, inplace=True)
typhoons_df.rename(columns={'Date': 'Start Date'}, inplace=True)

merged_df = forecast_df.merge(tracking_df, on=['ID', 'Start Date'])
final_df = merged_df.merge(typhoons_df, on=['ID', 'Start Date'])

for index, row in final_df.iterrows():
    print(index)
    start_date = pd.to_datetime(row['Start Date'], format='%Y%m%d%H')

    T = row['Forecast Hour'] // 6 + 1
    track_info = row['track_info']
    track_info = json.loads(track_info.replace("'", '"'))

    input_tensor = torch.zeros(T, 4)
    target_tensor = torch.zeros(T, 4)

    for t in range(T):
        forecast_time = start_date + pd.Timedelta(hours=t * 6)
        matching_row = typhoons_df[
            (typhoons_df['ID'] == row['ID']) & (typhoons_df['Start Date'] == int(forecast_time.strftime('%Y%m%d%H')))]
        input_tensor[t] = torch.tensor(
            [track_info['lons'][t], track_info['lats'][t], track_info['pmin'][t], track_info['vmax'][t]])
        if not matching_row.empty:
            target_tensor[t] = torch.tensor([
                matching_row.iloc[0]['Longitude'],
                matching_row.iloc[0]['Latitude'],
                matching_row.iloc[0]['Pressure'],
                matching_row.iloc[0]['Wind Speed']
            ])

    # 保存tensor
    torch.save(input_tensor,
               f'/bigdata4/wxz_data/typhoon_intensity_bc/value_data_extraction/{row["ID"]}_{row["Start Date"]}_{row["Forecast Hour"]}_input.pt')
    torch.save(target_tensor,
               f'/bigdata4/wxz_data/typhoon_intensity_bc/value_data_extraction/{row["ID"]}_{row["Start Date"]}_{row["Forecast Hour"]}_target.pt')
