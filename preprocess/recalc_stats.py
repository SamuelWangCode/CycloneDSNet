import os
import torch
import numpy as np
import glob
from tqdm import tqdm

# ================= 配置 =================
ROOT_DATASET = '/bigdata4/wxz_data/typhoon_intensity_bc/field_data_extraction'
SAVE_DIR = './data_file/stats'
SAVE_NAME = 'hr_minmax_stats.npz'


# =======================================

def recalc_stats():
    # 1. 准备路径
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    target_files = glob.glob(os.path.join(ROOT_DATASET, "*_target.pt"))
    if not target_files:
        print("[Error] 找不到文件！")
        return

    print(f"开始重新统计 Min/Max，共 {len(target_files)} 个文件...")

    # 初始化 Min/Max
    # 通道顺序已固定为: [0:U, 1:V, 2:T, 3:P]
    # 使用 float('inf') 初始化
    global_min = torch.full((4,), float('inf'))
    global_max = torch.full((4,), float('-inf'))

    # 2. 循环扫描
    for fpath in tqdm(target_files):
        try:
            # 加载 (4, H, W)
            data = torch.load(fpath, weights_only=False)
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data)

            # 展平 spatial 维度，变为 (4, -1)
            flat_data = data.view(4, -1).float()

            # 计算当前文件的 min/max
            curr_min = flat_data.min(dim=1)[0]  # (4,)
            curr_max = flat_data.max(dim=1)[0]  # (4,)

            # 更新全局 min/max
            global_min = torch.minimum(global_min, curr_min)
            global_max = torch.maximum(global_max, curr_max)

        except Exception as e:
            print(f"[Warn] 读取失败: {fpath}")

    # 3. 结果处理
    print("\n================ 统计完成 ================")
    channels = ['U (m/s)', 'V (m/s)', 'T (K)', 'P (Pa)']

    vmin = global_min.numpy()
    vmax = global_max.numpy()

    for i, name in enumerate(channels):
        print(f"Channel {i} [{name}]: Min={vmin[i]:.4f}, Max={vmax[i]:.4f}")

    # 检查 P 是否真的是 Pa
    if vmax[3] < 2000:
        print("\n气压最大值仍然小于 2000！请检查数据清洗是否成功！")
    else:
        print("\n气压数值看起来正常 (Pa 级)。")

    # 4. 保存
    save_path = os.path.join(SAVE_DIR, SAVE_NAME)
    np.savez(save_path, vmin=vmin, vmax=vmax)
    print(f"新的统计文件已保存至: {save_path}")


if __name__ == '__main__':
    recalc_stats()
