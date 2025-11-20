import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import PercentFormatter
import pandas as pd

from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch
# import matplotlib.patches as mpatches  # 新增的导入

import re
import shutil
from pathlib import Path
from scipy import stats

def plot_belowpic(data, subfolders, colors, num_bins=5):
    """
    自定义绘制箱型图和小提琴图，实现精确控制间隔和对齐
    
    核心改动：
    1. 手动计算箱型图位置，实现自定义间隔
    2. 小提琴图使用相同的位置计算逻辑
    3. 统一设置箱型图和小提琴图的样式参数
    """
    # ---------------- 字体设置 ----------------
    try:
        font_families = ['Arial', 'DejaVu Sans', 'Liberation Sans']
        available_fonts = [f.name for f in FontProperties().get_fontconfig_fonts()]
        font = next((fam for fam in font_families if fam in available_fonts), 'sans-serif')
        plt.rcParams["font.family"] = font
    except Exception as e:
        print(f"Font warning: {e}")
    plt.rcParams['axes.unicode_minus'] = False

    # ---------------- 数据预处理 ----------------
    # 提取IMU误差级别并排序
    valid_keyframes = [kf for kf in subfolders if kf in data]
    if not valid_keyframes:
        print("Error: No valid keyframes found!")
        return
    
    # 获取IMU误差列表并排序
    imu_errors = list(data[valid_keyframes[0]].keys())
    try:
        imu_errors = sorted(imu_errors, key=float)  # 数值排序
    except:
        print("Warning: IMU errors cannot be sorted numerically")
    
    # 收集所有数据
    all_data = []
    for kf in subfolders:
        if kf not in data:
            continue
        for imu_err in imu_errors:
            if imu_err not in data[kf]:
                continue
            for result in data[kf][imu_err]:
                imu_epe = np.array(result['IMU_epe_error'])
                epes = np.array(result['epes'])
                # 计算改进百分比
                valid_mask = imu_epe > 1e-6  # 避免除零
                if np.sum(valid_mask) > 0:
                    improvement = (imu_epe[valid_mask] - epes[valid_mask]) / imu_epe[valid_mask]
                    for val in improvement:
                        all_data.append({
                            'Keyframe': kf,
                            'IMU_Error': imu_err,
                            'Improvement': val
                        })
    
    if not all_data:
        print("Error: No valid data found!")
        return
    
    df = pd.DataFrame(all_data)
    
    # ---------------- 绘图核心逻辑 ----------------
    plt.figure(figsize=(21, 3.5))  # 调整画布大小
    ax = plt.gca()
    
    # 计算箱型图和小提琴图的位置参数
    n_keyframes = len(subfolders)
    n_imu_errors = len(imu_errors)
    mid_idx = (n_keyframes - 1) / 2  # 中间索引，用于计算偏移量
    spacing = 0.15  # 箱型图之间的间隔系数
    violin_width = 0.12  # 小提琴图宽度
    box_width = 0.06  # 箱型图宽度
    
    # 存储图例句柄
    legend_handles = []
    
    # 绘制小提琴图和箱型图
    for i, kf in enumerate(subfolders):
        kf_data = df[df['Keyframe'] == kf]
        color = colors[i]
        
        # 小提琴图数据
        violin_data = []
        positions = []
        
        for j, imu_err in enumerate(imu_errors):
            # 计算偏移位置
            offset = (i - mid_idx) * spacing
            pos = j + offset
            
            # 获取当前IMU误差的数据
            imu_data = kf_data[kf_data['IMU_Error'] == imu_err]['Improvement'].dropna()
            
            if not imu_data.empty:
                violin_data.append(imu_data)
                positions.append(pos)
        
        # 绘制小提琴图
        if violin_data:
            parts = ax.violinplot(
                violin_data, 
                positions=positions,
                widths=violin_width,
                showmeans=False,
                showmedians=False,
                showextrema=False
            )
            
            # 设置小提琴图样式
            for pc in parts['bodies']:
                pc.set_facecolor(color)
                pc.set_edgecolor('white')
                pc.set_alpha(0.3)
        
        # 箱型图样式设置 - 调整至与参考代码一致
        boxprops = {'color': 'gray', 'linewidth': 0.8}  # 边框为黑色，线宽1.5
        whiskerprops = {'color': 'gray', 'linewidth': 0.8}
        medianprops = {'color': 'black', 'linewidth': 0.9}
        meanprops = {
            'marker': 'o', 'markerfacecolor': 'white', 
            'markeredgecolor': 'gray', 'markersize': 6,
            'markeredgewidth': 0.8
        }
        flierprops = {
            'marker': '.', 'markersize': 2,
            'alpha': 0.2, 'markerfacecolor': 'gray',
            'markeredgecolor': 'gray'
        }
        
        # 箱型图数据
        boxplot_data = []
        box_positions = []
        
        for j, imu_err in enumerate(imu_errors):
            # 计算偏移位置
            offset = (i - mid_idx) * spacing
            pos = j + offset
            
            # 获取当前IMU误差的数据
            imu_data = kf_data[kf_data['IMU_Error'] == imu_err]['Improvement'].dropna()
            
            if not imu_data.empty:
                boxplot_data.append(imu_data)
                box_positions.append(pos)
        
        # 绘制箱型图
        if boxplot_data:
            box = ax.boxplot(
                boxplot_data, 
                positions=box_positions,
                widths=box_width,
                patch_artist=True,
                boxprops=boxprops,
                whiskerprops=whiskerprops,
                medianprops=medianprops,
                meanprops=meanprops,
                meanline=False,
                showmeans=True,
                flierprops=flierprops,
                manage_ticks=False,
                zorder=3  # 确保箱型图在最上层
            )
            
            # 设置箱型图填充颜色
            for patch in box['boxes']:
                patch.set_facecolor(color)
                patch.set_alpha(0.7)  # 略微透明的填充
        
        # 添加图例
        legend_handles.append(Patch(color=color, label=f"Keyframe = {kf[2:]}"))

    
    # ---------------- 美化与布局 ----------------
    # 设置x轴刻度和标签
    ax.set_xticks(range(len(imu_errors)))
    labels = []
    for imu_error in imu_errors:
        label = f'LEVEL:{imu_error}\nnoise_gyro={float(imu_error)*0.48:.2f} noise_accel={float(imu_error)*0.073:.2f}'
        labels.append(label)

    
    ax.set_xticklabels(labels)
    
    # 纵轴设置
    ax.set_ylim(-0.05, 1.05)  # 预留空间
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    
    # 坐标轴标签
    ax.set_xlabel('IMU Error Level', fontsize=16)
    ax.set_ylabel('Improvement over IMU', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    
    # 网格线
    ax.grid(True, linestyle='--', alpha=0.4, axis='y')
    
    # 图例
    ax.legend(handles=legend_handles,  fontsize=16, 
              loc='upper left', bbox_to_anchor=(1.02, 1))
    
    # # 标题
    # plt.title('(b) IMU Error vs Keyframe: Positioning Performance Improvement', 
    #           fontsize=15, fontweight='bold', y=1.05)
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(right=0.8)  # 为图例留出空间
    plt.savefig('raw_data/robustness_evaluation_below.png', dpi=300, bbox_inches='tight')
    # plt.show()


def load_data_total(root_file_dir, subfolders):
    "data结构为：data[keyrame][IMU error]['keys]"
    data = {}
    for key in root_file_dir:
        for subfolder in subfolders:
            # print(str(root_file_dir[key]))
            folder_path = os.path.join(root_file_dir[key], subfolder)
            if not os.path.exists(folder_path):
                continue
                
            # 确保子字典存在
            if subfolder not in data:
                data[subfolder] = {}
                
            # 初始化空列表
            data[subfolder][key] = []
            
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.pkl'):
                    file_path = os.path.join(folder_path, file_name)
                    with open(file_path, 'rb') as f:
                        data[subfolder][key].append(pickle.load(f))

    return data

def load_data(root_path,subfolders):
    data = {}
    for subfolder in subfolders:
        folder_path = os.path.join(root_path, subfolder)
        if not os.path.exists(folder_path):
            continue
        data[subfolder] = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.pkl'):
                file_path = os.path.join(folder_path, file_name)
                with open(file_path, 'rb') as f:
                    data[subfolder].append(pickle.load(f))
    return data


def plot_abovepic(data, subfolders, colors, num_bins=6):
    """
    基于plot_experiment_total_2模板重构的实验6绘图函数
    1. 箱型图样式与模板一致
    2. 密度图颜色参考小提琴图风格
    3. 密度图使用统一面积而非峰值归一化
    """
    # ---------------- 1. 字体设置（采用plot_experiment_total_2的配置） ----------------
    try:
        font_families = ['Arial', 'DejaVu Sans', 'Liberation Sans']
        available_fonts = [f.name for f in FontProperties().get_fontconfig_fonts()]
        font = next((fam for fam in font_families if fam in available_fonts), 'sans-serif')
        plt.rcParams["font.family"] = font
    except Exception as e:
        print(f"Font warning: {e}")
    plt.rcParams['axes.unicode_minus'] = False

    # ---------------- 2. 数据预处理（保留原实验6的epes区间逻辑） ----------------
    # 全局计算epes区间
    all_epes = []
    for subfolder in subfolders:
        if subfolder not in data:
            continue
        for result in data[subfolder]:
            all_epes.extend(result['epes'])
    
    if not all_epes:
        max_epes_global = 0.001
    else:
        max_epes_global = max(all_epes)
        max_epes_global = min(0.1, max_epes_global)
        max_epes_global = max_epes_global if max_epes_global > 1e-6 else 0.001
    
    min_epes_global = 0.0
    step_global = (max_epes_global - min_epes_global) / num_bins
    epes_bins_global = [min_epes_global + i * step_global for i in range(num_bins + 1)]
    
    epes_labels_global = []
    seen = set()
    for j in range(num_bins):
        lower = epes_bins_global[j]
        upper = epes_bins_global[j+1]
        if j < num_bins - 1:
            label = f'{lower:.2f}-{upper:.2f}'
        else:
            label = f'{lower:.2f}+'
        if label in seen:
            suffix = 1
            new_label = f"{label}_{suffix}"
            while new_label in seen:
                suffix += 1
                new_label = f"{label}_{suffix}"
            label = new_label
        seen.add(label)
        epes_labels_global.append(label)

    # ---------------- 3. 绘图核心逻辑（采用total_2模板风格） ----------------
    plt.figure(figsize=(21, 3.5))  # 使用total_2的画布比例
    ax = plt.gca()
    ax2 = ax.twinx()

    # 坐标轴层级设置（箱型图在上，密度图在下）
    ax.set_zorder(2)
    ax2.set_zorder(1)
    ax.patch.set_visible(False)

    # 计算位置参数（参考total_2的偏移逻辑）
    n_subfolders = len(subfolders)
    mid_idx = (n_subfolders - 1) / 2  # 中间索引用于计算偏移
    spacing = 0.15  # 箱型图间隔系数（与total_2一致）
    box_width = 0.1  # 箱型图宽度（与total_2一致）

    # 存储图例句柄
    legend_handles = []

 # 收集所有改进百分比数据，用于统计
    all_improvement_data = {}

    # 绘制箱型图和密度图
    for i, subfolder in enumerate(subfolders):
        # 数据提取与处理
        epes_values = []
        all_pcent = []
        if subfolder not in data:
            print(f'警告: 子文件夹 {subfolder} 不在数据中')
            continue
        for result in data[subfolder]:
            imu_errors = np.array(result['IMU_epe_error'])
            epes = np.array(result['epes'])
            pcent = (imu_errors - epes) / imu_errors  # 计算改进百分比
            epes_values.extend(epes)
            all_pcent.extend(pcent)
        
        if not epes_values:
            print(f'警告: 子文件夹 {subfolder} 没有有效的epes数据')
            continue
        # 保存改进百分比数据用于统计
        all_improvement_data[subfolder] = all_pcent

        # 按全局区间分组
        total = len(epes_values)
        df = pd.DataFrame({'epes': epes_values, 'all_pcent': all_pcent})
        try:
            df['epes_interval'] = pd.cut(
                df['epes'], 
                bins=epes_bins_global, 
                labels=epes_labels_global, 
                include_lowest=True
            )
        except Exception as e:
            print(f"警告: 无法为 {subfolder} 创建区间: {e}")
            continue

        # 箱型图数据准备
        boxplot_data = []
        positions = []
        for j, label in enumerate(epes_labels_global):
            # 计算偏移位置（与total_2逻辑一致）
            offset = (i - mid_idx) * spacing
            pos = j + offset
            positions.append(pos)
            
            # 获取当前区间数据
            interval_data = df[df['epes_interval'] == label]['all_pcent'].dropna().tolist()
            boxplot_data.append(interval_data)

        # 绘制箱型图（采用total_2的样式参数）
        boxprops = {'color': 'gray', 'linewidth': 0.8}  # 边框样式（与total_2一致）
        whiskerprops = {'color': 'gray', 'linewidth': 0.8}  # 须线样式
        medianprops = {'color': 'black', 'linewidth': 0.9}  # 中位数线样式
        meanprops = {
            'marker': 'o', 'markerfacecolor': 'white', 
            'markeredgecolor': 'gray', 'markersize': 6,
            'markeredgewidth': 0.8
        }  # 均值点样式
        flierprops = {
            'marker': 'o', 'markersize': 2,
            'alpha': 0.2, 'markerfacecolor': 'gray',
            'markeredgecolor': 'gray'
        }  # 离群点样式

        box = ax.boxplot(
            boxplot_data, 
            positions=positions,
            widths=box_width,  # 使用total_2的箱型图宽度
            patch_artist=True,
            boxprops=boxprops,
            whiskerprops=whiskerprops,
            medianprops=medianprops,
            meanprops=meanprops,
            meanline=False,
            showmeans=True,
            flierprops=flierprops,
            manage_ticks=False,
            zorder=3  # 确保在最上层
        )

        # 设置箱型图填充颜色（与total_2一致）
        for patch in box['boxes']:
            patch.set_facecolor(colors[i])
            patch.set_alpha(0.7)  # 与total_2的小提琴图alpha区分

        # 绘制密度图（参考小提琴图颜色风格）
        if len(epes_values) >= 2:
            # 转换epes值到区间坐标
            x_pos_epes = [(e / max_epes_global) * num_bins - 0.5 for e in epes_values]
            kde = stats.gaussian_kde(x_pos_epes)
            kde.set_bandwidth(bw_method=kde.factor * 0.5)  # 保持带宽设置
            
            # 密度计算与面积归一化（关键修改：面积一致）
            x_range = np.linspace(-0.8, num_bins - 0.2, 200)  # 匹配x轴范围
            density = kde(x_range)
            
            # 计算当前密度曲线的面积（积分），并调整至固定面积
            area = np.trapz(density, x_range)  # 计算积分面积
            if area > 1e-6:  # 避免除零
                target_area = 0.2  # 固定目标面积（可调整）
                density = density * (target_area / area)  # 缩放至目标面积
            
            # 绘制密度图（参考小提琴图样式）
            ax2.fill_between(
                x_range, density, 0, 
                color=colors[i], 
                alpha=0.3,  # 与小提琴图alpha一致
                zorder=1
            )
            ax2.plot(
                x_range, density, 
                color=colors[i], 
                alpha=0.7,  # 略高于填充透明度
                linewidth=0.8,  # 细线风格
                zorder=1
            )

        # 添加区间占比标注（保留原功能）
        for j in range(len(epes_labels_global)):
            lower = epes_bins_global[j]
            upper = epes_bins_global[j+1] if j < num_bins-1 else max_epes_global
            count = df[(df['epes'] >= lower) & (df['epes'] < upper)].shape[0]
            prop = count / total if total != 0 else 0

            # 使用颜色加深函数（关键修改）
            import matplotlib.colors as mcolors
            rgb = mcolors.to_rgb(colors[i])
            darker_color = tuple(max(0, c - 0.2) for c in rgb)  # 降低亮度

            # 固定标注位置，确保所有箱型标注在统一水平线上
            fixed_y_position = 1
            # print('=====================',fixed_y_position)
            ax.text(
                positions[j], 
                fixed_y_position,  # 使用固定的 y 位置
                f"{prop:.0%}", 
                ha='center', va='bottom', 
                fontsize=10, 
                color=darker_color
            )

        # 添加图例句柄
        values = all_improvement_data[subfolder]
        legend_handles.append(Patch(color=colors[i], label=f"KeyFrame = {subfolder[2:]}\nMean   = {np.mean(values):.2%}\nMedian = {np.median(values):.2%}"))

    # X轴设置
    ax.set_xlim(-0.8, num_bins - 0.2)
    ax.set_xticks(range(len(epes_labels_global)))
    ax.set_xticklabels(epes_labels_global, rotation=0, fontsize=14)  # 字体大小与total_2一致
    ax.set_xlabel('IMU epes error (m)', fontsize=16)  # 与total_2字体大小一致

    # Y轴设置（左侧：改进百分比）
    ax.set_ylabel('Improvement over IMU', fontsize=16)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylim(0, 1.1)  # 保留足够空间给占比标注
    ax.tick_params(axis='both', labelsize=14)  # 刻度字体大小

    # 右侧密度图轴设置
    ax2.set_yticks([])
    ax2.set_ylim(0, 0.5)  # 固定密度图范围
    ax2.set_ylabel('')

    # 网格线
    ax.grid(True, linestyle='--', alpha=0.4, axis='y')


    # 图例
    ax.legend(
        handles=legend_handles, 
        fontsize=16,
        loc='upper left', 
        bbox_to_anchor=(1.02, 1),
        markerscale=1.2, scatterpoints=2,
        labelspacing=1.5, handlelength=1.5, handleheight = 2
    )

    # # 标题
    # plt.title('Improvement over IMU by KeyFrame and Epes Interval', fontsize=16)

    # 布局调整
    plt.tight_layout()
    plt.subplots_adjust(right=0.8)  # 为图例留空间
    plt.savefig('raw_data/robustness_evaluation_above.png', dpi=300, bbox_inches='tight')
    # plt.show()

def plot_below():
    root_file_dir = {'0.25':'raw_data/Robustness_evaluation/checkpoint_epoch151_error025',
                     '0.5':'raw_data/Robustness_evaluation/checkpoint_epoch151_error05',
                     '1':'raw_data/Robustness_evaluation/checkpoint_epoch151_error1',
                     '2':'raw_data/Robustness_evaluation/checkpoint_epoch151_error2',}

    subfolders = ['kf1', 'kf2', 'kf3', 'kf4', 'kf5', 'kf6']
    colors = ['#F9D977', '#F7B78C', '#F79F98', '#C9A6C9', '#A6CFE7', '#69A2E0']

    data = load_data_total(root_file_dir,subfolders)
    plot_belowpic(data,subfolders,colors)

def plot_above():
    root_file = 'raw_data/Robustness_evaluation/checkpoint_epoch151_error1'

    subfolders = ['kf1', 'kf2', 'kf3', 'kf4', 'kf5', 'kf6']
    colors = ['#F9D977', '#F7B78C', '#F79F98', '#C9A6C9', '#A6CFE7', '#69A2E0']

    data = load_data(root_file,subfolders) 
    # plot_experiment_5_y(data,subfolders,colors)
    plot_abovepic(data,subfolders,colors)


if __name__ == '__main__':
    plot_above()
    plot_below()
    

