import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import PercentFormatter
import pandas as pd

from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch

import re
import shutil
from pathlib import Path
from scipy import stats

def plot_belowpic(data, subfolders, colors, num_bins=5):
    try:
        font_families = ['Arial', 'DejaVu Sans', 'Liberation Sans']
        available_fonts = [f.name for f in FontProperties().get_fontconfig_fonts()]
        font = next((fam for fam in font_families if fam in available_fonts), 'sans-serif')
        plt.rcParams["font.family"] = font
    except Exception as e:
        print(f"Font warning: {e}")
    plt.rcParams['axes.unicode_minus'] = False

    valid_keyframes = [kf for kf in subfolders if kf in data]
    if not valid_keyframes:
        print("Error: No valid keyframes found!")
        return
    
    imu_errors = list(data[valid_keyframes[0]].keys())
    try:
        imu_errors = sorted(imu_errors, key=float)
    except:
        print("Warning: IMU errors cannot be sorted numerically")
    
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
                valid_mask = imu_epe > 1e-6
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
    
    plt.figure(figsize=(21, 3.5))
    ax = plt.gca()
    
    n_keyframes = len(subfolders)
    n_imu_errors = len(imu_errors)
    mid_idx = (n_keyframes - 1) / 2
    spacing = 0.15
    violin_width = 0.12
    box_width = 0.06
    
    legend_handles = []
    
    for i, kf in enumerate(subfolders):
        kf_data = df[df['Keyframe'] == kf]
        color = colors[i]
        
        violin_data = []
        positions = []
        
        for j, imu_err in enumerate(imu_errors):
            offset = (i - mid_idx) * spacing
            pos = j + offset
            
            imu_data = kf_data[kf_data['IMU_Error'] == imu_err]['Improvement'].dropna()
            
            if not imu_data.empty:
                violin_data.append(imu_data)
                positions.append(pos)
        
        if violin_data:
            parts = ax.violinplot(
                violin_data, 
                positions=positions,
                widths=violin_width,
                showmeans=False,
                showmedians=False,
                showextrema=False
            )
            
            for pc in parts['bodies']:
                pc.set_facecolor(color)
                pc.set_edgecolor('white')
                pc.set_alpha(0.3)
        
        boxprops = {'color': 'gray', 'linewidth': 0.8}
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
        
        boxplot_data = []
        box_positions = []
        
        for j, imu_err in enumerate(imu_errors):
            offset = (i - mid_idx) * spacing
            pos = j + offset
            
            imu_data = kf_data[kf_data['IMU_Error'] == imu_err]['Improvement'].dropna()
            
            if not imu_data.empty:
                boxplot_data.append(imu_data)
                box_positions.append(pos)
        
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
                zorder=3
            )
            
            for patch in box['boxes']:
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        legend_handles.append(Patch(color=color, label=f"Keyframe = {kf[2:]}"))

    ax.set_xticks(range(len(imu_errors)))
    labels = []
    for imu_error in imu_errors:
        label = f'LEVEL:{imu_error}\nnoise_gyro={float(imu_error)*0.48:.2f} noise_accel={float(imu_error)*0.073:.2f}'
        labels.append(label)

    ax.set_xticklabels(labels)
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel('IMU Error Level', fontsize=16)
    ax.set_ylabel('Improvement over IMU', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, linestyle='--', alpha=0.4, axis='y')
    ax.legend(handles=legend_handles,  fontsize=16, 
              loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.subplots_adjust(right=0.8)
    plt.savefig('raw_data/robustness_evaluation_below.png', dpi=300, bbox_inches='tight')

def load_data_total(root_file_dir, subfolders):
    data = {}
    for key in root_file_dir:
        for subfolder in subfolders:
            folder_path = os.path.join(root_file_dir[key], subfolder)
            if not os.path.exists(folder_path):
                continue
                
            if subfolder not in data:
                data[subfolder] = {}
                
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
    try:
        font_families = ['Arial', 'DejaVu Sans', 'Liberation Sans']
        available_fonts = [f.name for f in FontProperties().get_fontconfig_fonts()]
        font = next((fam for fam in font_families if fam in available_fonts), 'sans-serif')
        plt.rcParams["font.family"] = font
    except Exception as e:
        print(f"Font warning: {e}")
    plt.rcParams['axes.unicode_minus'] = False

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

    plt.figure(figsize=(21, 3.5))
    ax = plt.gca()
    ax2 = ax.twinx()

    ax.set_zorder(2)
    ax2.set_zorder(1)
    ax.patch.set_visible(False)

    n_subfolders = len(subfolders)
    mid_idx = (n_subfolders - 1) / 2
    spacing = 0.15
    box_width = 0.1

    legend_handles = []

    all_improvement_data = {}

    for i, subfolder in enumerate(subfolders):
        epes_values = []
        all_pcent = []
        if subfolder not in data:
            print(f'Warning: Subfolder {subfolder} not in data')
            continue
        for result in data[subfolder]:
            imu_errors = np.array(result['IMU_epe_error'])
            epes = np.array(result['epes'])
            pcent = (imu_errors - epes) / imu_errors
            epes_values.extend(epes)
            all_pcent.extend(pcent)
        
        if not epes_values:
            print(f'Warning: Subfolder {subfolder} has no valid epes data')
            continue
        all_improvement_data[subfolder] = all_pcent

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
            print(f"Warning: Cannot create intervals for {subfolder}: {e}")
            continue

        boxplot_data = []
        positions = []
        for j, label in enumerate(epes_labels_global):
            offset = (i - mid_idx) * spacing
            pos = j + offset
            positions.append(pos)
            
            interval_data = df[df['epes_interval'] == label]['all_pcent'].dropna().tolist()
            boxplot_data.append(interval_data)

        boxprops = {'color': 'gray', 'linewidth': 0.8}
        whiskerprops = {'color': 'gray', 'linewidth': 0.8}
        medianprops = {'color': 'black', 'linewidth': 0.9}
        meanprops = {
            'marker': 'o', 'markerfacecolor': 'white', 
            'markeredgecolor': 'gray', 'markersize': 6,
            'markeredgewidth': 0.8
        }
        flierprops = {
            'marker': 'o', 'markersize': 2,
            'alpha': 0.2, 'markerfacecolor': 'gray',
            'markeredgecolor': 'gray'
        }

        box = ax.boxplot(
            boxplot_data, 
            positions=positions,
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
            zorder=3
        )

        for patch in box['boxes']:
            patch.set_facecolor(colors[i])
            patch.set_alpha(0.7)

        if len(epes_values) >= 2:
            x_pos_epes = [(e / max_epes_global) * num_bins - 0.5 for e in epes_values]
            kde = stats.gaussian_kde(x_pos_epes)
            kde.set_bandwidth(bw_method=kde.factor * 0.5)
            
            x_range = np.linspace(-0.8, num_bins - 0.2, 200)
            density = kde(x_range)
            
            area = np.trapz(density, x_range)
            if area > 1e-6:
                target_area = 0.2
                density = density * (target_area / area)
            
            ax2.fill_between(
                x_range, density, 0, 
                color=colors[i], 
                alpha=0.3,
                zorder=1
            )
            ax2.plot(
                x_range, density, 
                color=colors[i], 
                alpha=0.7,
                linewidth=0.8,
                zorder=1
            )

        for j in range(len(epes_labels_global)):
            lower = epes_bins_global[j]
            upper = epes_bins_global[j+1] if j < num_bins-1 else max_epes_global
            count = df[(df['epes'] >= lower) & (df['epes'] < upper)].shape[0]
            prop = count / total if total != 0 else 0

            import matplotlib.colors as mcolors
            rgb = mcolors.to_rgb(colors[i])
            darker_color = tuple(max(0, c - 0.2) for c in rgb)

            fixed_y_position = 1
            ax.text(
                positions[j], 
                fixed_y_position,
                f"{prop:.0%}", 
                ha='center', va='bottom', 
                fontsize=10, 
                color=darker_color
            )

        values = all_improvement_data[subfolder]
        legend_handles.append(Patch(color=colors[i], label=f"KeyFrame = {subfolder[2:]}\nMean   = {np.mean(values):.2%}\nMedian = {np.median(values):.2%}"))

    ax.set_xlim(-0.8, num_bins - 0.2)
    ax.set_xticks(range(len(epes_labels_global)))
    ax.set_xticklabels(epes_labels_global, rotation=0, fontsize=14)
    ax.set_xlabel('IMU epes error (m)', fontsize=16)

    ax.set_ylabel('Improvement over IMU', fontsize=16)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylim(0, 1.1)
    ax.tick_params(axis='both', labelsize=14)

    ax2.set_yticks([])
    ax2.set_ylim(0, 0.5)
    ax2.set_ylabel('')

    ax.grid(True, linestyle='--', alpha=0.4, axis='y')

    ax.legend(
        handles=legend_handles, 
        fontsize=16,
        loc='upper left', 
        bbox_to_anchor=(1.02, 1),
        markerscale=1.2, scatterpoints=2,
        labelspacing=1.5, handlelength=1.5, handleheight = 2
    )

    plt.tight_layout()
    plt.subplots_adjust(right=0.8)
    plt.savefig('raw_data/robustness_evaluation_above.png', dpi=300, bbox_inches='tight')

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
    plot_abovepic(data,subfolders,colors)

if __name__ == '__main__':
    plot_above()
    plot_below()
