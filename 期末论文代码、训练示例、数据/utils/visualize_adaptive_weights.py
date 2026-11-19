"""
自适应权重可视化工具

功能：
1. 从训练日志中提取权重信息
2. 绘制权重随训练变化的曲线
3. 分析权重变化模式
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def parse_log_file(log_path):
    """
    从训练日志中解析权重信息
    
    Args:
        log_path: 日志文件路径
        
    Returns:
        weights_data: {
            'epochs': [...],
            'w_mse': [...],
            'w_smooth_l1': [...]
        }
    """
    weights_data = {
        'epochs': [],
        'w_mse': [],
        'w_smooth_l1': []
    }
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 匹配格式：[Epoch X] Average adaptive weights: MSE=0.xxx, Smooth L1=0.xxx
            match = re.search(r'\[Epoch (\d+)\] Average adaptive weights: MSE=([\d.]+), Smooth L1=([\d.]+)', line)
            if match:
                epoch = int(match.group(1))
                w_mse = float(match.group(2))
                w_smooth = float(match.group(3))
                
                weights_data['epochs'].append(epoch)
                weights_data['w_mse'].append(w_mse)
                weights_data['w_smooth_l1'].append(w_smooth)
    
    return weights_data


def plot_weight_evolution(weights_data, save_path=None):
    """
    绘制权重随epoch变化的曲线
    
    Args:
        weights_data: parse_log_file返回的数据
        save_path: 保存图片的路径（可选）
    """
    if not weights_data['epochs']:
        print("警告：没有找到权重数据")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    epochs = weights_data['epochs']
    w_mse = weights_data['w_mse']
    w_smooth = weights_data['w_smooth_l1']
    
    # 子图1：权重对比
    axes[0].plot(epochs, w_mse, 'b-', linewidth=2, label='MSE Weight', marker='o', markersize=4)
    axes[0].plot(epochs, w_smooth, 'r-', linewidth=2, label='Smooth L1 Weight', marker='s', markersize=4)
    axes[0].axhline(y=0.8, color='b', linestyle='--', alpha=0.5, label='Fixed MSE (0.8)')
    axes[0].axhline(y=0.2, color='r', linestyle='--', alpha=0.5, label='Fixed Smooth L1 (0.2)')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Weight', fontsize=12)
    axes[0].set_title('Adaptive Loss Weights Evolution', fontsize=14, fontweight='bold')
    axes[0].legend(loc='best', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])
    
    # 子图2：权重差异
    weight_diff = np.array(w_mse) - np.array(w_smooth)
    axes[1].plot(epochs, weight_diff, 'g-', linewidth=2, marker='d', markersize=4)
    axes[1].axhline(y=0.6, color='k', linestyle='--', alpha=0.5, label='Fixed Difference (0.6)')
    axes[1].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Weight Difference (MSE - Smooth L1)', fontsize=12)
    axes[1].set_title('Weight Difference Over Training', fontsize=14, fontweight='bold')
    axes[1].legend(loc='best', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存至: {save_path}")
    
    plt.show()


def analyze_weights(weights_data):
    """
    分析权重变化统计信息
    
    Args:
        weights_data: parse_log_file返回的数据
    """
    if not weights_data['epochs']:
        print("没有数据可分析")
        return
    
    w_mse = np.array(weights_data['w_mse'])
    w_smooth = np.array(weights_data['w_smooth_l1'])
    
    print("\n" + "="*60)
    print("自适应权重统计分析")
    print("="*60)
    
    print(f"\n训练轮数: {len(weights_data['epochs'])}")
    print(f"Epoch范围: {weights_data['epochs'][0]} - {weights_data['epochs'][-1]}")
    
    print("\n【MSE权重统计】")
    print(f"  平均值: {w_mse.mean():.4f}")
    print(f"  标准差: {w_mse.std():.4f}")
    print(f"  最小值: {w_mse.min():.4f} (Epoch {weights_data['epochs'][w_mse.argmin()]})")
    print(f"  最大值: {w_mse.max():.4f} (Epoch {weights_data['epochs'][w_mse.argmax()]})")
    print(f"  初始值: {w_mse[0]:.4f}")
    print(f"  最终值: {w_mse[-1]:.4f}")
    print(f"  变化量: {w_mse[-1] - w_mse[0]:+.4f}")
    
    print("\n【Smooth L1权重统计】")
    print(f"  平均值: {w_smooth.mean():.4f}")
    print(f"  标准差: {w_smooth.std():.4f}")
    print(f"  最小值: {w_smooth.min():.4f} (Epoch {weights_data['epochs'][w_smooth.argmin()]})")
    print(f"  最大值: {w_smooth.max():.4f} (Epoch {weights_data['epochs'][w_smooth.argmax()]})")
    print(f"  初始值: {w_smooth[0]:.4f}")
    print(f"  最终值: {w_smooth[-1]:.4f}")
    print(f"  变化量: {w_smooth[-1] - w_smooth[0]:+.4f}")
    
    print("\n【与固定权重对比】")
    mse_deviation = np.abs(w_mse - 0.8).mean()
    smooth_deviation = np.abs(w_smooth - 0.2).mean()
    print(f"  MSE平均偏离固定值(0.8): {mse_deviation:.4f}")
    print(f"  Smooth L1平均偏离固定值(0.2): {smooth_deviation:.4f}")
    
    print("\n【权重变化趋势】")
    if w_mse[-1] > w_mse[0]:
        print(f"  MSE权重呈上升趋势 ↑ (增加 {(w_mse[-1] - w_mse[0])*100:.2f}%)")
    elif w_mse[-1] < w_mse[0]:
        print(f"  MSE权重呈下降趋势 ↓ (减少 {(w_mse[0] - w_mse[-1])*100:.2f}%)")
    else:
        print("  MSE权重保持稳定 →")
    
    if w_smooth[-1] > w_smooth[0]:
        print(f"  Smooth L1权重呈上升趋势 ↑ (增加 {(w_smooth[-1] - w_smooth[0])*100:.2f}%)")
    elif w_smooth[-1] < w_smooth[0]:
        print(f"  Smooth L1权重呈下降趋势 ↓ (减少 {(w_smooth[0] - w_smooth[-1])*100:.2f}%)")
    else:
        print("  Smooth L1权重保持稳定 →")
    
    print("\n" + "="*60 + "\n")


def compare_multiple_experiments(log_paths, labels, save_path=None):
    """
    对比多个实验的权重变化
    
    Args:
        log_paths: 日志文件路径列表
        labels: 对应的标签列表
        save_path: 保存图片路径（可选）
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['b', 'r', 'g', 'orange', 'purple', 'brown']
    
    for i, (log_path, label) in enumerate(zip(log_paths, labels)):
        weights_data = parse_log_file(log_path)
        if weights_data['epochs']:
            color = colors[i % len(colors)]
            ax.plot(weights_data['epochs'], weights_data['w_mse'], 
                   color=color, linewidth=2, marker='o', markersize=4,
                   label=f'{label} - MSE')
            ax.plot(weights_data['epochs'], weights_data['w_smooth_l1'], 
                   color=color, linewidth=2, linestyle='--', marker='s', markersize=4,
                   label=f'{label} - Smooth L1')
    
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='Fixed Baseline')
    ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Weight', fontsize=12)
    ax.set_title('Comparison of Adaptive Weights Across Experiments', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"对比图已保存至: {save_path}")
    
    plt.show()


def main():
    """主函数：示例用法"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法:")
        print("  python visualize_adaptive_weights.py <log_file_path>")
        print("  python visualize_adaptive_weights.py compare <log1> <log2> ...")
        print("\n示例:")
        print("  python visualize_adaptive_weights.py results/adaptive_weights/training.log")
        print("  python visualize_adaptive_weights.py compare results/exp1/training.log results/exp2/training.log")
        return
    
    if sys.argv[1] == 'compare':
        # 对比多个实验
        if len(sys.argv) < 3:
            print("错误：需要至少一个日志文件")
            return
        
        log_paths = sys.argv[2:]
        labels = [f"Exp {i+1}" for i in range(len(log_paths))]
        
        print(f"正在对比 {len(log_paths)} 个实验...")
        compare_multiple_experiments(log_paths, labels, save_path='weight_comparison.png')
        
    else:
        # 分析单个实验
        log_path = sys.argv[1]
        
        if not Path(log_path).exists():
            print(f"错误：文件不存在 - {log_path}")
            return
        
        print(f"正在解析日志文件: {log_path}")
        weights_data = parse_log_file(log_path)
        
        if not weights_data['epochs']:
            print("警告：没有找到自适应权重数据")
            print("请确保：")
            print("  1. 训练时启用了 use_adaptive_weights=True")
            print("  2. 日志文件格式正确")
            return
        
        # 统计分析
        analyze_weights(weights_data)
        
        # 可视化
        save_path = log_path.replace('.log', '_weights.png')
        plot_weight_evolution(weights_data, save_path=save_path)


if __name__ == '__main__':
    main()

