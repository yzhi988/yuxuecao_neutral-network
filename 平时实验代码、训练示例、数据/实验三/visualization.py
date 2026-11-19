"""
可视化分析模块
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os


def plot_loss_curves(mlp_history, cnn_history, save_dir='results'):
    """
    绘制损失函数曲线
    
    Args:
        mlp_history: MLP训练历史
        cnn_history: CNN训练历史
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    epochs = range(1, len(mlp_history['train_loss']) + 1)
    
    # 训练损失
    plt.subplot(1, 2, 1)
    plt.plot(epochs, mlp_history['train_loss'], 'b-', label='MLP Train', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, mlp_history['test_loss'], 'b--', label='MLP Test', linewidth=2, marker='s', markersize=4)
    plt.plot(epochs, cnn_history['train_loss'], 'r-', label='CNN Train', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, cnn_history['test_loss'], 'r--', label='CNN Test', linewidth=2, marker='s', markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Loss Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 测试损失对比（放大）
    plt.subplot(1, 2, 2)
    plt.plot(epochs, mlp_history['test_loss'], 'b--', label='MLP Test', linewidth=2.5, marker='s', markersize=5)
    plt.plot(epochs, cnn_history['test_loss'], 'r--', label='CNN Test', linewidth=2.5, marker='s', markersize=5)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Test Loss', fontsize=12)
    plt.title('Test Loss Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'loss_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"损失曲线已保存: {save_path}")


def plot_accuracy_curves(mlp_history, cnn_history, save_dir='results'):
    """
    绘制准确率曲线
    
    Args:
        mlp_history: MLP训练历史
        cnn_history: CNN训练历史
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    epochs = range(1, len(mlp_history['train_acc']) + 1)
    
    # 训练准确率
    plt.subplot(1, 2, 1)
    plt.plot(epochs, mlp_history['train_acc'], 'b-', label='MLP Train', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, mlp_history['test_acc'], 'b--', label='MLP Test', linewidth=2, marker='s', markersize=4)
    plt.plot(epochs, cnn_history['train_acc'], 'r-', label='CNN Train', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, cnn_history['test_acc'], 'r--', label='CNN Test', linewidth=2, marker='s', markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Accuracy Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 测试准确率对比（放大）
    plt.subplot(1, 2, 2)
    plt.plot(epochs, mlp_history['test_acc'], 'b--', label='MLP Test', linewidth=2.5, marker='s', markersize=5)
    plt.plot(epochs, cnn_history['test_acc'], 'r--', label='CNN Test', linewidth=2.5, marker='s', markersize=5)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'accuracy_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"准确率曲线已保存: {save_path}")


def plot_confusion_matrix(model, test_loader, classes, device, model_name, save_dir='results'):
    """
    绘制混淆矩阵
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        classes: 类别名称列表
        device: 设备
        model_name: 模型名称
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(f'Confusion Matrix - {model_name.upper()}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'confusion_matrix_{model_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存: {save_path}")
    
    return cm


def plot_class_accuracy(cm, classes, model_name, save_dir='results'):
    """
    绘制每个类别的准确率
    
    Args:
        cm: 混淆矩阵
        classes: 类别名称列表
        model_name: 模型名称
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 计算每个类别的准确率
    class_acc = cm.diagonal() / cm.sum(axis=1) * 100
    
    # 绘制柱状图
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(classes)), class_acc, color='steelblue', alpha=0.7, edgecolor='black')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(f'Per-Class Accuracy - {model_name.upper()}', fontsize=14, fontweight='bold')
    plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
    plt.ylim([0, 100])
    plt.grid(True, alpha=0.3, axis='y')
    
    # 在柱状图上添加数值
    for i, (bar, acc) in enumerate(zip(bars, class_acc)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'class_accuracy_{model_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"类别准确率图已保存: {save_path}")
    
    return class_acc


def plot_comprehensive_comparison(mlp_history, cnn_history, save_dir='results'):
    """
    绘制综合对比图
    
    Args:
        mlp_history: MLP训练历史
        cnn_history: CNN训练历史
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    epochs = range(1, len(mlp_history['train_loss']) + 1)
    
    # 损失曲线对比
    axes[0, 0].plot(epochs, mlp_history['train_loss'], 'b-', label='MLP Train', linewidth=2)
    axes[0, 0].plot(epochs, mlp_history['test_loss'], 'b--', label='MLP Test', linewidth=2)
    axes[0, 0].plot(epochs, cnn_history['train_loss'], 'r-', label='CNN Train', linewidth=2)
    axes[0, 0].plot(epochs, cnn_history['test_loss'], 'r--', label='CNN Test', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Loss', fontsize=11)
    axes[0, 0].set_title('Loss Curves', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 准确率曲线对比
    axes[0, 1].plot(epochs, mlp_history['train_acc'], 'b-', label='MLP Train', linewidth=2)
    axes[0, 1].plot(epochs, mlp_history['test_acc'], 'b--', label='MLP Test', linewidth=2)
    axes[0, 1].plot(epochs, cnn_history['train_acc'], 'r-', label='CNN Train', linewidth=2)
    axes[0, 1].plot(epochs, cnn_history['test_acc'], 'r--', label='CNN Test', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=11)
    axes[0, 1].set_title('Accuracy Curves', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)
    
    # MLP详细曲线
    ax2 = axes[1, 0]
    ax2_twin = ax2.twinx()
    line1 = ax2.plot(epochs, mlp_history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    line2 = ax2.plot(epochs, mlp_history['test_loss'], 'b--', label='Test Loss', linewidth=2)
    line3 = ax2_twin.plot(epochs, mlp_history['train_acc'], 'g-', label='Train Acc', linewidth=2)
    line4 = ax2_twin.plot(epochs, mlp_history['test_acc'], 'g--', label='Test Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Loss', fontsize=11, color='b')
    ax2_twin.set_ylabel('Accuracy (%)', fontsize=11, color='g')
    ax2.set_title('MLP Training History', fontsize=12, fontweight='bold')
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='center right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # CNN详细曲线
    ax3 = axes[1, 1]
    ax3_twin = ax3.twinx()
    line1 = ax3.plot(epochs, cnn_history['train_loss'], 'r-', label='Train Loss', linewidth=2)
    line2 = ax3.plot(epochs, cnn_history['test_loss'], 'r--', label='Test Loss', linewidth=2)
    line3 = ax3_twin.plot(epochs, cnn_history['train_acc'], 'g-', label='Train Acc', linewidth=2)
    line4 = ax3_twin.plot(epochs, cnn_history['test_acc'], 'g--', label='Test Acc', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Loss', fontsize=11, color='r')
    ax3_twin.set_ylabel('Accuracy (%)', fontsize=11, color='g')
    ax3.set_title('CNN Training History', fontsize=12, fontweight='bold')
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='center right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'comprehensive_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"综合对比图已保存: {save_path}")

