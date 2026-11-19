"""
过拟合现象对比实验
包括：基准模型、增大模型、延长训练轮数
"""
import torch
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from data_loader import get_cifar10_loaders
from mlp_model import create_mlp_model
from cnn_model import create_cnn_model
from train import train_model
from logger import setup_logger
from save_results import save_experiment_results


def create_mlp_baseline():
    """创建基准MLP模型"""
    return create_mlp_model(hidden_sizes=[512, 256], dropout_rate=0.5)


def create_mlp_large():
    """创建增大的MLP模型（更多层和神经元）"""
    return create_mlp_model(hidden_sizes=[2048, 1024, 512, 256, 128], dropout_rate=0.5)


def create_cnn_baseline():
    """创建基准CNN模型"""
    return create_cnn_model(num_classes=10)


def create_cnn_large():
    """创建增大的CNN模型（更多层和通道）"""
    import torch.nn as nn
    import torch.nn.functional as F
    
    class LargeCNN(nn.Module):
        """更大的CNN模型"""
        def __init__(self, num_classes=10):
            super(LargeCNN, self).__init__()
            
            # 第一个卷积块: 32x32x3 -> 32x32x64 -> 16x16x64
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(64)
            self.pool1 = nn.MaxPool2d(2, 2)
            
            # 第二个卷积块: 16x16x64 -> 16x16x128 -> 8x8x128
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(128)
            self.pool2 = nn.MaxPool2d(2, 2)
            
            # 第三个卷积块: 8x8x128 -> 8x8x256 -> 4x4x256
            self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
            self.bn3 = nn.BatchNorm2d(256)
            self.pool3 = nn.MaxPool2d(2, 2)
            
            # 第四个卷积块: 4x4x256 -> 4x4x512 -> 2x2x512
            self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
            self.bn4 = nn.BatchNorm2d(512)
            self.pool4 = nn.MaxPool2d(2, 2)
            
            # 第五个卷积块: 2x2x512 -> 2x2x512 -> 1x1x512
            self.conv5 = nn.Conv2d(512, 512, 3, padding=1)
            self.bn5 = nn.BatchNorm2d(512)
            self.pool5 = nn.MaxPool2d(2, 2)
            
            # 全连接层
            self.fc1 = nn.Linear(1 * 1 * 512, 2048)
            self.dropout1 = nn.Dropout(0.5)
            self.fc2 = nn.Linear(2048, 1024)
            self.dropout2 = nn.Dropout(0.5)
            self.fc3 = nn.Linear(1024, 512)
            self.dropout3 = nn.Dropout(0.5)
            self.fc4 = nn.Linear(512, num_classes)
        
        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.pool1(x)
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.pool2(x)
            x = F.relu(self.bn3(self.conv3(x)))
            x = self.pool3(x)
            x = F.relu(self.bn4(self.conv4(x)))
            x = self.pool4(x)
            x = F.relu(self.bn5(self.conv5(x)))
            x = self.pool5(x)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.dropout1(x)
            x = F.relu(self.fc2(x))
            x = self.dropout2(x)
            x = F.relu(self.fc3(x))
            x = self.dropout3(x)
            x = self.fc4(x)
            return x
        
        def count_parameters(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    return LargeCNN(num_classes=10)


def plot_overfitting_comparison(results_dict, save_dir='results'):
    """
    绘制过拟合对比图
    
    Args:
        results_dict: 包含所有实验结果的字典
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建多个子图
    fig = plt.figure(figsize=(20, 12))
    
    # 1. MLP基准模型 vs 增大模型
    ax1 = plt.subplot(2, 3, 1)
    if 'mlp_baseline' in results_dict:
        h = results_dict['mlp_baseline']['history']
        epochs = range(1, len(h['train_loss']) + 1)
        ax1.plot(epochs, h['train_acc'], 'b-', label='Baseline Train', linewidth=2)
        ax1.plot(epochs, h['test_acc'], 'b--', label='Baseline Test', linewidth=2)
    if 'mlp_large' in results_dict:
        h = results_dict['mlp_large']['history']
        epochs = range(1, len(h['train_loss']) + 1)
        ax1.plot(epochs, h['train_acc'], 'r-', label='Large Train', linewidth=2)
        ax1.plot(epochs, h['test_acc'], 'r--', label='Large Test', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Accuracy (%)', fontsize=11)
    ax1.set_title('MLP: Baseline vs Large Model', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(2, 3, 2)
    if 'mlp_baseline' in results_dict:
        h = results_dict['mlp_baseline']['history']
        epochs = range(1, len(h['train_loss']) + 1)
        ax2.plot(epochs, h['train_loss'], 'b-', label='Baseline Train', linewidth=2)
        ax2.plot(epochs, h['test_loss'], 'b--', label='Baseline Test', linewidth=2)
    if 'mlp_large' in results_dict:
        h = results_dict['mlp_large']['history']
        epochs = range(1, len(h['train_loss']) + 1)
        ax2.plot(epochs, h['train_loss'], 'r-', label='Large Train', linewidth=2)
        ax2.plot(epochs, h['test_loss'], 'r--', label='Large Test', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Loss', fontsize=11)
    ax2.set_title('MLP: Loss Comparison', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 2. CNN基准模型 vs 增大模型
    ax3 = plt.subplot(2, 3, 3)
    if 'cnn_baseline' in results_dict:
        h = results_dict['cnn_baseline']['history']
        epochs = range(1, len(h['train_loss']) + 1)
        ax3.plot(epochs, h['train_acc'], 'b-', label='Baseline Train', linewidth=2)
        ax3.plot(epochs, h['test_acc'], 'b--', label='Baseline Test', linewidth=2)
    if 'cnn_large' in results_dict:
        h = results_dict['cnn_large']['history']
        epochs = range(1, len(h['train_loss']) + 1)
        ax3.plot(epochs, h['train_acc'], 'r-', label='Large Train', linewidth=2)
        ax3.plot(epochs, h['test_acc'], 'r--', label='Large Test', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Accuracy (%)', fontsize=11)
    ax3.set_title('CNN: Baseline vs Large Model', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    ax4 = plt.subplot(2, 3, 4)
    if 'cnn_baseline' in results_dict:
        h = results_dict['cnn_baseline']['history']
        epochs = range(1, len(h['train_loss']) + 1)
        ax4.plot(epochs, h['train_loss'], 'b-', label='Baseline Train', linewidth=2)
        ax4.plot(epochs, h['test_loss'], 'b--', label='Baseline Test', linewidth=2)
    if 'cnn_large' in results_dict:
        h = results_dict['cnn_large']['history']
        epochs = range(1, len(h['train_loss']) + 1)
        ax4.plot(epochs, h['train_loss'], 'r-', label='Large Train', linewidth=2)
        ax4.plot(epochs, h['test_loss'], 'r--', label='Large Test', linewidth=2)
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('Loss', fontsize=11)
    ax4.set_title('CNN: Loss Comparison', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # 3. 延长训练轮数对比（使用基准模型）
    ax5 = plt.subplot(2, 3, 5)
    if 'mlp_baseline_short' in results_dict:
        h = results_dict['mlp_baseline_short']['history']
        epochs = range(1, len(h['train_loss']) + 1)
        ax5.plot(epochs, h['train_acc'], 'g-', label='Short Train', linewidth=2)
        ax5.plot(epochs, h['test_acc'], 'g--', label='Short Test', linewidth=2)
    if 'mlp_baseline_long' in results_dict:
        h = results_dict['mlp_baseline_long']['history']
        epochs = range(1, len(h['train_loss']) + 1)
        ax5.plot(epochs, h['train_acc'], 'm-', label='Long Train', linewidth=2)
        ax5.plot(epochs, h['test_acc'], 'm--', label='Long Test', linewidth=2)
    ax5.set_xlabel('Epoch', fontsize=11)
    ax5.set_ylabel('Accuracy (%)', fontsize=11)
    ax5.set_title('MLP: Short vs Long Training', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    ax6 = plt.subplot(2, 3, 6)
    if 'cnn_baseline_short' in results_dict:
        h = results_dict['cnn_baseline_short']['history']
        epochs = range(1, len(h['train_loss']) + 1)
        ax6.plot(epochs, h['train_acc'], 'g-', label='Short Train', linewidth=2)
        ax6.plot(epochs, h['test_acc'], 'g--', label='Short Test', linewidth=2)
    if 'cnn_baseline_long' in results_dict:
        h = results_dict['cnn_baseline_long']['history']
        epochs = range(1, len(h['train_loss']) + 1)
        ax6.plot(epochs, h['train_acc'], 'm-', label='Long Train', linewidth=2)
        ax6.plot(epochs, h['test_acc'], 'm--', label='Long Test', linewidth=2)
    ax6.set_xlabel('Epoch', fontsize=11)
    ax6.set_ylabel('Accuracy (%)', fontsize=11)
    ax6.set_title('CNN: Short vs Long Training', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'overfitting_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"过拟合对比图已保存: {save_path}")


def plot_overfitting_gap(results_dict, save_dir='results'):
    """
    绘制训练-测试准确率差距（过拟合指标）
    
    Args:
        results_dict: 包含所有实验结果的字典
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # MLP模型对比
    ax1 = axes[0, 0]
    if 'mlp_baseline' in results_dict:
        h = results_dict['mlp_baseline']['history']
        epochs = range(1, len(h['train_loss']) + 1)
        gap = np.array(h['train_acc']) - np.array(h['test_acc'])
        ax1.plot(epochs, gap, 'b-', label='Baseline', linewidth=2, marker='o', markersize=3)
    if 'mlp_large' in results_dict:
        h = results_dict['mlp_large']['history']
        epochs = range(1, len(h['train_loss']) + 1)
        gap = np.array(h['train_acc']) - np.array(h['test_acc'])
        ax1.plot(epochs, gap, 'r-', label='Large', linewidth=2, marker='s', markersize=3)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Train-Test Gap (%)', fontsize=11)
    ax1.set_title('MLP: Overfitting Gap (Train Acc - Test Acc)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # CNN模型对比
    ax2 = axes[0, 1]
    if 'cnn_baseline' in results_dict:
        h = results_dict['cnn_baseline']['history']
        epochs = range(1, len(h['train_loss']) + 1)
        gap = np.array(h['train_acc']) - np.array(h['test_acc'])
        ax2.plot(epochs, gap, 'b-', label='Baseline', linewidth=2, marker='o', markersize=3)
    if 'cnn_large' in results_dict:
        h = results_dict['cnn_large']['history']
        epochs = range(1, len(h['train_loss']) + 1)
        gap = np.array(h['train_acc']) - np.array(h['test_acc'])
        ax2.plot(epochs, gap, 'r-', label='Large', linewidth=2, marker='s', markersize=3)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Train-Test Gap (%)', fontsize=11)
    ax2.set_title('CNN: Overfitting Gap (Train Acc - Test Acc)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # 延长训练对比 - MLP
    ax3 = axes[1, 0]
    if 'mlp_baseline_short' in results_dict:
        h = results_dict['mlp_baseline_short']['history']
        epochs = range(1, len(h['train_loss']) + 1)
        gap = np.array(h['train_acc']) - np.array(h['test_acc'])
        ax3.plot(epochs, gap, 'g-', label='Short Training', linewidth=2, marker='o', markersize=3)
    if 'mlp_baseline_long' in results_dict:
        h = results_dict['mlp_baseline_long']['history']
        epochs = range(1, len(h['train_loss']) + 1)
        gap = np.array(h['train_acc']) - np.array(h['test_acc'])
        ax3.plot(epochs, gap, 'm-', label='Long Training', linewidth=2, marker='s', markersize=3)
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Train-Test Gap (%)', fontsize=11)
    ax3.set_title('MLP: Short vs Long Training Gap', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # 延长训练对比 - CNN
    ax4 = axes[1, 1]
    if 'cnn_baseline_short' in results_dict:
        h = results_dict['cnn_baseline_short']['history']
        epochs = range(1, len(h['train_loss']) + 1)
        gap = np.array(h['train_acc']) - np.array(h['test_acc'])
        ax4.plot(epochs, gap, 'g-', label='Short Training', linewidth=2, marker='o', markersize=3)
    if 'cnn_baseline_long' in results_dict:
        h = results_dict['cnn_baseline_long']['history']
        epochs = range(1, len(h['train_loss']) + 1)
        gap = np.array(h['train_acc']) - np.array(h['test_acc'])
        ax4.plot(epochs, gap, 'm-', label='Long Training', linewidth=2, marker='s', markersize=3)
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('Train-Test Gap (%)', fontsize=11)
    ax4.set_title('CNN: Short vs Long Training Gap', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'overfitting_gap.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"过拟合差距图已保存: {save_path}")


def main():
    """主函数"""
    # 创建结果目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'results/overfitting_experiment_{timestamp}'
    logs_dir = 'logs'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # 设置日志记录器
    log_file = f'overfitting_experiment_{timestamp}.log'
    logger = setup_logger(logs_dir, log_file)
    
    logger.info("=" * 80)
    logger.info("过拟合现象对比实验开始")
    logger.info("=" * 80)
    
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_msg = f"使用设备: {device}"
    print(device_msg)
    logger.info(device_msg)
    
    # 超参数设置
    batch_size = 128
    learning_rate = 0.001
    
    # 加载数据
    print("\n加载CIFAR-10数据集...")
    logger.info("开始加载CIFAR-10数据集")
    train_loader, test_loader, classes = get_cifar10_loaders(batch_size=batch_size)
    
    results_dict = {}
    
    # ========== 实验1: MLP基准模型 vs 增大模型 ==========
    print("\n" + "="*80)
    print("实验1: MLP基准模型 vs 增大模型")
    print("="*80)
    logger.info("="*80)
    logger.info("实验1: MLP基准模型 vs 增大模型")
    logger.info("="*80)
    
    # MLP基准模型
    print("\n训练MLP基准模型（2层隐藏层，20 epochs）...")
    mlp_baseline = create_mlp_baseline()
    mlp_baseline_params = mlp_baseline.count_parameters()
    print(f"参数量: {mlp_baseline_params:,}")
    logger.info(f"MLP基准模型参数量: {mlp_baseline_params:,}")
    
    mlp_baseline_history = train_model(
        model=mlp_baseline,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=20,
        learning_rate=learning_rate,
        device=device,
        model_name='mlp_baseline',
        logger=logger
    )
    results_dict['mlp_baseline'] = {
        'parameters': mlp_baseline_params,
        'history': mlp_baseline_history['training_history'],
        'best_test_acc': mlp_baseline_history['best_test_acc']
    }
    
    # MLP增大模型
    print("\n训练MLP增大模型（5层隐藏层，20 epochs）...")
    mlp_large = create_mlp_large()
    mlp_large_params = mlp_large.count_parameters()
    print(f"参数量: {mlp_large_params:,}")
    logger.info(f"MLP增大模型参数量: {mlp_large_params:,}")
    
    mlp_large_history = train_model(
        model=mlp_large,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=20,
        learning_rate=learning_rate,
        device=device,
        model_name='mlp_large',
        logger=logger
    )
    results_dict['mlp_large'] = {
        'parameters': mlp_large_params,
        'history': mlp_large_history['training_history'],
        'best_test_acc': mlp_large_history['best_test_acc']
    }
    
    # ========== 实验2: CNN基准模型 vs 增大模型 ==========
    print("\n" + "="*80)
    print("实验2: CNN基准模型 vs 增大模型")
    print("="*80)
    logger.info("="*80)
    logger.info("实验2: CNN基准模型 vs 增大模型")
    logger.info("="*80)
    
    # CNN基准模型
    print("\n训练CNN基准模型（4个卷积块，20 epochs）...")
    cnn_baseline = create_cnn_baseline()
    cnn_baseline_params = cnn_baseline.count_parameters()
    print(f"参数量: {cnn_baseline_params:,}")
    logger.info(f"CNN基准模型参数量: {cnn_baseline_params:,}")
    
    cnn_baseline_history = train_model(
        model=cnn_baseline,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=20,
        learning_rate=learning_rate,
        device=device,
        model_name='cnn_baseline',
        logger=logger
    )
    results_dict['cnn_baseline'] = {
        'parameters': cnn_baseline_params,
        'history': cnn_baseline_history['training_history'],
        'best_test_acc': cnn_baseline_history['best_test_acc']
    }
    
    # CNN增大模型
    print("\n训练CNN增大模型（5个卷积块，20 epochs）...")
    cnn_large = create_cnn_large()
    cnn_large_params = cnn_large.count_parameters()
    print(f"参数量: {cnn_large_params:,}")
    logger.info(f"CNN增大模型参数量: {cnn_large_params:,}")
    
    cnn_large_history = train_model(
        model=cnn_large,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=20,
        learning_rate=learning_rate,
        device=device,
        model_name='cnn_large',
        logger=logger
    )
    results_dict['cnn_large'] = {
        'parameters': cnn_large_params,
        'history': cnn_large_history['training_history'],
        'best_test_acc': cnn_large_history['best_test_acc']
    }
    
    # ========== 实验3: 延长训练轮数 ==========
    print("\n" + "="*80)
    print("实验3: 延长训练轮数（基准模型）")
    print("="*80)
    logger.info("="*80)
    logger.info("实验3: 延长训练轮数（基准模型）")
    logger.info("="*80)
    
    # 短训练（20 epochs）
    print("\n训练MLP基准模型（短训练：20 epochs）...")
    mlp_short = create_mlp_baseline()
    mlp_short_history = train_model(
        model=mlp_short,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=20,
        learning_rate=learning_rate,
        device=device,
        model_name='mlp_baseline_short',
        logger=logger
    )
    results_dict['mlp_baseline_short'] = {
        'parameters': mlp_baseline_params,
        'history': mlp_short_history['training_history'],
        'best_test_acc': mlp_short_history['best_test_acc']
    }
    
    # 长训练（60 epochs）
    print("\n训练MLP基准模型（长训练：60 epochs）...")
    mlp_long = create_mlp_baseline()
    mlp_long_history = train_model(
        model=mlp_long,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=60,
        learning_rate=learning_rate,
        device=device,
        model_name='mlp_baseline_long',
        logger=logger
    )
    results_dict['mlp_baseline_long'] = {
        'parameters': mlp_baseline_params,
        'history': mlp_long_history['training_history'],
        'best_test_acc': mlp_long_history['best_test_acc']
    }
    
    # CNN短训练
    print("\n训练CNN基准模型（短训练：20 epochs）...")
    cnn_short = create_cnn_baseline()
    cnn_short_history = train_model(
        model=cnn_short,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=20,
        learning_rate=learning_rate,
        device=device,
        model_name='cnn_baseline_short',
        logger=logger
    )
    results_dict['cnn_baseline_short'] = {
        'parameters': cnn_baseline_params,
        'history': cnn_short_history['training_history'],
        'best_test_acc': cnn_short_history['best_test_acc']
    }
    
    # CNN长训练
    print("\n训练CNN基准模型（长训练：60 epochs）...")
    cnn_long = create_cnn_baseline()
    cnn_long_history = train_model(
        model=cnn_long,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=60,
        learning_rate=learning_rate,
        device=device,
        model_name='cnn_baseline_long',
        logger=logger
    )
    results_dict['cnn_baseline_long'] = {
        'parameters': cnn_baseline_params,
        'history': cnn_long_history['training_history'],
        'best_test_acc': cnn_long_history['best_test_acc']
    }
    
    # ========== 生成可视化 ==========
    print("\n生成可视化图表...")
    logger.info("开始生成可视化图表")
    
    plot_overfitting_comparison(results_dict, results_dir)
    plot_overfitting_gap(results_dict, results_dir)
    
    # ========== 保存结果摘要 ==========
    print("\n保存实验结果摘要...")
    summary_path = os.path.join(results_dir, 'overfitting_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("过拟合现象对比实验报告\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("实验1: 基准模型 vs 增大模型\n")
        f.write("-" * 80 + "\n")
        f.write(f"MLP基准模型: 参数量={results_dict['mlp_baseline']['parameters']:,}, "
                f"最佳测试准确率={results_dict['mlp_baseline']['best_test_acc']:.2f}%\n")
        f.write(f"MLP增大模型: 参数量={results_dict['mlp_large']['parameters']:,}, "
                f"最佳测试准确率={results_dict['mlp_large']['best_test_acc']:.2f}%\n")
        f.write(f"CNN基准模型: 参数量={results_dict['cnn_baseline']['parameters']:,}, "
                f"最佳测试准确率={results_dict['cnn_baseline']['best_test_acc']:.2f}%\n")
        f.write(f"CNN增大模型: 参数量={results_dict['cnn_large']['parameters']:,}, "
                f"最佳测试准确率={results_dict['cnn_large']['best_test_acc']:.2f}%\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("实验2: 延长训练轮数\n")
        f.write("-" * 80 + "\n")
        f.write(f"MLP短训练(20 epochs): 最佳测试准确率={results_dict['mlp_baseline_short']['best_test_acc']:.2f}%\n")
        f.write(f"MLP长训练(60 epochs): 最佳测试准确率={results_dict['mlp_baseline_long']['best_test_acc']:.2f}%\n")
        f.write(f"CNN短训练(20 epochs): 最佳测试准确率={results_dict['cnn_baseline_short']['best_test_acc']:.2f}%\n")
        f.write(f"CNN长训练(60 epochs): 最佳测试准确率={results_dict['cnn_baseline_long']['best_test_acc']:.2f}%\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("过拟合分析\n")
        f.write("-" * 80 + "\n")
        f.write("过拟合现象表现为训练准确率持续上升，但测试准确率停滞或下降，\n")
        f.write("训练-测试准确率差距逐渐增大。\n")
        f.write("查看可视化图表以观察详细的过拟合趋势。\n")
    
    print(f"实验结果摘要已保存: {summary_path}")
    logger.info(f"实验结果摘要已保存: {summary_path}")
    
    completion_msg = f"""
    
{'='*80}
过拟合对比实验完成！
{'='*80}
所有结果已保存到目录: {results_dir}
日志文件: {os.path.join(logs_dir, log_file)}

生成的文件:
  - 过拟合对比图: overfitting_comparison.png
  - 过拟合差距图: overfitting_gap.png
  - 实验摘要: overfitting_summary.txt
{'='*80}
"""
    print(completion_msg)
    logger.info(completion_msg)
    logger.info("实验结束")


if __name__ == '__main__':
    main()

