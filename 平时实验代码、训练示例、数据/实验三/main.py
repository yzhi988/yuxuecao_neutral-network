"""
主程序：训练MLP和CNN模型对CIFAR-10进行分类
包含完整的日志记录、可视化和结果保存功能
"""
import torch
import os
from datetime import datetime
from data_loader import get_cifar10_loaders
from mlp_model import create_mlp_model
from cnn_model import create_cnn_model
from train import train_model
from logger import setup_logger
from visualization import (
    plot_loss_curves, 
    plot_accuracy_curves, 
    plot_confusion_matrix,
    plot_class_accuracy,
    plot_comprehensive_comparison
)
from save_results import save_experiment_results, save_summary_report


def main():
    """
    主函数
    """
    # 创建结果目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'results/experiment_{timestamp}'
    logs_dir = 'logs'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # 设置日志记录器
    log_file = f'experiment_{timestamp}.log'
    logger = setup_logger(logs_dir, log_file)
    
    logger.info("=" * 80)
    logger.info("CIFAR-10分类实验开始")
    logger.info("=" * 80)
    
    # 设置随机种子，保证结果可复现
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
    num_epochs = 40
    learning_rate = 0.001
    
    hyperparams_msg = f"\n超参数设置:\n  批次大小: {batch_size}\n  训练轮数: {num_epochs}\n  学习率: {learning_rate}"
    print(hyperparams_msg)
    logger.info(hyperparams_msg)
    
    # 加载数据
    print("\n加载CIFAR-10数据集...")
    logger.info("开始加载CIFAR-10数据集")
    train_loader, test_loader, classes = get_cifar10_loaders(batch_size=batch_size)
    data_msg = f"训练集大小: {len(train_loader.dataset)}\n测试集大小: {len(test_loader.dataset)}\n类别: {classes}"
    print(data_msg)
    logger.info(data_msg)
    
    # ========== 训练MLP模型 ==========
    print("\n" + "="*80)
    print("开始训练MLP模型")
    print("="*80)
    logger.info("="*80)
    logger.info("开始训练MLP模型")
    logger.info("="*80)
    
    mlp_model = create_mlp_model(hidden_sizes=[1024, 512, 256], dropout_rate=0.5)
    mlp_params = mlp_model.count_parameters()
    mlp_params_msg = f"MLP模型参数量: {mlp_params:,}"
    print(mlp_params_msg)
    logger.info(mlp_params_msg)
    
    mlp_history = train_model(
        model=mlp_model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        model_name='mlp',
        logger=logger
    )
    
    # 保存MLP模型权重到结果目录
    mlp_model_path = os.path.join(results_dir, 'mlp_best.pth')
    torch.save(mlp_model.state_dict(), mlp_model_path)
    logger.info(f"MLP模型权重已保存到: {mlp_model_path}")
    
    # ========== 训练CNN模型 ==========
    print("\n" + "="*80)
    print("开始训练CNN模型")
    print("="*80)
    logger.info("="*80)
    logger.info("开始训练CNN模型")
    logger.info("="*80)
    
    cnn_model = create_cnn_model(num_classes=10)
    cnn_params = cnn_model.count_parameters()
    cnn_params_msg = f"CNN模型参数量: {cnn_params:,}"
    print(cnn_params_msg)
    logger.info(cnn_params_msg)
    
    cnn_history = train_model(
        model=cnn_model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        model_name='cnn',
        logger=logger
    )
    
    # 保存CNN模型权重到结果目录
    cnn_model_path = os.path.join(results_dir, 'cnn_best.pth')
    torch.save(cnn_model.state_dict(), cnn_model_path)
    logger.info(f"CNN模型权重已保存到: {cnn_model_path}")
    
    # ========== 模型对比 ==========
    print("\n" + "="*80)
    print("模型对比结果")
    print("="*80)
    logger.info("="*80)
    logger.info("模型对比结果")
    logger.info("="*80)
    
    mlp_best_acc = mlp_history['best_test_acc']
    cnn_best_acc = cnn_history['best_test_acc']
    param_ratio = cnn_params / mlp_params
    
    comparison_msg = f"""
MLP模型:
  参数量: {mlp_params:,}
  最佳测试准确率: {mlp_best_acc:.2f}% (Epoch {mlp_history['best_epoch']})
  总训练时间: {mlp_history['total_time']/60:.2f} 分钟

CNN模型:
  参数量: {cnn_params:,}
  最佳测试准确率: {cnn_best_acc:.2f}% (Epoch {cnn_history['best_epoch']})
  总训练时间: {cnn_history['total_time']/60:.2f} 分钟

对比:
  参数量比例 (CNN/MLP): {param_ratio:.2f}
  准确率提升: {cnn_best_acc - mlp_best_acc:.2f}%
"""
    print(comparison_msg)
    logger.info(comparison_msg)
    
    # ========== 生成混淆矩阵和类别准确率 ==========
    print("\n生成混淆矩阵和类别准确率分析...")
    logger.info("开始生成混淆矩阵和类别准确率分析")
    
    # MLP混淆矩阵
    mlp_cm = plot_confusion_matrix(
        mlp_model, test_loader, classes, device, 'MLP', results_dir
    )
    mlp_class_acc = plot_class_accuracy(mlp_cm, classes, 'MLP', results_dir)
    
    # CNN混淆矩阵
    cnn_cm = plot_confusion_matrix(
        cnn_model, test_loader, classes, device, 'CNN', results_dir
    )
    cnn_class_acc = plot_class_accuracy(cnn_cm, classes, 'CNN', results_dir)
    
    # ========== 绘制可视化图表 ==========
    print("\n绘制可视化图表...")
    logger.info("开始绘制可视化图表")
    
    # 损失曲线
    plot_loss_curves(mlp_history['training_history'], cnn_history['training_history'], results_dir)
    
    # 准确率曲线
    plot_accuracy_curves(mlp_history['training_history'], cnn_history['training_history'], results_dir)
    
    # 综合对比图
    plot_comprehensive_comparison(mlp_history['training_history'], cnn_history['training_history'], results_dir)
    
    # ========== 保存实验结果 ==========
    print("\n保存实验结果...")
    logger.info("开始保存实验结果")
    
    # 准备结果数据
    mlp_results = {
        'parameters': mlp_params,
        'best_test_acc': mlp_best_acc,
        'best_epoch': mlp_history['best_epoch'],
        'total_time': mlp_history['total_time'],
        'history': mlp_history['training_history'],
        'confusion_matrix': mlp_cm,
        'class_accuracy': mlp_class_acc
    }
    
    cnn_results = {
        'parameters': cnn_params,
        'best_test_acc': cnn_best_acc,
        'best_epoch': cnn_history['best_epoch'],
        'total_time': cnn_history['total_time'],
        'history': cnn_history['training_history'],
        'confusion_matrix': cnn_cm,
        'class_accuracy': cnn_class_acc
    }
    
    # 保存JSON格式结果
    json_path = save_experiment_results(mlp_results, cnn_results, results_dir)
    logger.info(f"实验结果JSON已保存: {json_path}")
    
    # 保存文本摘要报告
    report_path = save_summary_report(mlp_results, cnn_results, results_dir)
    logger.info(f"实验摘要报告已保存: {report_path}")
    
    # ========== 实验完成 ==========
    completion_msg = f"""
    
{'='*80}
实验完成！
{'='*80}
所有结果已保存到目录: {results_dir}
日志文件: {os.path.join(logs_dir, log_file)}

生成的文件:
  - 模型权重: mlp_best.pth, cnn_best.pth
  - 损失曲线: loss_curves.png
  - 准确率曲线: accuracy_curves.png
  - 综合对比图: comprehensive_comparison.png
  - 混淆矩阵: confusion_matrix_mlp.png, confusion_matrix_cnn.png
  - 类别准确率: class_accuracy_mlp.png, class_accuracy_cnn.png
  - 实验结果: experiment_results.json
  - 摘要报告: experiment_summary.txt
{'='*80}
"""
    print(completion_msg)
    logger.info(completion_msg)
    logger.info("实验结束")


if __name__ == '__main__':
    main()
