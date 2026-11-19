"""
实验结果保存模块
"""
import json
import os
from datetime import datetime
import numpy as np


def save_experiment_results(mlp_results, cnn_results, save_dir='results'):
    """
    保存实验结果到JSON文件
    
    Args:
        mlp_results: MLP实验结果字典
        cnn_results: CNN实验结果字典
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 准备保存的数据
    results = {
        'experiment_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'mlp_model': {
            'parameters': mlp_results['parameters'],
            'best_test_accuracy': mlp_results['best_test_acc'],
            'best_epoch': mlp_results['best_epoch'],
            'total_training_time_minutes': mlp_results['total_time'] / 60,
            'final_train_loss': mlp_results['history']['train_loss'][-1],
            'final_train_acc': mlp_results['history']['train_acc'][-1],
            'final_test_loss': mlp_results['history']['test_loss'][-1],
            'final_test_acc': mlp_results['history']['test_acc'][-1],
            'training_history': {
                'train_loss': [float(x) for x in mlp_results['history']['train_loss']],
                'train_acc': [float(x) for x in mlp_results['history']['train_acc']],
                'test_loss': [float(x) for x in mlp_results['history']['test_loss']],
                'test_acc': [float(x) for x in mlp_results['history']['test_acc']]
            }
        },
        'cnn_model': {
            'parameters': cnn_results['parameters'],
            'best_test_accuracy': cnn_results['best_test_acc'],
            'best_epoch': cnn_results['best_epoch'],
            'total_training_time_minutes': cnn_results['total_time'] / 60,
            'final_train_loss': cnn_results['history']['train_loss'][-1],
            'final_train_acc': cnn_results['history']['train_acc'][-1],
            'final_test_loss': cnn_results['history']['test_loss'][-1],
            'final_test_acc': cnn_results['history']['test_acc'][-1],
            'training_history': {
                'train_loss': [float(x) for x in cnn_results['history']['train_loss']],
                'train_acc': [float(x) for x in cnn_results['history']['train_acc']],
                'test_loss': [float(x) for x in cnn_results['history']['test_loss']],
                'test_acc': [float(x) for x in cnn_results['history']['test_acc']]
            }
        },
        'comparison': {
            'parameter_ratio': cnn_results['parameters'] / mlp_results['parameters'],
            'accuracy_improvement': cnn_results['best_test_acc'] - mlp_results['best_test_acc'],
            'time_ratio': (cnn_results['total_time'] / 60) / (mlp_results['total_time'] / 60)
        }
    }
    
    # 如果有混淆矩阵和类别准确率，也保存
    if 'confusion_matrix' in mlp_results:
        results['mlp_model']['confusion_matrix'] = mlp_results['confusion_matrix'].tolist()
    if 'class_accuracy' in mlp_results:
        results['mlp_model']['class_accuracy'] = [float(x) for x in mlp_results['class_accuracy']]
    
    if 'confusion_matrix' in cnn_results:
        results['cnn_model']['confusion_matrix'] = cnn_results['confusion_matrix'].tolist()
    if 'class_accuracy' in cnn_results:
        results['cnn_model']['class_accuracy'] = [float(x) for x in cnn_results['class_accuracy']]
    
    # 保存到JSON文件
    json_path = os.path.join(save_dir, 'experiment_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"\n实验结果已保存到: {json_path}")
    return json_path


def save_summary_report(mlp_results, cnn_results, save_dir='results'):
    """
    保存文本格式的摘要报告
    
    Args:
        mlp_results: MLP实验结果字典
        cnn_results: CNN实验结果字典
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    report_path = os.path.join(save_dir, 'experiment_summary.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("CIFAR-10分类实验报告\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("MLP模型结果\n")
        f.write("-" * 80 + "\n")
        f.write(f"参数量: {mlp_results['parameters']:,}\n")
        f.write(f"最佳测试准确率: {mlp_results['best_test_acc']:.2f}% (Epoch {mlp_results['best_epoch']})\n")
        f.write(f"最终训练损失: {mlp_results['history']['train_loss'][-1]:.4f}\n")
        f.write(f"最终训练准确率: {mlp_results['history']['train_acc'][-1]:.2f}%\n")
        f.write(f"最终测试损失: {mlp_results['history']['test_loss'][-1]:.4f}\n")
        f.write(f"最终测试准确率: {mlp_results['history']['test_acc'][-1]:.2f}%\n")
        f.write(f"总训练时间: {mlp_results['total_time']/60:.2f} 分钟\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("CNN模型结果\n")
        f.write("-" * 80 + "\n")
        f.write(f"参数量: {cnn_results['parameters']:,}\n")
        f.write(f"最佳测试准确率: {cnn_results['best_test_acc']:.2f}% (Epoch {cnn_results['best_epoch']})\n")
        f.write(f"最终训练损失: {cnn_results['history']['train_loss'][-1]:.4f}\n")
        f.write(f"最终训练准确率: {cnn_results['history']['train_acc'][-1]:.2f}%\n")
        f.write(f"最终测试损失: {cnn_results['history']['test_loss'][-1]:.4f}\n")
        f.write(f"最终测试准确率: {cnn_results['history']['test_acc'][-1]:.2f}%\n")
        f.write(f"总训练时间: {cnn_results['total_time']/60:.2f} 分钟\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("模型对比\n")
        f.write("-" * 80 + "\n")
        param_ratio = cnn_results['parameters'] / mlp_results['parameters']
        acc_improvement = cnn_results['best_test_acc'] - mlp_results['best_test_acc']
        time_ratio = (cnn_results['total_time'] / 60) / (mlp_results['total_time'] / 60)
        
        f.write(f"参数量比例 (CNN/MLP): {param_ratio:.2f}\n")
        f.write(f"准确率提升: {acc_improvement:.2f}%\n")
        f.write(f"训练时间比例 (CNN/MLP): {time_ratio:.2f}\n\n")
        
        if 'class_accuracy' in mlp_results and 'class_accuracy' in cnn_results:
            f.write("-" * 80 + "\n")
            f.write("各类别准确率对比\n")
            f.write("-" * 80 + "\n")
            classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            f.write(f"{'类别':<10} {'MLP准确率':<15} {'CNN准确率':<15} {'提升':<10}\n")
            f.write("-" * 50 + "\n")
            for i, cls in enumerate(classes):
                mlp_acc = mlp_results['class_accuracy'][i]
                cnn_acc = cnn_results['class_accuracy'][i]
                improvement = cnn_acc - mlp_acc
                f.write(f"{cls:<10} {mlp_acc:>6.2f}%        {cnn_acc:>6.2f}%        {improvement:>+6.2f}%\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"实验摘要报告已保存到: {report_path}")
    return report_path

