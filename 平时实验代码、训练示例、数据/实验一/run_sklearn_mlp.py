import sys
import torch
import os
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 从我们之前的脚本中导入数据加载函数
from mnist_loader import load_preprocessed_data

# --- 辅助函数 ---
def log_and_print(message: str, logfile):
    """同时打印到终端并写入日志文件"""
    print(message)
    logfile.write(message + "\n")

def save_training_curves(model, save_path, logfile):
    """绘制并保存模型的训练损失和验证准确率曲线"""
    plt.figure(figsize=(12, 5))

    # 绘制验证准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(model.validation_scores_, label='Validation Accuracy')
    plt.title('Validation Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # 绘制训练损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(model.loss_curve_, label='Training Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    fig_path = os.path.join(save_path, 'training_curves.png')
    plt.savefig(fig_path)
    plt.close()
    log_and_print(f"训练曲线图已保存: {fig_path}", logfile)

def save_confusion_matrix(y_true, y_pred, save_path, class_names, logfile):
    """绘制并保存混淆矩阵图"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    fig_path = os.path.join(save_path, 'confusion_matrix.png')
    plt.savefig(fig_path)
    plt.close()
    log_and_print(f"混淆矩阵已保存: {fig_path}", logfile)

# --- 主程序 ---
def main():
    # --- 设置路径 ---
    DATA_PATH = './preprocessed_data'
    RESULTS_PATH = './sklearn_experiment_results'
    os.makedirs(RESULTS_PATH, exist_ok=True)

    # 打开日志文件
    log_path = os.path.join(RESULTS_PATH, 'experiment_log.txt')
    logfile = open(log_path, 'w', encoding='utf-8')
    log_and_print("===== Scikit-learn MLP 实验开始 =====", logfile)

    # --- 1. 加载并转换数据 ---
    log_and_print("正在加载预处理数据...", logfile)
    train_data, test_data = load_preprocessed_data(save_path=DATA_PATH)
    if train_data is None:
        log_and_print("加载数据失败，请先运行 mnist_loader.py 生成数据。", logfile)
        logfile.close()
        return

    # 将 PyTorch 张量转换为 Scikit-learn 使用的 NumPy 数组
    X_train, y_train = train_data[0].numpy(), train_data[1].numpy()
    X_test, y_test = test_data[0].numpy(), test_data[1].numpy()
    log_and_print(f"数据加载完成。训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}", logfile)

    # --- 2. 初始化并训练 MLPClassifier 模型 ---
    log_and_print("正在初始化 MLPClassifier...", logfile)
    model = MLPClassifier(
        hidden_layer_sizes=(100,),
        max_iter=30,
        activation='relu',
        solver='adam',
        batch_size=64,
        verbose=True,
        random_state=42,
        learning_rate_init=0.001,
        early_stopping=True, # 必须为True才能记录validation_scores_
        validation_fraction=0.1
    )

    log_and_print("开始训练模型...", logfile)
    # 重定向 stdout 到日志文件以捕获 verbose 输出
    original_stdout = sys.stdout
    sys.stdout = logfile
    model.fit(X_train, y_train)
    # 恢复 stdout
    sys.stdout = original_stdout
    log_and_print("模型训练完成。", logfile)

    # --- 3. 评估模型 ---
    log_and_print("正在评估模型...", logfile)
    y_pred = model.predict(X_test)

    # --- 4. 保存所有结果 ---
    log_and_print("正在保存实验结果...", logfile)
    class_names = [str(i) for i in range(10)]

    # 1. 生成并保存详细的分类报告
    log_and_print("\n===== 最终分类报告 =====", logfile)
    # 生成文本格式的报告以便写入日志
    report_text = classification_report(y_test, y_pred, target_names=class_names)
    log_and_print(report_text, logfile)

    # 生成字典格式的报告以便保存为JSON
    report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    report_dict['overall_accuracy'] = accuracy_score(y_test, y_pred)
    json_path = os.path.join(RESULTS_PATH, 'classification_results.json')
    with open(json_path, 'w') as f:
        json.dump(report_dict, f, indent=4)
    log_and_print(f"\n分类报告JSON文件已保存: {json_path}", logfile)

    # 2. 保存训练好的模型 (使用 joblib)
    model_path = os.path.join(RESULTS_PATH, 'sklearn_mlp_model.joblib')
    joblib.dump(model, model_path)
    log_and_print(f"模型已保存: {model_path}", logfile)

    # 3. 保存新的训练曲线图
    save_training_curves(model, RESULTS_PATH, logfile)

    # 3b. 保存原始曲线数据，便于后续比较实验
    curves_path = os.path.join(RESULTS_PATH, 'curves_data.json')
    with open(curves_path, 'w') as f:
        json.dump({
            'loss_curve': model.loss_curve_,
            'validation_scores': model.validation_scores_
        }, f)
    log_and_print(f"曲线原始数据已保存: {curves_path}", logfile)

    # 4. 保存混淆矩阵图
    save_confusion_matrix(y_test, y_pred, RESULTS_PATH, class_names, logfile)

    logfile.close()
    print(f"\n所有实验结果已保存在 '{RESULTS_PATH}' 文件夹中。")
    print(f"完整日志已写入: {log_path}")

if __name__ == '__main__':
    main()
