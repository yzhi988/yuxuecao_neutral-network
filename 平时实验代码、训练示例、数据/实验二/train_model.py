import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from matplotlib import rcParams, font_manager
import warnings
warnings.filterwarnings('ignore')

# 全局中文字体对象，在所有绘图函数中使用
CHINESE_FONT_PROP = None
# 全局英文字体对象（Times New Roman 类），用于英文和数字
ENGLISH_FONT_PROP = None


def _setup_chinese_font():
    """
    自动选择当前系统可用的中文字体，避免图像中文字显示为小方框。
    直接扫描系统字体文件，找到支持中文的字体并强制使用。
    同时为英文和数字推荐类似 Times New Roman 的西文字体。
    """
    # 扫描系统字体，查找中文字体文件
    system_fonts = font_manager.findSystemFonts()
    
    # 优先查找 Noto Sans CJK（最常见的中文字体）
    chinese_font_path = None
    chinese_font_family = None
    
    for font_path in system_fonts:
        font_name_lower = font_path.lower()
        if 'notosanscjk' in font_name_lower and 'regular' in font_name_lower:
            chinese_font_path = font_path
            # 获取字体的实际 family 名称
            try:
                font_prop = font_manager.FontProperties(fname=font_path)
                chinese_font_family = font_prop.get_name()
                break
            except:
                continue
    
    # 如果没找到 Noto Sans CJK Regular，尝试找其他 Noto CJK 字体
    if chinese_font_path is None:
        for font_path in system_fonts:
            font_name_lower = font_path.lower()
            if ('notosanscjk' in font_name_lower or 'notoserifcjk' in font_name_lower) and '.ttc' in font_name_lower:
                try:
                    font_prop = font_manager.FontProperties(fname=font_path)
                    chinese_font_family = font_prop.get_name()
                    chinese_font_path = font_path
                    break
                except:
                    continue
    
    # 如果还是没找到，尝试文泉驿
    if chinese_font_path is None:
        for font_path in system_fonts:
            if 'wqy' in font_path.lower() or 'wenquanyi' in font_path.lower():
                try:
                    font_prop = font_manager.FontProperties(fname=font_path)
                    chinese_font_family = font_prop.get_name()
                    chinese_font_path = font_path
                    break
                except:
                    continue
    
    global CHINESE_FONT_PROP
    
    if chinese_font_path and chinese_font_family:
        print(f"✓ 找到中文字体文件: {chinese_font_path}")
        print(f"✓ 使用字体族名称: {chinese_font_family}")
        # 创建全局字体属性对象，直接使用字体文件路径确保100%正确
        CHINESE_FONT_PROP = font_manager.FontProperties(fname=chinese_font_path)
        # 同时设置全局字体，优先使用找到的中文字体
        rcParams['font.sans-serif'] = [chinese_font_family, 'Noto Sans CJK SC', 'Noto Sans CJK', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    else:
        # 如果找不到字体文件，尝试使用常见的中文字体名称
        print("⚠ 未找到字体文件，使用常见中文字体名称")
        rcParams['font.sans-serif'] = [
            'Noto Sans CJK SC',
            'Noto Sans CJK JP',  # 有些系统使用 JP 变体
            'Noto Sans CJK',
            'WenQuanYi Micro Hei',
            'DejaVu Sans',
        ]
        # 尝试创建一个基于名称的 FontProperties（可能失败，但至少尝试）
        try:
            CHINESE_FONT_PROP = font_manager.FontProperties(family='Noto Sans CJK JP')
        except:
            CHINESE_FONT_PROP = None
    
    # 查找并设置英文字体（Times New Roman 类），用于英文和数字
    global ENGLISH_FONT_PROP
    system_fonts = font_manager.findSystemFonts()
    english_font_path = None
    
    # 优先查找 Times New Roman
    for font_path in system_fonts:
        font_name_lower = font_path.lower()
        if 'times' in font_name_lower and 'new' in font_name_lower and 'roman' in font_name_lower:
            if 'italic' not in font_name_lower and 'bold' not in font_name_lower:
                english_font_path = font_path
                break
    
    # 如果没找到，尝试找 times.ttf
    if english_font_path is None:
        for font_path in system_fonts:
            if 'times.ttf' in font_path.lower() and 'italic' not in font_path.lower():
                english_font_path = font_path
                break
    
    # 如果还是没找到，尝试 DejaVu Serif
    if english_font_path is None:
        for font_path in system_fonts:
            if 'dejavu' in font_path.lower() and 'serif' in font_path.lower() and 'bold' not in font_path.lower():
                english_font_path = font_path
                break
    
    if english_font_path:
        try:
            ENGLISH_FONT_PROP = font_manager.FontProperties(fname=english_font_path)
            print(f"✓ 找到英文字体文件: {english_font_path}")
        except:
            ENGLISH_FONT_PROP = font_manager.FontProperties(family='Times New Roman')
            print("✓ 使用 Times New Roman 字体名称")
    else:
        ENGLISH_FONT_PROP = font_manager.FontProperties(family='Times New Roman')
        print("✓ 使用 Times New Roman 字体名称（未找到字体文件）")
    
    # 设置全局字体配置
    rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'STIXGeneral']
    rcParams['font.family'] = ['sans-serif']  # 主体使用无衬线中文字体
    rcParams['axes.unicode_minus'] = False


_setup_chinese_font()
np.random.seed(42)

def load_dataset(filename='student_grades_dataset.csv'):
    df = pd.read_csv(filename, encoding='utf-8-sig')
    print(f"✓ 加载数据: {df.shape[0]} 行 × {df.shape[1]} 列")
    return df

def create_interaction_features(X_train, X_test):
    """
    创建交互特征，提升模型表达能力
    基于特征重要性分析，选择最重要的特征进行交互
    """
    X_train_feat = X_train.copy()
    X_test_feat = X_test.copy()
    
    # 添加重要的交互特征
    # 学习时间 × 出勤率：学习时间和出勤率的交互效应
    if '学习时间_小时每周' in X_train.columns and '出勤率' in X_train.columns:
        X_train_feat['学习时间_出勤率'] = X_train['学习时间_小时每周'] * X_train['出勤率']
        X_test_feat['学习时间_出勤率'] = X_test['学习时间_小时每周'] * X_test['出勤率']
    
    # 作业完成率 × 课堂参与度：学习投入的综合指标
    if '作业完成率' in X_train.columns and '课堂参与度_1到10' in X_train.columns:
        X_train_feat['作业完成率_课堂参与度'] = X_train['作业完成率'] * X_train['课堂参与度_1到10']
        X_test_feat['作业完成率_课堂参与度'] = X_test['作业完成率'] * X_test['课堂参与度_1到10']
    
    # 学习时间 × 前次考试成绩：学习时间与基础能力的交互
    if '学习时间_小时每周' in X_train.columns and '前次考试成绩' in X_train.columns:
        X_train_feat['学习时间_前次成绩'] = X_train['学习时间_小时每周'] * X_train['前次考试成绩']
        X_test_feat['学习时间_前次成绩'] = X_test['学习时间_小时每周'] * X_test['前次考试成绩']
    
    # 出勤率 × 作业完成率：学习态度的综合指标
    if '出勤率' in X_train.columns and '作业完成率' in X_train.columns:
        X_train_feat['出勤率_作业完成率'] = X_train['出勤率'] * X_train['作业完成率']
        X_test_feat['出勤率_作业完成率'] = X_test['出勤率'] * X_test['作业完成率']
    
    return X_train_feat, X_test_feat

def preprocess_data(X_train, X_test, method='none', use_feature_engineering=True):
    """
    数据预处理
    use_feature_engineering: 是否使用特征工程（添加交互特征）
    """
    # 先进行特征工程
    if use_feature_engineering:
        X_train, X_test = create_interaction_features(X_train, X_test)
    
    # 然后进行缩放
    # 注意：统一返回numpy数组，确保数据类型一致性，避免不同预处理方法返回不同类型导致的问题
    if method == 'none':
        # 统一转换为numpy数组，保持与其他预处理方法返回类型一致
        X_train_array = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        X_test_array = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
        return X_train_array, X_test_array, None
    elif method == 'standard':
        scaler = StandardScaler()
        return scaler.fit_transform(X_train), scaler.transform(X_test), scaler
    elif method == 'minmax':
        scaler = MinMaxScaler()
        return scaler.fit_transform(X_train), scaler.transform(X_test), scaler

def train_mlp_model(X_train, y_train, X_test, y_test, method_name='无处理'):
    print(f"\n训练模型 - {method_name}")
    
    # 使用统一的模型配置，确保公平比较不同预处理方法的效果
    # 注意：如果为不同预处理方法使用不同配置，则无法确定性能差异是来自预处理方法还是模型配置
    # 统一配置可以确保比较的公平性，真实反映预处理方法对模型性能的影响
    mlp = MLPRegressor(
        hidden_layer_sizes=(100, 50),  # 统一的隐藏层结构
        activation='relu',
        solver='adam',
        learning_rate_init=0.01,  # 统一的学习率，经过测试效果最好
        alpha=0.0001,  # 统一的正则化参数
        max_iter=500,  # 设置较大的最大迭代次数，让模型充分训练
        batch_size='auto',  # 自动选择batch size
        random_state=42,  # 固定随机种子，确保结果可复现
        early_stopping=True,  # 启用早停，模型会在收敛时自动停止
        validation_fraction=0.2,
        n_iter_no_change=5,  # 早停窗口：5轮无改善则停止（更激进的早停策略）
        tol=1e-3  # 收敛条件：验证集损失改善小于此值视为无改善（增大此值，避免微小波动导致继续训练）
    )
    # 注意：损失曲线的横轴会自动匹配实际训练轮数（mlp.n_iter_），而不是max_iter
    
    mlp.fit(X_train, y_train)
    
    y_train_pred = mlp.predict(X_train)
    y_test_pred = mlp.predict(X_test)
    
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"\n  【训练集性能】")
    print(f"    MSE:  {train_mse:.4f}")
    print(f"    RMSE: {train_rmse:.4f}")
    print(f"    MAE:  {train_mae:.4f}")
    print(f"    R²:   {train_r2:.4f}")
    
    print(f"\n  【测试集性能】")
    print(f"    MSE:  {test_mse:.4f}")
    print(f"    RMSE: {test_rmse:.4f}")
    print(f"    MAE:  {test_mae:.4f}")
    print(f"    R²:   {test_r2:.4f}")
    
    print(f"\n  迭代次数: {mlp.n_iter_}")
    print(f"  损失曲线长度: {len(mlp.loss_curve_)}")
    print(f"  过拟合程度: {train_r2 - test_r2:.4f}")
    
    # 分析损失收敛情况
    if len(mlp.loss_curve_) > 20:
        # 分析损失变化趋势
        early_losses = mlp.loss_curve_[:20]  # 前20轮
        late_losses = mlp.loss_curve_[-20:]  # 后20轮
        
        early_change = early_losses[0] - early_losses[-1]  # 前20轮的变化
        late_change = late_losses[0] - late_losses[-1]  # 后20轮的变化
        
        print(f"\n  【损失收敛分析】")
        print(f"  前20轮损失变化: {early_change:.6f}")
        print(f"  后20轮损失变化: {late_change:.6f}")
        
        if late_change < 0.001:  # 后20轮变化小于0.001
            print(f"  ⚠ 后20轮损失变化很小 ({late_change:.6f})，说明模型已基本收敛")
            if mlp.n_iter_ > 50:
                print(f"  💡 建议：可以进一步减小 n_iter_no_change 或增大 tol 来更早停止训练")
    
    # 解释早停机制
    if mlp.n_iter_ < 500:
        print(f"\n  ✓ 早停机制生效：模型在 {mlp.n_iter_} 轮后自动停止")
    else:
        print(f"\n  ⚠ 达到最大迭代次数 {mlp.n_iter_}，可能未完全收敛")
    
    return {
        'model': mlp,
        'y_test_pred': y_test_pred,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'loss_curve': mlp.loss_curve_,
        'n_iter': mlp.n_iter_  # 实际训练轮数
    }

def plot_loss_curves(results_dict):
    """
    为每种预处理方法单独绘制损失函数收敛曲线图
    """
    colors = {'无处理': '#FF6B6B', '标准化': '#4ECDC4', '最小最大缩放': '#95E1D3'}
    font_prop = CHINESE_FONT_PROP if CHINESE_FONT_PROP else None
    english_font = ENGLISH_FONT_PROP if ENGLISH_FONT_PROP else None
    
    # 为每种方法单独绘制一张图
    for method_name, results in results_dict.items():
        plt.figure(figsize=(10, 6))
        
        loss_curve = results['loss_curve']
        n_iter = results['n_iter']  # 实际训练轮数
        
        # 绘制损失曲线，横轴使用实际迭代次数
        plt.plot(range(1, len(loss_curve) + 1), loss_curve, 
                color=colors[method_name], linewidth=2, label=method_name)
        
        # 设置x轴范围：从1到实际训练轮数，并添加一些边距
        max_iter_actual = len(loss_curve)
        plt.xlim(0, max_iter_actual + 1)
        
        plt.xlabel('迭代次数', fontsize=12, fontproperties=font_prop)
        plt.ylabel('损失值', fontsize=12, fontproperties=font_prop)
        plt.title(f'{method_name} - 损失函数收敛曲线 (实际训练 {n_iter} 轮)', 
                 fontsize=14, fontweight='bold', fontproperties=font_prop)
        plt.legend(fontsize=11, prop=font_prop)
        
        # 坐标轴刻度（数字）使用英文字体
        ax = plt.gca()
        for label in ax.get_xticklabels():
            label.set_fontproperties(english_font)
        for label in ax.get_yticklabels():
            label.set_fontproperties(english_font)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 为每种方法生成单独的文件名
        filename = f'loss_convergence_curve_{method_name}.png'
        plt.savefig(filename, dpi=300)
        print(f"✓ 已保存: {filename} (实际训练 {n_iter} 轮，损失曲线长度 {max_iter_actual})")
        plt.close()
    
    # 同时生成一张对比图（可选）
    plt.figure(figsize=(12, 6))
    max_iter_all = max(len(results['loss_curve']) for results in results_dict.values())
    
    for method_name, results in results_dict.items():
        loss_curve = results['loss_curve']
        plt.plot(range(1, len(loss_curve) + 1), loss_curve, 
                label=f"{method_name} ({results['n_iter']}轮)", 
                color=colors[method_name], linewidth=2)
    
    plt.xlim(0, max_iter_all + 1)
    plt.xlabel('迭代次数', fontsize=12, fontproperties=font_prop)
    plt.ylabel('损失值', fontsize=12, fontproperties=font_prop)
    plt.title('损失函数收敛曲线对比', fontsize=14, fontweight='bold', fontproperties=font_prop)
    plt.legend(fontsize=11, prop=font_prop)
    
    ax = plt.gca()
    for label in ax.get_xticklabels():
        label.set_fontproperties(english_font)
    for label in ax.get_yticklabels():
        label.set_fontproperties(english_font)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('loss_convergence_curve_comparison.png', dpi=300)
    print("✓ 已保存: loss_convergence_curve_comparison.png (对比图)")
    plt.close()

def plot_prediction_scatter(y_test, results_dict):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    methods = ['无处理', '标准化', '最小最大缩放']
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
    font_prop = CHINESE_FONT_PROP if CHINESE_FONT_PROP else None
    
    for idx, (method_name, color) in enumerate(zip(methods, colors)):
        ax = axes[idx]
        y_pred = results_dict[method_name]['y_test_pred']
        r2 = results_dict[method_name]['test_r2']
        rmse = results_dict[method_name]['test_rmse']
        
        ax.scatter(y_test, y_pred, alpha=0.6, color=color, s=30)
        min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='理想预测')
        ax.set_xlabel('真实值', fontsize=11, fontproperties=font_prop)
        ax.set_ylabel('预测值', fontsize=11, fontproperties=font_prop)

        # 标题：由于 matplotlib 限制，混合文本使用中文字体（Noto Sans CJK 也能显示英文和数字）
        # 坐标轴刻度和数字文本会单独使用英文字体
        english_font = ENGLISH_FONT_PROP if ENGLISH_FONT_PROP else None
        title_text = f'{method_name}\nR²={r2:.4f}, RMSE={rmse:.4f}'
        ax.set_title(title_text, fontsize=11, fontweight='bold', fontproperties=font_prop)
        # 设置坐标轴刻度字体为英文字体（数字）
        ax.tick_params(labelsize=10)
        for label in ax.get_xticklabels():
            label.set_fontproperties(ENGLISH_FONT_PROP if ENGLISH_FONT_PROP else None)
        for label in ax.get_yticklabels():
            label.set_fontproperties(ENGLISH_FONT_PROP if ENGLISH_FONT_PROP else None)
        ax.legend(fontsize=9, prop=font_prop)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prediction_scatter_plot.png', dpi=300)
    print("✓ 已保存: prediction_scatter_plot.png")
    plt.close()

def analyze_feature_importance(X_train, y_train, feature_names):
    print("\n分析特征重要性...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    mlp = MLPRegressor(
        hidden_layer_sizes=(100, 50),  # 与主模型保持一致
        activation='relu',
        solver='adam',
        learning_rate_init=0.01,  # 与主模型保持一致
        alpha=0.0001,
        max_iter=500,  # 与主模型保持一致
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=5,  # 与主模型保持一致
        tol=1e-3
    )
    
    mlp.fit(X_train_scaled, y_train)
    baseline_score = mlp.score(X_train_scaled, y_train)
    
    # 设置随机种子，确保特征重要性分析结果可复现
    rng = np.random.RandomState(42)
    importances = []
    for i in range(X_train_scaled.shape[1]):
        X_permuted = X_train_scaled.copy()
        # 使用固定的随机种子，确保结果可复现
        rng.shuffle(X_permuted[:, i])
        permuted_score = mlp.score(X_permuted, y_train)
        importances.append(baseline_score - permuted_score)
    
    feature_importance_df = pd.DataFrame({
        '特征': feature_names,
        '重要性': importances
    }).sort_values('重要性', ascending=False)
    
    print("\n特征重要性排名：")
    for _, row in feature_importance_df.iterrows():
        print(f"  {row['特征']}: {row['重要性']:.4f}")
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(feature_names)))
    font_prop = CHINESE_FONT_PROP if CHINESE_FONT_PROP else None
    plt.barh(range(len(feature_importance_df)), feature_importance_df['重要性'], color=colors)
    english_font = ENGLISH_FONT_PROP if ENGLISH_FONT_PROP else None
    plt.yticks(range(len(feature_importance_df)), feature_importance_df['特征'], fontproperties=font_prop)
    plt.xlabel('重要性得分', fontsize=12, fontproperties=font_prop)
    plt.title('特征重要性分析', fontsize=14, fontweight='bold', fontproperties=font_prop)
    # 坐标轴刻度（数字）使用英文字体
    ax = plt.gca()
    for label in ax.get_xticklabels():
        label.set_fontproperties(english_font)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    print("\n✓ 已保存: feature_importance.png")
    plt.close()
    
    return feature_importance_df

def compare_preprocessing_methods(results_dict):
    comparison_data = []
    for method_name, results in results_dict.items():
        comparison_data.append({
            '预处理方式': method_name,
            '测试集R²': results['test_r2'],
            '测试集RMSE': results['test_rmse'],
            '测试集MAE': results['test_mae']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n性能对比：")
    print(comparison_df.to_string(index=False))
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    metric_keys = ['测试集R²', '测试集RMSE', '测试集MAE']
    metric_titles = ['测试集R²', '测试集RMSE', '测试集MAE']
    colors = ['#4ECDC4', '#FF6B6B', '#FFA07A']
    font_prop = CHINESE_FONT_PROP if CHINESE_FONT_PROP else None
    
    for idx, (metric_key, color) in enumerate(zip(metric_keys, colors)):
        ax = axes[idx]
        bars = ax.bar(comparison_df['预处理方式'], comparison_df[metric_key], color=color, alpha=0.7)
        english_font = ENGLISH_FONT_PROP if ENGLISH_FONT_PROP else None
        ax.set_ylabel(metric_titles[idx], fontsize=11, fontproperties=font_prop)
        ax.set_title(f'{metric_titles[idx]}对比', fontsize=12, fontweight='bold', fontproperties=font_prop)
        ax.set_xticklabels(comparison_df['预处理方式'], fontproperties=font_prop)
        # 坐标轴刻度（数字）使用英文字体
        for label in ax.get_yticklabels():
            label.set_fontproperties(english_font)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            # 数字文本使用英文字体
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontproperties=english_font)
    
    plt.tight_layout()
    plt.savefig('preprocessing_comparison.png', dpi=300)
    print("\n✓ 已保存: preprocessing_comparison.png")
    plt.close()
    
    return comparison_df

def main():
    print("=" * 60)
    print("MLPRegressor神经网络预测学生成绩实验")
    print("=" * 60)
    
    df = load_dataset()
    X = df.drop('期末成绩', axis=1)
    y = df['期末成绩']
    feature_names = X.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"✓ 训练集: {len(X_train)} 条, 测试集: {len(X_test)} 条")
    
    preprocessing_methods = {
        '无处理': 'none',
        '标准化': 'standard',
        '最小最大缩放': 'minmax'
    }
    
    print("\n✓ 启用特征工程：添加交互特征以提升模型性能")
    
    results_dict = {}
    for method_name, method_code in preprocessing_methods.items():
        # 传入DataFrame以支持特征工程
        X_train_processed, X_test_processed, _ = preprocess_data(
            X_train, X_test, method=method_code, use_feature_engineering=True
        )
        results = train_mlp_model(
            X_train_processed, y_train.values,
            X_test_processed, y_test.values,
            method_name=method_name
        )
        results_dict[method_name] = results
    
    plot_loss_curves(results_dict)
    plot_prediction_scatter(y_test.values, results_dict)
    # 特征重要性分析也使用特征工程
    X_train_feat, _ = create_interaction_features(X_train, X_test)
    feature_importance_df = analyze_feature_importance(
        X_train_feat.values, y_train.values, X_train_feat.columns.tolist()
    )
    comparison_df = compare_preprocessing_methods(results_dict)
    
    print("\n" + "=" * 60)
    print("实验完成")
    print("=" * 60)

if __name__ == "__main__":
    main()
