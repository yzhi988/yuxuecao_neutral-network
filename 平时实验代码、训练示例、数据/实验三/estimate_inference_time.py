"""
基于训练时间估算推理时间
"""
import math

def estimate_inference_time(training_time_minutes, num_epochs=40, train_samples=50000, batch_size=128):
    """
    从训练时间估算推理时间
    
    推理时间估算原理：
    1. 训练时间 = 前向传播时间 + 反向传播时间
    2. 反向传播时间 ≈ 2 × 前向传播时间（因为需要计算梯度）
    3. 所以：前向传播时间 ≈ 训练时间 / 3
    4. 推理只有前向传播，所以推理时间 ≈ 前向传播时间
    """
    # 总训练时间（秒）
    total_time_seconds = training_time_minutes * 60
    
    # 每个epoch的时间
    time_per_epoch = total_time_seconds / num_epochs
    
    # 每个batch的训练时间
    batches_per_epoch = math.ceil(train_samples / batch_size)
    time_per_batch_training = time_per_epoch / batches_per_epoch
    
    # 前向传播时间（训练时前向传播约占总时间的1/3）
    time_per_batch_forward = time_per_batch_training / 3
    
    # 每个样本的推理时间（推理只有前向传播）
    inference_time_per_sample_ms = (time_per_batch_forward / batch_size) * 1000
    
    return inference_time_per_sample_ms, time_per_batch_forward, batches_per_epoch


def main():
    print("=" * 80)
    print("基于训练时间估算推理时间")
    print("=" * 80)
    
    # 从日志中获取的数据
    mlp_training_time = 5.23  # 分钟
    cnn_training_time = 5.27  # 分钟
    num_epochs = 40
    train_samples = 50000
    batch_size = 128
    
    print(f"\n训练配置:")
    print(f"  训练轮数: {num_epochs}")
    print(f"  训练样本数: {train_samples:,}")
    print(f"  批次大小: {batch_size}")
    
    # MLP模型估算
    print(f"\n{'='*80}")
    print("MLP模型")
    print(f"{'='*80}")
    print(f"参数量: 3,809,034")
    print(f"训练时间: {mlp_training_time} 分钟 ({mlp_training_time*60:.1f} 秒)")
    
    mlp_inference_ms, mlp_batch_forward, mlp_batches = estimate_inference_time(
        mlp_training_time, num_epochs, train_samples, batch_size
    )
    
    print(f"\n每个epoch的batch数: {mlp_batches}")
    print(f"每个batch的训练时间: {mlp_training_time*60/num_epochs/mlp_batches*1000:.2f} ms")
    print(f"每个batch的前向传播时间: {mlp_batch_forward*1000:.2f} ms")
    print(f"\n估算结果:")
    print(f"  单样本推理时间: {mlp_inference_ms:.4f} ms")
    print(f"  吞吐量: {1000/mlp_inference_ms:.1f} 样本/秒")
    
    # CNN模型估算
    print(f"\n{'='*80}")
    print("CNN模型")
    print(f"{'='*80}")
    print(f"参数量: 4,181,002")
    print(f"训练时间: {cnn_training_time} 分钟 ({cnn_training_time*60:.1f} 秒)")
    
    cnn_inference_ms, cnn_batch_forward, cnn_batches = estimate_inference_time(
        cnn_training_time, num_epochs, train_samples, batch_size
    )
    
    print(f"\n每个epoch的batch数: {cnn_batches}")
    print(f"每个batch的训练时间: {cnn_training_time*60/num_epochs/cnn_batches*1000:.2f} ms")
    print(f"每个batch的前向传播时间: {cnn_batch_forward*1000:.2f} ms")
    print(f"\n估算结果:")
    print(f"  单样本推理时间: {cnn_inference_ms:.4f} ms")
    print(f"  吞吐量: {1000/cnn_inference_ms:.1f} 样本/秒")
    
    # 对比
    print(f"\n{'='*80}")
    print("对比分析")
    print(f"{'='*80}")
    print(f"{'指标':<25} {'MLP':<20} {'CNN':<20} {'比例(CNN/MLP)':<15}")
    print("-" * 80)
    print(f"{'参数量':<25} {'3,809,034':<20} {'4,181,002':<20} {4181002/3809034:.3f}")
    print(f"{'训练时间(分钟)':<25} {mlp_training_time:<20.2f} {cnn_training_time:<20.2f} {cnn_training_time/mlp_training_time:.3f}")
    print(f"{'单样本推理时间(ms)':<25} {mlp_inference_ms:<20.4f} {cnn_inference_ms:<20.4f} {cnn_inference_ms/mlp_inference_ms:.3f}")
    print(f"{'吞吐量(样本/秒)':<25} {1000/mlp_inference_ms:<20.1f} {1000/cnn_inference_ms:<20.1f} {(1000/mlp_inference_ms)/(1000/cnn_inference_ms):.3f}")
    
    # 不同场景的推理时间
    print(f"\n{'='*80}")
    print("不同场景的推理时间估算")
    print(f"{'='*80}")
    
    scenarios = [
        ("单张图片", 1),
        ("一个batch (128张)", 128),
        ("测试集 (10,000张)", 10000),
        ("训练集 (50,000张)", 50000),
    ]
    
    print(f"\n{'场景':<20} {'MLP时间':<20} {'CNN时间':<20}")
    print("-" * 60)
    for scenario, num_samples in scenarios:
        mlp_time = (mlp_inference_ms * num_samples) / 1000  # 秒
        cnn_time = (cnn_inference_ms * num_samples) / 1000  # 秒
        
        if mlp_time < 1:
            mlp_str = f"{mlp_time*1000:.2f} ms"
        elif mlp_time < 60:
            mlp_str = f"{mlp_time:.2f} 秒"
        else:
            mlp_str = f"{mlp_time/60:.2f} 分钟"
        
        if cnn_time < 1:
            cnn_str = f"{cnn_time*1000:.2f} ms"
        elif cnn_time < 60:
            cnn_str = f"{cnn_time:.2f} 秒"
        else:
            cnn_str = f"{cnn_time/60:.2f} 分钟"
        
        print(f"{scenario:<20} {mlp_str:<20} {cnn_str:<20}")
    
    print(f"\n{'='*80}")
    print("说明")
    print(f"{'='*80}")
    print("""
估算方法说明：
1. 训练时间包括：前向传播 + 反向传播 + 参数更新
2. 反向传播的计算量约为前向传播的2倍
3. 因此：前向传播时间 ≈ 训练时间 / 3
4. 推理只有前向传播，所以推理时间 ≈ 前向传播时间

注意：
- 这是理论估算值，实际推理时间会受到以下因素影响：
  * 硬件性能（CPU/GPU型号）
  * 批次大小（batch_size越大，每样本平均时间越短）
  * 数据加载和预处理时间
  * 内存带宽和缓存效率
  
- 建议运行 benchmark_inference.py 进行实际测试以获得准确值
    """)


if __name__ == '__main__':
    main()

