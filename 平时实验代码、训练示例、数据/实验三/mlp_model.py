"""
MLP（多层感知机）模型定义
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    多层感知机模型，用于CIFAR-10分类
    
    输入: 32x32x3的RGB图像，展平后为3072维向量
    输出: 10个类别的概率分布
    """
    def __init__(self, hidden_sizes=[1024, 512, 256], dropout_rate=0.5):
        """
        初始化MLP模型
        
        Args:
            hidden_sizes: 隐藏层的神经元数量列表，默认[1024, 512, 256]
            dropout_rate: Dropout比率，默认0.5
        """
        super(MLP, self).__init__()
        
        # CIFAR-10图像尺寸: 32x32x3 = 3072
        input_size = 32 * 32 * 3
        
        # 构建全连接层
        layers = []
        prev_size = input_size
        
        # 添加隐藏层
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))  # 添加BatchNorm提升性能
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # 输出层（10个类别）
        layers.append(nn.Linear(prev_size, 10))
        
        # 将所有层组合成Sequential模块
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为(batch_size, 3, 32, 32)
        
        Returns:
            输出张量，形状为(batch_size, 10)
        """
        # 展平图像: (batch_size, 3, 32, 32) -> (batch_size, 3072)
        x = x.view(x.size(0), -1)
        
        # 通过全连接网络
        x = self.network(x)
        
        return x
    
    def count_parameters(self):
        """
        计算模型参数量
        
        Returns:
            参数量（整数）
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_mlp_model(hidden_sizes=[1024, 512, 256], dropout_rate=0.5):
    """
    创建MLP模型的便捷函数
    
    Args:
        hidden_sizes: 隐藏层的神经元数量列表
        dropout_rate: Dropout比率
    
    Returns:
        MLP模型实例
    """
    return MLP(hidden_sizes=hidden_sizes, dropout_rate=dropout_rate)

