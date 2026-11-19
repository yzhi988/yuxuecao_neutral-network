"""
CNN（卷积神经网络）模型定义
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    卷积神经网络模型，用于CIFAR-10分类
    
    输入: 32x32x3的RGB图像
    输出: 10个类别的概率分布
    """
    def __init__(self, num_classes=10):
        """
        初始化CNN模型
        
        Args:
            num_classes: 分类数量，默认10（CIFAR-10）
        """
        super(CNN, self).__init__()
        
        # 第一个卷积块: 32x32x3 -> 32x32x64 -> 16x16x64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第二个卷积块: 16x16x64 -> 16x16x128 -> 8x8x128
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第三个卷积块: 8x8x128 -> 8x8x256 -> 4x4x256
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第四个卷积块: 4x4x256 -> 4x4x512 -> 2x2x512
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层
        # 经过4次池化后，图像尺寸: 32 -> 16 -> 8 -> 4 -> 2
        # 所以特征图尺寸为 2x2x512 = 2048
        self.fc1 = nn.Linear(2 * 2 * 512, 1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为(batch_size, 3, 32, 32)
        
        Returns:
            输出张量，形状为(batch_size, 10)
        """
        # 第一个卷积块
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # 第二个卷积块
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # 第三个卷积块
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # 第四个卷积块
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)
        
        # 展平: (batch_size, 512, 2, 2) -> (batch_size, 2048)
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x
    
    def count_parameters(self):
        """
        计算模型参数量
        
        Returns:
            参数量（整数）
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_cnn_model(num_classes=10):
    """
    创建CNN模型的便捷函数
    
    Args:
        num_classes: 分类数量
    
    Returns:
        CNN模型实例
    """
    return CNN(num_classes=num_classes)

