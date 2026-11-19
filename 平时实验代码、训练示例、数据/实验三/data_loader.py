"""
CIFAR-10数据加载模块
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_cifar10_loaders(batch_size=128, num_workers=2):
    """
    获取CIFAR-10数据集的训练和测试数据加载器
    
    Args:
        batch_size: 批次大小，默认128
        num_workers: 数据加载的进程数，默认2
    
    Returns:
        train_loader: 训练集数据加载器
        test_loader: 测试集数据加载器
    """
    # CIFAR-10数据集的均值和标准差（RGB三个通道）
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    
    # 训练集的数据增强和归一化（增强版）
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),      # 随机水平翻转
        transforms.RandomCrop(32, padding=4),        # 随机裁剪（填充4像素）
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
        transforms.RandomRotation(degrees=10),       # 随机旋转
        transforms.ToTensor(),                       # 转换为Tensor
        transforms.Normalize(mean=mean, std=std)     # 归一化
    ])
    
    # 测试集只进行归一化，不做数据增强
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # 下载并加载训练集
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )
    
    # 下载并加载测试集
    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=test_transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # CIFAR-10的类别名称
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    return train_loader, test_loader, classes

