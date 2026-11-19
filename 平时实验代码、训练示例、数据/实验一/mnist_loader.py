import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

def preprocess_and_save_mnist(
    save_path='./preprocessed_data',
    batch_size=1000, 
    augment=True
):
    """加载、预处理并保存MNIST数据集"""
    print("=== 开始精细化预处理并保存MNIST数据集 ===")
    
    # 1. 定义精细化的预处理管道
    # 训练集使用数据增强
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomRotation(10),  # 随机旋转±10度
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # 随机平移
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        print("训练集将使用数据增强 (随机旋转、平移)")
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        print("训练集不使用数据增强")

    # 测试集不使用数据增强，只进行标准化
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 2. 加载原始数据集
    print("正在加载原始MNIST数据集...")
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        transform=train_transform,
        download=True
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        transform=test_transform,
        download=True
    )
    
    # 使用DataLoader来应用变换
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. 迭代处理所有数据并收集
    def process_data(loader):
        all_images = []
        all_labels = []
        for images, labels in loader:
            # 展平图像用于MLP
            flattened_images = images.view(images.size(0), -1)
            all_images.append(flattened_images)
            all_labels.append(labels)
        return torch.cat(all_images, dim=0), torch.cat(all_labels, dim=0)

    print("正在处理训练集...")
    train_images, train_labels = process_data(train_loader)
    print("正在处理测试集...")
    test_images, test_labels = process_data(test_loader)
    
    # 4. 创建保存目录
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"创建目录: {save_path}")
    
    # 5. 保存为.pt文件
    train_save_file = os.path.join(save_path, 'train.pt')
    test_save_file = os.path.join(save_path, 'test.pt')
    
    print(f"正在保存预处理后的训练集到: {train_save_file}")
    torch.save((train_images, train_labels), train_save_file)
    
    print(f"正在保存预处理后的测试集到: {test_save_file}")
    torch.save((test_images, test_labels), test_save_file)
    
    # 6. 输出统计信息
    print("\n=== 预处理完成并已保存 ===")
    print(f"训练数据形状: {train_images.shape}")
    print(f"训练标签形状: {train_labels.shape}")
    print(f"测试数据形状: {test_images.shape}")
    print(f"测试标签形状: {test_labels.shape}")
    print(f"数据已保存到 '{save_path}' 目录中")


def load_preprocessed_data(save_path='./preprocessed_data'):
    """加载已保存的预处理数据"""
    train_file = os.path.join(save_path, 'train.pt')
    test_file = os.path.join(save_path, 'test.pt')
    
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print("未找到预处理文件，请先运行脚本生成数据。")
        return None, None
        
    print(f"正在从 '{save_path}' 加载预处理数据...")
    train_data = torch.load(train_file)
    test_data = torch.load(test_file)
    
    return train_data, test_data


if __name__ == "__main__":
    # 执行预处理和保存
    preprocess_and_save_mnist(augment=True)
    
    # 演示如何加载已保存的数据
    print("\n--- 演示加载已保存的数据 ---")
    train_data, test_data = load_preprocessed_data()
    
    if train_data and test_data:
        train_images, train_labels = train_data
        test_images, test_labels = test_data
        
        print("加载成功!")
        print(f"训练数据形状: {train_images.shape}")
        print(f"测试数据形状: {test_images.shape}")
        
        # 还可以将它们包装成DataLoader
        train_dataset = TensorDataset(train_images, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        print("已创建训练数据加载器 (DataLoader)")