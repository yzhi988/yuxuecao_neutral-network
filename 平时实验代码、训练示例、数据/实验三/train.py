"""
训练和评估函数
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import time
from tqdm import tqdm
import logging


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    训练一个epoch
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备（CPU或GPU）
    
    Returns:
        average_loss: 平均损失
        accuracy: 准确率
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 使用tqdm显示进度条
    pbar = tqdm(train_loader, desc='Training')
    for inputs, labels in pbar:
        # 将数据移到设备上
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    average_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return average_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    """
    评估模型
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        criterion: 损失函数
        device: 设备（CPU或GPU）
    
    Returns:
        average_loss: 平均损失
        accuracy: 准确率
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Evaluating')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    average_loss = running_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    return average_loss, accuracy


def train_model(model, train_loader, test_loader, num_epochs=20, 
                learning_rate=0.001, device='cuda', model_name='model', logger=None):
    """
    完整训练流程
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        num_epochs: 训练轮数
        learning_rate: 学习率
        device: 设备
        model_name: 模型名称（用于保存）
        logger: 日志记录器
    
    Returns:
        history: 训练历史记录（字典）
    """
    # 将模型移到设备上
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)  # 余弦退火学习率
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    # 用于返回的完整历史记录
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    # 记录开始时间
    start_time = time.time()
    
    log_msg = f"\n开始训练 {model_name}...\n"
    log_msg += f"模型参数量: {model.count_parameters():,}\n"
    log_msg += f"设备: {device}\n"
    log_msg += f"训练轮数: {num_epochs}\n"
    log_msg += f"学习率: {learning_rate}\n"
    log_msg += "-" * 60
    
    print(log_msg)
    if logger:
        logger.info(log_msg)
    
    best_test_acc = 0.0
    best_epoch = 0
    
    for epoch in range(num_epochs):
        epoch_msg = f"\nEpoch {epoch+1}/{num_epochs}"
        epoch_msg += f"\n当前学习率: {scheduler.get_last_lr()[0]:.6f}"
        print(epoch_msg)
        if logger:
            logger.info(epoch_msg)
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 评估
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['test_loss'].append(test_loss)
        training_history['test_acc'].append(test_acc)
        
        # 打印结果
        result_msg = f"训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%\n"
        result_msg += f"测试 - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%"
        print(result_msg)
        if logger:
            logger.info(result_msg)
        
        # 保存最佳模型
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f'{model_name}_best.pth')
            save_msg = f"✓ 保存最佳模型 (Epoch {best_epoch}, 测试准确率: {test_acc:.2f}%)"
            print(save_msg)
            if logger:
                logger.info(save_msg)
    
    # 计算总训练时间
    total_time = time.time() - start_time
    
    summary_msg = "\n" + "=" * 60 + "\n"
    summary_msg += f"训练完成!\n"
    summary_msg += f"总训练时间: {total_time/60:.2f} 分钟\n"
    summary_msg += f"最佳测试准确率: {best_test_acc:.2f}% (Epoch {best_epoch})\n"
    summary_msg += "=" * 60
    
    print(summary_msg)
    if logger:
        logger.info(summary_msg)
    
    # 添加训练时间到历史记录
    history['total_time'] = total_time
    history['best_test_acc'] = best_test_acc
    history['best_epoch'] = best_epoch
    history['training_history'] = training_history
    
    return history

