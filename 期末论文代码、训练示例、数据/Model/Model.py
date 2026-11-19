import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import grad
import math
from utils.util import AverageMeter, get_logger, eval_metrix
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MishActivation(nn.Module):
    """
    Mish Activation Function: x * tanh(softplus(x))
    Performs better than ReLU and GELU in many cases
    """
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))



class AdaptiveAttention(nn.Module):
    """
    自适应注意力机制：根据输入特征动态调整注意力权重
    """
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        # 动态权重生成器
        self.weight_generator = nn.Sequential(
            nn.Linear(dim, dim // 4),
            MishActivation(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )
        # 添加残差连接
        self.residual_proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        batch_size = x.shape[0]
        x_reshaped = x.unsqueeze(1)  # [B, 1, D]
        
        # 注意力处理
        normalized_x = self.norm(x_reshaped)
        attn_output, _ = self.attention(normalized_x, normalized_x, normalized_x)
        
        # 生成动态权重
        attention_weight = self.weight_generator(x)  # [B, D]
        
        # 应用动态权重和残差连接
        weighted_output = attn_output.squeeze(1) * attention_weight
        residual = self.residual_proj(x)
        
        return weighted_output + residual

class EnhancedResidualBlock(nn.Module):
    """
    增强型残差块：包含特征重校准和多分支处理
    """
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        # 主分支
        self.main_branch = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            MishActivation (),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        # 特征重校准 (SE-Net风格)
        self.se = nn.Sequential(
            nn.Linear(dim, dim // 8),
            MishActivation(),
            nn.Linear(dim // 8, dim),
            nn.Sigmoid()
        )
        # 添加分支注意力
        self.gating = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        identity = x
        main = self.main_branch(x)
        
        # 特征重校准
        se_weight = self.se(main)
        main = main * se_weight
        
        # 分支门控机制
        gate = self.gating(identity)
        
        return x + main * gate

class ImprovedMLP(nn.Module):
    def __init__(self, input_dim=17, output_dim=1, layers_num=4, hidden_dim=50, dropout=0.1):
        super().__init__()
        
        self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # 多尺度特征提取
        self.multi_scale_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                MishActivation()
            ) for _ in range(3)
        ])
        
        # 增强型残差层
        self.residual_layers = nn.ModuleList([
            EnhancedResidualBlock(hidden_dim, dropout) for _ in range(layers_num-1)
        ])
        
        # 自适应注意力
        self.attention = AdaptiveAttention(hidden_dim)
        
        # 输出层，使用两层以增强表达能力
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            MishActivation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.activation = MishActivation()
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        # 特征归一化
        x = self.input_norm(x)
        x = self.activation(self.input_layer(x))
        
        # 多尺度特征融合
        multi_scale_features = []
        for scale_layer in self.multi_scale_layers:
            multi_scale_features.append(scale_layer(x))
            
        if multi_scale_features:  # 确保列表非空
            # 特征融合采用注意力加权求和
            feature_stack = torch.stack(multi_scale_features, dim=1)  # [B, 3, hidden_dim]
            feature_weights = F.softmax(torch.mean(feature_stack, dim=2, keepdim=True), dim=1)  # [B, 3, 1]
            weighted_features = torch.sum(feature_stack * feature_weights, dim=1)  # [B, hidden_dim]
            x = x + weighted_features
        
        # 残差处理
        for res_layer in self.residual_layers:
            x = res_layer(x)
            
        # 注意力处理
        x = self.attention(x)
        x = self.dropout(x)
        
        # 输出映射
        x = self.output_layers(x)
        return x

# ImprovedPredictor类
class ImprovedPredictor(nn.Module):
    def __init__(self, input_dim=40):
        super().__init__()
        
        # 更丰富的预测网络架构
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Dropout(p=0.1),
            nn.Linear(input_dim, 128),
            MishActivation(),
            EnhancedResidualBlock(128, dropout=0.1),
            nn.Linear(128, 64),
            MishActivation(),
            EnhancedResidualBlock(64, dropout=0.1),
            nn.Dropout(p=0.1),
            nn.Linear(64, 32),
            MishActivation(),
            nn.Linear(32, 1)
        )
        self.input_dim = input_dim
        
    def forward(self, x):
        return self.net(x)

# ImprovedSolution类
class ImprovedSolution(nn.Module):
    def __init__(self):
        super().__init__()
        # 增加编码器的容量和复杂度
        self.encoder = ImprovedMLP(input_dim=17, output_dim=64, layers_num=5, hidden_dim=128, dropout=0.15)
        # 更强大的预测网络
        self.predictor = ImprovedPredictor(input_dim=64)
        self._init_weights()
        
    def get_embedding(self, x):
        return self.encoder(x)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.predictor(x)
        return x
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)





class PINN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.save_folder is not None and not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
        log_dir = args.log_dir if args.save_folder is None else os.path.join(args.save_folder, args.log_dir)
        self.logger = get_logger(log_dir)
        self._save_args()
        
        # 初始化改进的模型
        self.solution_u = ImprovedSolution().to(device)
        self.dynamical_F = ImprovedMLP(
            input_dim=35,
            output_dim=1,
            layers_num=args.F_layers_num,
            hidden_dim=args.F_hidden_dim,
            dropout=0.15  # 增加dropout以提高泛化能力
        ).to(device)
        
        # 使用AdamW优化器，提供更好的权重衰减策略
        self.optimizer1 = torch.optim.AdamW(
            self.solution_u.parameters(),
            lr=args.warmup_lr,
            weight_decay=1e-4,
            betas=(0.9, 0.999)  # 使用更适合深度网络的动量参数
        )
        self.optimizer2 = torch.optim.AdamW(
            self.dynamical_F.parameters(),
            lr=args.lr_F,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # 使用OneCycleLR调度器，在训练中逐渐提高学习率再降低
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer1,
            max_lr=args.lr,
            total_steps=args.epochs,
            pct_start=0.3,  # 30%的训练时间用于提高学习率
            div_factor=args.lr/args.warmup_lr,  # 初始学习率除以最大学习率
            final_div_factor=args.warmup_lr/args.final_lr  # 最终学习率降低的比例
        )
        
        # 添加第二个优化器的调度器
        self.scheduler2 = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer2,
            max_lr=args.lr_F * 2,  # 动力学模型使用较大的学习率
            total_steps=args.epochs,
            pct_start=0.3,
            div_factor=2,
            final_div_factor=10
        )
        
        self.loss_func = nn.MSELoss()
        self.huber_loss = nn.SmoothL1Loss()  # 添加Huber Loss提高鲁棒性
        self.relu = nn.ReLU()
        
        self.best_model = None
        self.alpha = self.args.alpha
        self.beta = self.args.beta
        
        # 添加学习率预热的标志
        self.warmup_done = False

    def forward(self, xt):
        xt.requires_grad = True
        x = xt[:, 0:-1]
        t = xt[:, -1:]
        
        u = self.solution_u(torch.cat((x, t), dim=1))
        
        u_t = grad(u.sum(), t, create_graph=True, retain_graph=True)[0]
        u_x = grad(u.sum(), x, create_graph=True, retain_graph=True)[0]
        
        F = self.dynamical_F(torch.cat([xt, u, u_x, u_t], dim=1))
        
        f = u_t - F
        return u, f

    def train_one_epoch(self, epoch, dataloader):
        self.train()
        loss1_meter = AverageMeter()
        loss2_meter = AverageMeter()
        loss3_meter = AverageMeter()
        
        for iter, (x1, x2, y1, y2) in enumerate(dataloader):
            x1, x2, y1, y2 = x1.to(device), x2.to(device), y1.to(device), y2.to(device)
            
            u1, f1 = self.forward(x1)
            u2, f2 = self.forward(x2)
            
            # 数据损失使用MSE和Huber Loss的组合
            loss1_mse = 0.5 * (self.loss_func(u1, y1) + self.loss_func(u2, y2))
            loss1_huber = 0.5 * (self.huber_loss(u1, y1) + self.huber_loss(u2, y2))
            loss1 = 0.8 * loss1_mse + 0.2 * loss1_huber  # 混合损失
            #loss1 = 0.5*self.loss_func(u1,y1) + 0.5*self.loss_func(u2,y2)
            
            # PDE损失
            loss2 = 0.5 * (self.loss_func(f1, torch.zeros_like(f1)) + 
                          self.loss_func(f2, torch.zeros_like(f2)))
            
            # 物理约束损失，使用平滑版本
            loss3 = self.relu(torch.mul(u2-u1, y1-y2)).mean()
            
            # 总损失计算，加入动态权重平衡
            epoch_ratio = min(epoch / (0.5 * self.args.epochs), 1.0)  # 随训练进程调整
            current_alpha = self.alpha * epoch_ratio  # 逐渐增加PDE损失权重
            current_beta = self.beta * epoch_ratio    # 逐渐增加物理约束损失权重
            
            loss = loss1 + self.alpha * loss2 + self.beta * loss3
            
            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            loss.backward()
            
            # 梯度裁剪，提高训练稳定性
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            self.optimizer1.step()
            self.optimizer2.step()
            
            loss1_meter.update(loss1.item())
            loss2_meter.update(loss2.item())
            loss3_meter.update(loss3.item())
            
            if (iter + 1) % 50 == 0:
                self.logger.info("[train] epoch:{} iter:{} data loss:{:.6f}, PDE loss:{:.6f}, physics loss:{:.6f}".format(
                    epoch, iter+1, loss1.item(), loss2.item(), loss3.item()
                ))
                
        return loss1_meter.avg, loss2_meter.avg, loss3_meter.avg

    def Train(self, trainloader, testloader=None, validloader=None):
        min_valid_mse = float('inf')
        early_stop = 0
        
        for epoch in range(1, self.args.epochs + 1):
            loss1, loss2, loss3 = self.train_one_epoch(epoch, trainloader)
            
            # 更新学习率调度器
            self.scheduler.step()
            self.scheduler2.step()
            
            current_lr = self.optimizer1.param_groups[0]['lr']
            
            total_loss = loss1 + self.alpha * loss2 + self.beta * loss3
            info = '[Train] epoch:{}, lr:{:.6f}, data loss:{:.6f}, PDE loss:{:.6f}, physics loss:{:.6f}, total loss:{:.6f}'.format(
                epoch, current_lr, loss1, loss2, loss3, total_loss
            )
            self.logger.info(info)
            
            if validloader is not None:
                valid_mse = self.Valid(validloader)
                info = '[Valid] epoch:{}, MSE: {}'.format(epoch, valid_mse)
                self.logger.info(info)
                
                if valid_mse < min_valid_mse:
                    min_valid_mse = valid_mse
                    early_stop = 0
                    
                    if testloader is not None:
                        true_label, pred_label = self.Test(testloader)
                        [MAE, MAPE, MSE, RMSE, R2] = eval_metrix(pred_label, true_label)
                        info = '[Test] MSE: {:.8f}, MAE: {:.6f}, MAPE: {:.6f}, RMSE: {:.6f}, R2: {:.6f}'.format(
                            MSE, MAE, MAPE, RMSE, R2
                        )
                        self.logger.info(info)
                        
                        if self.args.save_folder is not None:
                            np.save(os.path.join(self.args.save_folder, 'true_label.npy'), true_label)
                            np.save(os.path.join(self.args.save_folder, 'pred_label.npy'), pred_label)
                            
                        self.best_model = {
                            'solution_u': self.solution_u.state_dict(),
                            'dynamical_F': self.dynamical_F.state_dict(),
                            'epoch': epoch,
                            'valid_mse': valid_mse,
                            'test_metrics': {
                                'MAE': MAE,
                                'MAPE': MAPE,
                                'MSE': MSE,
                                'RMSE': RMSE,
                                'R2': R2
                            }
                        }
                else:
                    early_stop += 1
                    
            if self.args.early_stop is not None and early_stop > self.args.early_stop:
                info = 'early stop at epoch {}'.format(epoch)
                self.logger.info(info)
                break
                
        if self.args.save_folder is not None and self.best_model is not None:
            torch.save(self.best_model, os.path.join(self.args.save_folder, 'model.pth'))
            
        self.clear_logger()

    def Valid(self, validloader):
        self.eval()
        valid_losses = []
        with torch.no_grad():
            for x1, _, y1, _ in validloader:
                x1 = x1.to(device)
                y1 = y1.to(device)
                u1 = self.solution_u(x1)
                valid_losses.append(self.loss_func(u1, y1).item())
        return np.mean(valid_losses)

    def Test(self, testloader):
        self.eval()
        true_labels = []
        pred_labels = []
        with torch.no_grad():
            for x1, _, y1, _ in testloader:
                x1 = x1.to(device)
                u1 = self.solution_u(x1)
                true_labels.append(y1.numpy())
                pred_labels.append(u1.cpu().numpy())
        return np.concatenate(true_labels), np.concatenate(pred_labels)

    def _save_args(self):
        if self.args.log_dir is not None:
            self.logger.info("Args:")
            for k, v in self.args.__dict__.items():
                self.logger.critical(f"\t{k}:{v}")

    def clear_logger(self):
        if len(self.logger.handlers) > 0:
            self.logger.removeHandler(self.logger.handlers[0])
            self.logger.handlers.clear()




def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    
    # 数据相关参数
    parser.add_argument('--data', type=str, default='XJTU', help='数据集选择: XJTU, HUST, MIT, TJU')
    parser.add_argument('--batch_size', type=int, default=256, help='批次大小')
    parser.add_argument('--normalization_method', type=str, default='robust', help='归一化方法: min-max, z-score或robust')
    
    # 模型架构参数
    parser.add_argument('--F_layers_num', type=int, default=5, help='动力学模型层数')
    parser.add_argument('--F_hidden_dim', type=int, default=128, help='动力学模型隐藏层维度')
    
    # 训练策略参数
    parser.add_argument('--epochs', type=int, default=400, help='训练轮数')
    parser.add_argument('--lr', type=float, default=2e-3, help='基础学习率')
    parser.add_argument('--warmup_epochs', type=int, default=15, help='预热训练轮数')
    parser.add_argument('--warmup_lr', type=float, default=1e-3, help='预热阶段学习率')
    parser.add_argument('--final_lr', type=float, default=5e-6, help='最终学习率')
    parser.add_argument('--lr_F', type=float, default=2e-3, help='动力学模型学习率')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='权重衰减率')
    
    # 损失函数权重
    parser.add_argument('--alpha', type=float, default=1.0, help='PDE损失权重')
    parser.add_argument('--beta', type=float, default=0.5, help='物理约束损失权重')
    
    # 训练控制参数
    parser.add_argument('--early_stop', type=int, default=30, help='早停轮数')
    parser.add_argument('--save_folder', type=str, default='checkpoints', help='模型保存路径')
    parser.add_argument('--log_dir', type=str, default='logs', help='日志保存路径')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    model = PINN(args)
    print("Model Architecture:")
    print(model.solution_u)
    print("\nDynamical Model:")
    print(model.dynamical_F)