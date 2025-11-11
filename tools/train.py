"""
训练脚本
训练Social-PatchTST模型
"""

import os
import sys
import argparse
import logging
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import SocialPatchTST, create_model
from data.dataset import create_data_loaders
from config.config_manager import load_config


class Trainer:
    """
    训练器类
    """

    def __init__(self, config_path: str):
        """
        初始化训练器

        Args:
            config_path: 配置文件路径
        """
        self.config = load_config(config_path)
        self.setup_logging()
        self.device = self._setup_device()

        # 创建模型
        self.model = create_model(config_path)
        self.model.to(self.device)

        # 打印模型信息
        model_info = self.model.get_model_info()
        self.logger.info(f"模型信息: {model_info}")

        # 创建数据加载器
        self.train_loader, self.val_loader, self.test_loader, self.processor = create_data_loaders(
            config_path,
            batch_size=1,  # 数据加载器已经按批次组织
            num_workers=self.config.get('device.num_workers', 4)
        )

        # 设置优化器和学习率调度器
        self.setup_optimizer_scheduler()

        # 设置损失函数
        self.criterion = self.model.compute_loss

        # 设置混合精度训练
        self.use_amp = self.config.get('device.mixed_precision', True)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        # 设置TensorBoard
        self.setup_tensorboard()

        # 训练状态
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

    def _setup_device(self) -> torch.device:
        """设置训练设备"""
        gpu_ids = self.config.get('device.gpu_ids', [0])

        if torch.cuda.is_available() and gpu_ids:
            device = torch.device(f'cuda:{gpu_ids[0]}')
            self.logger.info(f"使用GPU: {gpu_ids}")

            # 多GPU训练
            if len(gpu_ids) > 1:
                self.logger.info(f"使用多GPU训练: {gpu_ids}")
        else:
            device = torch.device('cpu')
            self.logger.info("使用CPU训练")

        return device

    def setup_logging(self):
        """设置日志"""
        log_config = self.config.get('logging', {})
        log_dir = log_config.get('log_dir', './logs')
        os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_optimizer_scheduler(self):
        """设置优化器和学习率调度器"""
        training_config = self.config.training_config

        # 优化器
        optimizer_name = training_config.get('optimizer', 'AdamW')
        learning_rate = training_config.get('learning_rate', 0.0001)
        weight_decay = training_config.get('weight_decay', 0.01)

        if optimizer_name == 'AdamW':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"不支持的优化器: {optimizer_name}")

        # 学习率调度器
        scheduler_config = training_config.get('scheduler', {})
        scheduler_name = scheduler_config.get('name', 'CosineAnnealingLR')

        if scheduler_name == 'CosineAnnealingLR':
            T_max = scheduler_config.get('T_max', 100)
            eta_min = scheduler_config.get('eta_min', 0.00001)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=T_max, eta_min=eta_min
            )
        elif scheduler_name == 'StepLR':
            step_size = scheduler_config.get('step_size', 30)
            gamma = scheduler_config.get('gamma', 0.1)
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )
        else:
            self.scheduler = None

    def setup_tensorboard(self):
        """设置TensorBoard"""
        log_config = self.config.get('logging', {})
        tensorboard_dir = log_config.get('tensorboard_dir', './logs/tensorboard')
        os.makedirs(tensorboard_dir, exist_ok=True)

        self.writer = SummaryWriter(tensorboard_dir)

    def train_epoch(self) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1} Training")

        for batch_idx, batch in enumerate(pbar):
            # 将数据移动到设备
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # 前向传播
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    output = self.model(batch, teacher_forcing_ratio=0.5)
                    losses = self.criterion(
                        output['predictions'], batch['targets'], batch['distance_matrix']
                    )
            else:
                output = self.model(batch, teacher_forcing_ratio=0.5)
                losses = self.criterion(
                    output['predictions'], batch['targets'], batch['distance_matrix']
                )

            # 反向传播
            self.optimizer.zero_grad()

            if self.use_amp:
                self.scaler.scale(losses['total_loss']).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            # 记录损失
            total_loss += losses['total_loss'].item()
            num_batches += 1

            # 更新进度条
            pbar.set_postfix({
                'Loss': f"{losses['total_loss'].item():.6f}",
                'Pos': f"{losses['position_loss'].item():.6f}",
                'Alt': f"{losses['altitude_loss'].item():.6f}",
                'Vel': f"{losses['velocity_loss'].item():.6f}",
                'MD': f"{losses['mindist_loss'].item():.6f}"
            })

            # 记录到TensorBoard
            global_step = self.epoch * len(self.train_loader) + batch_idx
            for loss_name, loss_value in losses.items():
                self.writer.add_scalar(f'Train/{loss_name}', loss_value.item(), global_step)

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.train_losses.append(avg_loss)

        return avg_loss

    def validate_epoch(self) -> float:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {self.epoch+1} Validation")

            for batch_idx, batch in enumerate(pbar):
                # 将数据移动到设备
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # 前向传播
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        output = self.model(batch, teacher_forcing_ratio=0.0)
                        losses = self.criterion(
                            output['predictions'], batch['targets'], batch['distance_matrix']
                        )
                else:
                    output = self.model(batch, teacher_forcing_ratio=0.0)
                    losses = self.criterion(
                        output['predictions'], batch['targets'], batch['distance_matrix']
                    )

                # 记录损失
                total_loss += losses['total_loss'].item()
                num_batches += 1

                # 更新进度条
                pbar.set_postfix({
                    'Loss': f"{losses['total_loss'].item():.6f}",
                    'Pos': f"{losses['position_loss'].item():.6f}",
                    'Alt': f"{losses['altitude_loss'].item():.6f}",
                    'Vel': f"{losses['velocity_loss'].item():.6f}",
                    'MD': f"{losses['mindist_loss'].item():.6f}"
                })

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.val_losses.append(avg_loss)

        # 记录到TensorBoard
        self.writer.add_scalar('Validation/Total_Loss', avg_loss, self.epoch)

        return avg_loss

    def save_checkpoint(self, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config.config
        }

        # 保存最新检查点
        checkpoint_path = os.path.join(
            self.config.get('logging.log_dir', './logs'),
            'latest_checkpoint.pth'
        )
        torch.save(checkpoint, checkpoint_path)

        # 保存最佳模型
        if is_best:
            best_path = os.path.join(
                self.config.get('logging.log_dir', './logs'),
                'best_model.pth'
            )
            torch.save(checkpoint, best_path)

        # 定期保存
        save_freq = self.config.get('logging.save_freq', 10)
        if (self.epoch + 1) % save_freq == 0:
            epoch_path = os.path.join(
                self.config.get('logging.log_dir', './logs'),
                f'checkpoint_epoch_{self.epoch+1}.pth'
            )
            torch.save(checkpoint, epoch_path)

    def train(self):
        """主训练循环"""
        training_config = self.config.training_config
        epochs = training_config.get('epochs', 100)
        patience = training_config.get('patience', 10)

        self.logger.info(f"开始训练，总epochs: {epochs}")

        # 早停计数器
        patience_counter = 0

        for epoch in range(epochs):
            self.epoch = epoch

            # 训练
            train_loss = self.train_epoch()
            self.logger.info(f"Epoch {epoch+1} - Train Loss: {train_loss:.6f}")

            # 验证
            val_loss = self.validate_epoch()
            self.logger.info(f"Epoch {epoch+1} - Val Loss: {val_loss:.6f}")

            # 学习率调度
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('Learning_Rate', current_lr, epoch)
                self.logger.info(f"Learning Rate: {current_lr:.8f}")

            # 保存检查点
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                patience_counter = 0
                self.logger.info(f"新的最佳验证损失: {val_loss:.6f}")
            else:
                patience_counter += 1

            self.save_checkpoint(is_best)

            # 早停
            if patience_counter >= patience:
                self.logger.info(f"早停触发，patience: {patience}")
                break

        self.logger.info("训练完成！")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练Social-PatchTST模型')
    parser.add_argument('--config', type=str,
                       default='config/social_patchtst_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--resume', action='store_true',
                       help='从检查点恢复训练')

    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"配置文件不存在: {args.config}")
        return

    # 创建训练器
    trainer = Trainer(args.config)

    # 如果指定了恢复训练，加载检查点
    if args.resume:
        checkpoint_path = os.path.join(
            trainer.config.get('logging.log_dir', './logs'),
            'latest_checkpoint.pth'
        )
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=trainer.device)
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if trainer.scheduler and checkpoint['scheduler_state_dict']:
                trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            trainer.epoch = checkpoint['epoch']
            trainer.best_val_loss = checkpoint['best_val_loss']
            trainer.train_losses = checkpoint['train_losses']
            trainer.val_losses = checkpoint['val_losses']
            trainer.logger.info(f"从epoch {trainer.epoch}恢复训练，最佳验证损失: {trainer.best_val_loss:.6f}")
        else:
            trainer.logger.warning("未找到检查点文件，从头开始训练")

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()