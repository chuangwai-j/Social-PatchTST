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
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Tuple, Optional

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import SocialPatchTST, create_model
from data.dataset.scene_dataset import create_social_patchtst_loaders
from config.config_manager import load_config


# -----------------------------------------------------------------
# 论文性能指标计算函数 (Kimi的建议)
# -----------------------------------------------------------------

def calculate_rmse(pred, truth, feature_indices):
    """计算指定特征维度的RMSE"""
    if pred.ndim < 3 or truth.ndim < 3 or not feature_indices:
        return 0.0
    errors = pred[:, :, feature_indices] - truth[:, :, feature_indices]
    return np.sqrt(np.mean(errors**2))

def calculate_mae(pred, truth, feature_indices):
    """计算指定特征维度的MAE"""
    if pred.ndim < 3 or truth.ndim < 3 or not feature_indices:
        return 0.0
    errors = pred[:, :, feature_indices] - truth[:, :, feature_indices]
    return np.mean(np.abs(errors))

def calculate_far(pred, truth, cpa_threshold_nm=3.0, alt_threshold_ft=500):
    """
    计算虚警率 (False Alarm Rate) - 社交模型特有指标

    Args:
        pred: 预测结果 [N, T_out, features]
        truth: 真实标签 [N, T_out, features]
        cpa_threshold_nm: CPA阈值 (海里)
        alt_threshold_ft: 高度阈值 (英尺)

    Returns:
        far: 虚警率 (0-1之间的值)
    """
    # 根据您的数据格式，假设特征顺序为 [lat, lon, alt, vx, vy]
    # 预测的冲突数量：预测CPA < 阈值 或 高度差 > 阈值
    pred_conflicts = 0

    # 真实的冲突数量：实际CPA < 阈值 或 高度差 > 阈值
    truth_conflicts = 0

    # 简化实现：基于高度差和位置距离的简化冲突检测
    for i in range(pred.shape[0]):  # batch
        for t in range(pred.shape[1]):  # time
            # 计算位置距离 (简化为欧几里得距离)
            pos_dist = np.sqrt(pred[i, t, 0]**2 + pred[i, t, 1]**2)

            # 计算高度差
            alt_diff = abs(pred[i, t, 2] - truth[i, t, 2]) if pred.shape[2] > 2 else 0

            # 预测冲突判断
            if pos_dist < cpa_threshold_nm or alt_diff > alt_threshold_ft:
                pred_conflicts += 1

            # 真实冲突判断
            truth_pos_dist = np.sqrt(truth[i, t, 0]**2 + truth[i, t, 1]**2)
            truth_alt_diff = abs(truth[i, t, 2] - truth[i, t, 2]) if truth.shape[2] > 2 else 0
            if truth_pos_dist < cpa_threshold_nm or truth_alt_diff > alt_threshold_ft:
                truth_conflicts += 1

    total_predictions = pred.shape[0] * pred.shape[1]
    total_safe = total_predictions - truth_conflicts

    if total_safe == 0:
        return 0.0

    # FAR = 预测冲突但实际安全的数量 / 实际安全的总数
    false_alarms = max(0, pred_conflicts - truth_conflicts)
    return false_alarms / total_safe


class Trainer:
    """
    训练器类
    """

    def __init__(self, config_path: str, is_baseline: bool = False):
        """
        初始化训练器

        Args:
            config_path: 配置文件路径
            is_baseline: 是否运行Baseline模式（关闭社交模块）
        """
        self.config = load_config(config_path)
        self.is_baseline = is_baseline
        self.setup_logging()
        self.device = self._setup_device()

        # 创建模型
        self.model = create_model(config_path, is_baseline=self.is_baseline)
        mode_name = "Baseline (原版PatchTST)" if self.is_baseline else "Social-PatchTST"
        self.logger.info(f"运行模式: {mode_name}")
        self.model.to(self.device)

        # 打印模型信息
        model_info = self.model.get_model_info()
        self.logger.info(f"模型信息: {model_info}")

        # 创建场景数据加载器
        scenes_dir = self.config.get('data.scenes_dir', '/tmp/test_scenes')

        # 检查场景目录是否存在
        if not os.path.exists(scenes_dir):
            self.logger.error(f"场景数据目录不存在: {scenes_dir}")
            self.logger.error("请先运行场景数据生成器:")
            self.logger.error(f"python data/dataset/data_processor.py --input-dir /mnt/d/adsb --output-dir {os.path.dirname(scenes_dir)}")
            raise FileNotFoundError(f"场景数据目录不存在: {scenes_dir}")

        try:
            self.train_loader, self.val_loader, self.test_loader = create_social_patchtst_loaders(
                config_path=config_path,
                batch_size=self.config.get('training.batch_size', 4),
                max_neighbors=self.config.get('social_transformer.max_aircrafts', 50),
                sequence_length=self.config.get('data.history_length', 600),
                num_workers=self.config.get('device.num_workers', 4)
            )
        except Exception as e:
            self.logger.error(f"创建数据加载器失败: {e}")
            self.logger.error("请确保场景数据已正确生成")
            raise

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

        # 损失历史记录
        self.train_losses = []
        self.val_losses = []

        # 全面的指标历史记录 (Gemini建议)
        self.metrics_history = {
            # 训练损失指标
            'train_total_loss': [],
            'train_position_loss': [],
            'train_altitude_loss': [],
            'train_velocity_loss': [],
            'train_mindist_loss': [],

            # 验证损失指标
            'val_total_loss': [],
            'val_position_loss': [],
            'val_altitude_loss': [],
            'val_velocity_loss': [],
            'val_mindist_loss': [],

            # 性能指标 (RMSE, MAE)
            'val_position_rmse': [],
            'val_altitude_rmse': [],
            'val_velocity_rmse': [],
            'val_position_mae': [],
            'val_altitude_mae': [],
            'val_velocity_mae': [],

            # 社交模型特有指标
            'val_far': [],
            'val_mindist_mean': [],

            # 训练信息
            'learning_rates': [],
            'epoch_times': []
        }

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

    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch，返回详细的损失字典"""
        self.model.train()

        # 累计各项损失
        total_losses = {'total_loss': 0.0, 'position_loss': 0.0, 'altitude_loss': 0.0, 'velocity_loss': 0.0, 'mindist_loss': 0.0}
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
                        output['predictions'], batch['targets'], batch.get('distance_matrix', None)
                    )
            else:
                output = self.model(batch, teacher_forcing_ratio=0.5)
                losses = self.criterion(
                    output['predictions'],  # 这是4个预测头的字典
                    batch['targets'],       # 这是targets张量
                    batch.get('distance_matrix', None)
                )

            # 防空洞：空样本loss置0
            if batch['targets'].numel() == 0:
                losses = {'total_loss': 0.0 * output['predictions'].sum(), 'position_loss': 0.0 * output['predictions'].sum(),
                         'altitude_loss': 0.0 * output['predictions'].sum(), 'velocity_loss': 0.0 * output['predictions'].sum(),
                         'mindist_loss': 0.0 * output['predictions'].sum()}

            # 应用样本权重
            if 'sample_weight' in batch:
                sample_weights = batch['sample_weight'].to(self.device)
                # 对每个损失项应用样本权重
                for loss_name in losses:
                    if loss_name != 'total_loss':  # total_loss会在compute_loss中重新计算
                        losses[loss_name] = losses[loss_name] * sample_weights.mean()

                # 重新计算加权的总损失
                loss_weights = self.config.get('training.loss_weights', {})
                position_weight = loss_weights.get('position', 1.0)
                altitude_weight = loss_weights.get('altitude', 1.0)
                velocity_weight = loss_weights.get('velocity', 0.5)
                mindist_weight = loss_weights.get('mindist', 2.0)

                losses['total_loss'] = (
                    position_weight * losses['position_loss'] +
                    altitude_weight * losses['altitude_loss'] +
                    velocity_weight * losses['velocity_loss'] +
                    mindist_weight * losses['mindist_loss']
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

            # 累计损失
            for loss_name in total_losses:
                total_losses[loss_name] += losses[loss_name].item()
            num_batches += 1

            # NaN/Inf检测
            losses['total_loss'] = torch.nan_to_num(losses['total_loss'], nan=0.0, posinf=1.0, neginf=1e-6)

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

        # 计算平均损失
        avg_losses = {loss_name: total_loss / num_batches if num_batches > 0 else 0.0
                     for loss_name, total_loss in total_losses.items()}

        # 保持向后兼容性
        self.train_losses.append(avg_losses['total_loss'])

        return avg_losses

    def validate_epoch(self) -> Dict[str, float]:
        """验证一个epoch，返回详细的损失和性能指标"""
        self.model.eval()

        # 累计损失和性能指标
        total_losses = {'total_loss': 0.0, 'position_loss': 0.0, 'altitude_loss': 0.0, 'velocity_loss': 0.0, 'mindist_loss': 0.0}

        # 性能指标累计
        all_predictions = []
        all_targets = []
        all_mindist_predictions = []

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
                            output['predictions'], batch['targets'], batch.get('distance_matrix', None)
                        )
                else:
                    output = self.model(batch, teacher_forcing_ratio=0.0)
                    losses = self.criterion(
                        output['predictions'], batch['targets'], batch.get('distance_matrix', None)
                    )

                # 防空洞：空样本loss置0
                if batch['targets'].numel() == 0:
                    losses = {'total_loss': 0.0 * output['predictions'].sum(), 'position_loss': 0.0 * output['predictions'].sum(),
                             'altitude_loss': 0.0 * output['predictions'].sum(), 'velocity_loss': 0.0 * output['predictions'].sum(),
                             'mindist_loss': 0.0 * output['predictions'].sum()}

                # 累计损失
                for loss_name in total_losses:
                    total_losses[loss_name] += losses[loss_name].item()
                num_batches += 1

                # 收集预测和目标用于性能指标计算
                # 取ego飞机的预测和目标 (batch_size, pred_len, features)
                predictions = output['predictions']
                targets = batch['targets']

                if targets.numel() > 0:  # 确保不是空样本
                    # 取第一架飞机(ego)的预测和目标
                    ego_predictions = predictions[:, 0, :, :].cpu().numpy()  # [batch_size, pred_len, features]
                    ego_targets = targets[:, 0, :, :].cpu().numpy()          # [batch_size, pred_len, features]

                    all_predictions.append(ego_predictions)
                    all_targets.append(ego_targets)

                    # 收集mindist预测 (假设mindist是第4个预测头)
                    if 'mindist' in output:
                        mindist_pred = output['mindist'][:, 0, :, 0].cpu().numpy()  # [batch_size, pred_len]
                        all_mindist_predictions.append(mindist_pred)

                # 更新进度条
                pbar.set_postfix({
                    'Loss': f"{losses['total_loss'].item():.6f}",
                    'Pos': f"{losses['position_loss'].item():.6f}",
                    'Alt': f"{losses['altitude_loss'].item():.6f}",
                    'Vel': f"{losses['velocity_loss'].item():.6f}",
                    'MD': f"{losses['mindist_loss'].item():.6f}"
                })

        # 计算平均损失
        avg_losses = {loss_name: total_loss / num_batches if num_batches > 0 else 0.0
                     for loss_name, total_loss in total_losses.items()}

        # 保持向后兼容性
        self.val_losses.append(avg_losses['total_loss'])

        # 记录到TensorBoard
        self.writer.add_scalar('Validation/Total_Loss', avg_losses['total_loss'], self.epoch)

        # === 计算性能指标 (Gemini建议) ===
        performance_metrics = {}

        if all_predictions and all_targets:
            # 合并所有batch的预测和目标
            all_predictions = np.concatenate(all_predictions, axis=0)  # [total_samples, pred_len, features]
            all_targets = np.concatenate(all_targets, axis=0)          # [total_samples, pred_len, features]

            # 根据数据集特征顺序计算指标
            # 假设特征顺序为: [flight_level, latitude, longitude, vx, vy] 或类似
            try:
                # 位置指标 (latitude, longitude) - 假设是特征1和2
                position_rmse = calculate_rmse(all_predictions, all_targets, feature_indices=[1, 2])
                position_mae = calculate_mae(all_predictions, all_targets, feature_indices=[1, 2])

                # 高度指标 (flight_level) - 假设是特征0
                altitude_rmse = calculate_rmse(all_predictions, all_targets, feature_indices=[0])
                altitude_mae = calculate_mae(all_predictions, all_targets, feature_indices=[0])

                # 速度指标 (vx, vy) - 假设是特征3和4
                velocity_rmse = calculate_rmse(all_predictions, all_targets, feature_indices=[3, 4])
                velocity_mae = calculate_mae(all_predictions, all_targets, feature_indices=[3, 4])

                performance_metrics.update({
                    'val_position_rmse': position_rmse,
                    'val_position_mae': position_mae,
                    'val_altitude_rmse': altitude_rmse,
                    'val_altitude_mae': altitude_mae,
                    'val_velocity_rmse': velocity_rmse,
                    'val_velocity_mae': velocity_mae
                })

                # 计算虚警率 (FAR) - 社交模型特有指标
                far = calculate_far(all_predictions, all_targets)
                performance_metrics['val_far'] = far

            except Exception as e:
                self.logger.warning(f"性能指标计算失败: {e}")
                # 设置默认值
                performance_metrics.update({
                    'val_position_rmse': 0.0,
                    'val_position_mae': 0.0,
                    'val_altitude_rmse': 0.0,
                    'val_altitude_mae': 0.0,
                    'val_velocity_rmse': 0.0,
                    'val_velocity_mae': 0.0,
                    'val_far': 0.0
                })

        # Mindist指标
        if all_mindist_predictions:
            all_mindist_predictions = np.concatenate(all_mindist_predictions, axis=0)
            performance_metrics['val_mindist_mean'] = float(np.mean(all_mindist_predictions))
        else:
            performance_metrics['val_mindist_mean'] = 0.0

        # 合并损失和性能指标
        final_metrics = {**avg_losses, **performance_metrics}

        return final_metrics

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

        # 创建checkpoints目录
        checkpoints_dir = './checkpoints'
        os.makedirs(checkpoints_dir, exist_ok=True)

        # 保存最新检查点
        checkpoint_path = os.path.join(checkpoints_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)

        # 保存最佳模型到checkpoints/
        if is_best:
            best_path = os.path.join(checkpoints_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"💎 最佳模型已保存到: {best_path}")

        # 定期保存
        save_freq = self.config.get('logging.save_freq', 10)
        if (self.epoch + 1) % save_freq == 0:
            epoch_path = os.path.join(
                checkpoints_dir, f'checkpoint_epoch_{self.epoch+1}.pth'
            )
            torch.save(checkpoint, epoch_path)

    def save_metrics_to_json(self, filename: str = None):
        """保存所有训练指标到JSON文件"""
        if filename is None:
            # 创建metrics_calculation目录
            metrics_dir = './metrics_calculation'
            os.makedirs(metrics_dir, exist_ok=True)
            filename = f'{metrics_dir}/training_metrics_epoch_{self.epoch}.json'

        # 确保目录存在
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # 准备保存的数据
        results_data = {
            'experiment_info': {
                'model_name': 'Social-PatchTST',
                'config_file': self.config.config_path if hasattr(self.config, 'config_path') else 'unknown',
                'total_epochs': self.epoch + 1,
                'best_epoch': self.metrics_history['val_total_loss'].index(min(self.metrics_history['val_total_loss'])) if self.metrics_history['val_total_loss'] else 0,
                'best_val_loss': min(self.metrics_history['val_total_loss']) if self.metrics_history['val_total_loss'] else float('inf'),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'final_metrics': {
                'train_total_loss': self.metrics_history['train_total_loss'][-1] if self.metrics_history['train_total_loss'] else 0.0,
                'val_total_loss': self.metrics_history['val_total_loss'][-1] if self.metrics_history['val_total_loss'] else 0.0,
                'train_position_loss': self.metrics_history['train_position_loss'][-1] if self.metrics_history['train_position_loss'] else 0.0,
                'val_position_loss': self.metrics_history['val_position_loss'][-1] if self.metrics_history['val_position_loss'] else 0.0,
                'val_position_rmse': self.metrics_history['val_position_rmse'][-1] if self.metrics_history['val_position_rmse'] else 0.0,
                'val_position_mae': self.metrics_history['val_position_mae'][-1] if self.metrics_history['val_position_mae'] else 0.0,
                'val_altitude_rmse': self.metrics_history['val_altitude_rmse'][-1] if self.metrics_history['val_altitude_rmse'] else 0.0,
                'val_velocity_rmse': self.metrics_history['val_velocity_rmse'][-1] if self.metrics_history['val_velocity_rmse'] else 0.0,
                'val_far': self.metrics_history['val_far'][-1] if self.metrics_history['val_far'] else 0.0,
                'final_learning_rate': self.metrics_history['learning_rates'][-1] if self.metrics_history['learning_rates'] else 0.0
            },
            'best_epoch_metrics': {},
            'metrics_history': self.metrics_history
        }

        # 添加最佳epoch的指标
        if self.metrics_history['val_total_loss']:
            best_epoch = results_data['experiment_info']['best_epoch']
            results_data['best_epoch_metrics'] = {
                'epoch': best_epoch,
                'train_total_loss': self.metrics_history['train_total_loss'][best_epoch] if best_epoch < len(self.metrics_history['train_total_loss']) else 0.0,
                'val_total_loss': self.metrics_history['val_total_loss'][best_epoch],
                'val_position_rmse': self.metrics_history['val_position_rmse'][best_epoch] if best_epoch < len(self.metrics_history['val_position_rmse']) else 0.0,
                'val_position_mae': self.metrics_history['val_position_mae'][best_epoch] if best_epoch < len(self.metrics_history['val_position_mae']) else 0.0,
                'val_altitude_rmse': self.metrics_history['val_altitude_rmse'][best_epoch] if best_epoch < len(self.metrics_history['val_altitude_rmse']) else 0.0,
                'val_velocity_rmse': self.metrics_history['val_velocity_rmse'][best_epoch] if best_epoch < len(self.metrics_history['val_velocity_rmse']) else 0.0,
                'val_far': self.metrics_history['val_far'][best_epoch] if best_epoch < len(self.metrics_history['val_far']) else 0.0,
                'learning_rate': self.metrics_history['learning_rates'][best_epoch] if best_epoch < len(self.metrics_history['learning_rates']) else 0.0
            }

        # 保存到JSON文件
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"📊 训练指标已保存到: {filename}")

        # 打印最终结果摘要
        self.logger.info("=== 训练完成摘要 ===")
        self.logger.info(f"总训练轮数: {results_data['experiment_info']['total_epochs']}")
        self.logger.info(f"最佳验证损失: {results_data['experiment_info']['best_val_loss']:.6f} (epoch {results_data['experiment_info']['best_epoch']})")
        if results_data['best_epoch_metrics']:
            self.logger.info(f"最佳轮次位置RMSE: {results_data['best_epoch_metrics']['val_position_rmse']:.6f}")
            self.logger.info(f"最佳轮次虚警率: {results_data['best_epoch_metrics']['val_far']:.6f}")

        return filename

    def train(self):
        """主训练循环"""
        training_config = self.config.get('training', {})
        epochs = training_config.get('epochs', 100)
        patience = training_config.get('patience', 10)

        self.logger.info(f"开始训练，总epochs: {epochs}")

        # 早停计数器
        patience_counter = 0

        for epoch in range(epochs):
            self.epoch = epoch
            epoch_start_time = time.time()

            # 训练
            train_metrics = self.train_epoch()
            self.logger.info(f"Epoch {epoch+1} - Train Loss: {train_metrics['total_loss']:.6f}")

            # 验证
            val_metrics = self.validate_epoch()
            self.logger.info(f"Epoch {epoch+1} - Val Loss: {val_metrics['total_loss']:.6f}")

            # 记录epoch时间
            epoch_time = time.time() - epoch_start_time

            # === 记录所有指标到历史 (Gemini建议) ===
            self.metrics_history['train_total_loss'].append(train_metrics['total_loss'])
            self.metrics_history['train_position_loss'].append(train_metrics['position_loss'])
            self.metrics_history['train_altitude_loss'].append(train_metrics['altitude_loss'])
            self.metrics_history['train_velocity_loss'].append(train_metrics['velocity_loss'])
            self.metrics_history['train_mindist_loss'].append(train_metrics['mindist_loss'])

            self.metrics_history['val_total_loss'].append(val_metrics['total_loss'])
            self.metrics_history['val_position_loss'].append(val_metrics['position_loss'])
            self.metrics_history['val_altitude_loss'].append(val_metrics['altitude_loss'])
            self.metrics_history['val_velocity_loss'].append(val_metrics['velocity_loss'])
            self.metrics_history['val_mindist_loss'].append(val_metrics['mindist_loss'])

            # 性能指标 (如果有的话)
            self.metrics_history['val_position_rmse'].append(val_metrics.get('val_position_rmse', 0.0))
            self.metrics_history['val_altitude_rmse'].append(val_metrics.get('val_altitude_rmse', 0.0))
            self.metrics_history['val_velocity_rmse'].append(val_metrics.get('val_velocity_rmse', 0.0))
            self.metrics_history['val_position_mae'].append(val_metrics.get('val_position_mae', 0.0))
            self.metrics_history['val_altitude_mae'].append(val_metrics.get('val_altitude_mae', 0.0))
            self.metrics_history['val_velocity_mae'].append(val_metrics.get('val_velocity_mae', 0.0))

            # 社交模型特有指标
            self.metrics_history['val_far'].append(val_metrics.get('val_far', 0.0))
            self.metrics_history['val_mindist_mean'].append(val_metrics.get('val_mindist_mean', 0.0))

            # 训练信息
            self.metrics_history['epoch_times'].append(epoch_time)

            # 学习率调度
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                self.metrics_history['learning_rates'].append(current_lr)
                self.writer.add_scalar('Learning_Rate', current_lr, epoch)
                self.logger.info(f"Learning Rate: {current_lr:.8f}")

            # 保存检查点
            is_best = val_metrics['total_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['total_loss']
                patience_counter = 0
                self.logger.info(f"新的最佳验证损失: {val_metrics['total_loss']:.6f}")
            else:
                patience_counter += 1

            self.save_checkpoint(is_best)

            # 每5轮保存一次指标JSON
            if (epoch + 1) % 5 == 0:
                self.save_metrics_to_json(f'./metrics_calculation/training_metrics_epoch_{epoch+1}.json')

            # 早停
            if patience_counter >= patience:
                self.logger.info(f"早停触发，patience: {patience}")
                break

        # 训练完成后，保存最终的完整指标
        final_metrics_file = self.save_metrics_to_json('./metrics_calculation/final_training_metrics.json')

        self.logger.info("训练完成！")

        return final_metrics_file


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练Social-PatchTST模型')
    parser.add_argument('--config', type=str,
                       default='config/social_patchtst_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--resume', action='store_true',
                       help='从检查点恢复训练')
    parser.add_argument('--test', action='store_true',
                       help='运行测试模式而不是训练')
    parser.add_argument('--scenes_dir', type=str,
                       help='场景数据目录路径')
    parser.add_argument('--baseline', action='store_true',
                       help='运行原版PatchTST (Baseline) 模式，关闭社交模块')

    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"配置文件不存在: {args.config}")
        return

    # 如果指定了测试模式
    if args.test:
        from data.dataset.scene_dataset import SocialPatchTSTDataset
        from config.config_manager import load_config

        config = load_config(args.config)
        scenes_dir = args.scenes_dir or config.data_config.get('scenes_dir', '/tmp/test_scenes')

        print(f"测试场景数据加载...")
        print(f"场景目录: {scenes_dir}")

        # For testing, we need a paths file
        paths_file = None
        if os.path.exists(os.path.join(scenes_dir, "train_paths.txt")):
            paths_file = os.path.join(scenes_dir, "train_paths.txt")

        dataset = SocialPatchTSTDataset(
            data_dir=scenes_dir,
            max_neighbors=10,
            paths_file=paths_file
        )

        print(f"数据集大小: {len(dataset)}")

        if len(dataset) > 0:
            sample = dataset[0]
            print(f"样本形状:")
            print(f"  - 时序数据: {sample['temporal'].shape}")
            print(f"  - 空间数据: {sample['spatial'].shape}")
            print(f"  - 目标数据: {sample['targets'].shape}")
            print(f"  - 距离矩阵: {sample['distance_matrix'].shape}")

        print("✅ 测试完成!")
        return

    # 创建训练器
    trainer = Trainer(args.config, is_baseline=args.baseline)

    # 如果指定了恢复训练，加载检查点
    if args.resume:
        checkpoint_path = './checkpoints/latest_checkpoint.pth'
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