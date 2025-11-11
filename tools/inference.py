"""
推理脚本
使用训练好的Social-PatchTST模型进行预测
"""

import os
import sys
import argparse
import logging
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import SocialPatchTST, create_model
from data.dataset import ADSBDataProcessor
from config.config_manager import load_config


class Predictor:
    """
    预测器类
    """

    def __init__(self, config_path: str, model_path: str = None):
        """
        初始化预测器

        Args:
            config_path: 配置文件路径
            model_path: 模型权重路径
        """
        self.config = load_config(config_path)
        self.setup_logging()
        self.device = self._setup_device()

        # 创建模型
        self.model = create_model(config_path)
        self.model.to(self.device)

        # 加载模型权重
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(f"成功加载模型权重: {model_path}")
        else:
            # 尝试加载默认的最佳模型
            default_path = os.path.join(
                self.config.get('logging.log_dir', './logs'),
                'best_model.pth'
            )
            if os.path.exists(default_path):
                checkpoint = torch.load(default_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.logger.info(f"成功加载默��模型: {default_path}")
            else:
                self.logger.warning("未找到模型权重，使用随机初始化的模型")

        self.model.eval()

        # 创建数据处理器
        self.processor = ADSBDataProcessor(config_path)

    def _setup_device(self) -> torch.device:
        """设置推理设备"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            self.logger.info("使用GPU推理")
        else:
            device = torch.device('cpu')
            self.logger.info("使用CPU推理")
        return device

    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def predict_batch(self, batch: dict) -> dict:
        """
        批量预测

        Args:
            batch: 输入批次数据

        Returns:
            预测结果字典
        """
        # 将数据移动到设备
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

        with torch.no_grad():
            start_time = time.time()
            predictions = self.model.predict(batch)
            inference_time = time.time() - start_time

        return {
            'predictions': predictions.cpu(),
            'aircraft_ids': batch.get('aircraft_ids'),
            'inference_time': inference_time,
            'batch_size': predictions.size(0),
            'n_aircrafts': predictions.size(1),
            'prediction_length': predictions.size(2)
        }

    def predict_from_dataframe(self, df: pd.DataFrame, max_aircrafts: int = 50) -> dict:
        """
        从DataFrame进行预测

        Args:
            df: 输入数据DataFrame
            max_aircrafts: 最大飞机数量

        Returns:
            预测结果字典
        """
        self.logger.info(f"开始预测，输入数据: {len(df)} 行")

        # 数据预处理
        df_processed = self.processor.transform_data(df)

        # 创建输入序列
        inputs_list, _ = self.processor.create_sequences(df_processed)
        self.logger.info(f"生成了 {len(inputs_list)} 个序列")

        if not inputs_list:
            raise ValueError("没有生成有效的序列数据")

        # 创建批次
        batch_data = self.processor.create_multi_aircraft_batch(
            inputs_list, [], max_aircrafts
        )

        # 进行预测
        result = self.predict_batch(batch_data)

        # 后处理结果
        predictions = result['predictions']
        aircraft_ids = result['aircraft_ids']

        # 转换为更易理解的格式
        formatted_results = []

        for i, aircraft_id in enumerate(aircraft_ids):
            aircraft_predictions = predictions[i].numpy()  # [prediction_length, output_dim]

            # 转换预测结果为DataFrame
            pred_df = pd.DataFrame(aircraft_predictions, columns=[
                'flight_level', 'latitude', 'longitude', 'ground_speed', 'track_angle'
            ])

            # 添加时间信息
            start_time = batch_data['start_times'][i]
            time_interval = self.config.get('data.sampling_interval', 5)

            pred_df['timestamp'] = [start_time + (j+1) * time_interval for j in range(len(pred_df))]
            pred_df['aircraft_id'] = aircraft_id

            formatted_results.append(pred_df)

        # 合并所有预测结果
        final_result = pd.concat(formatted_results, ignore_index=True)

        return {
            'predictions': final_result,
            'inference_time': result['inference_time'],
            'n_aircrafts': result['n_aircrafts'],
            'prediction_length': result['prediction_length']
        }

    def evaluate_predictions(self, predictions: pd.DataFrame, targets: pd.DataFrame) -> dict:
        """
        评估预测结果

        Args:
            predictions: 预测结果DataFrame
            targets: 真实标签DataFrame

        Returns:
            评估指标字典
        """
        # 按飞机ID分组评估
        aircraft_ids = predictions['aircraft_id'].unique()
        metrics = {}

        all_position_errors = []
        all_altitude_errors = []
        all_speed_errors = []

        for aircraft_id in aircraft_ids:
            pred_aircraft = predictions[predictions['aircraft_id'] == aircraft_id]
            target_aircraft = targets[targets['aircraft_id'] == aircraft_id]

            if len(pred_aircraft) == 0 or len(target_aircraft) == 0:
                continue

            # 对齐时间戳
            merged = pd.merge(pred_aircraft, target_aircraft,
                             on=['aircraft_id', 'timestamp'],
                             suffixes=('_pred', '_target'))

            if len(merged) == 0:
                continue

            # 计算位置误差（欧氏距离）
            pos_errors = np.sqrt(
                (merged['latitude_pred'] - merged['latitude_target'])**2 +
                (merged['longitude_pred'] - merged['longitude_target'])**2
            )

            # 计算高度误差
            alt_errors = np.abs(merged['flight_level_pred'] - merged['flight_level_target'])

            # 计算速度误差
            speed_errors = np.abs(merged['ground_speed_pred'] - merged['ground_speed_target'])

            all_position_errors.extend(pos_errors)
            all_altitude_errors.extend(alt_errors)
            all_speed_errors.extend(speed_errors)

        # 计算总体指标
        if all_position_errors:
            metrics['position_rmse'] = np.sqrt(np.mean(np.array(all_position_errors)**2))
            metrics['position_mae'] = np.mean(np.array(all_position_errors))
            metrics['altitude_rmse'] = np.sqrt(np.mean(np.array(all_altitude_errors)**2))
            metrics['altitude_mae'] = np.mean(np.array(all_altitude_errors))
            metrics['speed_rmse'] = np.sqrt(np.mean(np.array(all_speed_errors)**2))
            metrics['speed_mae'] = np.mean(np.array(all_speed_errors))

        return metrics

    def save_predictions(self, predictions: pd.DataFrame, output_path: str):
        """
        保存预测结果

        Args:
            predictions: 预测结果DataFrame
            output_path: 输出文件路径
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        predictions.to_csv(output_path, index=False)
        self.logger.info(f"预测结果已保存到: {output_path}")

    def visualize_predictions(self, predictions: pd.DataFrame, sample_aircrafts: int = 3):
        """
        可视化预测结果（需要matplotlib）

        Args:
            predictions: 预测结果DataFrame
            sample_aircrafts: 采样飞机数量
        """
        try:
            import matplotlib.pyplot as plt

            aircraft_ids = predictions['aircraft_id'].unique()[:sample_aircrafts]

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Social-PatchTST预测结果', fontsize=16)

            # 轨迹图
            for i, aircraft_id in enumerate(aircraft_ids):
                aircraft_data = predictions[predictions['aircraft_id'] == aircraft_id]

                # 轨迹
                axes[0, 0].plot(aircraft_data['longitude'], aircraft_data['latitude'],
                               label=f'Aircraft {aircraft_id}', marker='o')
                axes[0, 0].set_xlabel('Longitude')
                axes[0, 0].set_ylabel('Latitude')
                axes[0, 0].set_title('轨迹预测')
                axes[0, 0].legend()
                axes[0, 0].grid(True)

                # 高度
                axes[0, 1].plot(aircraft_data['timestamp'], aircraft_data['flight_level'],
                               label=f'Aircraft {aircraft_id}')
                axes[0, 1].set_xlabel('Timestamp')
                axes[0, 1].set_ylabel('Flight Level')
                axes[0, 1].set_title('高度预测')
                axes[0, 1].legend()
                axes[0, 1].grid(True)

                # 速度
                axes[1, 0].plot(aircraft_data['timestamp'], aircraft_data['ground_speed'],
                               label=f'Aircraft {aircraft_id}')
                axes[1, 0].set_xlabel('Timestamp')
                axes[1, 0].set_ylabel('Ground Speed')
                axes[1, 0].set_title('速度预测')
                axes[1, 0].legend()
                axes[1, 0].grid(True)

                # 航向
                axes[1, 1].plot(aircraft_data['timestamp'], aircraft_data['track_angle'],
                               label=f'Aircraft {aircraft_id}')
                axes[1, 1].set_xlabel('Timestamp')
                axes[1, 1].set_ylabel('Track Angle')
                axes[1, 1].set_title('航向预测')
                axes[1, 1].legend()
                axes[1, 1].grid(True)

            plt.tight_layout()
            plt.savefig('prediction_visualization.png', dpi=300, bbox_inches='tight')
            plt.show()
            self.logger.info("预测可视化已保存为: prediction_visualization.png")

        except ImportError:
            self.logger.warning("matplotlib未安装，跳过可视化")
        except Exception as e:
            self.logger.error(f"可视化失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Social-PatchTST推理')
    parser.add_argument('--config', type=str,
                       default='../config/social_patchtst_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--model', type=str,
                       help='模型权重路径')
    parser.add_argument('--input', type=str, required=True,
                       help='输入数据文件路径')
    parser.add_argument('--output', type=str,
                       default='./predictions.csv',
                       help='输出文件路径')
    parser.add_argument('--max_aircrafts', type=int, default=50,
                       help='最大飞机数量')
    parser.add_argument('--visualize', action='store_true',
                       help='是否可视化预测结果')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"输入文件不存在: {args.input}")
        return

    # 创建预测器
    predictor = Predictor(args.config, args.model)

    # 读取输入数据
    print(f"读取输入数据: {args.input}")
    df = pd.read_csv(args.input)

    # 进行预测
    result = predictor.predict_from_dataframe(df, args.max_aircrafts)

    print(f"预测完成!")
    print(f"推理时间: {result['inference_time']:.3f}秒")
    print(f"飞机数量: {result['n_aircrafts']}")
    print(f"预测长度: {result['prediction_length']}个时间步")
    print(f"预测结果形状: {result['predictions'].shape}")

    # 保存预测结果
    predictor.save_predictions(result['predictions'], args.output)

    # 可视化
    if args.visualize:
        predictor.visualize_predictions(result['predictions'])

    print(f"预测完成，结果已保存到: {args.output}")


if __name__ == "__main__":
    main()