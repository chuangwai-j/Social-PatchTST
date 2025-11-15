"""
     Social-PatchTST 场景创建模块
     提供从原始数据到训练数据的完整处理流程：
     1. 数据处理 (data_processor.py)
     2. 索引生成 (generate_index.sh)
     3. 分层采样 (stratified_sampler.py)
"""
from .data_processor import process_adsb_data, Config

__all__ = [
    'process_adsb_data',
    'Config'
]