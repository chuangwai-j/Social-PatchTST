"""
配置管理工具
"""

import yaml
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Config:
    """配置类"""
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        return config

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值，支持嵌套键，如 'data.history_length'"""
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """设置配置值"""
        keys = key.split('.')
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def save(self, save_path: Optional[str] = None) -> None:
        """保存配置"""
        path = save_path or self.config_path
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)

    @property
    def data_config(self) -> Dict[str, Any]:
        return self.config['data']

    @property
    def patchtst_config(self) -> Dict[str, Any]:
        return self.config['patchtst']

    @property
    def social_config(self) -> Dict[str, Any]:
        return self.config['social_transformer']

    @property
    def decoder_config(self) -> Dict[str, Any]:
        return self.config['decoder']

    @property
    def training_config(self) -> Dict[str, Any]:
        return self.config['training']

    @property
    def device_config(self) -> Dict[str, Any]:
        return self.config['device']

    def __repr__(self) -> str:
        return f"Config(config_path='{self.config_path}')"


def load_config(config_path: str) -> Config:
    """加载配置文件的便捷函数"""
    return Config(config_path)