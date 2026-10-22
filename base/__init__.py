# base/__init__.py
# 统一对外暴露配置类和日志对象，供其他模块 from base import Config, logger 使用
from .config import Config
from .logger import logger

__all__ = ["Config", "logger"]
