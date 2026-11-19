"""
日志记录模块
"""
import logging
import os
from datetime import datetime


def setup_logger(log_dir='logs', log_name='experiment.log'):
    """
    设置日志记录器
    
    Args:
        log_dir: 日志目录
        log_name: 日志文件名
    
    Returns:
        logger: 配置好的日志记录器
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志文件路径
    log_path = os.path.join(log_dir, log_name)
    
    # 配置日志格式
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 配置日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),  # 文件处理器
            logging.StreamHandler()  # 控制台处理器
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志记录器已初始化，日志文件: {log_path}")
    
    return logger

