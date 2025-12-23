import logging
import os
from datetime import datetime


def setup_logger(log_dir, experiment_name):
    """
    设置日志记录器
    :param log_dir: 日志保存目录
    :param experiment_name: 实验名称
    :return: 配置好的logger
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)

    # 日志文件名
    log_file_name = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file_path = os.path.join(log_dir, log_file_name)

    # 创建logger
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.DEBUG)

    # 避免重复添加处理器
    if logger.handlers:
        return logger

    # 创建文件处理器
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 设置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"日志记录已初始化，日志文件保存至: {log_file_path}")
    return logger


def log_experiment_config(logger, config):
    """
    记录实验配置
    :param logger: 日志记录器
    :param config: 实验配置字典
    """
    logger.info("=" * 50)
    logger.info("实验配置信息")
    logger.info("=" * 50)
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info(f"{key}:")
            for sub_key, sub_value in value.items():
                logger.info(f"  {sub_key}: {sub_value}")
        else:
            logger.info(f"{key}: {value}")
    logger.info("=" * 50)


def log_metrics(logger, metrics, epoch=None, phase="Train"):
    """
    记录模型指标
    :param logger: 日志记录器
    :param metrics: 指标字典
    :param epoch: 轮次
    :param phase: 阶段（Train/Val/Test）
    """
    if epoch is not None:
        logger.info(f"Epoch {epoch} - {phase} Metrics:")
    else:
        logger.info(f"{phase} Metrics:")

    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, float):
            logger.info(f"  {metric_name}: {metric_value:.4f}")
        else:
            logger.info(f"  {metric_name}: {metric_value}")