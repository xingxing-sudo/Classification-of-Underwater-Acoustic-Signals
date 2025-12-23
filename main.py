#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N-PHYSound 项目主入口文件
"""

import argparse
from experiments.run_comparisons import run_all_experiments
from experiments.visualize import generate_visualizations


def main():
    parser = argparse.ArgumentParser(description='N-PHYSound项目主程序')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['train', 'evaluate', 'visualize', 'all'],
                        help='运行模式')
    args = parser.parse_args()

    if args.mode in ['train', 'all']:
        print("开始运行所有实验...")
        run_all_experiments()

    if args.mode in ['evaluate', 'all']:
        print("评估模型性能...")
        from experiments.evaluate import evaluate_all_models
        from utils.config import load_config
        config = load_config('configs/main_config.yaml')
        evaluate_all_models(config)

    if args.mode in ['visualize', 'all']:
        print("生成可视化结果...")
        generate_visualizations()


if __name__ == "__main__":
    main()