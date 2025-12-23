# fix_encoding.py
import os


def fix_temporal_file(file_path):
    """修复编码问题"""
    # 尝试用GBK编码读取（Windows中文系统常用）
    try:
        with open(file_path, 'r', encoding='gbk') as f:
            content = f.read()
        print("成功用GBK编码读取文件")
    except UnicodeDecodeError:
        # 尝试其他编码
        encodings = ['gb2312', 'gb18030', 'cp936', 'latin-1']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                print(f"成功用 {encoding} 编码读取文件")
                break
            except UnicodeDecodeError:
                continue

    # 添加UTF-8编码声明并保存
    lines = content.split('\n')
    if not lines[0].startswith('# -*- coding:'):
        lines.insert(0, '# -*- coding: utf-8 -*-')

    # 保存为UTF-8
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"文件已修复并保存为UTF-8编码")


# 修复文件
fix_temporal_file('D:/Another/UnderWater-Project/n-physound/preprocessing/temporal.py')