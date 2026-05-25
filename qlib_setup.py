#!/usr/bin/env python3
"""
Qlib 配置脚本

使用方法：
1. 如果你已经有Qlib格式的数据，直接修改 provider_uri 路径
2. 如果你有CSV数据，使用此脚本转换为Qlib格式
3. 运行：python qlib_setup.py
"""

import os
import sys
import qlib
from qlib.constant import REG_CN

# ============== 配置区域 ==============
# 设置你的Qlib数据目录路径
QLIB_DATA_PATH = os.path.expanduser("~/.qlib/qlib_data/cn_data")

# 如果你使用CSV数据，设置CSV路径
CSV_DATA_PATH = os.path.expanduser("~/your_csv_data")

# 选择初始化模式：'existing' 或 'csv'
INIT_MODE = 'existing'  # 修改这个值来选择模式
# ============== 配置区域 ==============

def init_with_existing_data():
    """初始化已存在的Qlib数据"""
    if not os.path.exists(QLIB_DATA_PATH):
        print(f"错误：数据目录不存在 {QLIB_DATA_PATH}")
        print("请下载Qlib数据或转换你的CSV数据")
        return False
    
    qlib.init(provider_uri=QLIB_DATA_PATH, region=REG_CN)
    print(f"✓ Qlib已成功初始化")
    print(f"  数据路径：{QLIB_DATA_PATH}")
    return True

def convert_csv_to_qlib():
    """将CSV数据转换为Qlib格式"""
    print(f"开始转换CSV数据...")
    print(f"  CSV路径：{CSV_DATA_PATH}")
    print(f"  输出路径：{QLIB_DATA_PATH}")
    
    # 导入转换工具
    try:
        from qlib.data import D
        
        # 使用dump_bin脚本转换
        cmd = f"""
        python -m qlib.data.dump_bin \
            --csv_path {CSV_DATA_PATH} \
            --qlib_dir {QLIB_DATA_PATH} \
            --include_fields open,close,high,low,volume,factor
        """
        
        print("\n运行转换命令...")
        print(f"命令: {cmd}")
        print("\n请在终端中运行上述命令完成数据转换")
        
        return True
        
    except Exception as e:
        print(f"转换过程中出错: {e}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("Qlib 配置脚本")
    print("=" * 60)
    
    if INIT_MODE == 'existing':
        success = init_with_existing_data()
    elif INIT_MODE == 'csv':
        success = convert_csv_to_qlib()
    else:
        print(f"错误：未知的初始化模式 '{INIT_MODE}'")
        print("请设置 INIT_MODE = 'existing' 或 'csv'")
        success = False
    
    if success:
        print("\n" + "=" * 60)
        print("✓ Qlib环境配置完成！")
        print("=" * 60)
        print("\n下一步：")
        print("1. 查看示例代码：qlib_init_example.py")
        print("2. 开始使用Qlib进行量化分析")
        print("3. 参考文档：https://github.com/microsoft/qlib")
    else:
        print("\n配置失败，请检查错误信息")
        sys.exit(1)

if __name__ == "__main__":
    main()
