#!/usr/bin/env python3
"""
重组Qlib加密货币数据目录结构

将数据重组为Qlib支持的频率格式
"""

import os
import shutil

data_path = "/workspace/qlib_data"

# 频率映射：目录名 -> Qlib频率格式
freq_mapping = {
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "60min",
    "4h": "240min"
}

print("重组Qlib加密货币数据目录结构...")
print("="*60)

for symbol in ["BTCUSDT", "ETHUSDT"]:
    print(f"\n处理 {symbol}...")
    
    for dir_freq, qlib_freq in freq_mapping.items():
        old_dir = os.path.join(data_path, symbol, dir_freq)
        new_dir = os.path.join(data_path, symbol, qlib_freq)
        
        if os.path.exists(old_dir):
            # 重命名目录
            if not os.path.exists(new_dir):
                os.rename(old_dir, new_dir)
                print(f"  ✓ {symbol}/{dir_freq} -> {symbol}/{qlib_freq}")
            else:
                print(f"  ⚠ {symbol}/{qlib_freq} 已存在，跳过")
            
            # 重命名特征文件
            feature_dir = os.path.join(new_dir, "features", symbol.lower())
            if os.path.exists(feature_dir):
                for file in os.listdir(feature_dir):
                    if dir_freq in file:
                        old_file = os.path.join(feature_dir, file)
                        new_file = os.path.join(feature_dir, file.replace(f".{dir_freq}", f".{qlib_freq}"))
                        if not os.path.exists(new_file):
                            os.rename(old_file, new_file)
                            print(f"    ✓ 重命名 {file} -> {file.replace(f'.{dir_freq}', f'.{qlib_freq}')}")
            
            # 重命名日历文件
            calendar_dir = os.path.join(new_dir, "calendars")
            if os.path.exists(calendar_dir):
                for file in os.listdir(calendar_dir):
                    if file == f"{dir_freq}.txt":
                        old_file = os.path.join(calendar_dir, file)
                        new_file = os.path.join(calendar_dir, f"{qlib_freq}.txt")
                        if not os.path.exists(new_file):
                            os.rename(old_file, new_file)
                            print(f"    ✓ 重命名日历 {file} -> {qlib_freq}.txt")

print("\n" + "="*60)
print("✓ 目录重组完成！")
print("\n新的目录结构:")
for symbol in ["BTCUSDT", "ETHUSDT"]:
    print(f"\n{symbol}:")
    symbol_dir = os.path.join(data_path, symbol)
    if os.path.exists(symbol_dir):
        for freq in sorted(os.listdir(symbol_dir)):
            freq_dir = os.path.join(symbol_dir, freq)
            if os.path.isdir(freq_dir):
                cal_count = len(os.listdir(os.path.join(freq_dir, "calendars"))) if os.path.exists(os.path.join(freq_dir, "calendars")) else 0
                print(f"  - {freq}")
