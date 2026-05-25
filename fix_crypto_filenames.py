#!/usr/bin/env python3
"""
修复Qlib加密货币数据文件名

之前的脚本错误地将 5min 重命名为 5minin，需要修复
"""

import os

data_path = "/workspace/qlib_data"

print("修复文件名...")
print("="*60)

# 修复文件名
for symbol in ["BTCUSDT", "ETHUSDT"]:
    print(f"\n处理 {symbol}...")
    
    for freq in ["5min", "15min", "30min"]:
        feature_dir = os.path.join(data_path, symbol, freq, "features", symbol.lower())
        
        if os.path.exists(feature_dir):
            for file in os.listdir(feature_dir):
                if "minin" in file:
                    old_file = os.path.join(feature_dir, file)
                    new_file = os.path.join(feature_dir, file.replace("minin", "min"))
                    os.rename(old_file, new_file)
                    print(f"  ✓ 修复 {file} -> {file.replace('minin', 'min')}")

print("\n" + "="*60)
print("✓ 文件名修复完成！")
print("\n验证特征文件:")
for symbol in ["BTCUSDT", "ETHUSDT"]:
    print(f"\n{symbol}:")
    for freq in ["5min", "15min", "30min", "60min", "240min"]:
        feature_dir = os.path.join(data_path, symbol, freq, "features", symbol.lower())
        if os.path.exists(feature_dir):
            files = sorted(os.listdir(feature_dir))
            print(f"  {freq}: {', '.join(files)}")
