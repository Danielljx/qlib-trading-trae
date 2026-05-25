#!/usr/bin/env python3
"""
重新组织Qlib加密货币数据目录结构

将数据重组为Qlib期望的标准结构
"""

import os
import shutil

source_path = "/workspace/qlib_data"
target_path = "/workspace/qlib_data_new"

print("重新组织Qlib加密货币数据目录结构...")
print("="*70)

# 创建新的目录结构
os.makedirs(target_path, exist_ok=True)
os.makedirs(os.path.join(target_path, "calendars"), exist_ok=True)
os.makedirs(os.path.join(target_path, "instruments"), exist_ok=True)
os.makedirs(os.path.join(target_path, "features"), exist_ok=True)

# 频率列表
freqs = ["5min", "15min", "30min", "60min", "240min"]

# 处理每个交易对和频率
for symbol in ["BTCUSDT", "ETHUSDT"]:
    symbol_lower = symbol.lower()
    
    for freq in freqs:
        source_freq_dir = os.path.join(source_path, symbol, freq)
        
        if not os.path.exists(source_freq_dir):
            continue
        
        # 复制日历文件
        source_cal = os.path.join(source_freq_dir, "calendars", f"{freq}.txt")
        target_cal = os.path.join(target_path, "calendars", f"{freq}.txt")
        
        if os.path.exists(source_cal) and not os.path.exists(target_cal):
            shutil.copy2(source_cal, target_cal)
            print(f"✓ 复制日历: {freq}.txt")
        
        # 复制特征文件
        source_feature_dir = os.path.join(source_freq_dir, "features", symbol_lower)
        target_feature_dir = os.path.join(target_path, "features", symbol_lower)
        
        if os.path.exists(source_feature_dir):
            os.makedirs(target_feature_dir, exist_ok=True)
            
            for file in os.listdir(source_feature_dir):
                if freq in file:
                    target_file = os.path.join(target_feature_dir, file)
                    if not os.path.exists(target_file):
                        shutil.copy2(os.path.join(source_feature_dir, file), target_file)
                        print(f"✓ 复制特征: {symbol_lower}/{file}")

# 创建 instruments 文件
for symbol in ["BTCUSDT", "ETHUSDT"]:
    symbol_lower = symbol.lower()
    instrument_file = os.path.join(target_path, "instruments", f"{symbol_lower}.txt")
    
    with open(instrument_file, 'w') as f:
        f.write(f"{symbol}\n")
    print(f"✓ 创建交易对文件: instruments/{symbol_lower}.txt")

# 创建 all.txt
all_file = os.path.join(target_path, "instruments", "all.txt")
with open(all_file, 'w') as f:
    f.write("btcusdt\nethusdt\n")
print(f"✓ 创建 all.txt")

print("\n" + "="*70)
print(f"✓ 数据重组完成！")
print(f"\n新目录结构: {target_path}")
print("\n验证目录结构:")
os.system(f"tree {target_path} -L 3 -I '__pycache__' | head -50")

print("\n下一步:")
print(f"1. 将旧数据备份: mv {source_path} {source_path}_old")
print(f"2. 使用新数据: mv {target_path} {source_path}")
print(f"3. 测试: python3 test_crypto_data.py")
