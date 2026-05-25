#!/usr/bin/env python3
"""
创建Qlib加密货币数据的配置文件

Qlib需要一个特殊的配置来识别不同的时间周期频率
"""

import json
import os

# Qlib数据根目录
data_path = "/workspace/qlib_data"

# 创建 provider_uri.json 配置文件
config = {
    "calendar_provider": {
        "class": "LocalCalendarProvider",
        "module_path": "qlib.data.data",
        "kwargs": {
            "remote_url": None
        }
    },
    "instrument_provider": {
        "class": "LocalInstrumentProvider",
        "module_path": "qlib.data.data",
        "kwargs": {
            "remote_url": None
        }
    },
    "feature_provider": {
        "class": "LocalFeatureProvider",
        "module_path": "qlib.data.data",
        "kwargs": {
            "remote_url": None
        }
    },
    "dataset_provider": {
        "class": "LocalDatasetProvider",
        "module_path": "qlib.data.data",
        "kwargs": {
            "remote_url": None
        }
    },
    "freq_1h": {
        "calendar": ["1h.txt"],
        "instrument": ["all.txt"],
        "feature": ["close.1h.bin", "open.1h.bin", "high.1h.bin", "low.1h.bin", "volume.1h.bin"]
    },
    "freq_4h": {
        "calendar": ["4h.txt"],
        "instrument": ["all.txt"],
        "feature": ["close.4h.bin", "open.4h.bin", "high.4h.bin", "low.4h.bin", "volume.4h.bin"]
    },
    "freq_5min": {
        "calendar": ["5min.txt"],
        "instrument": ["all.txt"],
        "feature": ["close.5min.bin", "open.5min.bin", "high.5min.bin", "low.5min.bin", "volume.5min.bin"]
    },
    "freq_15min": {
        "calendar": ["15min.txt"],
        "instrument": ["all.txt"],
        "feature": ["close.15min.bin", "open.15min.bin", "high.15min.bin", "low.15min.bin", "volume.15min.bin"]
    },
    "freq_30min": {
        "calendar": ["30min.txt"],
        "instrument": ["all.txt"],
        "feature": ["close.30min.bin", "open.30min.bin", "high.30min.bin", "low.30min.bin", "volume.30min.bin"]
    }
}

# 写入配置文件
config_path = os.path.join(data_path, "provider_uri.json")
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f"✓ 配置文件已创建: {config_path}")

# 创建各个频率的链接
print("\n创建频率目录链接...")
for symbol in ["BTCUSDT", "ETHUSDT"]:
    for freq in ["1h", "4h", "5min", "15min", "30min"]:
        src_cal = os.path.join(data_path, symbol, freq, "calendars", f"{freq}.txt")
        dst_cal = os.path.join(data_path, symbol, freq, f"{freq}.txt")
        
        if os.path.exists(src_cal) and not os.path.exists(dst_cal):
            # 复制日历文件到根目录
            with open(src_cal, 'r') as sf:
                with open(dst_cal, 'w') as df:
                    df.write(sf.read())
            print(f"  ✓ 复制 {symbol}/{freq}/calendars/{freq}.txt -> {symbol}/{freq}/{freq}.txt")

print("\n配置完成！")
