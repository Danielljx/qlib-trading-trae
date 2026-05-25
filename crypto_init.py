"""
Qlib 加密货币数据配置脚本

使用你提交的加密货币数据进行初始化
"""

import os
import qlib
from qlib.constant import REG_CN

# Qlib数据路径
QLIB_DATA_PATH = "/workspace/qlib_data"

# 初始化Qlib
qlib.init(provider_uri=QLIB_DATA_PATH, region=REG_CN)

print("=" * 70)
print("✓ Qlib加密货币数据环境配置成功！")
print("=" * 70)
print(f"\n数据路径: {QLIB_DATA_PATH}")
print("\n可用的交易对:")
print("  - BTCUSDT (比特币)")
print("  - ETHUSDT (以太坊)")
print("\n可用的时间周期:")
print("  - 5m  (5分钟)")
print("  - 15m (15分钟)")
print("  - 30m (30分钟)")
print("  - 1h  (1小时)")
print("  - 4h  (4小时)")
print("\n下一步:")
print("  1. 运行: python crypto_query_example.py 查看数据")
print("  2. 开始训练模型和回测策略")
print("  3. 参考: qlib系统框架.md 了解Qlib完整功能")
print("=" * 70)
