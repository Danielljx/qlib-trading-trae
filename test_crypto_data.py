"""
Qlib 加密货币数据快速测试
"""

import qlib
from qlib.constant import REG_CN
from qlib.data import D

# 初始化Qlib
print("初始化Qlib...")
provider_uri = "/workspace/qlib_data"
qlib.init(provider_uri=provider_uri, region=REG_CN)

print("✓ Qlib初始化成功！")
print("\n测试数据查询...")
print("="*70)

# 测试1: 查询BTCUSDT 60分钟数据
try:
    print("\n1. 查询BTCUSDT 60分钟数据...")
    features = D.features(
        instruments=['btcusdt'],
        fields=['$close', '$volume'],
        start_time='2024-01-01',
        end_time='2024-01-02',
        freq='60min'
    )
    print(f"   ✓ 成功！查询到 {len(features)} 条记录")
    print(features.head())
except Exception as e:
    print(f"   ✗ 失败: {e}")
    import traceback
    traceback.print_exc()

# 测试2: 查询ETHUSDT 240分钟数据
try:
    print("\n2. 查询ETHUSDT 240分钟(4小时)数据...")
    features = D.features(
        instruments=['ethusdt'],
        fields=['$close', '$volume'],
        start_time='2024-01-01',
        end_time='2024-01-03',
        freq='240min'
    )
    print(f"   ✓ 成功！查询到 {len(features)} 条记录")
    print(features.head())
except Exception as e:
    print(f"   ✗ 失败: {e}")
    import traceback
    traceback.print_exc()

# 测试3: 查询5分钟数据
try:
    print("\n3. 查询BTCUSDT 5分钟数据...")
    features = D.features(
        instruments=['btcusdt'],
        fields=['$close'],
        start_time='2024-01-01',
        end_time='2024-01-01 01:00:00',
        freq='5min'
    )
    print(f"   ✓ 成功！查询到 {len(features)} 条记录")
    print(features.head())
except Exception as e:
    print(f"   ✗ 失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("✓ 数据查询测试完成！")
print("="*70)
