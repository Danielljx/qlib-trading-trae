"""
Qlib 加密货币数据查询示例

展示如何使用Qlib查询加密货币数据
运行前请确保已正确初始化Qlib
"""

import qlib
from qlib.constant import REG_CN
from qlib.data import D

# 初始化Qlib（使用加密货币数据）
provider_uri = "/workspace/qlib_data"
qlib.init(provider_uri=provider_uri, region=REG_CN)

print("=" * 70)
print("Qlib 加密货币数据查询示例")
print("=" * 70)

# 1. 查询1小时周期的交易日历
print("\n1. 查询BTCUSDT 1小时周期日历（最近10个时间点）:")
try:
    calendar = D.calendar(start_time='2024-01-01', end_time='2024-01-10', freq='1h')
    print(f"   共 {len(calendar)} 个时间点")
    print(f"   最近5个: {calendar[-5:]}")
except Exception as e:
    print(f"   查询失败: {e}")
    print("   尝试其他频率格式...")

# 2. 查询可用交易对
print("\n2. 查询BTCUSDT交易对信息:")
instruments = D.instruments('btcusdt')
print(f"   交易对: {instruments}")

# 3. 查询1小时K线数据
print("\n3. 查询BTCUSDT 1小时K线数据:")
try:
    features = D.features(
        instruments=['btcusdt'],
        fields=['$close', '$open', '$high', '$low', '$volume'],
        start_time='2024-01-01',
        end_time='2024-01-05',
        freq='1h'
    )
    print(f"   查询到 {len(features)} 条记录")
    print("\n前5条数据:")
    print(features.head())
except Exception as e:
    print(f"   查询失败: {e}")

# 4. 使用表达式计算收益率
print("\n4. 计算收益率和技术指标:")
try:
    features = D.features(
        instruments=['btcusdt'],
        fields=[
            '$close',                      # 收盘价
            'Ref($close, 1)',              # 前1小时收盘价
            '$close/Ref($close, 1)-1',    # 收益率
            'Mean($close, 24)',            # 24小时均线（约1天）
            'Mean($close, 168)',           # 168小时均线（约1周）
        ],
        start_time='2024-01-01',
        end_time='2024-01-05',
        freq='1h'
    )
    print(features.head(10))
except Exception as e:
    print(f"   查询失败: {e}")

# 5. 查询ETHUSDT数据
print("\n5. 查询ETHUSDT 4小时数据:")
try:
    features = D.features(
        instruments=['ethusdt'],
        fields=['$close', '$volume'],
        start_time='2024-01-01',
        end_time='2024-01-03',
        freq='4h'
    )
    print(f"   查询到 {len(features)} 条记录")
    print(features)
except Exception as e:
    print(f"   查询失败: {e}")

# 6. 批量查询多个交易对
print("\n6. 批量查询BTC和ETH的5分钟数据:")
try:
    features = D.features(
        instruments=['btcusdt', 'ethusdt'],
        fields=['$close', '$volume'],
        start_time='2024-01-01',
        end_time='2024-01-01 12:00:00',
        freq='5min'
    )
    print(f"   共查询到 {len(features)} 条记录")
    if len(features) > 0:
        print(f"   BTCUSDT记录数: {len(features[features.index.get_level_values('instrument') == 'btcusdt'])}")
        print(f"   ETHUSDT记录数: {len(features[features.index.get_level_values('instrument') == 'ethusdt'])}")
except Exception as e:
    print(f"   查询失败: {e}")

print("\n" + "=" * 70)
print("✓ 查询完成！")
print("=" * 70)
print("\n下一步:")
print("- 使用Alpha158构建特征工程")
print("- 训练LightGBM/LSTM等模型预测价格")
print("- 使用回测模块测试策略")
print("- 参考qlib系统框架.md学习完整功能")
