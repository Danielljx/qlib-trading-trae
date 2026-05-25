"""
Qlib 快速查询示例

此脚本展示了Qlib的基本数据查询功能
运行前请确保已正确初始化Qlib（运行 qlib_init_example.py）
"""

import qlib
from qlib.constant import REG_CN
from qlib.data import D

# 初始化Qlib（请确保数据已准备好）
provider_uri = "~/.qlib/qlib_data/cn_data"
qlib.init(provider_uri=provider_uri, region=REG_CN)

print("=" * 60)
print("Qlib 数据查询示例")
print("=" * 60)

# 1. 查询交易日历
print("\n1. 查询交易日历（最近10天）:")
calendar = D.calendar(start_time='2024-01-01', end_time='2024-12-31')
print(f"   2024年共有 {len(calendar)} 个交易日")
print(f"   最近5个交易日: {calendar[-5:]}")

# 2. 查询股票列表
print("\n2. 查询沪深300成分股:")
instruments = D.instruments('csi300')
print(f"   沪深300共有 {len(instruments)} 只股票")
print(f"   前5只股票: {instruments[:5]}")

# 3. 查询股票数据
print("\n3. 查询贵州茅台(SH600519)的历史数据:")
features = D.features(
    instruments=['SH600519'],
    fields=['$close', '$volume', '$turnover'],
    start_time='2024-01-01',
    end_time='2024-01-10'
)
print(features.head())

# 4. 使用表达式引擎
print("\n4. 使用表达式计算技术指标:")
features = D.features(
    instruments=['SH600519'],
    fields=[
        '$close',                                    # 收盘价
        'Ref($close, 1)',                           # 前日收盘价
        'Mean($close, 5)',                          # 5日均线
        '$close/Ref($close, 1)-1',                  # 日收益率
    ],
    start_time='2024-01-01',
    end_time='2024-01-10'
)
print(features)

# 5. 批量查询多只股票
print("\n5. 批量查询多只股票:")
features = D.features(
    instruments=['SH600519', 'SH600036', 'SH601318'],
    fields=['$close', '$volume'],
    start_time='2024-01-01',
    end_time='2024-01-05'
)
print(f"   查询到 {len(features)} 条记录")
print(features.head(10))

print("\n" + "=" * 60)
print("✓ 查询完成！")
print("=" * 60)
print("\n下一步：")
print("- 使用 Alpha158 或 Alpha360 构建特征")
print("- 训练机器学习模型")
print("- 进行策略回测")
print("- 参考 /data/user/skills/qlib-skill 获取完整文档")
