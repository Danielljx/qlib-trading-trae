# Qlib 加密货币数据环境

## 📊 数据概况

你提交的加密货币数据已成功配置！

### 数据详情

- **交易对**: BTCUSDT (比特币), ETHUSDT (以太坊)
- **时间周期**: 5种频率
  - 5min (5分钟)
  - 15min (15分钟)
  - 30min (30分钟)
  - 60min (1小时)
  - 240min (4小时)
- **数据范围**: 2017-12-31 至 2026-05-23
- **特征字段**: close, open, high, low, volume

## 🚀 快速开始

### 1. 初始化Qlib

```python
import qlib
from qlib.constant import REG_CN

provider_uri = "/workspace/qlib_data"
qlib.init(provider_uri=provider_uri, region=REG_CN)
```

### 2. 查询数据

```python
from qlib.data import D

# 查询BTCUSDT 1小时K线数据
features = D.features(
    instruments=['btcusdt'],
    fields=['$close', '$volume'],
    start_time='2024-01-01',
    end_time='2024-01-05',
    freq='60min'
)
print(features)
```

### 3. 使用表达式计算技术指标

```python
features = D.features(
    instruments=['btcusdt'],
    fields=[
        '$close',                        # 收盘价
        'Ref($close, 1)',               # 前1小时收盘价
        '$close/Ref($close, 1)-1',     # 收益率
        'Mean($close, 24)',             # 24小时均线
        'Mean($close, 168)',            # 168小时均线（约1周）
    ],
    start_time='2024-01-01',
    end_time='2024-01-05',
    freq='60min'
)
```

## 📁 文件说明

### 数据目录结构

```
qlib_data/
├── calendars/           # 交易日历
│   ├── 5min.txt
│   ├── 15min.txt
│   ├── 30min.txt
│   ├── 60min.txt
│   └── 240min.txt
├── instruments/         # 交易对列表
│   ├── btcusdt.txt
│   ├── ethusdt.txt
│   └── all.txt
└── features/            # K线数据（bin格式）
    ├── btcusdt/
    │   ├── close.5min.bin
    │   ├── close.15min.bin
    │   ├── close.30min.bin
    │   ├── close.60min.bin
    │   ├── close.240min.bin
    │   └── ...
    └── ethusdt/
        └── ...
```

## 🔧 可用脚本

- **crypto_init.py** - Qlib初始化示例
- **crypto_query_example.py** - 数据查询示例
- **test_crypto_data.py** - 数据可用性测试

## 📚 学习资源

- **qlib系统框架.md** - Qlib完整技术文档
- [Qlib官方文档](https://github.com/microsoft/qlib)

## ⚠️ 注意事项

1. 频率格式：Qlib使用"min"而非"m"
   - ✅ 正确: `freq='60min'`
   - ❌ 错误: `freq='1h'`

2. 交易对名称：使用小写
   - ✅ 正确: `instruments=['btcusdt']`
   - ❌ 错误: `instruments=['BTCUSDT']`

## 🎯 下一步

1. 使用Alpha158/Alpha360构建特征工程
2. 训练LightGBM/LSTM模型预测价格
3. 实现加密货币交易策略
4. 使用回测模块测试策略效果

---

**状态**: ✅ 环境配置完成
**最后更新**: 2026-05-25
