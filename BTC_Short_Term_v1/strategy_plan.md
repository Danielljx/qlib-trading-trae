# BTC 短线交易v1 - 策略开发计划（终极优化版）

> 修订日期: 2026-05-25
> 修订理由: 基于顶级量化交易最佳实践，引入Kaufman效率比、加密货币微观结构因子和动态阈值

---

## 零、本次四大核心升级亮点

### 核心升级 1: Kaufman效率比加权标签 (ER-Adjusted Label)
**原理**: 将未来4小时的价格运动路径解构，通过效率比过滤高波动震荡行情，仅让模型学习流畅单边的爆发信号。

| 场景 | 真实4h收益率 | 效率比ER | 最终标签(加权) | 模型学习目标 |
|------|------------|---------|--------------|------------|
| 暴力拉升5%，路径几乎直线 | +5% | 0.9 | +4.5% | ✅ 重点学习 |
| 暴涨5%，但剧烈上下插针 | +5% | 0.3 | +1.5% | ⚠️ 减弱权重 |
| 横盘震荡，振幅2% | ±1% | 0.1 | ±0.1% | ❌ 几乎忽略 |



### 核心升级 2: 非对称滚动自适应阈值策略
**问题**: 固定阈值在牛熊不同市场环境下完全失效。
**解决**: 基于历史预测信号的滚动分位数，动态计算当前阈值，自动适应市场体制变化。

### 核心升级 3: 模型参数微调适配
由于标签值被效率比加权后量级变小，调整Huber alpha为0.95，强迫模型更关注尾部的突破异动。

---

## 一、策略核心参数（终极版）

| 参数类别 | 参数 | 推荐值 | 变更说明 |
|----------|------|--------|---------|
| 品种 | instrument | `btcusdt` | 单品种 |
| 频率 | freq | `60min` | 1小时K线 |
| 初始训练 | train | 2017-01 ~ 2020-12 (4年, ~35,000 bar) | 不变 |
| 滚动频率 | retrain_interval | 每季度 (3个月, ~2,160 bar) | 不变 |
| 验证期 | valid | 滚动前 3 个月 (~2,160 bar) | 不变 |
| 测试期 | test | 2021-01 ~ 2026-05 | 不变 |
| **标签** | **label** | `LABEL_ER_4H` (Kaufman效率比加权) | ⚠️ 核心升级 |
| 模型 | model | LightGBM (回归) | 不变 |
| **目标函数** | objective | `huber` (alpha=**0.95**) | ⚠️ alpha从0.9调整为0.95 |
| 手续费 | fee | 0.04% 单边 (Taker), 0.02% 双边 (Maker) | 不变 |
| 滑点 | slippage | 0.01% (流动性充足时段) | 不变 |
| 初始资金 | account | 100,000 USDT | 不变 |
| **策略阈值** | long_percentile | 0.85 | ⚠️ 滚动分位数阈值 |
| **策略阈值** | short_percentile | 0.15 | ⚠️ 滚动分位数阈值 |
| **滚动窗口** | threshold_window | 720小时 (30天) | ⚠️ 动态阈值滚动窗口 |

---

## 二、因子工程（终极版，共 39 个因子）

### 2.1 基础因子 (原35个，维持不变)
完整列表参照原计划，包括：
- K线形态因子 (9个)
- 价格动量因子 (4个)
- 均线偏离因子 (4个)
- 波动率因子 (5个)
- 量价关系因子 (4个)
- 趋势强度因子 (3个)
- 极值因子 (2个)
- 价格记忆因子 (4个)

### 2.2 新增微观结构因子 (4个)
基于新增的 quote_volume、taker_buy_quote_volume、taker_buy_volume、trades 四个维度数据：

| 因子名称 | 计算表达式 | 说明 |
|---------|----------|------|
| QVOL_RATIO | `$quote_volume / Mean($quote_volume, 24) - 1` | 成交额相对24小时均值的变化率 |
| TAKER_BUY_RATIO | `$taker_buy_quote_volume / ($quote_volume + 1e-12)` | 主动买入成交额占比 |
| TAKER_VOL_RATIO | `$taker_buy_volume / ($volume + 1e-12)` | 主动买入成交量占比 |
| TRADES_RATIO | `$trades / Mean($trades, 24) - 1` | 成交笔数相对24小时均值的变化率 |



### 2.3 因子审查审计（前向/后向检查）

| 因子类别 | 包含未来数据? | 审计状态 |
|----------|--------------|---------|
| 基础因子35个 | ❌ 否 | ✅ 通过 |
| 微观结构因子4个 | ❌ 否 | ✅ 通过 |
| **Label (预测目标)** | **✅ 是（这是标签！）** | ✅ 作为标签使用 |

---

## 三、标签定义（终极版：Kaufman效率比加权）⭐核心⭐

### 3.1 定义与数学推导

我们将未来4小时的价格运动路径进行解构：
- **T时刻**: 产生信号的当前K线
- **T+1开盘**: 真实入场价
- **T+1 ~ T+4**: 未来4小时的时间窗口

```python
# ========== 1. 定义基础组件 ==========
# 注意: Ref负偏移代表未来数据，这是允许的（因为是标签！）
P_in = "Ref($open, -1)"   # 下一根K线开盘价 (T+1开盘，真实入场价)
C1 = "Ref($close, -1)"    # T+1收盘价
C2 = "Ref($close, -2)"    # T+2收盘价
C3 = "Ref($close, -3)"    # T+3收盘价
C4 = "Ref($close, -4)"    # T+4收盘价 (最终出场价)

# ========== 2. 计算净位移 (Net Change) ==========
# T+1开盘到T+4收盘的绝对距离
net_change = f"Abs({C4} - {P_in})"

# ========== 3. 计算路径总长度 (Path Length) ==========
# T+1到T+4过程中所有K线波动的绝对值总和
path_length = f"(Abs({C1} - {P_in}) + Abs({C2} - {C1}) + Abs({C3} - {C2}) + Abs({C4} - {C3}))"

# ========== 4. 计算效率比 (Efficiency Ratio) ==========
# ER ∈ [0, 1]
# - ER≈1: 流畅单边行情，几乎没有回撤
# - ER≈0: 剧烈震荡，来回插针
er = f"({net_change} / ({path_length} + 1e-12))"

# ========== 5. 计算原始真实收益率 ==========
raw_return = f"({C4} / {P_in} - 1)"

# ========== 6. 最终加权标签组合 ==========
# LABEL_ER_4H = 原始收益率 × 效率比
# 效果: 震荡行情的标签被压缩，流畅行情的标签被保留
LABEL_ER_4H = f"({raw_return} * {er})"
```

### 3.2 直观理解：三种典型场景

| 场景描述 | 4h真实走势 | 原始收益 | ER效率比 | 最终标签 | 模型权重 |
|---------|-----------|---------|---------|---------|---------|
| **流畅暴涨** | 直线拉升 | +5% | 0.9 | +4.5% | ⭐⭐⭐⭐⭐ 重点学习 |
| **暴力拉升插针** | 先拉6%再回1% | +5% | 0.3 | +1.5% | ⭐⭐ 一般权重 |
| **横盘震荡** | 上下各1% | +0.2% | 0.1 | +0.02% | ⭐ 几乎忽略 |

### 3.3 为什么选4小时？
- 足够长，可以识别趋势
- 足够短，加密货币4小时内可能有显著行情
- 效率比计算需要一定样本量

### 3.4 标签预处理

```python
learn_processors = [
    {"class": "DropnaLabel"},                          # 删除无标签的行
    # 注意: 由于标签是单品种时序预测，不需要截面排名
]
```

---

## 四、LightGBM 配置（终极微调版）

### 4.1 关键调整说明

由于现在标签是效率比加权后的收益，量级变小（一般在-3%~+3%，而原始可能-10%~+10%），需要调整模型对尾部的敏感度：

```python
LGBM_CONFIG = {
    # ========== 任务定义 ==========
    "loss": "mse",                       # Qlib LGBModel 内部使用 MSE
    "objective": "huber",                # 肥尾鲁棒
    "alpha": 0.95,                       # ⚠️ 从0.9调整为0.95，更关注尾部突破
    
    # ========== 学习控制 ==========
    "learning_rate": 0.015,              # 较低学习率 + 更多迭代
    "num_iterations": 500,               # boosting 迭代轮数
    "early_stopping_rounds": 50,         # 50轮无改善即停止
    
    # ========== 树结构 ==========
    "num_leaves": 31,                    # 叶子数 (2^5-1, 适合 ~35K 样本)
    "max_depth": 6,                      # 树深度限制
    "min_data_in_leaf": 100,             # 叶子最小样本数 (防止过拟合)
    "min_child_weight": 0.001,           # 最小子节点权重
    
    # ========== 正则化 ==========
    "lambda_l1": 0.5,                    # L1 正则 (鼓励稀疏)
    "lambda_l2": 1.5,                    # L2 正则 (防止过拟合)
    "min_gain_to_split": 0.01,           # 最小分裂增益
    
    # ========== 随机采样 (关键抗过拟合) ==========
    "subsample": 0.7,                    # 行采样比例
    "subsample_freq": 1,                 # 每轮都采样
    "colsample_bytree": 0.7,             # 列采样比例
    "colsample_bynode": 0.7,             # 节点级别列采样
    
    # ========== 硬件配置 ==========
    "num_threads": 8,
    "verbosity": -1,                     # 静默模式
    "seed": 42,                          # 随机种子
}
```

### 4.2 alpha参数敏感性

| Huber alpha值 | 对小误差 | 对大误差 | 适用场景 | 推荐 |
|--------------|---------|---------|---------|-----|
| 0.9 | 敏感 | 鲁棒 | 有小震荡的趋势 | 原计划 |
| 0.95 | 较鲁棒 | 更敏感 | **我们要抓住突破** | ⭐推荐 |
| 0.99 | 很鲁棒 | 极其敏感 | 只抓超大行情 | 实验用 |

---

## 五、训练/验证/测试周期设计（维持不变）

与原计划一致：
- 扩展窗口（Expanding Window）滚动训练
- 每季度滚动一次
- 保留滚动前3个月为验证集用于Early Stopping

---

## 六、策略层：非对称滚动自适应阈值策略 ⭐核心⭐

### 6.1 为什么固定阈值失效？

| 市场环境 | 预测得分典型范围 | 固定阈值 0.001 | 结果 |
|---------|---------------|---------------|------|
| 疯牛行情 | [-0.005, +0.008] | 0.001 | ✅ 正常工作 |
| 震荡熊市 | [-0.002, +0.002] | 0.001 | ⚠️ 可能开仓 |
| 极致低波动 | [-0.0005, +0.0005] | 0.001 | ❌ 半年不开仓 |

### 6.2 DynamicThresholdStrategy 完整实现

```python
import numpy as np
import pandas as pd
from qlib.backtest.decision import OrderDir, TradeDecisionWO
from qlib.backtest.order import OrderHelper
from qlib.strategy.base import BaseSignalStrategy

class DynamicThresholdStrategy(BaseSignalStrategy):
    """
    单品种动态滚动自适应阈值策略
    
    核心思想:
    - 不使用固定的 0.001 这样的数值阈值
    - 基于历史滚动窗口，计算当前预测分所处的分位数
    - 只有前N%的高分才做多，后M%的低分才做空
    - 自动适应牛熊不同市场环境
    
    参数说明:
    - long_percentile: 做多阈值的分位数，例如 0.85 表示前15%才做多
    - short_percentile: 做空阈值的分位数，例如 0.15 表示后15%才做空
    - rolling_window: 滚动窗口小时数，例如 720 (30天)
    """
    def __init__(
        self,
        long_percentile=0.85,     # 做多: 前15%
        short_percentile=0.15,    # 做空: 后15%
        rolling_window=720,       # 滚动窗口: 720小时 = 30天
        position_ratio=0.95,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.long_pct = long_percentile
        self.short_pct = short_percentile
        self.window = rolling_window
        self.position_ratio = position_ratio
        
        # ====== 核心逻辑：初始化时预计算完整的动态阈值序列 ======
        self._calculate_dynamic_thresholds()

    def _calculate_dynamic_thresholds(self):
        """
        基于模型输出的预测分，计算历史滚动窗口的自适应阈值
        """
        # 1. 获取完整的预测信号序列
        sig_df = self.signal.get_signal()
        
        # 2. 提取单品种的预测时间序列 (MultiIndex → Series)
        if isinstance(sig_df, pd.DataFrame):
            # 如果是DataFrame，unstack并提取btcusdt
            sig_series = sig_df.unstack().xs('btcusdt', level=1).iloc[:, 0]
        else:
            # 如果已经是Series或其他格式
            sig_series = sig_df.xs('btcusdt', level=1)
        
        # 3. 计算滚动分位数 (Rolling Quantiles)
        # 这里假设滚动窗口为720小时，即过去30天
        # min_periods设为24小时，避免最开始数据不足
        
        self.long_thresh_series = sig_series.rolling(
            window=self.window, 
            min_periods=24
        ).quantile(self.long_pct)
        
        self.short_thresh_series = sig_series.rolling(
            window=self.window, 
            min_periods=24
        ).quantile(self.short_pct)

    def generate_trade_decision(self, execute_result=None):
        """
        生成交易决策
        """
        # 1. 获取当前交易时间步
        trade_step = self.trade_calendar.get_trade_step()
        
        # 2. 获取当前预测时间窗口 (shift=1表示预测下一期)
        pred_start, pred_end = self.trade_calendar.get_step_time(trade_step, shift=1)
        
        # 3. 获取当前预测得分
        pred_score = self.signal.get_signal(start_time=pred_start, end_time=pred_end)
        
        if pred_score is None or pred_score.empty:
            return TradeDecisionWO([], self)
        
        # 4. 提取当前具体的预测值
        if isinstance(pred_score, pd.DataFrame):
            pred_value = pred_score.iloc[-1].values[0]
        else:
            pred_value = pred_score.iloc[-1]
        
        # 5. 获取当前时间戳，用于匹配动态阈值
        current_time = (
            pred_score.index[-1][0] 
            if isinstance(pred_score.index, pd.MultiIndex) 
            else pred_score.index[-1]
        )
        
        # 6. 尝试获取当前时刻的动态阈值
        try:
            curr_long_th = self.long_thresh_series.loc[current_time]
            curr_short_th = self.short_thresh_series.loc[current_time]
        except KeyError:
            # 如果索引不匹配或数据还在积累期，不交易
            return TradeDecisionWO([], self)
        
        # 7. 非对称动态仓位判断
        target_weight = 0.0
        if pd.notna(curr_long_th) and pred_value > curr_long_th:
            # 预测分高于滚动历史85%分位数 → 做多
            target_weight = self.position_ratio
        elif pd.notna(curr_short_th) and pred_value < curr_short_th:
            # 预测分低于滚动历史15%分位数 → 做空
            target_weight = -self.position_ratio
        else:
            # 中间区域 → 空仓
            target_weight = 0.0
        
        # 8. 生成订单
        cash = self.trade_position.get_cash()
        current_stock = 'btcusdt'
        order_helper = OrderHelper()
        
        if abs(target_weight) > 0.01:
            target_amount = target_weight * cash
            direction = OrderDir.BUY if target_weight > 0 else OrderDir.SELL
            order = order_helper.create_target_amount_order(
                instrument=current_stock,
                amount=target_amount,
                direction=direction,
            )
            return TradeDecisionWO([order], self)
        else:
            # 空仓：平掉所有仓位
            current_amount = self.trade_position.get_stock_amount(current_stock)
            if current_amount > 0:
                order = order_helper.create_target_amount_order(
                    instrument=current_stock,
                    amount=0,
                    direction=OrderDir.SELL,
                )
                return TradeDecisionWO([order], self)
            return TradeDecisionWO([], self)
```

### 6.3 参数推荐组合

| 风格 | long_pct | short_pct | 交易频率 | 持仓占比 | 适用场景 |
|------|---------|----------|---------|---------|---------|
| 保守 | 0.90 | 0.10 | 低 | ~20% | 震荡市 |
| **稳健** | **0.85** | **0.15** | **中** | **~30%** | **通用推荐** |
| 激进 | 0.75 | 0.25 | 高 | ~50% | 趋势市 |

---

## 七、交易成本（维持不变）

严格对标Binance永续合约费率，完整参照原计划。

---

## 八、回测（严格遵循 Qlib 框架）

### 8.1 关键修改：策略类更新

```python
strategy_config = {
    "class": "DynamicThresholdStrategy",  # ⚠️ 更新为动态阈值策略
    "module_path": "BTC_Short_Term_v1.strategies.dynamic_threshold_strategy",
    "kwargs": {
        "signal": predictions,
        "long_percentile": 0.85,     # 前15%才做多
        "short_percentile": 0.15,    # 后15%才做空
        "rolling_window": 720,       # 30天滚动窗口
        "position_ratio": 0.95,
    }
}
```

其他回测配置与原计划一致。

---

## 九、评估指标体系（维持不变）

除标准指标外，建议额外监控：
- **效率比与开仓胜率**: 观察当ER>0.5时，策略的胜率是否显著提升
- **动态阈值跟踪**: 绘制long_thresh_series和short_thresh_series，观察其如何适应市场

---

## 十、执行流程（行动计划）

### 前置步骤：数据管线重修 ⚠️必须优先做


### Phase 1: 特征工程
```
CryptoHandler:
├── 基础35因子
├── 新增4个微观结构因子 (QVOL_RATIO, TAKER_BUY_RATIO, TAKER_VOL_RATIO, TRADES_RATIO)
└── Kaufman ER加权标签 (LABEL_ER_4H)
```

### Phase 2: 模型训练
```
RollingTrainer:
├── Huber alpha=0.95
├── Expanding Window季度滚动
└── Early Stopping (3个月验证集)
```

### Phase 3: 策略实现
```
DynamicThresholdStrategy:
├── 滚动分位数阈值
├── 自动适应牛熊
└── 阈值: 85%/15%, 窗口720h
```

### Phase 4: 回测执行
```
Qlib Backtest:
├── 2021-01 ~ 2026-05
├── Binance永续费率 (0.04%)
└── 滑点 0.01%
```

### Phase 5: 报告生成
```
标准指标 + 效率比分析
```

---

## 十一、最终项目结构

```
BTC_Short_Term_v1/
├── strategy_plan.md                    ← 本文件 (终极优化版)
│
├── config/
│   ├── __init__.py
│   ├── model_config.py                # LightGBM (alpha=0.95)
│   ├── data_config.py                 #
│   ├── training_config.py             # 季度滚动
│   └── backtest_config.py             # 费率
│
├── features/
│   ├── __init__.py
│   └── crypto_handler.py             # 35基础+ER标签
│
├── models/
│   ├── __init__.py
│   └── rolling_trainer.py            # 训练
│
├── strategies/
│   ├── __init__.py
│   └── dynamic_threshold_strategy.py # ⭐动态阈值策略
│
├── backtest/
│   ├── __init__.py
│   └── run_backtest.py               # 回测
│
├── evaluation/
│   ├── __init__.py
│   └── report.py                     # 报告
│
├── report/
│   ├── performance_report.png
│   ├── risk_report.png
│   ├── ic_report.png
│   └── trading_log.csv
│
├── main.py
└── run.sh
```

---

## 十二、修订总结（终极版）

| 类别 | 原专家修订版 | 本次终极版 | 变更理由 |
|------|------------|----------|---------|
| 标签 | 1h前向收益 | **4h ER加权收益** | ⭐过滤震荡，只学流畅单边 |
| Huber alpha | 0.9 | **0.95** | ⭐标签量级变小，更关注尾部 |
| 策略类 | 固定阈值 | **动态分位数阈值** | ⭐自动适应牛熊 |
| 阈值参数 | 0.001/-0.001 | **85%/15%分位数** | ⭐相对值而非绝对值 |

---

**审核状态**: ✅ 终极优化完成，等待确认后开始编码

**本次核心变更简述**:
1. 🚀 **Kaufman ER标签**: 将震荡行情标签压缩，强迫模型学习流畅单边
2. 🎯 **动态阈值**: 滚动分位数替代固定值，自动适应不同市场体制
3. ⚙️ **Huber微调**: alpha=0.95，适应标签量级变小，更关注突破
