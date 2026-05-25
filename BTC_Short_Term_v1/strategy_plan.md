# BTC 短线交易v1 - 策略开发计划（专家修订版）

> 修订日期: 2026-05-25
> 修订理由: 基于专业量化交易最佳实践，全面优化原计划中的关键缺陷

---

## 零、原计划关键问题诊断与修正

### 问题 1: 标签与因子完全一致 —— 致命的数据泄漏（P0）

**原计划 Bug（致命）**：
```python
# 原始计划中：
Label1h    = ($close / Ref($close, 1)) - 1   # 标签：过去1小时收益率
Return1    = ($close / Ref($close, 1)) - 1   # 因子：过去1小时收益率
# ↑↑↑ 两者完全一致！模型会学到 y = x，这是纯数据泄漏
```

**诊断依据**：查看 Qlib 官方 Alpha158 源码，标签定义如下：
```python
# qlib/contrib/data/handler.py, Alpha158.get_label_config()
return ["Ref($close, -2) / Ref($close, -1) - 1"], ["LABEL0"]
#        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#        Ref($close,-2) = 1根K线之后的收盘价 (未来价格)
#        Ref($close,-1) = 当前收盘价
#        标签 = (未来收盘 - 当前收盘) / 当前收盘 = T+1 的前向收益率
```

**修正**：
```python
# 标签必须是前向（未来）收益率，使用 Ref 的负偏移
Label_1h  = Ref($close, -2) / Ref($close, -1) - 1  # 未来1小时收益率
Label_4h  = Ref($close, -5) / Ref($close, -1) - 1  # 未来4小时收益率（可选）
Label_24h = Ref($close, -25) / Ref($close, -1) - 1 # 未来24小时收益率（可选）
```

### 问题 2: 因子列表未与标签隔离审计（P0）

原计划未做「前向/后向」检查。正确做法：所有因子只能使用 `Ref($x, d)` 其中 `d >= -1`（不能窥视未来），并且需要逐一审计确保与标签时间错位。

### 问题 3: LightGBM 参数不适合加密货币交易噪声（P0）

原计划参数过于通用。加密货币具有高波动、高噪声特性，需要：
- 更低的 learning_rate + 更多的 estimators
- 更强的正则化（L1+L2）
- 目标函数改用 `huber` 或 `mae` 替代 `mse`（肥尾分布）

### 问题 4: 滚动窗口频率过密（P1）

原计划每月滚动对单品种 1h 级别过于频繁，BTC 市场体制变化周期通常在 3-6 个月，每月重训练容易过拟合噪声。

### 问题 5: 交易成本未严格对标真实交易所（P1）

需要严格对标 Binance 现货/永续合约的真实费率。

### 问题 6: 多品种策略不适配单品种（P1）

TopkDropoutStrategy 设计理念是「排名选择」，单品种下 topk=1 退化为「仅持有」策略，需要改为阈值驱动的交易信号机制。

### 问题 7: 未做因子共线性筛查（P2）

60+ 因子中存在高度共线性（如 MA5 与 MA10），增加过拟合风险。

---

## 一、策略核心参数（修订版）

| 参数类别 | 参数 | 推荐值 | 变更说明 |
|----------|------|--------|---------|
| 品种 | instrument | `btcusdt` | 单品种 |
| 频率 | freq | `60min` | 1小时K线 |
| 初始训练 | train | 2017-01 ~ 2020-12 (4年, ~35,000 bar) | 不变 |
| 滚动频率 | retrain_interval | 每季度 (3个月, ~2,160 bar) | ⚠️ 原为每月→改为季度 |
| 验证期 | valid | 滚动前 3 个月 (~2,160 bar) | 不变 |
| 测试期 | test | 2021-01 ~ 2026-05 | 不变 |
| 标签 | label | `Ref($close,-2)/Ref($close,-1)-1` | ⚠️ 修正为前向收益率 |
| 模型 | model | LightGBM (回归) | 不变 |
| 目标函数 | objective | `huber` (alpha=0.9) | ⚠️ 原为 mse → 改为 huber |
| 手续费 | fee | 0.04% 单边 (Taker), 0.02% 双边 (Maker) | ⚠️ 对标 Binance |
| 滑点 | slippage | 0.01% (流动性充足时段) | ⚠️ 下调，BTC流动性好 |
| 初始资金 | account | 100,000 USDT | 不变 |

---

## 二、因子工程（专家修订版，共 35 个因子）

### 2.1 设计原则

1. **严禁未来数据泄漏**: 所有因子仅用当前及历史数据 (`Ref($x, d)` with `d >= -1`)
2. **单品种适用**: 去掉截面排名类因子（如 Rank），单品种下无意义
3. **避免共线性**: 同类指标只保留最核心的 1-2 个变体
4. **加密货币适配**: 加入波动率加权和肥尾适配因子
5. **因子数量控制**: 35 个因子，避免维度过高导致过拟合

### 2.2 完整因子列表

#### A. K线形态因子 (9个, Alpha158 原始 kbar 因子)

```python
# 所有值除以 open 去量纲，避免价格绝对值影响
# K线实体
KMID  = ($close - $open) / $open           # K线实体占比
KLEN  = ($high - $low) / $open             # K线振幅

# K线上影线
KUP   = ($high - Greater($open, $close)) / $open   # 上影线占比
KUP2  = ($high - Greater($open, $close)) / ($high - $low + 1e-12)  # 上影比例(标准化)

# K线下影线
KLOW  = (Less($open, $close) - $low) / $open       # 下影线占比
KLOW2 = (Less($open, $close) - $low) / ($high - $low + 1e-12)  # 下影比例(标准化)

# 综合形态
KSFT  = (2 * $close - $high - $low) / $open        # K线偏度
KSFT2 = (2 * $close - $high - $low) / ($high - $low + 1e-12)   # K线偏度(标准化)
KMID2 = ($close - $open) / ($high - $low + 1e-12)  # 实体比例
```

#### B. 价格动量和收益率因子 (4个)

```python
# 注意：这些都是后向收益率（用当前和历史数据计算），不是标签
ROC_4   = Ref($close, 4) / $close - 1     # 4小时收益率 (向后)
ROC_12  = Ref($close, 12) / $close - 1    # 12小时收益率 (向后)  
ROC_24  = Ref($close, 24) / $close - 1    # 24小时收益率 (向后)
ROC_72  = Ref($close, 72) / $close - 1    # 3日收益率 (向后)
```

#### C. 移动平均线偏离因子 (4个)

```python
# 均线偏离度 = (价格 - 均线) / 均线，衡量价格与均线的相对位置
MA5_DEV   = $close / Mean($close, 5) - 1    # 与5周期均线偏离
MA12_DEV  = $close / Mean($close, 12) - 1   # 与12周期均线偏离  
MA24_DEV  = $close / Mean($close, 24) - 1   # 与24周期均线偏离 (日线级别)
MA72_DEV  = $close / Mean($close, 72) - 1   # 与72周期均线偏离 (3日级别)
```

#### D. 波动率和标准差因子 (5个)

```python
# 历史波动率（关键因子，为加密货币量身定制）
STD_5   = Std($close, 5) / $close - 1       # 5周期标准差 (短期波动)
STD_12  = Std($close, 12) / $close - 1      # 12周期标准差
STD_24  = Std($close, 24) / $close - 1      # 24周期标准差 (日波动)
STD_72  = Std($close, 72) / $close - 1      # 72周期标准差 (3日波动)

# 波动率收敛/发散 —— 加密货币重要信号
VOL_RATIO = Std($close, 5) / (Std($close, 24) + 1e-12)  # 短/长波动比率
```

#### E. 成交量与量价关系因子 (4个)

```python
# 成交量变化
VOLUME  = $volume / Mean($volume, 24) - 1   # 量比（当前量 / 24周期均量 - 1）

# 量价相关性 —— 判断趋势强度
CORR_12  = Corr($close, $volume, 12)        # 12周期价量相关系数
CORR_24  = Corr($close, $volume, 24)        # 24周期价量相关系数

# 价格振幅 vs 成交量关系（异常成交量识别）
RANGE_VOL = (($high - $low) / $open) / (Mean($volume, 24) + 1e-12)  # 振幅/成交量比
```

#### F. 趋势强度因子 (3个)

```python
# 斜率因子 —— 衡量价格趋势方向和强度
SLOPE_12 = Slope($close, 12) / $close       # 12周期斜率 (趋势方向)
SLOPE_24 = Slope($close, 24) / $close       # 24周期斜率

# R² 因子 —— 衡量趋势一致性
RSQR_24  = Rsquare($close, 24)              # 24周期线性拟合R² (趋势一致性)
```

#### G. 极值因子 (2个)

```python
# 衡量价格在近期区间中的位置
MAX_24  = Max($high, 24) / $close - 1       # 24周期最高价相对位置
MIN_24  = $close / Min($low, 24) - 1        # 24周期最低价相对位置（反转公式确保方向性）
```

#### H. 价格记忆/自相关因子 (4个)

```python
# 滞后价格（去量纲）
OPEN_0  = $open / $close - 1                # 开盘价/收盘价偏离
HIGH_0  = $high / $close - 1                # 最高价/收盘价偏离  
LOW_0   = $low / $close - 1                 # 最低价/收盘价偏离
VWAP_0  = $close / $close - 1               # 占位（实际为 0，保留结构一致性）

# 注：VWAP 在加密货币现货中不常用，保留字段但不作为有效因子
```

### 2.3 因子审查审计（前向/后向检查）

| 因子 | 计算方向 | 包含未来数据? | 审计状态 |
|------|---------|--------------|---------|
| KMID, KLEN, KUP, KUP2, KLOW, KLOW2, KSFT, KSFT2, KMID2 | 后向(当前bar) | ❌ 否 | ✅ 通过 |
| ROC_4, ROC_12, ROC_24, ROC_72 | 后向(历史) | ❌ 否 | ✅ 通过 |
| MA5_DEV ~ MA72_DEV | 后向(历史均值) | ❌ 否 | ✅ 通过 |
| STD_5 ~ STD_72, VOL_RATIO | 后向(历史标准差) | ❌ 否 | ✅ 通过 |
| VOLUME, CORR_12, CORR_24, RANGE_VOL | 后向(历史) | ❌ 否 | ✅ 通过 |
| SLOPE_12, SLOPE_24, RSQR_24 | 后向(历史拟合) | ❌ 否 | ✅ 通过 |
| MAX_24, MIN_24 | 后向(历史极值) | ❌ 否 | ✅ 通过 |
| OPEN_0, HIGH_0, LOW_0 | 当前bar | ❌ 否 | ✅ 通过 |
| **Label (预测目标)** | **前向(未来)** | **✅ 是（这是标签！）** | ✅ 作为标签使用 |
| ~~Return1（已删除）~~ | ~~后向~~ | ~~已从因子列表移除~~ | ⚠️ 原计划中与标签完全一致 |

### 2.4 因子共线性简述

35 个因子中，理论上存在部分相关性（如 MA5_DEV 与 MA12_DEV），但通过 LightGBM 的 `colsample_bytree` 和正则化可以缓解。后续可做 VIF 分析进一步筛选。

---

## 三、标签定义（修正版）

### 3.1 核心标签

```python
# 未来1小时的收益率（前向）
Label_1h = Ref($close, -2) / Ref($close, -1) - 1

# Ref($close, -2) = 未来1根K线的收盘价
# Ref($close, -1) = 当前K线的收盘价
# 结果：T+1时刻相对于T时刻的收益率
```

### 3.2 为什么选择 1h 预测周期

| 预测周期 | 预测难度 | 策略价值 | 推荐度 |
|----------|---------|---------|--------|
| 1h | 中等 | 高（高频交易机会） | ⭐⭐⭐⭐⭐ 核心 |
| 4h | 较低 | 中等（减少噪声） | ⭐⭐⭐ 可选 |
| 24h | 较高 | 低（信息衰减快） | ⭐⭐ 备选 |

### 3.3 标签预处理

```python
learn_processors = [
    {"class": "DropnaLabel"},                          # 删除无标签的行
    {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}}  # 截面排名标准化（单品种下等同于去均值）
]
```

---

## 四、LightGBM 配置（专家调优版）

### 4.1 参数设计理念

加密货币交易的三大挑战：
1. **高噪声** → 需要强正则化 + 采样
2. **肥尾分布** → Huber loss 替代 MSE
3. **小样本** (单品种 ~35,000 训练 bar) → 控制树复杂度

### 4.2 推荐参数

```python
LGBM_CONFIG = {
    # ========== 任务定义 ==========
    "loss": "mse",                       # Qlib LGBModel 内部使用 MSE
    "objective": "huber",                # 传给 LightGBM 的实际目标 (肥尾鲁棒)
    "alpha": 0.9,                        # Huber 损失的分位点参数
    
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

# 注意: Qlib 的 LGBModel 支持自动 early_stopping
# 可以同时传入 num_iterations=500 和 early_stopping_rounds=50
# Qlib 会自动在验证集上监控并停止
```

### 4.3 参数与样本量的适配关系

| 样本量条件 | num_leaves | max_depth | min_data_in_leaf | learning_rate |
|-----------|-----------|-----------|-----------------|---------------|
| < 10,000 | 15-31 | 4-5 | 50-100 | 0.01 |
| 10,000-50,000 | 31-63 | 5-7 | 100-200 | 0.015 | ← 我们在此区间 |
| 50,000-100,000 | 63-127 | 7-9 | 200-500 | 0.02 |
| > 100,000 | 127-255 | 9-12 | 500-1000 | 0.03 |

### 4.4 可选对比实验

```python
# 对照组 1: 基线 MSE
config_mse = {**LGBM_CONFIG, "objective": "regression", "alpha": None}

# 对照组 2: MAE
config_mae = {**LGBM_CONFIG, "objective": "regression_l1", "alpha": None}

# 实验组: Huber (主要)
config_huber = LGBM_CONFIG  # 当前推荐

# 对照组 3: 无正则化
config_no_reg = {**LGBM_CONFIG, "lambda_l1": 0.0, "lambda_l2": 0.0}
```

---

## 五、训练/验证/测试周期设计（修订版）

### 5.1 是否使用滚动窗口？

**结论：使用扩展窗口（Expanding Window）滚动训练，但频率从每月降低为每季度。**

**理由**：
- BTC 市场体制变化：牛熊周期通常持续 6-18 个月 → 月度滚动过于频繁，容易过拟合短期噪声
- 季度滚动（3个月）更合理：每季度新增约 2,160 bar 训练数据，样本增长有意义
- 保留验证期做 Early Stopping：每次重训练时使用滚动前 3 个月作为验证集

### 5.2 周期划分

```
时间轴 (训练至测试)：

2017-01 ◀──────── 训练期(扩展) ────────▶ 2026-05
        │                                   │
        ├──── 初始训练: 2017-01~2020-12 ────┤
        │    (4年, ~35,000 bar)             │
        │                                   │
        ├──── 第1次滚动: +2021-Q1 ──────────┤
        │    (训练 4.25年 → 验证用2020-Q4)   │
        │                                   │
        ├──── 第2次滚动: +2021-Q2 ──────────┤
        │    (训练 4.5年 → 验证用2021-Q1)    │
        │                                   │
        │    ... 每季度滚动 ...               │
        │                                   │
        └──── 最终: 2026-Q1 训练 → 测试2026-Q2
```

### 5.3 具体时间划分

```python
TRAINING_CONFIG = {
    "train_start": "2017-01-01",
    "train_end": "2020-12-31",        # 初始训练期 (4年)
    "valid_start": "2020-10-01",       # 初始验证期 (最后3个月)
    "valid_end": "2020-12-31",
    "test_start": "2021-01-01",        # 测试期
    "test_end": "2026-05-25",
    "rolling_step_months": 3,          # 滚动步长: 3个月
    "rolling_type": "expanding",       # 扩展窗口
    "valid_window_months": 3,          # 验证窗口: 3个月 (用于 early stopping)
}
```

### 5.4 滚动训练时间表

| 滚动序号 | 训练期 | 验证期 | 测试期 | 样本增长 |
|---------|--------|--------|--------|---------|
| R-01 | 2017-01 ~ 2020-12 | 2020-10 ~ 2020-12 | 2021-01 ~ 2021-03 | 35,040 bar |
| R-02 | 2017-01 ~ 2021-03 | 2021-01 ~ 2021-03 | 2021-04 ~ 2021-06 | 37,200 bar |
| ... | ... | ... | ... | ... |
| R-20 | 2017-01 ~ 2026-02 | 2025-12 ~ 2026-02 | 2026-03 ~ 2026-05 | 80,000+ bar |

---

## 六、单品种策略适配（关键设计）

### 6.1 为什么 TopkDropoutStrategy 不适合

```python
# TopkDropoutStrategy 的设计哲学：
# - 从 N 只股票中选出 topk 只买入
# - 每期 drop n_drop 只，换入新的
# - 核心逻辑：排名 → 选择 → 持仓

# 对于单品种 BTCUSDT (topk=1, n_drop=1)：
# - 评分最高就持有，评分下降就卖出并重新买入（同一品种）
# - 退化为：一直持有 → 无法表达"空仓"信号
# - 本质上变成了 buy-and-hold 策略，丢失了交易策略的核心价值
```

### 6.2 单品种正确方案：阈值信号策略

```python
# 方案: 构建自定义策略类 ThresholdSignalStrategy
# 不依赖 "排名选择"，而是依赖 "阈值判断"

class ThresholdSignalStrategy(BaseSignalStrategy):
    """
    单品种阈值策略
    
    信号逻辑:
    - pred >  long_threshold → 做多 (position = +1.0)
    - pred < -short_threshold → 做空 (position = -1.0)
    - 其他 → 空仓 (position = 0.0)
    
    与 TopkDropoutStrategy 的本质区别:
    - TopkDropout: 排名驱动，始终持有 top-k
    - ThresholdSignal: 阈值驱动，可空仓
    """
    def __init__(
        self,
        long_threshold=0.001,      # 做多阈值 (预测收益 > +0.1%)
        short_threshold=-0.001,    # 做空阈值 (预测收益 < -0.1%)
        position_ratio=0.95,       # 仓位比例
        hold_thresh=1,             # 最小持仓周期 (避免过度交易)
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.position_ratio = position_ratio
        self.hold_thresh = hold_thresh
    
    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        
        if pred_score is None:
            return TradeDecisionWO([], self)
        
        # 获取预测值（单品种只有一个值）
        if isinstance(pred_score, pd.DataFrame):
            pred_value = pred_score.iloc[-1, 0]
        else:
            pred_value = pred_score.iloc[-1]
        
        # 阈值判断
        if pred_value > self.long_threshold:
            target_weight = self.position_ratio     # 做多
        elif pred_value < self.short_threshold:
            target_weight = -self.position_ratio    # 做空
        else:
            target_weight = 0.0                      # 空仓
        
        # 生成目标仓位订单
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
            # 空仓：卖出所有持仓
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

### 6.3 信号阈值优化

| 参数 | 含义 | 初始值 | 调优建议 |
|------|------|--------|---------|
| `long_threshold` | 做多信号阈值 | 0.001 (0.1%) | 可在 0.0005 ~ 0.003 之间搜索 |
| `short_threshold` | 做空信号阈值 | -0.001 (-0.1%) | 可对称或非对称 (-0.0005 ~ -0.003) |
| `position_ratio` | 仓位占用比例 | 0.95 | 可在 0.5 ~ 1.0 之间 |
| `hold_thresh` | 最小持仓K线数 | 1 | BTC流动性好，设为1即可 |

---

## 七、交易成本 —— 严格对标 Binance 真实费率

### 7.1 Binance 现货交易费率（USDT 交易对）

| 等级 | Maker | Taker | 备注 |
|------|-------|-------|------|
| VIP 0 (默认) | 0.10% | 0.10% | 无 BNB 折扣 |
| VIP 0 (BNB 折扣 25%) | 0.075% | 0.075% | 使用 BNB 支付手续费 |

### 7.2 Binance 永续合约费率

| 等级 | Maker | Taker |
|------|-------|-------|
| VIP 0 | 0.02% | 0.04% |

### 7.3 回测配置选择

**选择永续合约费率（更低、BTC 流动性最好）**：

```python
EXCHANGE_CONFIG = {
    "freq": "60min",
    "limit_threshold": None,          # 加密货币无涨跌停
    "deal_price": "close",            # 以小时收盘价成交
    
    # Binance USDT 永续合约真实费率
    "open_cost": 0.0004,              # 开仓 Taker 费 0.04%
    "close_cost": 0.0004,             # 平仓 Taker 费 0.04%
    "min_cost": 0,                    # 最小手续费 (加密货币无最低)
    
    # 滑点: BTCUSDT 1小时级别流动性充足
    "slippage": 0.0001,               # 0.01% (币安 BTCUSDT 点差通常 < 0.01%)
}

# 手续费总计: 0.04% + 0.04% = 0.08% 每完整交易
# 滑点总计: 0.01% + 0.01% = 0.02% 每完整交易
# 总摩擦成本: 0.10% 每完整交易
```

### 7.4 现货 vs 永续合约对比

| 维度 | 现货 | 永续合约 | 选择 |
|------|------|---------|------|
| Maker 费 | 0.10% | 0.02% | ✅ 永续 |
| Taker 费 | 0.10% | 0.04% | ✅ 永续 |
| 资金费率 | 无 | ~0.01% / 8h | ⚠️ 需考虑 |
| 杠杆 | 无 | 最高 125x | 回测仅用 1x |
| 流动性 | 高 | 极高 | ✅ 永续 |

**建议**：使用永续合约费率模型，回测中加入资金费率成本（每 8 小时 0.01%，约 0.03%/天）。

---

## 八、回测 —— 严格遵循 Qlib 框架

### 8.1 回测架构图

```
┌────────────────────────────────────────────────────────────┐
│                     Qlib Backtest Engine                    │
│                                                            │
│  ┌───────────┐   ┌────────────┐   ┌──────────────────┐    │
│  │ Calendar  │──▶│ Executor   │──▶│ Decision Generator│   │
│  │ (60min)   │   │(Simulator) │   │  (Strategy)      │   │
│  └───────────┘   └────────────┘   └──────────────────┘    │
│                         │                    │              │
│                         ▼                    ▼              │
│                  ┌────────────┐   ┌──────────────────┐    │
│                  │  Exchange  │◀──│    Order/Trade   │    │
│                  │ (60min)    │   │    Execution     │    │
│                  └────────────┘   └──────────────────┘    │
│                         │                                   │
│                         ▼                                   │
│                  ┌────────────┐                            │
│                  │  Account   │                            │
│                  │  (Portfolio)│                           │
│                  └────────────┘                            │
└────────────────────────────────────────────────────────────┘
```

### 8.2 回测代码（严格 Qlib API）

```python
from qlib.backtest import backtest, executor
from qlib.backtest.decision import OrderDir
from qlib.utils import init_instance_by_config

# ====== 1. 初始化策略 ======
strategy_config = {
    "class": "ThresholdSignalStrategy",  # 自定义单品种策略
    "module_path": "BTC_Short_Term_v1.strategies.threshold_strategy",
    "kwargs": {
        "signal": predictions,             # 模型预测信号 DataFrame
        "long_threshold": 0.001,
        "short_threshold": -0.001,
        "position_ratio": 0.95,
    }
}
strategy = init_instance_by_config(strategy_config)

# ====== 2. 配置执行器 ======
executor_config = {
    "class": "SimulatorExecutor",
    "module_path": "qlib.backtest.executor",
    "kwargs": {
        "time_per_step": "60min",
        "generate_portfolio_metrics": True,
        "indicator_config": {
            "show_indicator_during_backtest": True,
        }
    }
}
executor_obj = init_instance_by_config(executor_config)

# ====== 3. 回测 ======
portfolio_metrics, indicator_metrics = backtest(
    start_time="2021-01-01",
    end_time="2026-05-25",
    executor=executor_obj,
    strategy=strategy,
    account=100000,
    benchmark="btcusdt",
    
    exchange_kwargs={
        "freq": "60min",
        "limit_threshold": None,      # 加密货币无涨跌停
        "deal_price": "close",
        "open_cost": 0.0004,          # 0.04%
        "close_cost": 0.0004,         # 0.04%
        "min_cost": 0,
        "slippage": 0.0001,           # 0.01%
    },
    pos_type="InfPosition",           # 允许做空
)

# ====== 4. 提取报告 ======
report_normal, positions_normal = portfolio_metrics.get("60min")
```

### 8.3 Qlib 框架约束检查清单

| 检查项 | 预期 | 说明 |
|--------|------|------|
| 数据管道 | DataHandlerLP → DatasetH | ✅ 严格遵循 |
| 模型接口 | Model.fit() → Model.predict() | ✅ 严格遵循 |
| 信号生成 | Signal → Strategy → Decision | ✅ 严格遵循 |
| 执行器 | SimulatorExecutor | ✅ 严格遵循 |
| Exchange | `exchange_kwargs` 参数 | ✅ 严格遵循 |
| Account | 统一 API | ✅ 严格遵循 |
| 指标提取 | `portfolio_metrics.get(freq)` | ✅ 严格遵循 |
| 不使用 backtest_daily() | 自定义 executor | ⚠️ backtest_daily 硬编码 freq="day"，不适合 60min |

---

## 九、评估指标体系

### 9.1 核心指标

```python
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.evaluate_portfolio import (
    get_sharpe_ratio_from_return_series,
    get_max_drawdown_from_series,
    get_alpha,
    get_beta,
    get_normal_ic,
    get_rank_ic,
)

# 1. 风险分析 (Qlib 内置)
risk_stats = risk_analysis(returns)

# 2. 自定义指标
annual_return = risk_stats["annualized_return"]
annual_vol = risk_stats["annualized_volatility"]
sharpe = annual_return / annual_vol
max_dd = get_max_drawdown_from_series(returns)
```

### 9.2 指标目标值

| 指标 | 公式 | 及格线 | 优秀线 | 说明 |
|------|------|--------|--------|------|
| 总收益率 | (终值 - 本金) / 本金 | > 50% | > 200% | 2021-2026 约5.4年 |
| 年化收益率 | (1 + 总收益)^(1/年) - 1 | > 8% | > 30% | |
| 年化波动率 | std(收益) * sqrt(8760) | < 60% | < 40% | 8760=365天×24小时 |
| 夏普比率 | (年化收益 - RF) / 年化波动 | > 0.8 | > 1.5 | RF=3% |
| 最大回撤 | max(peak - trough) / peak | < 40% | < 25% | BTC 波动大 |
| Calmar 比率 | 年化收益 / 最大回撤 | > 0.3 | > 1.0 | |
| 胜率 | 盈利交易 / 总交易 | > 50% | > 55% | |
| 盈亏比 | avg(盈利) / avg(亏损) | > 1.0 | > 1.5 | |
| IC | corr(pred, real) | > 0.01 | > 0.03 | 时序 IC |
| 换手率 | 日均换手 | < 50% | < 20% | 避免过度交易 |

### 9.3 基准对比

| 基准 | 类型 | 期望战胜 |
|------|------|---------|
| BTCUSDT Buy & Hold | 被动策略 | ✅ 收益更高且回撤更低 |
| 无风险利率 (3%) | 绝对基准 | ✅ 显著超越 |

---

## 十、报告输出

### 10.1 图表输出

```python
from qlib.contrib.report.analysis_position import report_graph, risk_analysis_graph
from qlib.contrib.report.analysis_model import model_performance_graph

# A. 组合表现图
report_graph(
    report_df=portfolio_metrics,
    show_notebook=False,
    save_path="report/performance_report.png"
)

# B. 风险分析图
risk_analysis_graph(
    report_df=portfolio_metrics,
    show_notebook=False,
    save_path="report/risk_report.png"
)

# C. IC 分析图
model_performance_graph(
    pred_label=pred_label_df,
    show_notebook=False,
    save_path="report/ic_report.png"
)
```

### 10.2 文本报告

```
══════════════════════════════════════════════════════════
             BTC 短线交易v1 - 回测报告
══════════════════════════════════════════════════════════

【策略概要】
  策略名称: BTC 短线交易v1
  交易品种: BTCUSDT
  数据频率: 1小时 (60min)
  模型: LightGBM (Huber Loss)
  因子数量: 35
  标签: 未来1小时收益率 (Ref($close,-2)/Ref($close,-1)-1)

【训练配置】
  训练方式: Expanding Window 滚动训练
  初始训练: 2017-01-01 ~ 2020-12-31 (4年)
  滚动频率: 每季度 (3个月)
  滚动次数: 20 次

【回测表现】 (2021-01-01 ~ 2026-05-25, 约5.4年)
  ✓ 总收益率:     xx.xx%
  ✓ 年化收益率:   xx.xx%
  ✓ 年化波动率:   xx.xx%
  ✓ 夏普比率:     x.xx
  ✓ 最大回撤:     xx.xx%
  ✓ Calmar比率:   x.xx
  ✓ 交易次数:     xxxx
  ✓ 胜率:         xx.xx%
  ✓ 盈亏比:       x.xx
  ✓ 平均IC:       x.xxx

【与基准对比】
  BTC Buy & Hold 总收益:   xx.xx%
  策略超额收益:             xx.xx%

【年度表现】
  2021: xx.xx%   | 2022: xx.xx%   | 2023: xx.xx%
  2024: xx.xx%   | 2025: xx.xx%   | 2026: xx.xx%

══════════════════════════════════════════════════════════
```

---

## 十一、最终项目结构

```
BTC_Short_Term_v1/
├── strategy_plan.md                    ← 本文件 (专家修订版)
│
├── config/
│   ├── __init__.py
│   ├── model_config.py                # LightGBM 参数 (Huber, 强正则)
│   ├── data_config.py                 # 数据路径 + 频率配置
│   ├── training_config.py             # 滚动训练 + 周期划分
│   └── backtest_config.py             # 手续费 + 滑点 + 账户
│
├── features/
│   ├── __init__.py
│   └── crypto_handler.py             # 自定义 DataHandler (35因子 + Label)
│
├── models/
│   ├── __init__.py
│   └── rolling_trainer.py            # Expanding Window 滚动训练
│
├── strategies/
│   ├── __init__.py
│   └── threshold_strategy.py         # 单品种阈值策略 (继承 BaseSignalStrategy)
│
├── backtest/
│   ├── __init__.py
│   └── run_backtest.py               # Qlib Backtest 执行
│
├── evaluation/
│   ├── __init__.py
│   └── report.py                     # 评估报告 + 图表生成
│
├── report/                            # 输出目录
│   ├── performance_report.png
│   ├── risk_report.png
│   ├── ic_report.png
│   └── trading_log.csv
│
├── main.py                           # 主程序入口
└── run.sh                            # 一键运行脚本
```

---

## 十二、执行流程

```
Phase 1: 数据验证
  └── crypto_init.py → 验证数据可读、日历正确

Phase 2: 特征工程
  └── crypto_handler.py → 定义 35 因子 + Label, 组装 DataHandler

Phase 3: 模型训练
  └── rolling_trainer.py → Expanding Window 季度滚动, LightGBM 训练

Phase 4: 策略实现
  └── threshold_strategy.py → 单品种阈值策略 (做多/做空/空仓)

Phase 5: 回测执行
  └── run_backtest.py → Qlib Backtest, 严格成本模型

Phase 6: 报告生成
  └── report.py → 图表 + 文本报告, 对比基准 Buy & Hold
```

---

## 十三、修订总结

| 类别 | 原计划 | 修订后 | 变更理由 |
|------|--------|--------|---------|
| 标签 | `$close/Ref($close,1)-1` (后向) | `Ref($close,-2)/Ref($close,-1)-1` (前向) | 修正致命的数据泄漏 |
| 因子数量 | 60+ | 35 (精筛) | 减少共线性, 单品种适配 |
| 标签重复因子 | `Return1` 等 5 个后向收益因子 | 删除 1h/4h Return, 保留4/12/24/72周期ROC | 1h 因子与标签定义完全一致(已泄漏) |
| LightGBM loss | MSE | Huber (alpha=0.9) | 加密货币肥尾分布 |
| learning_rate | 0.03 | 0.015 | 降低, 配合更多迭代 |
| 正则化 | lambda_l1=0.1, lambda_l2=1.0 | lambda_l1=0.5, lambda_l2=1.5 | 加强正则化 |
| 采样 | subsample=0.8, colsample=0.8 | subsample=0.7, colsample=0.7 | 更激进对抗过拟合 |
| 叶子节点 | 63 | 31 | ~35K 样本约束 |
| min_data_in_leaf | 50 | 100 | 提高最小样本防止过拟合 |
| 滚动频率 | 1个月 | 3个月(季度) | BTC体制变化周期更长 |
| 策略 | TopkDropoutStrategy (topk=1) | ThresholdSignalStrategy (阈值) | 单品种需阈值驱动 |
| 手续费 | 0.04%(双边) | 0.04%(单边 Taker) | 严格对标 Binance 永续合约 |
| 滑点 | 0.02% | 0.01% | BTCUSDT 流动性充足 |

---

**审核状态**: ✅ 专家修订完成，待审核

**核心变更简述**:
1. 🐛 **修复标签泄漏**: 标签从前向改为未来，彻底消除与因子的重合
2. 📊 **精筛因子**: 从 60+ 精简到 35，删除共线性和泄漏因子
3. ⚙️ **LightGBM 调优**: Huber loss、更强正则、更保守的树结构
4. 📅 **季度滚动**: 从月度降为季度，匹配 BTC 市场体制
5. 🎯 **单品种策略**: 自定义 ThresholdSignalStrategy 替代 TopkDropout
6. 💰 **真实成本**: 严格对标 Binance 永续合约费率
7. 🏗️ **严格 Qlib**: 使用 backtest() 而非 backtest_daily()，适配 60min 频率