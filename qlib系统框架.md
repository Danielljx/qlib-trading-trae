# Qlib 量化投资平台 — 完整系统框架文档

> 基于 Microsoft Qlib 源码全面梳理，涵盖所有核心模块的类、方法、配置项和模块间关系。

---

## 一、Qlib 概述

### 1.1 定位与目标

Qlib 是微软开源的 **面向 AI 的量化投资平台**，旨在实现 AI 技术在量化投资中的潜力。它覆盖了从数据获取到策略执行的完整量化投资链条。

**核心目标：**
- 端到端量化投资流程：数据获取 → 特征工程 → 模型训练 → 回测 → 策略执行 → 结果分析
- AI 驱动：利用机器学习挖掘市场复杂非线性模式
- 模块化设计：松耦合组件，每个模块可独立使用
- 研究友好：支持从想法探索到生产实现的全流程

### 1.2 支持的学习范式

| 范式 | 说明 | 典型场景 |
|------|------|----------|
| 监督学习 | 传统 ML/DL 模型训练 | Alpha 预测、风险建模 |
| 强化学习 | MDP 框架下的策略学习 | 订单执行优化、组合构建 |
| 元学习 | 动态适应市场变化 | DDG-DA 跨域适应 |
| 滚动训练 | 时间序列滚动建模 | 模型持续更新 |

### 1.3 支持的市场

| 区域标识 | 常量 | 市场 |
|----------|------|------|
| `cn` | `REG_CN` | 中国 A 股 |
| `us` | `REG_US` | 美国 |
| `tw` | `REG_TW` | 中国台湾 |

---

## 二、整体架构

### 2.1 四层架构

```
┌──────────────────────────────────────────────────────────────────────┐
│                     Interface Layer (接口层)                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│  │ Analyser │ │ Recorder │ │   qrun   │ │  Tuner   │ │ Operator │ │
│  │ 分析报告  │ │ 实验管理  │ │ 自动工作流│ │ 超参调优 │ │ 在线交易  │ │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ │
├──────────────────────────────────────────────────────────────────────┤
│                     Workflow Layer (工作流层)                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │   Strategy   │  │   Executor   │  │        Exchange         │  │
│  │  策略决策生成  │  │  执行器控制   │  │  模拟交易所(撮合/成本)   │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │   Account    │  │   Position   │  │   Signal / Report       │  │
│  │  账户管理     │  │  持仓管理     │  │  信号接口 / 指标报告     │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
├──────────────────────────────────────────────────────────────────────┤
│                   Learning Framework Layer (学习框架层)               │
│  ┌────────────────────┐  ┌────────────────────────────────────┐    │
│  │ Supervised Learning │  │     Reinforcement Learning         │    │
│  │  Model / Trainer    │  │  Simulator / Reward / Interpreter  │    │
│  │  Ensemble / Meta    │  │  Policy (PPO / OPDS)              │    │
│  └────────────────────┘  └────────────────────────────────────┘    │
├──────────────────────────────────────────────────────────────────────┤
│                  Infrastructure Layer (基础设施层)                    │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐              │
│  │ Data Server  │ │   Cache      │ │   Config     │              │
│  │ Provider体系  │ │ 多级缓存系统  │ │  全局配置     │              │
│  │ Storage体系   │ │ Mem/Disk/Exp │ │  日志系统     │              │
│  └──────────────┘ └──────────────┘ └──────────────┘              │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.2 模块依赖关系总览

```
qlib/
├── __init__.py          # 入口：qlib.init()
├── config.py            # 全局配置 QlibConfig (C)
├── constant.py          # 常量 REG_CN, REG_US, EPS 等
├── typehint.py          # 类型提示兼容层
├── log.py               # 日志系统 QlibLogger, TimeInspector
├── utils/               # 通用工具函数
│
├── data/                # 数据层
│   ├── base.py          # Expression 表达式体系
│   ├── data.py          # Provider 体系 (Calendar/Instrument/Feature/Expression/Dataset)
│   ├── client.py        # 客户端 SocketIO 通信
│   ├── cache.py         # 缓存体系 (MemCache/DiskCache/ExpressionCache/DatasetCache)
│   ├── filter.py        # 动态股票池过滤 (NameDFilter/ExpressionDFilter)
│   ├── ops.py           # 运算符引擎 (Rolling/Pair/Elem/Cython加速)
│   ├── dataset/
│   │   ├── handler.py   # DataHandler / DataHandlerLP
│   │   ├── loader.py    # DataLoader (Qlib/Static/Nested/DH)
│   │   ├── processor.py # Processor (ZScoreNorm/Fillna/Dropna等)
│   │   ├── storage.py   # Handler存储后端
│   │   ├── weight.py    # 样本权重 Reweighter
│   │   └── utils.py     # 数据集工具函数
│   └── storage/
│       ├── storage.py   # 存储抽象基类
│       └── file_storage.py  # 文件存储实现 (Calendar/Instrument/Feature)
│
├── model/               # 模型层
│   ├── base.py          # BaseModel / Model / ModelFT
│   ├── trainer.py       # Trainer / TrainerR / TrainerRM (单任务/多任务/延迟)
│   ├── utils.py         # PyTorch训练辅助 (ConcatDataset/IndexSampler)
│   ├── ens/
│   │   ├── ensemble.py  # Ensemble / RollingEnsemble / AverageEnsemble
│   │   └── group.py     # Group / RollingGroup
│   ├── interpret/       # 特征解释 (FeatureInt / LightGBMFInt)
│   ├── riskmodel/       # 风险模型 (POETCov/ShrinkCov/StructuredCov)
│   └── meta/            # 元学习 (MetaTask / MetaTaskDataset)
│
├── backtest/            # 回测层
│   ├── __init__.py      # 入口：backtest() / get_exchange() / get_strategy_executor()
│   ├── backtest.py      # 回测主循环 backtest_loop / collect_data_loop
│   ├── exchange.py      # Exchange 模拟交易所
│   ├── executor.py      # BaseExecutor / NestedExecutor / SimulatorExecutor
│   ├── account.py       # Account 账户管理
│   ├── position.py      # Position / InfPosition 持仓管理
│   ├── decision.py      # Order / OrderDir / BaseTradeDecision
│   ├── signal.py        # Signal / SignalWCache / ModelSignal
│   ├── report.py        # PortfolioMetrics / Indicator
│   ├── profit_attribution.py  # Brinson 归因分析
│   └── utils.py         # TradeCalendarManager / CommonInfrastructure
│
├── strategy/            # 策略层
│   └── base.py          # BaseStrategy / RLStrategy / RLIntStrategy
│
├── rl/                  # 强化学习层
│   ├── simulator.py     # Simulator 模拟器基类
│   ├── reward.py        # Reward / RewardCombination
│   ├── interpreter.py   # StateInterpreter / ActionInterpreter
│   ├── seed.py          # 初始状态类型定义
│   ├── trainer/         # RL训练器 (Trainer/Vessel/Checkpoint/EarlyStopping)
│   ├── order_execution/ # 订单执行RL (PPO/OPDS/SAOEState/PA奖励)
│   └── contrib/backtest.py  # RL回测工具
│
├── workflow/            # 工作流层
│   └── __init__.py      # R (QlibRecorder) 实验管理全局入口
│
└── contrib/             # 扩展实现
    ├── data/
    │   ├── handler.py   # Alpha158 / Alpha360 数据处理器
    │   └── loader.py    # Alpha158DL / Alpha360DL 数据加载器
    ├── model/
    │   ├── gbdt.py      # LGBModel (LightGBM)
    │   ├── xgboost.py   # XGBModel
    │   ├── catboost_model.py  # CatBoostModel
    │   ├── linear.py    # LinearModel (OLS/NNLS/Ridge/Lasso)
    │   ├── pytorch_nn.py    # DNNModelPytorch
    │   ├── pytorch_lstm_ts.py  # LSTM
    │   ├── pytorch_transformer.py  # Transformer
    │   ├── pytorch_tcts.py  # TCTS
    │   ├── pytorch_gats.py  # GATs
    │   ├── pytorch_alstm.py  # ALSTM
    │   ├── pytorch_gru.py   # GRU
    │   ├── pytorch_tcn.py   # TCN
    │   ├── pytorch_sfm.py   # SFM
    │   ├── pytorch_tabnet.py  # TabNet
    │   └── double_ensemble.py  # DEnsembleModel
    ├── strategy/
    │   ├── signal_strategy.py  # TopkDropout / EnhancedIndexing / WeightStrategy
    │   ├── rule_strategy.py    # TWAP / SBB / AC / Random / FileOrder
    │   ├── order_generator.py  # OrderGenWInteract / OrderGenWOInteract
    │   ├── cost_control.py     # SoftTopkStrategy
    │   └── optimizer/
    │       ├── base.py         # BaseOptimizer
    │       ├── optimizer.py    # PortfolioOptimizer (GMV/MVO/RP/INV)
    │       └── enhanced_indexing.py  # EnhancedIndexingOptimizer (CVXPY)
    ├── evaluate.py          # risk_analysis / backtest_daily / long_short_backtest
    ├── evaluate_portfolio.py # 持仓评估指标 (Sharpe/MaxDD/Beta/Alpha/IC)
    ├── workflow/
    │   └── record_temp.py  # MultiSegRecord / SignalMseRecord
    ├── report/
    │   ├── graph.py        # BaseGraph / ScatterGraph / BarGraph / SubplotsGraph
    │   ├── analysis_model/ # 模型性能图表 (分组收益/IC/自相关/换手率)
    │   └── analysis_position/  # 持仓分析图表 (报告/累积收益/风险/IC/排名)
    ├── tuner/
    │   ├── tuner.py        # Tuner / QLibTuner (Hyperopt)
    │   └── space.py        # 预设搜索空间
    ├── rolling/
    │   └── base.py         # Rolling 滚动训练
    ├── online/
    │   ├── manager.py      # UserManager 用户管理
    │   ├── user.py         # User 用户实体
    │   ├── online_model.py # ScoreFileModel
    │   ├── operator.py     # Operator 在线交易操作
    │   └── utils.py        # 序列化/反序列化工具
    └── eva/
        └── alpha.py        # Alpha因子评估 (多空精度/多空收益)
```

---

## 三、基础设施层详解

### 3.1 全局配置 (`qlib/config.py`)

```python
from qlib.config import C  # 全局配置单例

# 核心配置项
C["provider_uri"]       # 数据路径
C["region"]             # 市场区域 (cn/us/tw)
C["calendar_provider"]  # 日历提供者配置
C["expression_cache"]   # 表达式缓存配置
C["dataset_cache"]      # 数据集缓存配置
C["kernels"]            # 并行计算核数
C["redis_host"]         # Redis主机
C["redis_port"]         # Redis端口
C["logging_level"]      # 日志级别
C["exp_manager"]        # 实验管理器配置
```

**QlibConfig.DataPathManager** — 数据路径管理：
- 处理 `provider_uri` 和 `mount_path`
- 支持本地路径和 NFS 挂载路径
- 自动检测路径类型并转换

### 3.2 日志系统 (`qlib/log.py`)

| 组件 | 功能 |
|------|------|
| `QlibLogger` | 自定义日志器，封装 Python logging |
| `get_module_logger(name)` | 获取模块级日志器 |
| `TimeInspector` | 计时工具：`set_time_mark()` / `log_cost_time()` / `logt()` 上下文管理器 |
| `LogFilter` | 日志过滤器，支持正则匹配 |

### 3.3 常量定义 (`qlib/constant.py`)

| 常量 | 值 | 说明 |
|------|-----|------|
| `REG_CN` | `"cn"` | 中国市场 |
| `REG_US` | `"us"` | 美国市场 |
| `REG_TW` | `"tw"` | 台湾市场 |
| `EPS` | `1e-12` | 防除零小量 |
| `INF` | `int(1e18)` | 整数无穷大 |
| `ONE_DAY` | — | 一天时间增量 |
| `ONE_MIN` | — | 一分钟时间增量 |

---

## 四、数据层详解

### 4.1 Provider 体系 (`qlib/data/data.py`)

Provider 是数据访问的核心抽象，分为 6 种类型：

| Provider | 功能 | 本地实现 | 客户端实现 |
|----------|------|----------|-----------|
| `CalendarProvider` | 交易日历 | `LocalCalendarProvider` | `ClientCalendarProvider` |
| `InstrumentProvider` | 股票池定义 | `LocalInstrumentProvider` | `ClientInstrumentProvider` |
| `FeatureProvider` | 特征数据 | `LocalFeatureProvider` | — |
| `PITProvider` | PIT 数据 | `LocalPITProvider` | — |
| `ExpressionProvider` | 表达式计算 | `LocalExpressionProvider` | — |
| `DatasetProvider` | 数据集 | `LocalDatasetProvider` | `ClientDatasetProvider` |

**统一访问接口 `BaseProvider`：**
- `calendar(start_time, end_time, freq)` → 交易日历
- `instruments(instruments)` → 股票池
- `features(instruments, fields, start_time, end_time, freq)` → 特征数据

**Instrument 类型常量：**
- `LIST` — 列表模式
- `DICT` — 字典模式（带时间范围）
- `CONF` — 配置模式

### 4.2 表达式引擎 (`qlib/data/base.py` + `qlib/data/ops.py`)

**Expression 类层次：**
```
Expression (基类，支持运算符重载 +, -, *, /, >, <)
├── Feature           # 静态特征 ($close, $open...)
├── PFeature          # PIT特征 ($$close...)
└── ExpressionOps     # 运算符表达式
    ├── ElemOperator  # 一元运算符
    │   ├── NpElemOperator  # Abs, Sign, Log, Mask, Not
    │   └── ChangeInstrument  # 切换instrument
    └── PairOperator  # 二元运算符
        └── NpPairOperator   # Add, Sub, Mul, Div, Gt, Lt...
```

**内置运算符函数：**

| 类别 | 运算符 | 说明 |
|------|--------|------|
| 滞后 | `Ref($x, n)` | 前/后 n 日数据 |
| 滚动统计 | `Mean/Std/Sum/Max/Min/Median/Mad($x, d)` | d 日滚动统计 |
| 截面 | `Rank($x)` / `Quantile($x, q)` | 截面排名/分位数 |
| 相关 | `Corr($x, $y, d)` / `Cov($x, $y, d)` | d 日滚动相关/协方差 |
| 技术指标 | `RSI($x, d)` / `MA($x, d)` / `EMA($x, d)` / `WMA($x, d)` | 技术分析 |
| Cython加速 | `rolling_slope/rolling_rsquare/rolling_resi` | 高性能滚动计算 |

### 4.3 缓存体系 (`qlib/data/cache.py`)

| 缓存类型 | 类 | 说明 |
|----------|-----|------|
| 内存缓存 | `MemCache` | 全局实例 `H`，含日历(c)/股票池(i)/特征(f)三区 |
| 内存条目限制 | `MemCacheLengthUnit` | 按 LRU 条目数淘汰 |
| 内存大小限制 | `MemCacheSizeofUnit` | 按内存大小淘汰 |
| 带过期缓存 | `MemCacheExpire` | 支持过期时间 |
| 表达式缓存 | `ExpressionCache` / `DiskExpressionCache` | 磁盘缓存 + Redis 锁 |
| 数据集缓存 | `DatasetCache` / `DiskDatasetCache` | HDF 格式 + 索引 |
| 简单数据集缓存 | `SimpleDatasetCache` | Pickle 格式 |
| URI缓存 | `DatasetURICache` | URI 机制缓存 |

### 4.4 数据存储 (`qlib/data/storage/`)

**存储抽象基类：**

| 类 | 接口风格 | 说明 |
|-----|----------|------|
| `CalendarStorage` | list 风格 (`__getitem__`, `__setitem__`, `extend`, `index`) | 日历存储 |
| `InstrumentStorage` | dict 风格 | 股票池存储 |
| `FeatureStorage` | 二进制 (`data`, `start_index`, `end_index`, `write`, `rebase`, `rewrite`) | 特征存储 |

**文件存储实现 (`file_storage.py`)：**
- `FileCalendarStorage` — 读写 txt 文件，支持缓存和频率重采样
- `FileInstrumentStorage` — 读写 tab 分隔的 txt 文件
- `FileFeatureStorage` — 读写 bin 二进制文件，支持追加和重写

### 4.5 数据过滤 (`qlib/data/filter.py`)

| 过滤器 | 说明 |
|--------|------|
| `NameDFilter` | 基于名称正则表达式过滤（如 `SH[0-9]{6}`） |
| `ExpressionDFilter` | 基于表达式过滤（如 `$close/$open > 5`），支持时间范围和 keep 参数 |

### 4.6 数据目录结构

```
data/
├── calendars/           # 交易日历 (day.txt, 1min.txt)
├── instruments/         # 股票池 (all.txt, csi300.txt, csi500.txt)
├── features/            # 特征数据 (bin格式)
│   └── sh600000/
│       ├── open.day.bin
│       ├── close.day.bin
│       ├── high.day.bin
│       ├── low.day.bin
│       └── volume.day.bin
├── calculated_features/ # 计算特征缓存
└── cache/               # 数据集缓存
```

---

## 五、数据集层详解

### 5.1 DataHandler (`qlib/data/dataset/handler.py`)

**类层次：**
```
DataHandlerABC (接口)
└── DataHandler (基础实现)
    └── DataHandlerLP (支持可学习处理器)
```

**DataHandlerLP 核心特性：**
- 维护三份数据：`_data`(原始)、`_infer`(推理用)、`_learn`(学习用)
- `fit()` — 在 fit 时间段学习处理器参数
- `process_data()` — 应用处理器转换数据
- `fit_process_data()` — 一站式：fit + process
- `fetch(selector, level, col_set, data_key)` — 获取数据
- `get_cols(col_set)` — 获取列名
- `get_range_iterator()` — 按时间范围迭代

**关键常量：**
- 列集：`CS_ALL`, `CS_RAW`
- 数据键：`DK_R`(原始), `DK_I`(推理), `DK_L`(学习)
- 处理类型：`PTYPE_I`(推理), `PTYPE_A`(全部)

### 5.2 DataLoader (`qlib/data/dataset/loader.py`)

| 加载器 | 说明 |
|--------|------|
| `QlibDataLoader` | 通过 `D.features()` 从 Qlib 加载 |
| `StaticDataLoader` | 从文件或 DataFrame 加载 |
| `NestedDataLoader` | 组合多个 DataLoader |
| `DataLoaderDH` | 基于 DataHandler 加载 |

### 5.3 Processor (`qlib/data/dataset/processor.py`)

| 处理器 | 说明 |
|--------|------|
| `DropnaProcessor` | 删除缺失值 |
| `DropnaLabel` | 删除标签缺失值 |
| `Fillna(value=0)` | 用指定值填充 |
| `CSZFillna` | 截面填充 |
| `ProcessInf` | 处理无穷值 |
| `TanhProcess` | tanh 处理噪声 |
| `ZScoreNorm` | Z-Score 标准化 |
| `MinMaxNorm` | Min-Max 归一化 |
| `RobustZScoreNorm` | 稳健 Z-Score |
| `CSZScoreNorm` | 截面 Z-Score |
| `CSRankNorm` | 截面排名归一化 |

### 5.4 DatasetH (`qlib/data/dataset/__init__.py`)

```python
dataset = DatasetH(
    handler=handler,
    segments={
        "train": ("2008-01-01", "2014-12-31"),
        "valid": ("2015-01-01", "2016-12-31"),
        "test":  ("2017-01-01", "2020-08-01"),
    }
)
train_data = dataset.prepare("train")  # 准备训练数据
```

### 5.5 Reweighter (`qlib/data/dataset/weight.py`)

样本权重重加权器基类，定义 `reweight(data)` 抽象方法，用于对训练样本赋权。

### 5.6 预定义数据处理器 (`qlib/contrib/data/`)

| 处理器 | 特征数 | 加载器 | 说明 |
|--------|--------|--------|------|
| `Alpha158` | 158 | `Alpha158DL` | kbar + price + volume + rolling 特征 |
| `Alpha360` | 360 | `Alpha360DL` | 60天 CLOSE/OPEN/HIGH/LOW/VWAP/VOLUME 的 Ref 归一化 |

---

## 六、模型层详解

### 6.1 模型接口 (`qlib/model/base.py`)

```
BaseModel (Serializable)
├── predict(dataset) → pd.Series    # 预测
└── __call__(dataset)               # 语法糖
    │
Model (BaseModel)
├── fit(dataset, reweighter)        # 训练
├── predict(dataset, segment)       # 预测
    │
ModelFT (Model)
└── finetune(dataset)               # 微调
```

### 6.2 训练器 (`qlib/model/trainer.py`)

| 训练器 | 说明 |
|--------|------|
| `Trainer` | 基类：`train(tasks)` + `end_train(models)` 两阶段模式 |
| `TrainerR` | 基于 Recorder 的线性训练器，按顺序训练 |
| `DelayTrainerR` | 延迟训练器：`train` 仅保存配置，`end_train` 实际训练 |
| `TrainerRM` | 多进程训练器，支持 worker 并行 |
| `DelayTrainerRM` | 延迟版 TrainerRM |

**便捷函数：**
- `task_train(task_config, experiment_name)` — 完整单任务训练
- `begin_task_train(task_config, experiment_name)` — 开始任务（仅保存配置）
- `end_task_train(rec, experiment_name)` — 完成任务（实际拟合）

### 6.3 集成 (`qlib/model/ens/`)

| 类 | 说明 |
|-----|------|
| `Ensemble` | 集成基类，`__call__(ensemble_dict)` |
| `SingleKeyEnsemble` | 单键值提取，支持递归 |
| `RollingEnsemble` | 合并滚动预测结果，按 datetime 拼接去重 |
| `AverageEnsemble` | 标准化后取平均 |
| `Group` | 分组归约，支持并行 |
| `RollingGroup` | 滚动分组，按最后一个 key 分组 |

### 6.4 内置模型 (`qlib/contrib/model/`)

| 模型 | 类名 | 文件 | 特点 |
|------|------|------|------|
| LightGBM | `LGBModel` | `gbdt.py` | 支持 early stopping、样本权重、特征重要性、finetune |
| XGBoost | `XGBModel` | `xgboost.py` | 支持 early stopping、样本权重、特征重要性 |
| CatBoost | `CatBoostModel` | `catboost_model.py` | 支持 GPU 加速、特征重要性 |
| 线性模型 | `LinearModel` | `linear.py` | OLS/NNLS/Ridge/Lasso 四种估计器 |
| DNN | `DNNModelPytorch` | `pytorch_nn.py` | 自定义网络结构、Adam/SGD、early stopping、DataParallel |
| LSTM | `LSTM` | `pytorch_lstm_ts.py` | d_feat, hidden_size, num_layers, dropout |
| GRU | `GRU` | `pytorch_gru.py` | 同 LSTM |
| ALSTM | `ALSTM` | `pytorch_alstm.py` | 带注意力的 LSTM |
| Transformer | `Transformer` | `pytorch_transformer.py` | 注意力机制 |
| GATs | `GATs` | `pytorch_gats.py` | 图注意力网络 |
| TCN | `TCN` | `pytorch_tcn.py` | 时序卷积网络 |
| TFT | `TFT` | `pytorch_tfts.py` | Temporal Fusion Transformer |
| TCTS | `TCTS` | `pytorch_tcts.py` | 时序交叉 Transformer |
| SFM | `SFM` | `pytorch_sfm.py` | Simple Factor Model |
| TabNet | `TabNetModel` | `pytorch_tabnet.py` | TabNet 架构 |
| Double Ensemble | `DEnsembleModel` | `double_ensemble.py` | 样本重加权 + 特征选择 |

### 6.5 特征解释 (`qlib/model/interpret/`)

| 类 | 说明 |
|-----|------|
| `FeatureInt` | 特征解释器抽象基类，`get_feature_importance() → pd.Series` |
| `LightGBMFInt` | LightGBM 特征解释器，调用 `model.feature_importance()` |

### 6.6 风险模型 (`qlib/model/riskmodel/`)

| 类 | 说明 |
|-----|------|
| `RiskModel` | 风险模型基类 |
| `POETCovEstimator` | POET 协方差估计 |
| `ShrinkCovEstimator` | 收缩协方差估计 |
| `StructuredCovEstimator` | 结构化协方差估计 |

### 6.7 元学习 (`qlib/model/meta/`)

| 类 | 说明 |
|-----|------|
| `MetaTask` | 元学习任务定义 |
| `MetaTaskDataset` | 元学习任务数据集 |

---

## 七、回测层详解

### 7.1 回测入口 (`qlib/backtest/__init__.py`)

| 函数 | 说明 |
|------|------|
| `backtest(...)` | 一站式回测：初始化策略+执行器 → 运行循环 → 返回组合指标+交易指标 |
| `get_exchange(...)` | 创建模拟交易所 |
| `create_account_instance(...)` | 创建账户（支持 float/int/dict 初始化） |
| `get_strategy_executor(...)` | 初始化策略与执行器 |
| `collect_data(...)` | 生成器形式收集交易决策（用于 RL 训练） |
| `format_decisions(...)` | 将决策格式化为树状结构 |

**返回类型：**
- `PORT_METRIC = Dict[str, Tuple[pd.DataFrame, dict]]` — 组合指标
- `INDICATOR_METRIC = Dict[str, Tuple[pd.DataFrame, Indicator]]` — 交易指标

### 7.2 回测主循环 (`qlib/backtest/backtest.py`)

| 函数 | 说明 |
|------|------|
| `backtest_loop(start_time, end_time, trade_strategy, trade_executor)` | 主循环：收集数据 → 汇总指标 |
| `collect_data_loop(...)` | 生成器：重置 → 策略生成决策 → 执行器执行 → 策略后处理 → 循环 |

### 7.3 Exchange 模拟交易所 (`qlib/backtest/exchange.py`)

| 方法 | 说明 |
|------|------|
| `check_stock_limit(stock_id, start_time, end_time, direction)` | 检查涨跌停 |
| `check_stock_suspended(stock_id, start_time, end_time)` | 检查停牌 |
| `is_stock_tradable(...)` | 综合判断可交易性 |
| `deal_order(order, trade_account, position, dealt_order_amount)` | 撮合订单 → (trade_val, trade_cost, trade_price) |
| `get_deal_price(...)` | 获取成交价格 |
| `get_close/get_volume/get_factor(...)` | 获取行情数据 |
| `generate_amount_position_from_weight_position(...)` | 权重→数量仓位 |
| `generate_order_for_target_amount_position(...)` | 目标仓位→订单列表 |
| `calculate_amount_position_value(...)` | 计算仓位市值 |
| `round_amount_by_trade_unit(...)` | 按交易单位取整 |

**涨跌停限制类型：**
- `LT_TP_EXP = "(exp)"` — 表达式模式
- `LT_FLT = "float"` — 浮动比例模式
- `LT_NONE = "none"` — 无限制

### 7.4 Executor 执行器 (`qlib/backtest/executor.py`)

| 类 | 说明 |
|-----|------|
| `BaseExecutor` | 基类：管理日历、账户更新、指标计算 |
| `NestedExecutor` | 嵌套执行器：内含子策略和子执行器，支持多频率嵌套 |
| `SimulatorExecutor` | 模拟执行器：串行或并行执行订单 |

**执行类型：**
- `TT_SERIAL = "serial"` — 串行执行（先卖后买）
- `TT_PARAL = "parallel"` — 并行执行

### 7.5 Account 账户 (`qlib/backtest/account.py`)

| 方法 | 说明 |
|------|------|
| `get_cash()` | 获取当前现金 |
| `update_order(order, trade_val, cost, trade_price)` | 更新订单 |
| `update_bar_end(...)` | 每步结束更新（持仓价格、组合指标、交易指标） |
| `get_portfolio_metrics()` | 获取历史组合指标 DataFrame |
| `get_trade_indicator()` | 获取交易指标 Indicator |

### 7.6 Position 持仓 (`qlib/backtest/position.py`)

| 类 | 说明 |
|-----|------|
| `Position` | 标准持仓：cash, stock amount/price/weight/count |
| `InfPosition` | 无限持仓：资金和数量无限，用于测试 |

**结算模式：**
- `ST_CASH = "cash"` — 现金结算
- `ST_NO = "None"` — 无结算

### 7.7 Decision 决策 (`qlib/backtest/decision.py`)

| 类 | 说明 |
|-----|------|
| `OrderDir` | 方向枚举：`SELL=0`, `BUY=1` |
| `Order` | 订单：stock_id, amount, direction, start_time, end_time, deal_amount, factor |
| `OrderHelper` | 订单创建辅助工具 |
| `BaseTradeDecision` | 决策基类：`get_decision()` / `update()` / `empty()` |
| `EmptyTradeDecision` | 空决策 |
| `TradeDecisionWO` | 带订单的决策 |
| `TradeDecisionWithDetails` | 带详细信息的决策 |

### 7.8 Signal 信号 (`qlib/backtest/signal.py`)

| 类 | 说明 |
|-----|------|
| `Signal` | 信号抽象基类：`get_signal(start_time, end_time)` |
| `SignalWCache` | 缓存信号：接收 pd.Series/DataFrame，自动重采样 |
| `ModelSignal` | 模型信号：用模型预测生成信号 |

**工厂函数：** `create_signal_from(obj)` — 从 Signal/Model+Dataset/List/Dict/Text/pd.Series/DataFrame 创建信号

### 7.9 Report 报告 (`qlib/backtest/report.py`)

| 类 | 说明 |
|-----|------|
| `PortfolioMetrics` | 组合指标：account_value, return, cost, turnover, bench |
| `Indicator` | 交易指标：FFR(完成率), PA(价格优势), POS(胜率) |

**FFR 聚合方式：** mean / amount_weighted / value_weighted

**PA 基准价：** twap / vwap

### 7.10 Brison 归因 (`qlib/backtest/profit_attribution.py`)

| 函数 | 说明 |
|------|------|
| `brinson_pa(positions, bench, group_field, ...)` | Brinson 归因分析 |

**输出：** RAA(资产配置超额) / RSS(个股选择超额) / RIN(交互超额) / RTotal(总超额)

### 7.11 基础设施 (`qlib/backtest/utils.py`)

| 类 | 说明 |
|-----|------|
| `TradeCalendarManager` | 交易日历管理：步进、时间范围、频率 |
| `CommonInfrastructure` | 公共基础设施：持有 trade_account + trade_exchange |
| `LevelInfrastructure` | 层级基础设施：持有 calendar + sub_infra + common_infra + executor |

---

## 八、策略层详解

### 8.1 策略基类 (`qlib/strategy/base.py`)

```
BaseStrategy
├── generate_trade_decision(execute_result) → BaseTradeDecision  # [抽象] 生成决策
├── reset(level_infra, common_infra, ...)                         # 重置
├── update_trade_decision(trade_decision, trade_calendar)         # 每步更新
├── post_upper_level_exe_step()                                   # 上层执行后钩子
├── post_exe_step(execute_result)                                 # 执行后钩子
│
├── RLStrategy          # RL策略基类
└── RLIntStrategy       # 带解释器的RL策略
    ├── interpret_state → policy.step → interpret_action
    └── 返回 TradeDecision
```

### 8.2 信号策略 (`contrib/strategy/signal_strategy.py`)

| 策略 | 说明 | 关键参数 |
|------|------|----------|
| `TopkDropoutStrategy` | TopK 选股 + Drop 调仓 | topk, n_drop, method_sell, method_buy, hold_thresh |
| `WeightStrategyBase` | 权重策略基类 | 通过 OrderGenerator 生成订单 |
| `EnhancedIndexingStrategy` | 增强指数策略 | riskmodel_root, market, turn_limit, optimizer_kwargs |

### 8.3 规则策略 (`contrib/strategy/rule_strategy.py`)

| 策略 | 说明 |
|------|------|
| `TWAPStrategy` | 时间加权平均：均匀拆分大单 |
| `SBBStrategyBase` | SBB 策略：选择更优 bar 执行 |
| `SBBStrategyEMA` | 基于 EMA 信号的 SBB |
| `ACStrategy` | Almgren-Chriss 自适应执行 |
| `RandomOrderStrategy` | 随机订单（测试用） |
| `FileOrderStrategy` | 从 CSV 读取订单 |

### 8.4 成本控制策略 (`contrib/strategy/cost_control.py`)

| 策略 | 说明 |
|------|------|
| `SoftTopkStrategy` | Soft Topk：确定性卖出 → 预算计算 → 比例买入 |

### 8.5 订单生成器 (`contrib/strategy/order_generator.py`)

| 生成器 | 说明 |
|--------|------|
| `OrderGenWInteract` | 带交互：使用交易日实际价格和可交易状态 |
| `OrderGenWOInteract` | 不带交互：使用预测日价格 |

### 8.6 组合优化器 (`contrib/strategy/optimizer/`)

| 优化器 | 方法 | 说明 |
|--------|------|------|
| `PortfolioOptimizer` | GMV/MVO/RP/INV | 全局最小方差/均值方差/风险平价/逆波动率 |
| `EnhancedIndexingOptimizer` | CVXPY 二次规划 | 控制跟踪误差，最大化超额收益 |

**PortfolioOptimizer 方法常量：**
- `OPT_GMV = "gmv"` / `OPT_MVO = "mvo"` / `OPT_RP = "rp"` / `OPT_INV = "inv"`

---

## 九、强化学习层详解

### 9.1 核心组件 (`qlib/rl/`)

| 组件 | 类 | 说明 |
|------|-----|------|
| 模拟器 | `Simulator` | `step(action)` / `get_state()` / `done()` |
| 奖励 | `Reward` / `RewardCombination` | `reward(simulator_state) → float` |
| 状态解释器 | `StateInterpreter` | 模拟器状态 → 策略观测 |
| 动作解释器 | `ActionInterpreter` | 策略动作 → 模拟器动作 |

### 9.2 订单执行 RL (`qlib/rl/order_execution/`)

| 组件 | 说明 |
|------|------|
| `SAOEState` | 单资产订单执行状态（order, cur_time, position, history_exec, metrics） |
| `SAOEMetrics` | 执行指标（stock_id, datetime, direction, market_volume, deal_amount, ffr, pa） |
| `PPO` | PPO 策略（PPOActor + PPOCritic） |
| `PAPenaltyReward` | PA 奖励 + 大单量惩罚 |
| `PPOReward` | 基于 VWAP/TWAP 比率的奖励（-1/0/1 三档） |
| `AllOne` | 全1动作策略（TWAP 基线） |
| `SingleAssetOrderExecutionSimple` | 单资产订单执行环境 |

### 9.3 RL 训练器 (`qlib/rl/trainer/`)

| 组件 | 说明 |
|------|------|
| `Trainer` | 以 "collect" 为迭代单位，支持 max_iters、val_every_n_iters、callbacks |
| `TrainingVessel` | 训练容器 |
| `Checkpoint` | 检查点保存 |
| `EarlyStopping` | 早停 |
| `MetricsWriter` | 指标写入 |

---

## 十、工作流层详解

### 10.1 实验管理 R (`qlib/workflow/__init__.py`)

`R` 是全局 `QlibRecorder` 实例，提供完整的实验管理 API：

| 方法 | 说明 |
|------|------|
| `R.start(experiment_name=...)` | 上下文管理器启动实验 |
| `R.end_exp(recorder_status)` | 手动结束实验 |
| `R.search_records(**kwargs)` | 搜索实验记录 |
| `R.list_experiments()` | 列出所有实验 |
| `R.list_recorders(experiment_name=...)` | 列出记录器 |
| `R.get_exp(experiment_name=...)` | 获取实验 |
| `R.delete_exp(experiment_name=...)` | 删除实验 |
| `R.get_recorder(recorder_id=...)` | 获取记录器 |
| `R.save_objects(**kwargs)` | 保存对象为产物 |
| `R.load_object(name)` | 加载产物对象 |
| `R.log_params(**kwargs)` | 记录参数 |
| `R.log_metrics(step=..., **kwargs)` | 记录指标 |
| `R.log_artifact(local_path, artifact_path)` | 记录文件产物 |
| `R.set_tags(**kwargs)` | 设置标签 |

### 10.2 记录模板 (`qlib/contrib/workflow/record_temp.py`)

| 模板 | 说明 |
|------|------|
| `SignalRecord` | 记录模型预测信号 |
| `SigAnaRecord` | 记录信号分析（IC/Rank IC） |
| `PortAnaRecord` | 记录回测分析结果 |
| `MultiSegRecord` | 多段信号记录 |
| `SignalMseRecord` | 信号 MSE 记录 |

---

## 十一、在线交易系统详解

### 11.1 系统架构 (`qlib/contrib/online/`)

| 组件 | 类 | 说明 |
|------|-----|------|
| 用户管理 | `UserManager` | 管理所有用户的账户/策略/模型持久化 |
| 用户实体 | `User` | 封装 account + strategy + model |
| 在线模型 | `ScoreFileModel` | 从 CSV 加载预测分数 |
| 操作入口 | `Operator` | 完整在线交易生命周期 |

### 11.2 Operator 操作流程

```
init() → add_user() → [generate() → execute() → update()] 循环 → show()
```

| 方法 | 说明 |
|------|------|
| `init(client, path, date)` | 初始化 UserManager，获取预测/交易日期 |
| `add_user(id, config, path, date)` | 添加用户 |
| `generate(date, path)` | 生成订单（预测 → 更新策略 → 生成决策） |
| `execute(date, exchange_config, path)` | 执行订单 |
| `update(date, path, type)` | 更新账户状态 |
| `simulate(id, config, ..., start, end)` | 完整在线模拟 |
| `show(id, path, bench)` | 展示风险分析报告 |

---

## 十二、评估与报告详解

### 12.1 评估函数 (`qlib/contrib/evaluate.py`)

| 函数 | 说明 |
|------|------|
| `risk_analysis(r, N, freq, mode)` | 风险分析：均值、标准差、年化收益、信息比率、最大回撤 |
| `indicator_analysis(df, method)` | 交易指标：PA、POS、FFR |
| `backtest_daily(...)` | 日频回测入口 |
| `long_short_backtest(pred, topk, ...)` | 多空策略回测 |

### 12.2 持仓评估 (`qlib/contrib/evaluate_portfolio.py`)

| 函数 | 说明 |
|------|------|
| `get_position_value(date, position)` | 持仓市值 |
| `get_daily_return_series_from_positions(positions)` | 日收益率序列 |
| `get_annual_return_from_positions(positions)` | 年化收益率 |
| `get_sharpe_ratio_from_return_series(r)` | 夏普比率 |
| `get_max_drawdown_from_series(r)` | 最大回撤 |
| `get_beta(r, b)` | Beta 系数 |
| `get_alpha(r, b, risk_free_rate)` | Alpha |
| `get_rank_ic(a, b)` | Rank IC (Spearman) |
| `get_normal_ic(a, b)` | Normal IC (Pearson) |

### 12.3 报告图表 (`qlib/contrib/report/`)

**图表基类 (`graph.py`)：**
- `BaseGraph` → `ScatterGraph` / `BarGraph` / `DistplotGraph` / `HeatmapGraph` / `HistogramGraph`
- `SubplotsGraph` — 子图组合器

**模型性能图表 (`analysis_model/`)：**
- `model_performance_graph()` — 分组收益 / IC 分析 / 自相关 / 换手率

**持仓分析图表 (`analysis_position/`)：**
- `report_graph()` — 综合回测报告（7行子图）
- `cumulative_return_graph()` — 买入/卖出/持仓累积收益
- `risk_analysis_graph()` — 风险指标柱状图 + 月度趋势
- `score_ic_graph()` — IC 和 Rank IC
- `rank_label_graph()` — 标签排名百分比

### 12.4 超参数调优 (`qlib/contrib/tuner/`)

| 组件 | 说明 |
|------|------|
| `QLibTuner` | 基于 Hyperopt 的调优器，TPE 算法 |
| `TopkAmountStrategySpace` | Topk 策略搜索空间 |
| `QLibDataLabelSpace` | 数据标签搜索空间 |

---

## 十三、滚动训练与元学习

### 13.1 滚动训练 (`qlib/contrib/rolling/base.py`)

| 类 | 说明 |
|-----|------|
| `Rolling` | 将单个任务按时间切分为多个滚动任务，使用 TrainerR 训练，RollingEnsemble 合并预测 |

### 13.2 Alpha 评估 (`qlib/contrib/eva/alpha.py`)

| 函数 | 说明 |
|------|------|
| `calc_long_short_prec(pred, label, quantile)` | 多空精度 |
| `calc_long_short_return(pred, label, quantile)` | 多空收益 |

---

## 十四、完整工作流

### 14.1 qrun 配置文件方式

```bash
qrun workflow_config.yaml
```

### 14.2 编程方式核心流程

```python
import qlib
from qlib.workflow import R

# 1. 初始化
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)

# 2. 构建数据集
dataset = DatasetH(handler=Alpha158(...), segments={...})

# 3. 训练模型
with R.start(experiment_name="exp"):
    model = LGBModel(...)
    model.fit(dataset)
    R.save_objects(model=model)

    # 4. 预测
    pred = model.predict(dataset)

    # 5. 回测
    strategy = TopkDropoutStrategy(topk=50, n_drop=5, signal=pred)
    report, positions = backtest_daily(start_time=..., end_time=..., strategy=strategy)

    # 6. 分析
    analysis = risk_analysis(report["return"])
```

---

## 十五、性能对比

| 存储方案 | 创建14特征数据集 (800股票, 2007-2020) |
|----------|--------------------------------------|
| HDF5 | 184.4s |
| MySQL | 365.3s |
| MongoDB | 253.6s |
| InfluxDB | 368.2s |
| **Qlib +E +D** | **7.4s** |

---

*文档版本: 基于 Qlib 源码全面梳理*
*最后更新: 2026-05-14*
