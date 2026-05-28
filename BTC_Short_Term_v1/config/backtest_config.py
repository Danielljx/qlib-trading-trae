BACKTEST_CONFIG: dict = {
    "start_time": "2021-01-01",
    "end_time": "2026-05-23",
    "init_cash": 100000,
    "pos_type": "Position",
    "freq": "60min",
    "codes": "all",
    "deal_price": "close",

    "long_percentile": 0.90,
    "short_percentile": 0.10,
    "exit_long_percentile": 0.20,
    "exit_short_percentile": 0.80,
    "rolling_window": 720,
    "position_ratio": 0.40,
    "pos_side": "long",

    "max_hold_bars": 8,
    "atr_stop_multiplier": 2.0,
    "risk_reward_ratio": 2.0,
    "partial_tp_ratio": 0.50,
    "risk_per_trade": 0.02,
    "cooldown_bars": 2,

    "smart_hold_extension": True,
    "smart_hold_atr_threshold": 1.0,

    "use_trend_filter": True,
    "trend_filter_ma_period": 120,
    "trend_filter_ma_fast_period": 20,

    "adx_threshold": 25,

    "limit_threshold": None,
    "open_cost": 0.0004,
    "close_cost": 0.0004,
    "min_cost": 0,
    "slippage": 0.0001,
}
