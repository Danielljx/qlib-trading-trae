BACKTEST_CONFIG: dict = {
    "start_time": "2021-01-01",
    "end_time": "2026-05-23",
    "init_cash": 100000,
    "pos_type": "Position",
    "freq": "60min",
    "codes": "all",
    "deal_price": "close",

    "long_percentile": 0.95,
    "short_percentile": 0.05,
    "rolling_window": 720,
    "position_ratio": 0.40,

    "limit_threshold": None,
    "open_cost": 0.0004,
    "close_cost": 0.0004,
    "min_cost": 0,
    "slippage": 0.0001,
}