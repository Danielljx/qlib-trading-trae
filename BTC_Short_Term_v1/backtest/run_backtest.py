import pandas as pd
import numpy as np
from qlib.backtest.executor import SimulatorExecutor
from qlib.strategy.base import BaseStrategy
from qlib.backtest.utils import CommonInfrastructure
from qlib.utils import init_instance_by_config
from qlib.backtest.account import Account
from qlib.backtest.exchange import Exchange
from qlib.backtest import backtest_loop
from qlib.backtest.decision import BaseTradeDecision
from qlib.backtest.utils import TradeCalendarManager


_orig_init = BaseTradeDecision.__init__
_orig_get_step_time = TradeCalendarManager.get_step_time


def _patched_init(self, strategy, trade_range=None):
    self.strategy = strategy
    try:
        self.start_time, self.end_time = strategy.trade_calendar.get_step_time()
    except IndexError:
        cal = strategy.trade_calendar
        idx = cal.start_index + cal.get_trade_step()
        self.start_time = cal._calendar[idx]
        self.end_time = self.start_time + pd.Timedelta(hours=1)
    self.total_step = None
    from qlib.backtest.decision import TradeRange
    if isinstance(trade_range, tuple):
        from qlib.backtest.decision import IdxTradeRange
        trade_range = IdxTradeRange(*trade_range)
    self.trade_range = trade_range


def _patched_get_step_time(self, trade_step=None, shift=0):
    if trade_step is None:
        trade_step = self.get_trade_step()
    calendar_index = self.start_index + trade_step - shift
    if calendar_index + 1 >= len(self._calendar):
        return self._calendar[calendar_index], self._calendar[calendar_index] + pd.Timedelta(hours=1)
    return self._calendar[calendar_index], self._calendar[calendar_index + 1]


def run_backtest(predictions, backtest_config, start_time, end_time):
    BaseTradeDecision.__init__ = _patched_init
    TradeCalendarManager.get_step_time = _patched_get_step_time

    try:
        return _run_backtest_impl(predictions, backtest_config, start_time, end_time)
    finally:
        BaseTradeDecision.__init__ = _orig_init
        TradeCalendarManager.get_step_time = _orig_get_step_time


def _run_backtest_impl(predictions, backtest_config, start_time, end_time):
    predictions = predictions.sort_index()
    signal_df = predictions.to_frame(name="score")

    freq = backtest_config["freq"]
    init_cash = float(backtest_config["account"])

    trade_exchange = Exchange(
        freq=freq,
        start_time=start_time,
        end_time=end_time,
        codes="all",
        limit_threshold=backtest_config["limit_threshold"],
        deal_price=backtest_config["deal_price"],
        open_cost=backtest_config["open_cost"],
        close_cost=backtest_config["close_cost"],
        min_cost=backtest_config["min_cost"],
        impact_cost=backtest_config.get("slippage", 0.0001),
    )

    trade_account = Account(
        init_cash=init_cash,
        position_dict={},
        freq=freq,
        benchmark_config={"benchmark": backtest_config.get("benchmark", None)},
        pos_type=backtest_config["pos_type"],
        port_metr_enabled=True,
    )

    common_infra = CommonInfrastructure(
        trade_account=trade_account,
        trade_exchange=trade_exchange,
    )

    strategy_config = {
        "class": "BTC_Short_Term_v1.strategies.ThresholdSignalStrategy",
        "module_path": "BTC_Short_Term_v1.strategies",
        "kwargs": {
            "signal": signal_df,
            "long_threshold": backtest_config["long_threshold"],
            "short_threshold": backtest_config["short_threshold"],
            "position_ratio": backtest_config["position_ratio"],
        },
    }

    trade_strategy = init_instance_by_config(strategy_config, accept_types=BaseStrategy)
    trade_strategy.reset_common_infra(common_infra)

    trade_executor = SimulatorExecutor(time_per_step=freq, generate_portfolio_metrics=True)
    trade_executor.reset_common_infra(common_infra)

    portfolio_metrics, indicator_metrics = backtest_loop(
        start_time=start_time,
        end_time=end_time,
        trade_strategy=trade_strategy,
        trade_executor=trade_executor,
    )

    pm_df = pd.DataFrame()
    if portfolio_metrics is not None:
        for freq_key, (df, info) in portfolio_metrics.items():
            if df is not None and len(df) > 0:
                pm_df = df
                break

    trade_df = _extract_trades_from_indicator(indicator_metrics)
    metrics_df = _compute_metrics(pm_df, init_cash, trade_df)

    return metrics_df, pm_df, trade_df, portfolio_metrics, indicator_metrics


def _extract_trades_from_indicator(indicator_metrics):
    records = []
    if indicator_metrics is None:
        return pd.DataFrame()

    for freq_key, (ind_df, indicator) in indicator_metrics.items():
        if ind_df is None or len(ind_df) == 0:
            continue
        for idx, row in ind_df.iterrows():
            deal_val = row.get("value", None)
            deal_amt = row.get("deal_amount", None)
            count = row.get("count", None)
            if deal_val is None:
                continue
            if isinstance(deal_val, (pd.Series, dict)):
                if isinstance(deal_val, pd.Series) and len(deal_val) > 0:
                    deal_val = deal_val.iloc[0]
                elif isinstance(deal_val, dict):
                    deal_val = list(deal_val.values())[0] if deal_val else 0
                else:
                    deal_val = 0
                if isinstance(deal_amt, pd.Series) and len(deal_amt) > 0:
                    deal_amt = deal_amt.iloc[0]
                elif isinstance(deal_amt, dict):
                    deal_amt = list(deal_amt.values())[0] if deal_amt else 0
                else:
                    deal_amt = 0
                if isinstance(count, pd.Series) and len(count) > 0:
                    count = count.iloc[0]
                elif isinstance(count, dict):
                    count = list(count.values())[0] if count else 0
                else:
                    count = 0
            tv = float(deal_val) if deal_val else 0
            if tv <= 0:
                continue
            records.append({
                "datetime": idx,
                "trade_value": tv,
                "deal_amount": float(deal_amt) if deal_amt else 0,
                "order_count": int(count) if count else 0,
            })

    return pd.DataFrame(records)


def _compute_metrics(pm_df, init_cash, trade_df):
    if pm_df is None or len(pm_df) == 0:
        return pd.DataFrame([{
            "init_cash": init_cash,
            "final_value": init_cash,
            "total_return": 0.0,
            "annual_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "n_trades": 0,
            "n_periods": 0,
        }])

    account_series = pm_df["account"]
    final_value = float(account_series.iloc[-1])
    total_return = (final_value / init_cash - 1)

    n_periods = len(account_series)
    hours_per_year = 365 * 24

    if n_periods > 0 and total_return > -1:
        annual_return = (1 + total_return) ** (hours_per_year / n_periods) - 1
    else:
        annual_return = total_return

    returns = account_series.pct_change().fillna(0)
    mean_ret = returns.mean()
    std_ret = returns.std()
    sharpe = (mean_ret / std_ret * np.sqrt(hours_per_year)) if std_ret > 0 else 0

    cummax = account_series.expanding().max()
    drawdown = (account_series - cummax) / cummax
    max_drawdown = float(drawdown.min())

    win_rate = float((returns > 0).sum() / n_periods) if n_periods > 0 else 0

    n_trades = len(trade_df) if trade_df is not None and len(trade_df) > 0 else 0

    result = {
        "init_cash": init_cash,
        "final_value": final_value,
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "n_trades": n_trades,
        "n_periods": n_periods,
    }

    return pd.DataFrame([result])