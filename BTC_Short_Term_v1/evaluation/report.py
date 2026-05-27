import os
import numpy as np
import pandas as pd


def generate_report(metrics_df, pm_df, trade_df, config=None):
    if metrics_df is None or len(metrics_df) == 0:
        return "ERROR: No backtest metrics available."

    row = metrics_df.iloc[0]
    init_cash = row["init_cash"]
    final_value = row["final_value"]
    total_return = row["total_return"]
    annual_return = row["annual_return"]
    sharpe = row["sharpe_ratio"]
    max_dd = row["max_drawdown"]
    win_rate = row["win_rate"]
    n_trades = int(row.get("n_trades", 0))
    n_periods = int(row.get("n_periods", 0))

    calmar = annual_return / abs(max_dd) if max_dd and abs(max_dd) > 0 else np.nan

    benchmark_return = np.nan
    if pm_df is not None and len(pm_df) > 0 and "bench" in pm_df.columns:
        bench_series = pm_df["bench"].dropna()
        if len(bench_series) > 0:
            benchmark_return = (1 + bench_series).prod() - 1

    annual_vol = np.nan
    if pm_df is not None and len(pm_df) > 0 and "return" in pm_df.columns:
        returns = pm_df["return"].dropna()
        if len(returns) > 0:
            annual_vol = returns.std() * np.sqrt(365 * 24)

    turnover_mean = 0
    if pm_df is not None and len(pm_df) > 0 and "turnover" in pm_df.columns:
        turnover_mean = pm_df["turnover"].mean()

    profit_loss_ratio = np.nan
    if pm_df is not None and len(pm_df) > 0 and "return" in pm_df.columns:
        rets = pm_df["return"].dropna()
        pos_rets = rets[rets > 0]
        neg_rets = rets[rets < 0]
        if len(pos_rets) > 0 and len(neg_rets) > 0:
            profit_loss_ratio = pos_rets.mean() / abs(neg_rets.mean())

    training_config = config.get("training", {}) if config else {}
    backtest_config = config.get("backtest", {}) if config else {}
    model_config = config.get("model", {}) if config else {}

    lines = []
    lines.append("=" * 60)
    lines.append("  BTCUSDT Short-Term Strategy v1 - Backtest Report")
    lines.append("  (Kaufman ER-Adjusted Label + Dynamic Threshold)")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Strategy Overview")
    lines.append("  Name:           BTC Short-Term v1 (Ultimate)")
    lines.append("  Instrument:     BTCUSDT")
    lines.append("  Frequency:      1 Hour (60min)")
    lines.append(f"  Model:          LightGBM (Huber Loss, alpha={model_config.get('alpha', 0.95)})")
    lines.append(f"  Features:       39 (35 base + 4 microstructure)")
    lines.append("  Label:          LABEL_ER_4H (Kaufman ER-Weighted 4H Return)")
    lines.append("  Strategy:       DynamicThresholdStrategy")
    lines.append(f"  Long Percentile:  {backtest_config.get('long_percentile', 0.85)} (top 15%)")
    lines.append(f"  Short Percentile: {backtest_config.get('short_percentile', 0.15)} (bottom 15%)")
    lines.append(f"  Rolling Window:   {backtest_config.get('rolling_window', 720)}h (30 days)")
    lines.append("")
    lines.append("Training Configuration")
    lines.append("  Method:     Expanding Window Rolling")
    lines.append(f"  Initial:    {training_config.get('train_start', 'N/A')} ~ {training_config.get('train_end', 'N/A')}")
    lines.append("  Rolling:    Quarterly (3-month step)")
    lines.append(f"  Valid:      {training_config.get('valid_window_months', 3)} months before test")
    lines.append("")
    lines.append("Backtest Performance")
    lines.append(f"  Period:           {training_config.get('test_start', 'N/A')} ~ {training_config.get('test_end', 'N/A')}")
    lines.append("")
    lines.append(f"  Initial Capital:      {init_cash:>12,.2f} USDT")
    lines.append(f"  Final Value:          {final_value:>12,.2f} USDT")
    lines.append(f"  Total Return:         {total_return*100:>11.2f}%")
    if not np.isnan(benchmark_return):
        lines.append(f"  Benchmark (B&H):      {benchmark_return*100:>11.2f}%")
    else:
        lines.append(f"  Benchmark (B&H):      {'N/A':>11}")
    lines.append(f"  Annual Return:        {annual_return*100:>11.2f}%")
    if not np.isnan(annual_vol):
        lines.append(f"  Annual Volatility:    {annual_vol*100:>11.2f}%")
    else:
        lines.append(f"  Annual Volatility:    {'N/A':>11}")
    lines.append(f"  Sharpe Ratio:         {sharpe:>11.4f}")
    lines.append(f"  Max Drawdown:         {max_dd*100:>11.2f}%")
    if not np.isnan(calmar):
        lines.append(f"  Calmar Ratio:         {calmar:>11.4f}")
    else:
        lines.append(f"  Calmar Ratio:         {'N/A':>11}")
    lines.append(f"  Win Rate:             {win_rate*100:>11.2f}%")
    if not np.isnan(profit_loss_ratio):
        lines.append(f"  Profit/Loss Ratio:    {profit_loss_ratio:>11.4f}")
    else:
        lines.append(f"  Profit/Loss Ratio:    {'N/A':>11}")
    lines.append(f"  Total Trades:         {n_trades:>11d}")
    if turnover_mean:
        lines.append(f"  Avg Turnover:         {turnover_mean*100:>11.2f}%")
    else:
        lines.append(f"  Avg Turnover:         {'N/A':>11}")
    lines.append("")
    lines.append("Trading Costs (Binance USDT Perpetual)")
    lines.append(f"  Open Fee:       {backtest_config.get('open_cost', 0)*100:.2f}% (Taker)")
    lines.append(f"  Close Fee:      {backtest_config.get('close_cost', 0)*100:.2f}% (Taker)")
    lines.append(f"  Slippage:       {backtest_config.get('slippage', 0)*100:.2f}%")
    lines.append(f"  Total/round:    {(backtest_config.get('open_cost',0)+backtest_config.get('close_cost',0)+backtest_config.get('slippage',0)*2)*100:.2f}%")
    lines.append("")
    lines.append("=" * 60)

    report_text = "\n".join(lines)

    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "report")
    os.makedirs(save_dir, exist_ok=True)
    report_path = os.path.join(save_dir, "backtest_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)

    return report_text
