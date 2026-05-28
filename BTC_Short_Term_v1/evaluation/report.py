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
    lines.append("  BTCUSDT Short-Term Strategy v2 - Backtest Report")
    lines.append("  (Schmitt Trigger + ATR Trailing Stop + Vol-Targeting)")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Strategy Overview")
    lines.append("  Name:           BTC Short-Term v2 (Enhanced)")
    lines.append("  Instrument:     BTCUSDT")
    lines.append("  Frequency:      1 Hour (60min)")
    lines.append(f"  Model:          LightGBM (Huber Loss, alpha={model_config.get('alpha', 0.95)})")
    lines.append(f"  Features:       39 (35 base + 4 microstructure)")
    lines.append("  Label:          LABEL_ER_4H (Kaufman ER-Weighted 4H Return)")
    lines.append("  Strategy:       DynamicThresholdStrategy (Enhanced)")
    lines.append("")
    lines.append("Schmitt Trigger (Hysteresis)")
    lines.append(f"  Entry Long:     {backtest_config.get('long_percentile', 0.95):.0%} percentile (strict)")
    lines.append(f"  Exit Long:      {backtest_config.get('exit_long_percentile', 0.50):.0%} percentile (relaxed)")
    lines.append(f"  Entry Short:    {backtest_config.get('short_percentile', 0.05):.0%} percentile")
    lines.append(f"  Exit Short:     {backtest_config.get('exit_short_percentile', 0.50):.0%} percentile")
    lines.append(f"  Rolling Window: {backtest_config.get('rolling_window', 720)}h")
    lines.append(f"  Pos Side:       {backtest_config.get('pos_side', 'long')}")
    lines.append("")
    lines.append("CTA Exit Mechanism")
    lines.append(f"  Max Hold Bars:  {backtest_config.get('max_hold_bars', 4)} (4h label alignment)")
    lines.append(f"  ATR Stop Mult:  {backtest_config.get('atr_stop_multiplier', 2.0)}x")
    lines.append(f"  Risk-Reward:    1:{backtest_config.get('risk_reward_ratio', 2.0)}")
    lines.append(f"  Partial TP:     {backtest_config.get('partial_tp_ratio', 0.50):.0%} at TP1")
    lines.append(f"  Cooldown:       {backtest_config.get('cooldown_bars', 2)} bars after exit")
    lines.append("")
    lines.append("Volatility-Targeting Position Sizing")
    lines.append(f"  Risk/Trade:     {backtest_config.get('risk_per_trade', 0.02):.0%} of capital")
    lines.append(f"  Max Position:   {backtest_config.get('position_ratio', 0.30):.0%} of capital")
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
    lines.append(f"  Annual Return:        {annual_return*100:>11.2f}%")
    if not np.isnan(annual_vol):
        lines.append(f"  Annual Volatility:    {annual_vol*100:>11.2f}%")
    lines.append(f"  Sharpe Ratio:         {sharpe:>11.4f}")
    lines.append(f"  Max Drawdown:         {max_dd*100:>11.2f}%")
    if not np.isnan(calmar):
        lines.append(f"  Calmar Ratio:         {calmar:>11.4f}")
    lines.append(f"  Win Rate:             {win_rate*100:>11.2f}%")
    if not np.isnan(profit_loss_ratio):
        lines.append(f"  Profit/Loss Ratio:    {profit_loss_ratio:>11.4f}")
    lines.append(f"  Total Trades:         {n_trades:>11d}")
    if turnover_mean:
        lines.append(f"  Avg Turnover:         {turnover_mean*100:>11.2f}%")
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