import os
import pandas as pd
import numpy as np


class TradeAnalyzer:

    def __init__(self, trade_log_df, output_dir="."):
        self.trade_log = trade_log_df.copy() if trade_log_df is not None and len(trade_log_df) > 0 else pd.DataFrame()
        self.output_dir = output_dir

    def export_trade_log(self, filename="trade_behavior_log.csv"):
        if len(self.trade_log) == 0:
            print("[TradeAnalyzer] No trade records to export.")
            return None

        filepath = os.path.join(self.output_dir, filename)
        self.trade_log.to_csv(filepath, index=False)
        print(f"[TradeAnalyzer] Exported {len(self.trade_log)} trades to {filepath}")
        return filepath

    def generate_summary(self):
        if len(self.trade_log) == 0:
            return "No trade records available for summary."

        df = self.trade_log
        lines = []
        lines.append("=" * 90)
        lines.append("  Trade Behavior Analysis Summary")
        lines.append("=" * 90)
        lines.append("")

        lines.append(f"  Total Trades: {len(df)}")
        long_trades = df[df["Direction"] == "Long"]
        short_trades = df[df["Direction"] == "Short"]
        lines.append(f"  Long Trades: {len(long_trades)}  |  Short Trades: {len(short_trades)}")
        lines.append("")

        win_mask = df["PnL_Percent"] > 0
        win_rate = win_mask.mean() * 100
        lines.append(f"  Overall Win Rate: {win_rate:.2f}%")
        if len(long_trades) > 0:
            lr = (long_trades["PnL_Percent"] > 0).mean() * 100
            lines.append(f"  Long Win Rate: {lr:.2f}%")
        if len(short_trades) > 0:
            sr = (short_trades["PnL_Percent"] > 0).mean() * 100
            lines.append(f"  Short Win Rate: {sr:.2f}%")
        lines.append("")

        avg_pnl = df["PnL_Percent"].mean()
        med_pnl = df["PnL_Percent"].median()
        lines.append(f"  Avg PnL%: {avg_pnl:.4f}%  |  Median PnL%: {med_pnl:.4f}%")

        gross_profit = df.loc[win_mask, "PnL_Percent"].sum() if win_mask.any() else 0
        gross_loss = abs(df.loc[~win_mask, "PnL_Percent"].sum()) if (~win_mask).any() else 1e-8
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        lines.append(f"  Profit Factor: {pf:.4f}")
        lines.append("")

        avg_hold = df["Hold_Bars"].mean()
        lines.append(f"  Avg Hold Bars: {avg_hold:.1f}  |  Median Hold Bars: {df['Hold_Bars'].median():.0f}")
        lines.append("")

        if "Exit_Reason" in df.columns:
            lines.append("  Exit Reason Distribution:")
            reason_counts = df["Exit_Reason"].value_counts()
            for reason, cnt in reason_counts.items():
                reason_trades = df[df["Exit_Reason"] == reason]
                avg_r = reason_trades["PnL_Percent"].mean()
                wr = (reason_trades["PnL_Percent"] > 0).mean() * 100
                lines.append(f"    {reason:<16}: {cnt:>4} trades  |  Avg PnL%: {avg_r:>8.4f}%  |  Win Rate: {wr:.1f}%")
        lines.append("")

        if "MFE_Percent" in df.columns:
            avg_mfe = df["MFE_Percent"].mean()
            lines.append(f"  Avg MFE%: {avg_mfe:.4f}%")
            winners = df[win_mask]
            if len(winners) > 0:
                avg_mfe_win = winners["MFE_Percent"].mean()
                lines.append(f"  Avg MFE% (Winners): {avg_mfe_win:.4f}%")
                avg_pnl_win = winners["PnL_Percent"].mean()
                efficiency = avg_pnl_win / avg_mfe_win * 100 if avg_mfe_win > 0 else 0
                lines.append(f"  TP Efficiency (PnL/MFE): {efficiency:.1f}%  <- lower = TP too tight")

        lines.append("")
        lines.append("=" * 90)
        return "\n".join(lines)

    def generate_rolling_performance_matrix(self, pm_df, step_months=3):
        if len(self.trade_log) == 0:
            print("[TradeAnalyzer] No trade records for rolling matrix.")
            return pd.DataFrame()

        trade_df = self.trade_log.copy()
        if "Exit_Time" not in trade_df.columns:
            print("[TradeAnalyzer] Missing Exit_Time column.")
            return pd.DataFrame()

        trade_df["Exit_Time"] = pd.to_datetime(trade_df["Exit_Time"])
        trade_df["Window_Period"] = trade_df["Exit_Time"].apply(
            lambda t: f"{t.year}-Q{(t.month - 1) // 3 + 1}"
        )

        rows = []
        for window, group in trade_df.groupby("Window_Period"):
            total_trades = len(group)
            win_rate = (group["PnL_Percent"] > 0).mean() * 100 if total_trades > 0 else 0
            avg_hold = group["Hold_Bars"].mean()

            gross_profit = group.loc[group["PnL_Percent"] > 0, "PnL_Percent"].sum()
            gross_loss = abs(group.loc[group["PnL_Percent"] < 0, "PnL_Percent"].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

            window_return = group["PnL_Percent"].sum()

            rows.append({
                "Window_Period": window,
                "Total_Trades": total_trades,
                "Win_Rate": round(win_rate, 2),
                "Window_Return": round(window_return, 4),
                "Max_Drawdown": self._compute_window_drawdown(pm_df, window),
                "Avg_Hold_Bars": round(avg_hold, 1),
                "Profit_Factor": round(profit_factor, 4),
            })

        result_df = pd.DataFrame(rows)
        result_df = result_df.sort_values("Window_Period").reset_index(drop=True)
        return result_df

    def _compute_window_drawdown(self, pm_df, window_period):
        if pm_df is None or len(pm_df) == 0:
            return 0.0

        try:
            year_str, q_str = window_period.split("-Q")
            year = int(year_str)
            quarter = int(q_str)
            q_start_month = (quarter - 1) * 3 + 1
            q_end_month = q_start_month + 2
            start = pd.Timestamp(f"{year}-{q_start_month:02d}-01")
            if q_end_month > 12:
                end = pd.Timestamp(f"{year + 1}-01-01") - pd.Timedelta(days=1)
            else:
                end = pd.Timestamp(f"{year}-{q_end_month:02d}-01") - pd.Timedelta(days=1)

            pm_index = pm_df.index
            if isinstance(pm_index, pd.MultiIndex):
                pm_index = pm_index.get_level_values("datetime")

            mask = (pm_index >= start) & (pm_index <= end)
            window_account = pm_df.loc[mask, "account"] if "account" in pm_df.columns else pd.Series()

            if len(window_account) == 0:
                return 0.0

            cummax = window_account.expanding().max()
            drawdown = (window_account - cummax) / cummax
            return round(float(drawdown.min()) * 100, 4)
        except Exception:
            return 0.0

    def export_rolling_matrix(self, matrix_df, filename="rolling_performance_matrix.csv"):
        if matrix_df is None or len(matrix_df) == 0:
            print("[TradeAnalyzer] No rolling matrix data to export.")
            return None

        filepath = os.path.join(self.output_dir, filename)
        matrix_df.to_csv(filepath, index=False)
        print(f"[TradeAnalyzer] Exported rolling matrix ({len(matrix_df)} windows) to {filepath}")
        return filepath

    def generate_rolling_matrix_report(self, matrix_df):
        if matrix_df is None or len(matrix_df) == 0:
            return "No rolling performance data available."

        lines = []
        lines.append("=" * 110)
        lines.append("  Rolling Performance Matrix (Per-Quarter Trading Metrics)")
        lines.append("=" * 110)
        lines.append("")

        header = (
            f"{'Window':<10} {'Trades':>7} {'WinRate%':>9} {'Return%':>9} "
            f"{'MaxDD%':>9} {'AvgHold':>8} {'PF':>8}"
        )
        lines.append(header)
        lines.append("-" * 110)

        for _, row in matrix_df.iterrows():
            line = (
                f"{row['Window_Period']:<10} {row['Total_Trades']:>7} "
                f"{row['Win_Rate']:>8.2f}% {row['Window_Return']:>8.4f}% "
                f"{row['Max_Drawdown']:>8.4f}% {row['Avg_Hold_Bars']:>7.1f} "
                f"{row['Profit_Factor']:>8.4f}"
            )
            lines.append(line)

        lines.append("-" * 110)

        avg_trades = matrix_df["Total_Trades"].mean()
        avg_wr = matrix_df["Win_Rate"].mean()
        avg_ret = matrix_df["Window_Return"].mean()
        avg_dd = matrix_df["Max_Drawdown"].mean()
        avg_hold = matrix_df["Avg_Hold_Bars"].mean()
        avg_pf = matrix_df["Profit_Factor"].replace([float("inf")], np.nan).mean()

        lines.append(
            f"{'AVERAGE':<10} {avg_trades:>7.1f} {avg_wr:>8.2f}% {avg_ret:>8.4f}% "
            f"{avg_dd:>8.4f}% {avg_hold:>7.1f} {avg_pf:>8.4f}"
        )
        lines.append("")
        lines.append("=" * 110)
        return "\n".join(lines)
