import os
import pandas as pd
import numpy as np


class TradeAnalyzer:

    def __init__(self, trade_log_df, output_dir=".", filter_stats=None):
        self.trade_log = trade_log_df.copy() if trade_log_df is not None and len(trade_log_df) > 0 else pd.DataFrame()
        self.output_dir = output_dir
        self.filter_stats = filter_stats or {}

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

        lines.append(f"  Total Trade Records: {len(df)}")

        if "Close_Ratio" in df.columns:
            unique_trades = df.groupby(["Entry_Time", "Direction"]).first().reset_index()
            lines.append(f"  Unique Positions: {len(unique_trades)}")
        else:
            unique_trades = df

        long_trades = df[df["Direction"] == "Long"]
        short_trades = df[df["Direction"] == "Short"]
        lines.append(f"  Long Records: {len(long_trades)}  |  Short Records: {len(short_trades)}")
        lines.append("")

        win_mask = df["PnL_Percent"] > 0
        win_rate = win_mask.mean() * 100
        lines.append(f"  Per-Record Win Rate: {win_rate:.2f}%")

        if "Weighted_PnL" in df.columns:
            total_weighted_pnl = df["Weighted_PnL"].sum()
            lines.append(f"  Total Weighted PnL%: {total_weighted_pnl:.4f}%")
            lines.append(f"  Avg Weighted PnL%: {df['Weighted_PnL'].mean():.4f}%")

            weighted_gross_profit = df.loc[win_mask, "Weighted_PnL"].sum() if win_mask.any() else 0
            weighted_gross_loss = abs(df.loc[~win_mask, "Weighted_PnL"].sum()) if (~win_mask).any() else 1e-8
            weighted_pf = weighted_gross_profit / weighted_gross_loss if weighted_gross_loss > 0 else float("inf")
            lines.append(f"  Weighted Profit Factor: {weighted_pf:.4f}")
        else:
            avg_pnl = df["PnL_Percent"].mean()
            med_pnl = df["PnL_Percent"].median()
            lines.append(f"  Avg PnL%: {avg_pnl:.4f}%  |  Median PnL%: {med_pnl:.4f}%")

            gross_profit = df.loc[win_mask, "PnL_Percent"].sum() if win_mask.any() else 0
            gross_loss = abs(df.loc[~win_mask, "PnL_Percent"].sum()) if (~win_mask).any() else 1e-8
            pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
            lines.append(f"  Profit Factor (unweighted): {pf:.4f}")

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
                if "Close_Ratio" in reason_trades.columns:
                    avg_cr = reason_trades["Close_Ratio"].mean()
                    lines.append(f"    {reason:<20}: {cnt:>4} records | Avg PnL%: {avg_r:>8.4f}% | WinRate: {wr:.1f}% | Avg CloseRatio: {avg_cr:.2f}")
                else:
                    lines.append(f"    {reason:<20}: {cnt:>4} records | Avg PnL%: {avg_r:>8.4f}% | WinRate: {wr:.1f}%")
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
                lines.append(f"  TP Efficiency (PnL/MFE): {efficiency:.1f}%")

        lines.append("")

        if self.filter_stats:
            filtered_long = self.filter_stats.get("filtered_long", 0)
            filtered_short = self.filter_stats.get("filtered_short", 0)
            total_filtered = filtered_long + filtered_short
            if total_filtered > 0:
                lines.append("  Trend Filter Statistics:")
                lines.append(f"    Filtered Long Signals:  {filtered_long}")
                lines.append(f"    Filtered Short Signals: {filtered_short}")
                lines.append(f"    Total Filtered Signals: {total_filtered}")
                lines.append(f"    Trend Filter Exits:     {self.filter_stats.get('trend_filter_exits', 0)}")
                lines.append(f"    Trend Filter Protected: {self.filter_stats.get('trend_filter_protected', 0)}")

        lines.append("")
        lines.append("=" * 90)
        return "\n".join(lines)

    def generate_rolling_performance_matrix(self, pm_df, step_months=1):
        if len(self.trade_log) == 0:
            print("[TradeAnalyzer] No trade records for rolling matrix.")
            return pd.DataFrame()

        trade_df = self.trade_log.copy()
        if "Exit_Time" not in trade_df.columns:
            print("[TradeAnalyzer] Missing Exit_Time column.")
            return pd.DataFrame()

        trade_df["Exit_Time"] = pd.to_datetime(trade_df["Exit_Time"])

        if step_months == 1:
            trade_df["Window_Period"] = trade_df["Exit_Time"].dt.strftime("%Y-%m")
        else:
            trade_df["Window_Period"] = trade_df["Exit_Time"].apply(
                lambda t: f"{t.year}-Q{(t.month - 1) // 3 + 1}"
            )

        rows = []
        for window, group in trade_df.groupby("Window_Period"):
            total_records = len(group)

            unique_positions = group.groupby(["Entry_Time", "Direction"]).first().reset_index()
            total_trades = len(unique_positions)

            win_rate = (unique_positions["PnL_Percent"] > 0).mean() * 100 if total_trades > 0 else 0
            avg_hold = group["Hold_Bars"].mean()

            pnl_col = "Weighted_PnL" if "Weighted_PnL" in group.columns else "PnL_Percent"
            gross_profit = group.loc[group[pnl_col] > 0, pnl_col].sum()
            gross_loss = abs(group.loc[group[pnl_col] < 0, pnl_col].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

            account_return = self._compute_window_account_return(pm_df, window, step_months)

            rows.append({
                "Window_Period": window,
                "Total_Trades": total_trades,
                "Total_Records": total_records,
                "Win_Rate": round(win_rate, 2),
                "Window_Return": round(account_return, 4),
                "Weighted_PnL_Sum": round(group[pnl_col].sum(), 4) if pnl_col in group.columns else 0,
                "Max_Drawdown": self._compute_window_drawdown(pm_df, window, step_months),
                "Avg_Hold_Bars": round(avg_hold, 1),
                "Profit_Factor": round(profit_factor, 4),
            })

        result_df = pd.DataFrame(rows)
        result_df = result_df.sort_values("Window_Period").reset_index(drop=True)
        return result_df

    def _compute_window_account_return(self, pm_df, window_period, step_months=1):
        if pm_df is None or len(pm_df) == 0:
            return 0.0

        try:
            if step_months == 1:
                year = int(window_period[:4])
                month = int(window_period[5:7])
                start = pd.Timestamp(f"{year}-{month:02d}-01")
                if month == 12:
                    end = pd.Timestamp(f"{year + 1}-01-01") - pd.Timedelta(days=1)
                else:
                    end = pd.Timestamp(f"{year}-{month + 1:02d}-01") - pd.Timedelta(days=1)
            else:
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

            if len(window_account) < 2:
                return 0.0

            start_val = float(window_account.iloc[0])
            end_val = float(window_account.iloc[-1])

            if start_val <= 0:
                return 0.0

            return round((end_val / start_val - 1) * 100, 4)
        except Exception:
            return 0.0

    def _compute_window_drawdown(self, pm_df, window_period, step_months=1):
        if pm_df is None or len(pm_df) == 0:
            return 0.0

        try:
            if step_months == 1:
                year = int(window_period[:4])
                month = int(window_period[5:7])
                start = pd.Timestamp(f"{year}-{month:02d}-01")
                if month == 12:
                    end = pd.Timestamp(f"{year + 1}-01-01") - pd.Timedelta(days=1)
                else:
                    end = pd.Timestamp(f"{year}-{month + 1:02d}-01") - pd.Timedelta(days=1)
            else:
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
        lines.append("=" * 120)
        lines.append("  Rolling Performance Matrix (Account-Level Returns from Equity Curve)")
        lines.append("=" * 120)
        lines.append("")

        header = (
            f"{'Window':<10} {'Trades':>7} {'WinRate%':>9} {'AcctRet%':>9} "
            f"{'MaxDD%':>9} {'AvgHold':>8} {'PF':>8}"
        )
        lines.append(header)
        lines.append("-" * 120)

        for _, row in matrix_df.iterrows():
            line = (
                f"{row['Window_Period']:<10} {row['Total_Trades']:>7} "
                f"{row['Win_Rate']:>8.2f}% {row['Window_Return']:>8.4f}% "
                f"{row['Max_Drawdown']:>8.4f}% {row['Avg_Hold_Bars']:>7.1f} "
                f"{row['Profit_Factor']:>8.4f}"
            )
            lines.append(line)

        lines.append("-" * 120)

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

        positive_months = (matrix_df["Window_Return"] > 0).sum()
        total_months = len(matrix_df)
        lines.append(f"  Positive Return Months: {positive_months}/{total_months} ({positive_months/total_months*100:.1f}%)")

        cum_return = 1.0
        for _, row in matrix_df.iterrows():
            cum_return *= (1 + row["Window_Return"] / 100)
        lines.append(f"  Compounded Account Return: {(cum_return - 1) * 100:.4f}%")

        lines.append("")
        lines.append("=" * 120)
        return "\n".join(lines)
