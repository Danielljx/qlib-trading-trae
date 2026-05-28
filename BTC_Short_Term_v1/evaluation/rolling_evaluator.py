import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from qlib.contrib.model.gbdt import LGBModel
from qlib.data.dataset import DatasetH


class RollingEvaluator:

    def __init__(self, handler, model_config, training_config):
        self.handler = handler
        self.model_config = model_config.copy()
        self.training_config = training_config

    def _generate_expanding_segments(self):
        train_start = pd.Timestamp(self.training_config["train_start"])
        test_start = pd.Timestamp(self.training_config["test_start"])
        test_end = pd.Timestamp(self.training_config["test_end"])
        step_months = self.training_config["rolling_step_months"]
        valid_months = self.training_config["valid_window_months"]

        segments = []
        current_test_start = test_start

        while current_test_start < test_end:
            current_test_end = min(
                current_test_start + pd.DateOffset(months=step_months) - pd.Timedelta(days=1),
                test_end,
            )
            valid_start = current_test_start - pd.DateOffset(months=valid_months)
            valid_end = current_test_start - pd.Timedelta(days=1)
            if valid_start < train_start:
                valid_start = current_test_start - pd.Timedelta(days=60)

            segments.append({
                "train": (train_start.strftime("%Y-%m-%d"), current_test_start.strftime("%Y-%m-%d")),
                "valid": (valid_start.strftime("%Y-%m-%d"), valid_end.strftime("%Y-%m-%d")),
                "test": (current_test_start.strftime("%Y-%m-%d"), current_test_end.strftime("%Y-%m-%d")),
            })
            current_test_start = current_test_end + pd.Timedelta(days=1)

        return segments

    def run(self):
        segments = self._generate_expanding_segments()
        results = []

        for i, seg in enumerate(segments):
            print(f"  Evaluating segment {i+1}/{len(segments)}: test={seg['test']}")

            dataset = DatasetH(
                handler=self.handler,
                segments={
                    "train": seg["train"],
                    "valid": seg["valid"],
                    "test": seg["test"],
                },
            )

            model = LGBModel(**self.model_config)
            model.fit(dataset)

            pred = model.predict(dataset, segment="test")
            label_df = dataset.prepare(segments="test", col_set="label", data_key=DataHandlerLP.DK_R)

            if pred is None or len(pred) == 0:
                continue

            pred_aligned, label_aligned = self._align_pred_label(pred, label_df)
            if len(pred_aligned) == 0:
                continue

            ic = pred_aligned.corr(label_aligned)
            rank_ic = pred_aligned.corr(label_aligned, method="spearman")

            pred_sign = (pred_aligned > 0).astype(int)
            label_sign = (label_aligned > 0).astype(int)
            direction_accuracy = (pred_sign == label_sign).mean()

            long_mask = pred_aligned > pred_aligned.quantile(0.85)
            short_mask = pred_aligned < pred_aligned.quantile(0.15)

            long_win = (label_aligned[long_mask] > 0).mean() if long_mask.sum() > 0 else np.nan
            short_win = (label_aligned[short_mask] < 0).mean() if short_mask.sum() > 0 else np.nan

            long_avg_ret = label_aligned[long_mask].mean() if long_mask.sum() > 0 else np.nan
            short_avg_ret = -label_aligned[short_mask].mean() if short_mask.sum() > 0 else np.nan

            decay_days = self._compute_signal_decay(pred_aligned, label_aligned)

            test_start = pd.Timestamp(seg["test"][0])
            test_end = pd.Timestamp(seg["test"][1])
            n_days = max((test_end - test_start).days, 1)

            results.append({
                "segment": f"{seg['test'][0]}~{seg['test'][1]}",
                "train_end": seg["train"][1],
                "n_samples": len(pred_aligned),
                "ic": round(ic, 4) if not np.isnan(ic) else None,
                "rank_ic": round(rank_ic, 4) if not np.isnan(rank_ic) else None,
                "direction_accuracy": round(direction_accuracy, 4),
                "long_win_rate": round(long_win, 4) if not np.isnan(long_win) else None,
                "short_win_rate": round(short_win, 4) if not np.isnan(short_win) else None,
                "long_avg_ret": round(long_avg_ret, 6) if not np.isnan(long_avg_ret) else None,
                "short_avg_ret": round(short_avg_ret, 6) if not np.isnan(short_avg_ret) else None,
                "signal_decay_days": decay_days,
                "n_days": n_days,
            })

        return pd.DataFrame(results)

    def _align_pred_label(self, pred, label_df):
        if isinstance(pred, pd.DataFrame):
            pred_series = pred.iloc[:, 0]
        else:
            pred_series = pred

        if isinstance(label_df, pd.DataFrame):
            label_series = label_df.iloc[:, 0]
        else:
            label_series = label_df

        if isinstance(pred_series.index, pd.MultiIndex):
            pred_series = pred_series.droplevel("instrument") if "instrument" in pred_series.index.names else pred_series
        if isinstance(label_series.index, pd.MultiIndex):
            label_series = label_series.droplevel("instrument") if "instrument" in label_series.index.names else label_series

        pred_series = pred_series.sort_index()
        label_series = label_series.sort_index()

        common_idx = pred_series.index.intersection(label_series.index)
        return pred_series.loc[common_idx], label_series.loc[common_idx]

    def _compute_signal_decay(self, pred, label, week_hours=168):
        if len(pred) < week_hours * 2:
            return len(pred) // 24

        n_weeks = len(pred) // week_hours
        weekly_ics = []

        for w in range(n_weeks):
            start = w * week_hours
            end = min((w + 1) * week_hours, len(pred))
            p = pred.iloc[start:end]
            l = label.iloc[start:end]
            if len(p) > 10:
                ic = p.corr(l)
                weekly_ics.append(ic)

        decay_days = len(pred) // 24
        for i, ic in enumerate(weekly_ics):
            if not np.isnan(ic) and ic <= 0:
                decay_days = (i + 1) * 7
                break

        return decay_days

    def generate_report(self, eval_df):
        if eval_df is None or len(eval_df) == 0:
            return "No evaluation data available."

        lines = []
        lines.append("=" * 100)
        lines.append("  Rolling Training Evaluation Report (Per-Window IC / Rank IC / Signal Decay)")
        lines.append("=" * 100)
        lines.append("")

        header = (
            f"{'Segment':<24} {'IC':>8} {'RankIC':>8} {'DirAcc':>8} "
            f"{'LongWR':>8} {'ShortWR':>8} {'LongRet':>10} {'ShortRet':>10} "
            f"{'Decay(d)':>10} {'N':>6}"
        )
        lines.append(header)
        lines.append("-" * 100)

        for _, row in eval_df.iterrows():
            ic_str = f"{row['ic']:.4f}" if row['ic'] is not None else "N/A"
            ric_str = f"{row['rank_ic']:.4f}" if row['rank_ic'] is not None else "N/A"
            da_str = f"{row['direction_accuracy']:.4f}" if row['direction_accuracy'] is not None else "N/A"
            lwr_str = f"{row['long_win_rate']:.4f}" if row['long_win_rate'] is not None else "N/A"
            swr_str = f"{row['short_win_rate']:.4f}" if row['short_win_rate'] is not None else "N/A"
            lret_str = f"{row['long_avg_ret']:.6f}" if row['long_avg_ret'] is not None else "N/A"
            sret_str = f"{row['short_avg_ret']:.6f}" if row['short_avg_ret'] is not None else "N/A"
            decay_str = f"{row['signal_decay_days']}"
            n_str = f"{row['n_samples']}"

            line = (
                f"{row['segment']:<24} {ic_str:>8} {ric_str:>8} {da_str:>8} "
                f"{lwr_str:>8} {swr_str:>8} {lret_str:>10} {sret_str:>10} "
                f"{decay_str:>10} {n_str:>6}"
            )
            lines.append(line)

        lines.append("-" * 100)

        avg_ic = eval_df["ic"].mean()
        avg_ric = eval_df["rank_ic"].mean()
        avg_da = eval_df["direction_accuracy"].mean()
        avg_lwr = eval_df["long_win_rate"].mean()
        avg_swr = eval_df["short_win_rate"].mean()

        lines.append(f"{'AVERAGE':<24} {avg_ic:>8.4f} {avg_ric:>8.4f} {avg_da:>8.4f} "
                      f"{avg_lwr:>8.4f} {avg_swr:>8.4f}")
        lines.append("")

        ic_positive = (eval_df["ic"].dropna() > 0).sum()
        ic_total = eval_df["ic"].dropna().shape[0]
        lines.append(f"IC > 0 windows: {ic_positive}/{ic_total} ({ic_positive/ic_total*100:.1f}%)")

        avg_decay = eval_df["signal_decay_days"].mean()
        lines.append(f"Average signal decay: {avg_decay:.0f} days")

        if avg_lwr is not None and avg_swr is not None:
            if avg_lwr > avg_swr + 0.05:
                lines.append("")
                lines.append("WARNING: Long win rate significantly higher than short win rate.")
                lines.append("  Recommendation: Consider switching to LONG-ONLY mode.")

        lines.append("")
        lines.append("=" * 100)

        return "\n".join(lines)


from qlib.data.dataset.handler import DataHandlerLP