import sys
import os
import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data import D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    print("=" * 60)
    print("  BTCUSDT Short-Term Strategy v3")
    print("  Sliding Window (24mo) + Monthly Step + MA120 Trend Filter")
    print("=" * 60)

    print("\n[1/7] Initializing Qlib...")
    qlib.init(provider_uri="/workspace/BTC_Short_Term_v1/btcusdt_1h_full", region=REG_CN)
    from BTC_Short_Term_v1.config import LGBM_CONFIG, DATA_CONFIG, TRAINING_CONFIG, BACKTEST_CONFIG

    print(f"      Data provider: {DATA_CONFIG['provider_uri']}")
    print(f"      Rolling type: {TRAINING_CONFIG.get('rolling_type', 'expanding')}")
    print(f"      Step: {TRAINING_CONFIG.get('rolling_step_months', 3)} month(s)")
    print(f"      Train window: {TRAINING_CONFIG.get('train_window_months', 'N/A')} months")
    print(f"      Trend filter: {BACKTEST_CONFIG.get('use_trend_filter', False)}")

    print("\n[2/7] Building CryptoDataHandler...")
    from BTC_Short_Term_v1.features import CryptoDataHandler

    handler = CryptoDataHandler(
        instruments=DATA_CONFIG["instrument"],
        start_time=TRAINING_CONFIG["train_start"],
        end_time=TRAINING_CONFIG["test_end"],
        freq=DATA_CONFIG["freq"],
        fit_start_time=TRAINING_CONFIG["train_start"],
        fit_end_time=TRAINING_CONFIG["test_start"],
    )
    feat_cols = handler.get_cols("feature")
    label_cols = handler.get_cols("label")
    print(f"      Features: {len(feat_cols)}, Label: {label_cols[0] if label_cols else 'N/A'}")

    print("\n[3/7] Rolling Training (Monthly Sliding Window, 24mo)...")
    from BTC_Short_Term_v1.models import RollingTrainer

    trainer = RollingTrainer(
        handler=handler,
        model_config=LGBM_CONFIG,
        training_config=TRAINING_CONFIG,
    )
    predictions = trainer.run()
    print(f"      Predictions: {len(predictions)} rows")
    print(f"      Period: {predictions.index.get_level_values('datetime').min()} ~ {predictions.index.get_level_values('datetime').max()}")

    print("\n[4/7] Computing ATR + MA120 data for strategy...")
    atr_df = D.features(
        instruments=D.instruments("all"),
        fields=["Mean($high - $low, 24)"],
        start_time=TRAINING_CONFIG["test_start"],
        end_time=TRAINING_CONFIG["test_end"],
        freq="60min",
    )
    atr_series = atr_df.iloc[:, 0]
    if isinstance(atr_series.index, pd.MultiIndex):
        atr_series = atr_series.droplevel("instrument")
    atr_series = atr_series.sort_index().dropna()
    print(f"      ATR data: {len(atr_series)} rows, mean={atr_series.mean():.2f}")

    ma_period = BACKTEST_CONFIG.get("trend_filter_ma_period", 120)
    close_df = D.features(
        instruments=D.instruments("all"),
        fields=["$close"],
        start_time=TRAINING_CONFIG["test_start"],
        end_time=TRAINING_CONFIG["test_end"],
        freq="60min",
    )
    close_series = close_df.iloc[:, 0]
    if isinstance(close_series.index, pd.MultiIndex):
        close_series = close_series.droplevel("instrument")
    close_series = close_series.sort_index()
    ma_series = close_series.rolling(window=ma_period, min_periods=ma_period).mean()
    ma_series = ma_series.dropna()
    print(f"      MA{ma_period} data: {len(ma_series)} rows, latest={ma_series.iloc[-1]:.2f}")

    print("\n[5/7] Backtesting (Sliding Window + Trend Filter)...")
    from BTC_Short_Term_v1.backtest import run_backtest

    metrics_df, pm_df, trade_df, port_metrics, ind_metrics, trade_log, filter_stats = run_backtest(
        predictions=predictions,
        backtest_config=BACKTEST_CONFIG,
        start_time=TRAINING_CONFIG["test_start"],
        end_time=TRAINING_CONFIG["test_end"],
        atr_series=atr_series,
        ma_series=ma_series,
    )
    print(f"      Metrics computed: {len(metrics_df)} rows")
    print(f"      Trade log entries: {len(trade_log)}")

    print("\n[6/7] Trade Behavior Analysis...")
    from BTC_Short_Term_v1.evaluation.trade_analyzer import TradeAnalyzer

    output_dir = os.path.dirname(os.path.abspath(__file__))

    analyzer = TradeAnalyzer(trade_log, output_dir=output_dir, filter_stats=filter_stats)

    trade_csv_path = analyzer.export_trade_log()
    trade_summary = analyzer.generate_summary()
    print("\n" + trade_summary)

    step_months = TRAINING_CONFIG.get("rolling_step_months", 1)
    rolling_matrix = analyzer.generate_rolling_performance_matrix(pm_df, step_months=step_months)
    if rolling_matrix is not None and len(rolling_matrix) > 0:
        matrix_csv_path = analyzer.export_rolling_matrix(rolling_matrix)
        matrix_report = analyzer.generate_rolling_matrix_report(rolling_matrix)
        print("\n" + matrix_report)

    print("\n[7/7] Generating Reports...")
    from BTC_Short_Term_v1.evaluation import generate_report

    report_text = generate_report(
        metrics_df=metrics_df,
        pm_df=pm_df,
        trade_df=trade_df,
        config={
            "model": LGBM_CONFIG,
            "training": TRAINING_CONFIG,
            "backtest": BACKTEST_CONFIG,
        },
    )

    print("\n" + "=" * 60)
    print(report_text)
    print("=" * 60)

    report_path = os.path.join(output_dir, "backtest_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\nReport saved to: {report_path}")

    full_report_path = os.path.join(output_dir, "full_analysis_report.txt")
    with open(full_report_path, "w") as f:
        f.write(report_text)
        f.write("\n\n")
        f.write(trade_summary)
        f.write("\n\n")
        if rolling_matrix is not None and len(rolling_matrix) > 0:
            f.write(matrix_report)
    print(f"Full analysis report saved to: {full_report_path}")

    return report_text


if __name__ == "__main__":
    main()
