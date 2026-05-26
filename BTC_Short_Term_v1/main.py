import sys
import os
import qlib
from qlib.constant import REG_CN

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    print("=" * 60)
    print("  BTCUSDT Short-Term Strategy v1")
    print("=" * 60)

    print("\n[1/5] Initializing Qlib...")
    qlib.init(provider_uri="/workspace/qlib_data", region=REG_CN)
    from BTC_Short_Term_v1.config import LGBM_CONFIG, DATA_CONFIG, TRAINING_CONFIG, BACKTEST_CONFIG

    print(f"      Data provider: {DATA_CONFIG['provider_uri']}")
    print(f"      Instrument: {DATA_CONFIG['instrument']}")
    print(f"      Frequency: {DATA_CONFIG['freq']}")

    print("\n[2/5] Building CryptoDataHandler...")
    from BTC_Short_Term_v1.features import CryptoDataHandler

    handler = CryptoDataHandler(
        instruments=DATA_CONFIG["instrument"],
        start_time=TRAINING_CONFIG["train_start"],
        end_time=TRAINING_CONFIG["test_end"],
        freq=DATA_CONFIG["freq"],
        fit_start_time=TRAINING_CONFIG["train_start"],
        fit_end_time=TRAINING_CONFIG["train_end"],
    )
    feat_cols = handler.get_cols("feature")
    label_cols = handler.get_cols("label")
    print(f"      Features: {len(feat_cols)}, Label: {label_cols[0] if label_cols else 'N/A'}")

    print("\n[3/5] Rolling Training (Quarterly Expanding Window)...")
    from BTC_Short_Term_v1.models import RollingTrainer

    trainer = RollingTrainer(
        handler=handler,
        model_config=LGBM_CONFIG,
        training_config=TRAINING_CONFIG,
    )
    predictions = trainer.run()
    print(f"      Predictions: {len(predictions)} rows")
    print(f"      Period: {predictions.index.get_level_values('datetime').min()} ~ {predictions.index.get_level_values('datetime').max()}")

    print("\n[4/5] Backtesting...")
    from BTC_Short_Term_v1.backtest import run_backtest

    metrics_df, pm_df, trade_df, port_metrics, ind_metrics = run_backtest(
        predictions=predictions,
        backtest_config=BACKTEST_CONFIG,
        start_time=TRAINING_CONFIG["test_start"],
        end_time=TRAINING_CONFIG["test_end"],
    )
    print(f"      Metrics computed: {len(metrics_df)} rows")

    print("\n[5/5] Generating Report...")
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

    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\nReport saved to: {report_path}")

    return report_text


if __name__ == "__main__":
    main()