from qlib.data.dataset.handler import DataHandlerLP


def _get_config():
    ALPHA_35 = [
        ("KMID", "($close - $open) / $open"),
        ("KLEN", "($high - $low) / $open"),
        ("KUP", "($high - Greater($open, $close)) / $open"),
        ("KUP2", "($high - Greater($open, $close)) / ($high - $low + 1e-12)"),
        ("KLOW", "(Less($open, $close) - $low) / $open"),
        ("KLOW2", "(Less($open, $close) - $low) / ($high - $low + 1e-12)"),
        ("KSFT", "(2 * $close - $high - $low) / $open"),
        ("KSFT2", "(2 * $close - $high - $low) / ($high - $low + 1e-12)"),
        ("KMID2", "($close - $open) / ($high - $low + 1e-12)"),
        ("ROC_4", "Ref($close, 4) / $close - 1"),
        ("ROC_12", "Ref($close, 12) / $close - 1"),
        ("ROC_24", "Ref($close, 24) / $close - 1"),
        ("ROC_72", "Ref($close, 72) / $close - 1"),
        ("MA5_DEV", "$close / Mean($close, 5) - 1"),
        ("MA12_DEV", "$close / Mean($close, 12) - 1"),
        ("MA24_DEV", "$close / Mean($close, 24) - 1"),
        ("MA72_DEV", "$close / Mean($close, 72) - 1"),
        ("STD_5", "Std($close, 5) / $close - 1"),
        ("STD_12", "Std($close, 12) / $close - 1"),
        ("STD_24", "Std($close, 24) / $close - 1"),
        ("STD_72", "Std($close, 72) / $close - 1"),
        ("VOL_RATIO", "Std($close, 5) / (Std($close, 24) + 1e-12)"),
        ("VOLUME", "$volume / Mean($volume, 24) - 1"),
        ("CORR_12", "Corr($close, $volume, 12)"),
        ("CORR_24", "Corr($close, $volume, 24)"),
        ("RANGE_VOL", "(($high - $low) / $open) / (Mean($volume, 24) + 1e-12)"),
        ("SLOPE_12", "Slope($close, 12) / $close"),
        ("SLOPE_24", "Slope($close, 24) / $close"),
        ("RSQR_24", "Rsquare($close, 24)"),
        ("MAX_24", "Max($high, 24) / $close - 1"),
        ("MIN_24", "$close / Min($low, 24) - 1"),
        ("OPEN_0", "$open / $close - 1"),
        ("HIGH_0", "$high / $close - 1"),
        ("LOW_0", "$low / $close - 1"),
        ("VWAP_0", "$close / $close - 1"),
    ]

    LABEL_1H_EXPR = "Ref($close, -2) / Ref($close, -1) - 1"
    LABEL_1H_NAME = "LABEL_1H"

    return ALPHA_35, LABEL_1H_EXPR, LABEL_1H_NAME


class CryptoDataHandler(DataHandlerLP):
    def __init__(
        self,
        instruments="btcusdt",
        start_time=None,
        end_time=None,
        freq="60min",
        infer_processors=None,
        learn_processors=None,
        fit_start_time=None,
        fit_end_time=None,
    ):
        ALPHA_35, LABEL_1H_EXPR, LABEL_1H_NAME = _get_config()

        fields = [expr for _, expr in ALPHA_35]
        names = [name for name, _ in ALPHA_35]

        if infer_processors is None:
            infer_processors = [
                {"class": "RobustZScoreNorm", "kwargs": {
                    "fit_start_time": fit_start_time,
                    "fit_end_time": fit_end_time,
                    "clip_outlier": True,
                }},
                {"class": "Fillna", "kwargs": {"fill_value": 0}},
            ]
        if learn_processors is None:
            learn_processors = [
                {"class": "DropnaLabel"},
            ]

        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader={
                "class": "QlibDataLoader",
                "kwargs": {
                    "config": {
                        "feature": (fields, names),
                        "label": ([LABEL_1H_EXPR], [LABEL_1H_NAME]),
                    },
                    "freq": freq,
                },
            },
            infer_processors=infer_processors,
            learn_processors=learn_processors,
        )