import pandas as pd
from qlib.contrib.model.gbdt import LGBModel
from qlib.data.dataset import DatasetH


class RollingTrainer:
    def __init__(self, handler, model_config, training_config):
        self.handler = handler
        self.model_config = model_config.copy()
        self.training_config = training_config

    def _generate_expanding_segments(self):
        train_start = pd.Timestamp(self.training_config["train_start"])
        train_end = pd.Timestamp(self.training_config["train_end"])
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

            expanding_train_end = current_test_start - pd.Timedelta(days=1)
            valid_start = expanding_train_end - pd.DateOffset(months=valid_months) + pd.Timedelta(days=1)
            valid_end = expanding_train_end

            if valid_start < train_start:
                valid_start = train_start

            segments.append({
                "train": (train_start.strftime("%Y-%m-%d"), expanding_train_end.strftime("%Y-%m-%d")),
                "valid": (valid_start.strftime("%Y-%m-%d"), valid_end.strftime("%Y-%m-%d")),
                "test": (current_test_start.strftime("%Y-%m-%d"), current_test_end.strftime("%Y-%m-%d")),
            })

            current_test_start = current_test_end + pd.Timedelta(days=1)

        return segments

    def run(self):
        segments = self._generate_expanding_segments()
        print(f"Rolling segments: {len(segments)}")

        all_pred = []

        for i, seg in enumerate(segments):
            print(f"Segment {i+1}/{len(segments)}: train={seg['train']}, valid={seg['valid']}, test={seg['test']}")

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
            all_pred.append(pred)

        if all_pred:
            result = pd.concat(all_pred).sort_index()
            result = result[~result.index.duplicated(keep="first")]
            return result
        return pd.Series(dtype=float)