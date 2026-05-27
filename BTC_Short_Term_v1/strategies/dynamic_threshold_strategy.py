import numpy as np
import pandas as pd
from qlib.contrib.strategy.signal_strategy import BaseSignalStrategy
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO


class DynamicThresholdStrategy(BaseSignalStrategy):
    def __init__(
        self,
        *,
        signal=None,
        long_percentile=0.85,
        short_percentile=0.15,
        rolling_window=720,
        position_ratio=0.95,
        **kwargs,
    ):
        self._raw_signal = signal
        self.long_pct = long_percentile
        self.short_pct = short_percentile
        self.window = rolling_window
        self.position_ratio = position_ratio

        super().__init__(signal=signal, **kwargs)

        self._calculate_dynamic_thresholds()

    def _extract_instrument(self, sig_series):
        if not isinstance(sig_series.index, pd.MultiIndex):
            return sig_series
        inst_values = sig_series.index.get_level_values('instrument').unique().tolist()
        if len(inst_values) == 1:
            return sig_series.droplevel('instrument')
        for name in ['BTCUSDT', 'btcusdt']:
            if name in inst_values:
                return sig_series.xs(name, level='instrument')
        return sig_series.droplevel('instrument')

    def _get_instrument_code(self, pred_score):
        if isinstance(pred_score.index, pd.MultiIndex):
            codes = list(pred_score.index.get_level_values("instrument").unique())
            return codes[0] if codes else 'BTCUSDT'
        if isinstance(pred_score.index, pd.Index) and pred_score.index.name == 'instrument':
            return pred_score.index[0] if len(pred_score) > 0 else 'BTCUSDT'
        return 'BTCUSDT'

    def _calculate_dynamic_thresholds(self):
        sig_df = self._raw_signal

        if isinstance(sig_df, pd.DataFrame):
            sig_series = sig_df.iloc[:, 0]
        else:
            sig_series = sig_df

        sig_series = self._extract_instrument(sig_series)

        self.long_thresh_series = sig_series.rolling(
            window=self.window,
            min_periods=24,
        ).quantile(self.long_pct)

        self.short_thresh_series = sig_series.rolling(
            window=self.window,
            min_periods=24,
        ).quantile(self.short_pct)

    def _get_current_pred(self, pred_score):
        if isinstance(pred_score, pd.DataFrame):
            if len(pred_score.columns) > 1:
                val = pred_score.iloc[-1, -1]
            else:
                val = pred_score.iloc[-1, 0]
        else:
            val = pred_score.iloc[-1]
        return float(val)

    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)

        pred_start_time, pred_end_time = trade_start_time, trade_end_time
        try:
            pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        except (IndexError, Exception):
            pass

        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)

        if pred_score is None or (hasattr(pred_score, '__len__') and len(pred_score) == 0):
            return TradeDecisionWO([], self)

        current_pred = self._get_current_pred(pred_score)
        code = self._get_instrument_code(pred_score)

        current_time = trade_start_time

        try:
            curr_long_th = self.long_thresh_series.loc[current_time]
            curr_short_th = self.short_thresh_series.loc[current_time]
        except (KeyError, TypeError):
            try:
                idx = self.long_thresh_series.index.get_indexer([current_time], method='pad')
                if idx[0] >= 0:
                    curr_long_th = self.long_thresh_series.iloc[idx[0]]
                    curr_short_th = self.short_thresh_series.iloc[idx[0]]
                else:
                    return TradeDecisionWO([], self)
            except Exception:
                return TradeDecisionWO([], self)

        if pd.isna(curr_long_th) or pd.isna(curr_short_th):
            return TradeDecisionWO([], self)

        current_position = self.trade_position
        cash = current_position.get_cash()
        current_amount = current_position.get_stock_amount(code)

        target_weight = 0.0
        if current_pred > curr_long_th:
            target_weight = self.position_ratio
        elif current_pred < curr_short_th:
            target_weight = -self.position_ratio

        order_list = []

        if abs(target_weight) < 0.01:
            if abs(current_amount) > 0:
                close_dir = OrderDir.SELL if current_amount > 0 else OrderDir.BUY
                order_list.append(Order(
                    stock_id=code,
                    amount=abs(current_amount),
                    direction=close_dir,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                ))
            return TradeDecisionWO(order_list, self)

        if (target_weight > 0 and current_amount > 0) or \
           (target_weight < 0 and current_amount < 0):
            return TradeDecisionWO([], self)

        rate = self.trade_exchange.get_deal_price(
            stock_id=code,
            start_time=trade_start_time,
            end_time=trade_end_time,
            direction=OrderDir.BUY if target_weight > 0 else OrderDir.SELL,
        )
        if rate is None or rate <= 0:
            return TradeDecisionWO([], self)

        post_close_cash = cash
        if abs(current_amount) > 0:
            close_dir = OrderDir.SELL if current_amount > 0 else OrderDir.BUY
            order_list.append(Order(
                stock_id=code,
                amount=abs(current_amount),
                direction=close_dir,
                start_time=trade_start_time,
                end_time=trade_end_time,
            ))
            if current_amount > 0:
                post_close_cash = cash + abs(current_amount) * rate * (1 - 0.0004)
            else:
                post_close_cash = cash - abs(current_amount) * rate * (1 + 0.0004)

        target_value = post_close_cash * abs(target_weight)
        trade_amount = target_value / rate
        factor = self.trade_exchange.get_factor(code, trade_start_time, trade_end_time)
        trade_amount = self.trade_exchange.round_amount_by_trade_unit(trade_amount, factor)

        if trade_amount > 0:
            direction = OrderDir.BUY if target_weight > 0 else OrderDir.SELL
            order_list.append(Order(
                stock_id=code,
                amount=trade_amount,
                direction=direction,
                start_time=trade_start_time,
                end_time=trade_end_time,
            ))

        return TradeDecisionWO(order_list, self)
