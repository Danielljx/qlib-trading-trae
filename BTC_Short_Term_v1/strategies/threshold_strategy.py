import numpy as np
import pandas as pd
from qlib.contrib.strategy.signal_strategy import BaseSignalStrategy
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO


class ThresholdSignalStrategy(BaseSignalStrategy):
    def __init__(
        self,
        *,
        long_threshold=0.001,
        short_threshold=-0.001,
        position_ratio=0.95,
        hold_thresh=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.position_ratio = position_ratio
        self.hold_thresh = hold_thresh
        self._last_trade_step = -999

    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)

        pred_start_time, pred_end_time = trade_start_time, trade_end_time
        try:
            pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        except IndexError:
            pass

        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)

        if isinstance(pred_score, pd.DataFrame):
            pred_score = pred_score.iloc[:, 0]

        if pred_score is None or len(pred_score) == 0:
            return TradeDecisionWO([], self)

        current_pred = pred_score.iloc[0]

        codes = list(pred_score.index.get_level_values("instrument").unique())
        if not codes:
            return TradeDecisionWO([], self)
        code = codes[0]

        current_position = self.trade_position
        cash = current_position.get_cash()
        current_amount = current_position.get_stock_amount(code)

        if current_pred > self.long_threshold:
            target_direction = OrderDir.BUY
        elif current_pred < self.short_threshold:
            target_direction = OrderDir.SELL
        else:
            target_direction = None

        order_list = []

        if target_direction is None:
            if abs(current_amount) > 0:
                close_dir = OrderDir.SELL if current_amount > 0 else OrderDir.BUY
                order_list.append(Order(
                    stock_id=code,
                    amount=abs(current_amount),
                    direction=close_dir,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                ))
                self._last_trade_step = trade_step
            return TradeDecisionWO(order_list, self)

        if (target_direction == OrderDir.BUY and current_amount > 0) or \
           (target_direction == OrderDir.SELL and current_amount < 0):
            return TradeDecisionWO([], self)

        if trade_step - self._last_trade_step < self.hold_thresh:
            return TradeDecisionWO([], self)

        rate = self.trade_exchange.get_deal_price(
            stock_id=code,
            start_time=trade_start_time,
            end_time=trade_end_time,
            direction=target_direction,
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

        target_value = post_close_cash * self.position_ratio
        trade_amount = target_value / rate
        factor = self.trade_exchange.get_factor(code, trade_start_time, trade_end_time)
        trade_amount = self.trade_exchange.round_amount_by_trade_unit(trade_amount, factor)

        if trade_amount > 0:
            order_list.append(Order(
                stock_id=code,
                amount=trade_amount,
                direction=target_direction,
                start_time=trade_start_time,
                end_time=trade_end_time,
            ))
            self._last_trade_step = trade_step

        return TradeDecisionWO(order_list, self)