import numpy as np
import pandas as pd
from qlib.contrib.strategy.signal_strategy import BaseSignalStrategy
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO


class DynamicThresholdStrategy(BaseSignalStrategy):

    def __init__(
        self,
        *,
        signal=None,
        atr_series=None,
        long_percentile=0.95,
        short_percentile=0.05,
        exit_long_percentile=0.50,
        exit_short_percentile=0.50,
        rolling_window=720,
        position_ratio=0.30,
        pos_side="long",
        max_hold_bars=4,
        atr_stop_multiplier=2.0,
        risk_reward_ratio=2.0,
        partial_tp_ratio=0.50,
        risk_per_trade=0.02,
        cooldown_bars=2,
        **kwargs,
    ):
        self._raw_signal = signal
        self._atr_series = atr_series
        self.entry_long_pct = long_percentile
        self.entry_short_pct = short_percentile
        self.exit_long_pct = exit_long_percentile
        self.exit_short_pct = exit_short_percentile
        self.window = rolling_window
        self.position_ratio = position_ratio
        self.pos_side = pos_side
        self.max_hold_bars = max_hold_bars
        self.atr_stop_mult = atr_stop_multiplier
        self.risk_reward = risk_reward_ratio
        self.partial_tp_ratio = partial_tp_ratio
        self.risk_per_trade = risk_per_trade
        self.cooldown_bars = cooldown_bars

        super().__init__(signal=signal, **kwargs)

        self._calculate_dynamic_thresholds()

        self._state = "flat"
        self._entry_price = None
        self._bars_held = 0
        self._highest_price = None
        self._lowest_price = None
        self._trailing_stop = None
        self._stop_loss = None
        self._take_profit = None
        self._partial_tp_hit = False
        self._cooldown = 0
        self._trade_count = 0

    def _calculate_dynamic_thresholds(self):
        sig_df = self._raw_signal
        if isinstance(sig_df, pd.DataFrame):
            sig_series = sig_df.iloc[:, 0]
        else:
            sig_series = sig_df

        if isinstance(sig_series.index, pd.MultiIndex):
            inst_values = sig_series.index.get_level_values("instrument").unique().tolist()
            if len(inst_values) == 1:
                sig_series = sig_series.droplevel("instrument")
            else:
                for name in ["btcusdt", "BTCUSDT"]:
                    if name in inst_values:
                        sig_series = sig_series.xs(name, level="instrument")
                        break
                else:
                    sig_series = sig_series.droplevel("instrument")

        sig_series = sig_series.sort_index()

        self.entry_long_thresh = sig_series.rolling(
            window=self.window, min_periods=24
        ).quantile(self.entry_long_pct)

        self.entry_short_thresh = sig_series.rolling(
            window=self.window, min_periods=24
        ).quantile(self.entry_short_pct)

        self.exit_long_thresh = sig_series.rolling(
            window=self.window, min_periods=24
        ).quantile(self.exit_long_pct)

        self.exit_short_thresh = sig_series.rolling(
            window=self.window, min_periods=24
        ).quantile(self.exit_short_pct)

    def _get_instrument_code(self, pred_score):
        if isinstance(pred_score.index, pd.MultiIndex):
            codes = list(pred_score.index.get_level_values("instrument").unique())
            return codes[0] if codes else "BTCUSDT"
        return "BTCUSDT"

    def _get_current_pred(self, pred_score):
        if isinstance(pred_score, pd.DataFrame):
            return float(pred_score.iloc[-1, 0])
        return float(pred_score.iloc[-1])

    def _get_atr(self, current_time):
        if self._atr_series is None:
            return None
        try:
            return float(self._atr_series.loc[current_time])
        except (KeyError, TypeError):
            try:
                idx = self._atr_series.index.get_indexer([current_time], method="pad")
                if idx[0] >= 0:
                    return float(self._atr_series.iloc[idx[0]])
            except Exception:
                pass
        return None

    def _get_threshold(self, thresh_series, current_time):
        try:
            val = thresh_series.loc[current_time]
            return float(val) if pd.notna(val) else None
        except (KeyError, TypeError):
            try:
                idx = thresh_series.index.get_indexer([current_time], method="pad")
                if idx[0] >= 0:
                    val = thresh_series.iloc[idx[0]]
                    return float(val) if pd.notna(val) else None
            except Exception:
                pass
        return None

    def _calc_position_size(self, capital, current_price, atr_value):
        if atr_value is not None and atr_value > 0:
            pos_btc = (capital * self.risk_per_trade) / atr_value
        else:
            pos_btc = (capital * self.position_ratio * 0.5) / current_price
        max_pos_btc = (capital * self.position_ratio) / current_price
        return min(pos_btc, max_pos_btc)

    def _reset_state(self):
        self._state = "flat"
        self._entry_price = None
        self._bars_held = 0
        self._highest_price = None
        self._lowest_price = None
        self._trailing_stop = None
        self._stop_loss = None
        self._take_profit = None
        self._partial_tp_hit = False
        self._cooldown = self.cooldown_bars

    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)

        current_stock = "BTCUSDT"

        current_price = self.trade_exchange.get_deal_price(
            stock_id=current_stock,
            start_time=trade_start_time,
            end_time=trade_end_time,
            direction=OrderDir.BUY,
        )
        if current_price is None or current_price <= 0:
            return TradeDecisionWO([], self)

        pred_start_time, pred_end_time = trade_start_time, trade_end_time
        try:
            pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        except (IndexError, Exception):
            pass

        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        current_pred = None
        if pred_score is not None and hasattr(pred_score, "__len__") and len(pred_score) > 0:
            current_pred = self._get_current_pred(pred_score)

        current_time = trade_start_time
        entry_long_th = self._get_threshold(self.entry_long_thresh, current_time)
        exit_long_th = self._get_threshold(self.exit_long_thresh, current_time)
        entry_short_th = self._get_threshold(self.entry_short_thresh, current_time)
        exit_short_th = self._get_threshold(self.exit_short_thresh, current_time)

        atr_value = self._get_atr(current_time)

        cash = self.trade_position.get_cash()
        current_amount = self.trade_position.get_stock_amount(current_stock)
        portfolio_value = self.trade_position.calculate_value()

        if abs(current_amount) < 1e-8:
            if self._state != "flat":
                self._reset_state()
            self._state = "flat"
        elif current_amount > 0:
            if self._state != "long":
                self._state = "long"
                self._entry_price = current_price
                self._bars_held = 0
                self._highest_price = current_price
                self._lowest_price = current_price
                self._trailing_stop = None
                self._stop_loss = None
                self._take_profit = None
                self._partial_tp_hit = False
        else:
            if self._state != "short":
                self._state = "short"
                self._entry_price = current_price
                self._bars_held = 0
                self._highest_price = current_price
                self._lowest_price = current_price
                self._trailing_stop = None
                self._stop_loss = None
                self._take_profit = None
                self._partial_tp_hit = False

        order_list = []

        if self._state != "flat":
            self._bars_held += 1
            if current_price > (self._highest_price or 0):
                self._highest_price = current_price
            if current_price < (self._lowest_price or float("inf")):
                self._lowest_price = current_price

            if self._state == "long" and atr_value is not None:
                new_trail = self._highest_price - self.atr_stop_mult * atr_value
                if self._trailing_stop is None or new_trail > self._trailing_stop:
                    self._trailing_stop = new_trail
            elif self._state == "short" and atr_value is not None:
                new_trail = self._lowest_price + self.atr_stop_mult * atr_value
                if self._trailing_stop is None or new_trail < self._trailing_stop:
                    self._trailing_stop = new_trail

        if self._cooldown > 0:
            self._cooldown -= 1

        # ========== STATE MACHINE ==========

        if self._state == "flat":
            if self._cooldown > 0:
                return TradeDecisionWO([], self)

            if current_pred is not None and entry_long_th is not None and current_pred > entry_long_th:
                pos_btc = self._calc_position_size(portfolio_value, current_price, atr_value)
                if pos_btc > 1e-8:
                    order_list.append(Order(
                        stock_id=current_stock,
                        amount=pos_btc,
                        direction=OrderDir.BUY,
                        start_time=trade_start_time,
                        end_time=trade_end_time,
                    ))
                    self._state = "long"
                    self._entry_price = current_price
                    self._bars_held = 0
                    self._highest_price = current_price
                    self._lowest_price = current_price
                    self._partial_tp_hit = False
                    if atr_value is not None and atr_value > 0:
                        self._stop_loss = current_price - self.atr_stop_mult * atr_value
                        risk_per_btc = current_price - self._stop_loss
                        self._take_profit = current_price + risk_per_btc * self.risk_reward
                    else:
                        self._stop_loss = current_price * 0.97
                        self._take_profit = current_price * 1.06
                    self._trailing_stop = self._stop_loss
                    self._trade_count += 1

            elif (self._state == "flat" and self.pos_side == "both"
                  and current_pred is not None and entry_short_th is not None
                  and current_pred < entry_short_th):
                pos_btc = self._calc_position_size(portfolio_value, current_price, atr_value)
                if pos_btc > 1e-8:
                    order_list.append(Order(
                        stock_id=current_stock,
                        amount=pos_btc,
                        direction=OrderDir.SELL,
                        start_time=trade_start_time,
                        end_time=trade_end_time,
                    ))
                    self._state = "short"
                    self._entry_price = current_price
                    self._bars_held = 0
                    self._highest_price = current_price
                    self._lowest_price = current_price
                    self._partial_tp_hit = False
                    if atr_value is not None and atr_value > 0:
                        self._stop_loss = current_price + self.atr_stop_mult * atr_value
                        risk_per_btc = self._stop_loss - current_price
                        self._take_profit = current_price - risk_per_btc * self.risk_reward
                    else:
                        self._stop_loss = current_price * 1.03
                        self._take_profit = current_price * 0.94
                    self._trailing_stop = self._stop_loss
                    self._trade_count += 1

        elif self._state == "long":
            should_exit = False

            if self._stop_loss is not None and current_price <= self._stop_loss:
                should_exit = True

            if not should_exit and self._trailing_stop is not None and current_price <= self._trailing_stop:
                should_exit = True

            if not should_exit and self._bars_held >= self.max_hold_bars:
                should_exit = True

            if not should_exit and current_pred is not None and exit_long_th is not None:
                if current_pred < exit_long_th:
                    should_exit = True

            if not should_exit and not self._partial_tp_hit and self._take_profit is not None:
                if current_price >= self._take_profit:
                    partial_amount = abs(current_amount) * self.partial_tp_ratio
                    if partial_amount > 1e-8:
                        order_list.append(Order(
                            stock_id=current_stock,
                            amount=partial_amount,
                            direction=OrderDir.SELL,
                            start_time=trade_start_time,
                            end_time=trade_end_time,
                        ))
                    self._partial_tp_hit = True
                    self._stop_loss = self._entry_price
                    self._trailing_stop = max(
                        self._trailing_stop or 0, self._entry_price
                    )
                    self._take_profit = None

            if should_exit:
                if abs(current_amount) > 1e-8:
                    order_list.append(Order(
                        stock_id=current_stock,
                        amount=abs(current_amount),
                        direction=OrderDir.SELL,
                        start_time=trade_start_time,
                        end_time=trade_end_time,
                    ))
                self._reset_state()

        elif self._state == "short":
            should_exit = False

            if self._stop_loss is not None and current_price >= self._stop_loss:
                should_exit = True

            if not should_exit and self._trailing_stop is not None and current_price >= self._trailing_stop:
                should_exit = True

            if not should_exit and self._bars_held >= self.max_hold_bars:
                should_exit = True

            if not should_exit and current_pred is not None and exit_short_th is not None:
                if current_pred > exit_short_th:
                    should_exit = True

            if not should_exit and not self._partial_tp_hit and self._take_profit is not None:
                if current_price <= self._take_profit:
                    partial_amount = abs(current_amount) * self.partial_tp_ratio
                    if partial_amount > 1e-8:
                        order_list.append(Order(
                            stock_id=current_stock,
                            amount=partial_amount,
                            direction=OrderDir.BUY,
                            start_time=trade_start_time,
                            end_time=trade_end_time,
                        ))
                    self._partial_tp_hit = True
                    self._stop_loss = self._entry_price
                    self._trailing_stop = min(
                        self._trailing_stop or float("inf"), self._entry_price
                    )
                    self._take_profit = None

            if should_exit:
                if abs(current_amount) > 1e-8:
                    order_list.append(Order(
                        stock_id=current_stock,
                        amount=abs(current_amount),
                        direction=OrderDir.BUY,
                        start_time=trade_start_time,
                        end_time=trade_end_time,
                    ))
                self._reset_state()

        return TradeDecisionWO(order_list, self)