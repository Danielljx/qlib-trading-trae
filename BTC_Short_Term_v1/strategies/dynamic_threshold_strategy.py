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
        atr_mean_series=None,
        ma_series=None,
        ma_fast_series=None,
        adx_series=None,
        long_percentile=0.90,
        short_percentile=0.10,
        exit_long_percentile=0.20,
        exit_short_percentile=0.80,
        rolling_window=720,
        position_ratio=0.30,
        pos_side="both",
        max_hold_bars=12,
        atr_stop_multiplier=2.0,
        risk_reward_ratio=2.0,
        partial_tp_ratio=0.50,
        risk_per_trade=0.02,
        cooldown_bars=2,
        smart_hold_extension=True,
        smart_hold_atr_threshold=1.0,
        use_trend_filter=True,
        adx_strong=30,
        adx_weak=20,
        trend_buffer=0.015,
        tp1_multiplier=1.0,
        tp2_multiplier=1.5,
        tp3_multiplier=2.0,
        tp1_ratio=0.50,
        tp2_ratio=0.30,
        tp3_ratio=0.20,
        vol_tp_scale=True,
        **kwargs,
    ):
        self._raw_signal = signal
        self._atr_series = atr_series
        self._atr_mean_series = atr_mean_series
        self._ma_series = ma_series
        self._ma_fast_series = ma_fast_series
        self._adx_series = adx_series
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
        self.smart_hold_ext = smart_hold_extension
        self.smart_hold_atr_thresh = smart_hold_atr_threshold
        self.use_trend_filter = use_trend_filter
        self.adx_strong = adx_strong
        self.adx_weak = adx_weak
        self.trend_buffer = trend_buffer
        self.tp1_mult = tp1_multiplier
        self.tp2_mult = tp2_multiplier
        self.tp3_mult = tp3_multiplier
        self.tp1_ratio = tp1_ratio
        self.tp2_ratio = tp2_ratio
        self.tp3_ratio = tp3_ratio
        self.vol_tp_scale = vol_tp_scale

        super().__init__(signal=signal, **kwargs)

        self._calculate_dynamic_thresholds()

        self._state = "flat"
        self._entry_price = None
        self._entry_time = None
        self._direction = None
        self._bars_held = 0
        self._highest_price = None
        self._lowest_price = None
        self._trailing_stop = None
        self._stop_loss = None
        self._tp1 = None
        self._tp2 = None
        self._tp3 = None
        self._tp1_hit = False
        self._tp2_hit = False
        self._cooldown = 0
        self._trade_count = 0
        self._trend_hold_bars = 0

        self._trade_log = []
        self._filtered_long_count = 0
        self._filtered_short_count = 0
        self._filtered_adx_count = 0
        self._trend_filter_exits = 0
        self._trend_filter_protected = 0

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

    def _get_current_pred(self, pred_score):
        if isinstance(pred_score, pd.DataFrame):
            return float(pred_score.iloc[-1, 0])
        return float(pred_score.iloc[-1])

    def _lookup_series(self, series, current_time):
        if series is None:
            return None
        try:
            val = series.loc[current_time]
            return float(val) if pd.notna(val) else None
        except (KeyError, TypeError):
            try:
                idx = series.index.get_indexer([current_time], method="pad")
                if idx[0] >= 0:
                    val = series.iloc[idx[0]]
                    return float(val) if pd.notna(val) else None
            except Exception:
                pass
        return None

    def _get_atr(self, current_time):
        return self._lookup_series(self._atr_series, current_time)

    def _get_atr_mean(self, current_time):
        return self._lookup_series(self._atr_mean_series, current_time)

    def _get_ma(self, current_time):
        return self._lookup_series(self._ma_series, current_time)

    def _get_ma_fast(self, current_time):
        return self._lookup_series(self._ma_fast_series, current_time)

    def _get_adx(self, current_time):
        return self._lookup_series(self._adx_series, current_time)

    def _get_threshold(self, thresh_series, current_time):
        return self._lookup_series(thresh_series, current_time)

    def _calc_position_size(self, capital, current_price, atr_value):
        if atr_value is not None and atr_value > 0:
            pos_btc = (capital * self.risk_per_trade) / atr_value
        else:
            pos_btc = (capital * self.position_ratio * 0.5) / current_price
        max_pos_btc = (capital * self.position_ratio) / current_price
        return min(pos_btc, max_pos_btc)

    def _has_significant_profit(self, current_price, atr_value):
        if not self.smart_hold_ext or atr_value is None or self._entry_price is None:
            return False
        if self._direction == "long":
            return (current_price - self._entry_price) > self.smart_hold_atr_thresh * atr_value
        else:
            return (self._entry_price - current_price) > self.smart_hold_atr_thresh * atr_value

    def _calc_tp_levels(self, current_price, atr_value, direction):
        if atr_value is None or atr_value <= 0:
            return None, None, None

        atr_mean = self._last_atr_mean
        scale = 1.0
        if self.vol_tp_scale and atr_mean is not None and atr_mean > 0:
            vol_ratio = atr_value / atr_mean
            if vol_ratio > 1.5:
                scale = 0.8
            elif vol_ratio < 0.7:
                scale = 1.2

        if direction == "long":
            tp1 = current_price + self.tp1_mult * scale * atr_value
            tp2 = current_price + self.tp2_mult * scale * atr_value
            tp3 = current_price + self.tp3_mult * scale * atr_value
        else:
            tp1 = current_price - self.tp1_mult * scale * atr_value
            tp2 = current_price - self.tp2_mult * scale * atr_value
            tp3 = current_price - self.tp3_mult * scale * atr_value

        return tp1, tp2, tp3

    def _check_trend_filter_entry(self, direction, current_price, ma_slow, adx_value, atr_value):
        if not self.use_trend_filter or ma_slow is None:
            return True

        if direction == "long":
            if adx_value is not None and adx_value > self.adx_strong:
                return current_price > ma_slow
            elif adx_value is not None and adx_value >= self.adx_weak:
                return current_price > ma_slow * (1 - self.trend_buffer)
            else:
                if atr_value is not None:
                    atr_mean = self._last_atr_mean
                    if atr_mean is not None and atr_mean > 0:
                        return atr_value < atr_mean * 0.8
                return False
        elif direction == "short":
            if adx_value is not None and adx_value > self.adx_strong:
                return current_price < ma_slow
            elif adx_value is not None and adx_value >= self.adx_weak:
                return current_price < ma_slow * (1 + self.trend_buffer)
            else:
                return False

        return True

    def _check_trend_filter_exit(self, current_price, ma_slow, adx_value, atr_value):
        if not self.use_trend_filter or ma_slow is None:
            return False, "none"

        if self._direction == "long":
            if adx_value is not None and adx_value > self.adx_strong:
                if current_price < ma_slow:
                    return True, "Trend_Filter_Exit"
            elif adx_value is not None and adx_value >= self.adx_weak:
                if current_price < ma_slow * (1 - self.trend_buffer):
                    return True, "Trend_Filter_Exit"
            else:
                pass

        elif self._direction == "short":
            if adx_value is not None and adx_value > self.adx_strong:
                if current_price > ma_slow:
                    return True, "Trend_Filter_Exit"
            elif adx_value is not None and adx_value >= self.adx_weak:
                if current_price > ma_slow * (1 + self.trend_buffer):
                    return True, "Trend_Filter_Exit"

        return False, "none"

    def _apply_trend_exit_protection(self, current_price, current_amount, trade_start_time, trade_end_time):
        if self._entry_price is None:
            return None

        pnl_pct = 0
        if self._direction == "long":
            pnl_pct = (current_price - self._entry_price) / self._entry_price
        else:
            pnl_pct = (self._entry_price - current_price) / self._entry_price

        if pnl_pct > 0.005:
            self._stop_loss = self._entry_price
            if self._direction == "long":
                self._trailing_stop = max(self._trailing_stop or 0, self._entry_price)
            else:
                self._trailing_stop = min(self._trailing_stop or float("inf"), self._entry_price)
            self._trend_hold_bars = 2
            self._trend_filter_protected += 1
            return None

        elif pnl_pct > 0:
            partial_amount = abs(current_amount) * 0.5
            if partial_amount > 1e-8:
                direction = OrderDir.SELL if self._direction == "long" else OrderDir.BUY
                order = Order(
                    stock_id="BTCUSDT",
                    amount=partial_amount,
                    direction=direction,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                )
                self._trend_hold_bars = 2
                return order
            return None

        else:
            self._trend_filter_exits += 1
            return "full_exit"

    def _record_exit(self, exit_time, exit_price, exit_reason, current_amount):
        if self._entry_price is None or self._entry_time is None:
            return
        direction = "Long" if self._direction == "long" else "Short"
        if direction == "Long":
            pnl_pct = (exit_price - self._entry_price) / self._entry_price
            mfe_pct = (self._highest_price - self._entry_price) / self._entry_price if self._highest_price else 0
        else:
            pnl_pct = (self._entry_price - exit_price) / self._entry_price
            mfe_pct = (self._entry_price - self._lowest_price) / self._entry_price if self._lowest_price else 0

        self._trade_count += 1
        self._trade_log.append({
            "Trade_ID": self._trade_count,
            "Direction": direction,
            "Entry_Time": self._entry_time,
            "Exit_Time": exit_time,
            "Hold_Bars": self._bars_held,
            "Entry_Price": round(self._entry_price, 2),
            "Exit_Price": round(exit_price, 2),
            "Exit_Reason": exit_reason,
            "PnL_Percent": round(pnl_pct * 100, 4),
            "MFE_Percent": round(mfe_pct * 100, 4),
        })

    def _reset_state(self, exit_time=None, exit_price=None, exit_reason=None, current_amount=0):
        if exit_time is not None and exit_price is not None and exit_reason is not None:
            self._record_exit(exit_time, exit_price, exit_reason, current_amount)
        self._state = "flat"
        self._entry_price = None
        self._entry_time = None
        self._direction = None
        self._bars_held = 0
        self._highest_price = None
        self._lowest_price = None
        self._trailing_stop = None
        self._stop_loss = None
        self._tp1 = None
        self._tp2 = None
        self._tp3 = None
        self._tp1_hit = False
        self._tp2_hit = False
        self._trend_hold_bars = 0
        self._cooldown = self.cooldown_bars

    def get_trade_log(self):
        return pd.DataFrame(self._trade_log) if self._trade_log else pd.DataFrame()

    def get_filter_stats(self):
        return {
            "filtered_long": self._filtered_long_count,
            "filtered_short": self._filtered_short_count,
            "filtered_adx": self._filtered_adx_count,
            "trend_filter_exits": self._trend_filter_exits,
            "trend_filter_protected": self._trend_filter_protected,
        }

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
        atr_mean = self._get_atr_mean(current_time)
        self._last_atr_mean = atr_mean
        ma_slow = self._get_ma(current_time)
        ma_fast = self._get_ma_fast(current_time)
        adx_value = self._get_adx(current_time)

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
                self._entry_time = current_time
                self._direction = "long"
                self._bars_held = 0
                self._highest_price = current_price
                self._lowest_price = current_price
                self._trailing_stop = None
                self._stop_loss = None
                self._tp1, self._tp2, self._tp3 = self._calc_tp_levels(current_price, atr_value, "long")
                self._tp1_hit = False
                self._tp2_hit = False
        else:
            if self._state != "short":
                self._state = "short"
                self._entry_price = current_price
                self._entry_time = current_time
                self._direction = "short"
                self._bars_held = 0
                self._highest_price = current_price
                self._lowest_price = current_price
                self._trailing_stop = None
                self._stop_loss = None
                self._tp1, self._tp2, self._tp3 = self._calc_tp_levels(current_price, atr_value, "short")
                self._tp1_hit = False
                self._tp2_hit = False

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

        if self._trend_hold_bars > 0:
            self._trend_hold_bars -= 1

        if self._state == "flat":
            if self._cooldown > 0:
                return TradeDecisionWO([], self)

            want_long = (current_pred is not None and entry_long_th is not None and current_pred > entry_long_th)
            want_short = (self.pos_side in ("both", "short")
                          and current_pred is not None and entry_short_th is not None
                          and current_pred < entry_short_th)

            if self.use_trend_filter:
                if want_long and not self._check_trend_filter_entry("long", current_price, ma_slow, adx_value, atr_value):
                    want_long = False
                    self._filtered_long_count += 1
                if want_short and not self._check_trend_filter_entry("short", current_price, ma_slow, adx_value, atr_value):
                    want_short = False
                    self._filtered_short_count += 1

            if want_long:
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
                    self._entry_time = current_time
                    self._direction = "long"
                    self._bars_held = 0
                    self._highest_price = current_price
                    self._lowest_price = current_price
                    self._tp1_hit = False
                    self._tp2_hit = False
                    self._trend_hold_bars = 0
                    if atr_value is not None and atr_value > 0:
                        self._stop_loss = current_price - self.atr_stop_mult * atr_value
                        self._tp1, self._tp2, self._tp3 = self._calc_tp_levels(current_price, atr_value, "long")
                    else:
                        self._stop_loss = current_price * 0.97
                        self._tp1 = current_price * 1.01
                        self._tp2 = current_price * 1.015
                        self._tp3 = current_price * 1.02
                    self._trailing_stop = self._stop_loss

            elif want_short:
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
                    self._entry_time = current_time
                    self._direction = "short"
                    self._bars_held = 0
                    self._highest_price = current_price
                    self._lowest_price = current_price
                    self._tp1_hit = False
                    self._tp2_hit = False
                    self._trend_hold_bars = 0
                    if atr_value is not None and atr_value > 0:
                        self._stop_loss = current_price + self.atr_stop_mult * atr_value
                        self._tp1, self._tp2, self._tp3 = self._calc_tp_levels(current_price, atr_value, "short")
                    else:
                        self._stop_loss = current_price * 1.03
                        self._tp1 = current_price * 0.99
                        self._tp2 = current_price * 0.985
                        self._tp3 = current_price * 0.98
                    self._trailing_stop = self._stop_loss

        elif self._state == "long":
            should_exit = False
            exit_reason = None

            if self._stop_loss is not None and current_price <= self._stop_loss:
                should_exit = True
                exit_reason = "SL_Hit"

            if not should_exit and self._trailing_stop is not None and current_price <= self._trailing_stop:
                should_exit = True
                exit_reason = "Trailing_Stop"

            if not should_exit and self._trend_hold_bars <= 0:
                trend_exit, trend_reason = self._check_trend_filter_exit(current_price, ma_slow, adx_value, atr_value)
                if trend_exit:
                    protection_result = self._apply_trend_exit_protection(
                        current_price, current_amount, trade_start_time, trade_end_time
                    )
                    if protection_result is None:
                        pass
                    elif protection_result == "full_exit":
                        should_exit = True
                        exit_reason = trend_reason
                    else:
                        order_list.append(protection_result)

            if not should_exit and self._bars_held >= self.max_hold_bars:
                if self._has_significant_profit(current_price, atr_value):
                    pass
                else:
                    should_exit = True
                    exit_reason = "Time_Exit"

            if not should_exit and current_pred is not None and exit_long_th is not None:
                if current_pred < exit_long_th:
                    if self._has_significant_profit(current_price, atr_value):
                        pass
                    else:
                        should_exit = True
                        exit_reason = "Signal_Exit"

            if not should_exit and not self._tp1_hit and self._tp1 is not None:
                if current_price >= self._tp1:
                    tp1_amount = abs(current_amount) * self.tp1_ratio
                    if tp1_amount > 1e-8:
                        order_list.append(Order(
                            stock_id=current_stock,
                            amount=tp1_amount,
                            direction=OrderDir.SELL,
                            start_time=trade_start_time,
                            end_time=trade_end_time,
                        ))
                    self._tp1_hit = True
                    self._stop_loss = self._entry_price
                    self._trailing_stop = max(self._trailing_stop or 0, self._entry_price)
                    self._record_exit(current_time, current_price, "TP1_Hit", current_amount)

            if not should_exit and self._tp1_hit and not self._tp2_hit and self._tp2 is not None:
                if current_price >= self._tp2:
                    remaining = abs(current_amount) * self.tp2_ratio
                    if remaining > 1e-8:
                        order_list.append(Order(
                            stock_id=current_stock,
                            amount=remaining,
                            direction=OrderDir.SELL,
                            start_time=trade_start_time,
                            end_time=trade_end_time,
                        ))
                    self._tp2_hit = True
                    self._trailing_stop = max(self._trailing_stop or 0, self._tp1 or self._entry_price)
                    self._record_exit(current_time, current_price, "TP2_Hit", current_amount)

            if not should_exit and self._tp2_hit and self._tp3 is not None:
                if current_price >= self._tp3:
                    should_exit = True
                    exit_reason = "TP3_Hit"

            if should_exit:
                if abs(current_amount) > 1e-8:
                    order_list.append(Order(
                        stock_id=current_stock,
                        amount=abs(current_amount),
                        direction=OrderDir.SELL,
                        start_time=trade_start_time,
                        end_time=trade_end_time,
                    ))
                self._reset_state(current_time, current_price, exit_reason, current_amount)

        elif self._state == "short":
            should_exit = False
            exit_reason = None

            if self._stop_loss is not None and current_price >= self._stop_loss:
                should_exit = True
                exit_reason = "SL_Hit"

            if not should_exit and self._trailing_stop is not None and current_price >= self._trailing_stop:
                should_exit = True
                exit_reason = "Trailing_Stop"

            if not should_exit and self._trend_hold_bars <= 0:
                trend_exit, trend_reason = self._check_trend_filter_exit(current_price, ma_slow, adx_value, atr_value)
                if trend_exit:
                    protection_result = self._apply_trend_exit_protection(
                        current_price, current_amount, trade_start_time, trade_end_time
                    )
                    if protection_result is None:
                        pass
                    elif protection_result == "full_exit":
                        should_exit = True
                        exit_reason = trend_reason
                    else:
                        order_list.append(protection_result)

            if not should_exit and self._bars_held >= self.max_hold_bars:
                if self._has_significant_profit(current_price, atr_value):
                    pass
                else:
                    should_exit = True
                    exit_reason = "Time_Exit"

            if not should_exit and current_pred is not None and exit_short_th is not None:
                if current_pred > exit_short_th:
                    if self._has_significant_profit(current_price, atr_value):
                        pass
                    else:
                        should_exit = True
                        exit_reason = "Signal_Exit"

            if not should_exit and not self._tp1_hit and self._tp1 is not None:
                if current_price <= self._tp1:
                    tp1_amount = abs(current_amount) * self.tp1_ratio
                    if tp1_amount > 1e-8:
                        order_list.append(Order(
                            stock_id=current_stock,
                            amount=tp1_amount,
                            direction=OrderDir.BUY,
                            start_time=trade_start_time,
                            end_time=trade_end_time,
                        ))
                    self._tp1_hit = True
                    self._stop_loss = self._entry_price
                    self._trailing_stop = min(self._trailing_stop or float("inf"), self._entry_price)
                    self._record_exit(current_time, current_price, "TP1_Hit", current_amount)

            if not should_exit and self._tp1_hit and not self._tp2_hit and self._tp2 is not None:
                if current_price <= self._tp2:
                    remaining = abs(current_amount) * self.tp2_ratio
                    if remaining > 1e-8:
                        order_list.append(Order(
                            stock_id=current_stock,
                            amount=remaining,
                            direction=OrderDir.BUY,
                            start_time=trade_start_time,
                            end_time=trade_end_time,
                        ))
                    self._tp2_hit = True
                    self._trailing_stop = min(self._trailing_stop or float("inf"), self._tp1 or self._entry_price)
                    self._record_exit(current_time, current_price, "TP2_Hit", current_amount)

            if not should_exit and self._tp2_hit and self._tp3 is not None:
                if current_price <= self._tp3:
                    should_exit = True
                    exit_reason = "TP3_Hit"

            if should_exit:
                if abs(current_amount) > 1e-8:
                    order_list.append(Order(
                        stock_id=current_stock,
                        amount=abs(current_amount),
                        direction=OrderDir.BUY,
                        start_time=trade_start_time,
                        end_time=trade_end_time,
                    ))
                self._reset_state(current_time, current_price, exit_reason, current_amount)

        return TradeDecisionWO(order_list, self)
