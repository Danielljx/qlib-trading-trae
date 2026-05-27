import logging

from qlib.backtest.exchange import Exchange
from qlib.backtest.decision import Order, OrderDir

logger = logging.getLogger(__name__)


class CryptoExchange(Exchange):

    def _calc_trade_info_by_order(self, order, trade_account=None, dealt_order_amount=None, position=None):
        if order.direction != Order.SELL:
            return super()._calc_trade_info_by_order(
                order, trade_account, dealt_order_amount
            )

        trade_price = self.get_deal_price(
            str(order.stock_id),
            order.start_time,
            order.end_time,
            OrderDir.SELL,
        )
        if trade_price is None or trade_price <= 1e-8:
            return trade_price, 0.0, 0.0

        volume = self.get_volume(order.stock_id, order.start_time, order.end_time)
        total_trade_val = volume * trade_price

        adj_cost_ratio = self.impact_cost * (order.amount * trade_price / (total_trade_val + 1e-8)) ** 2
        cost_ratio = self.close_cost + adj_cost_ratio

        order.factor = self.get_factor(
            str(order.stock_id),
            order.start_time,
            order.end_time,
        )
        order.deal_amount = order.amount

        self._clip_amount_by_volume(order, dealt_order_amount)

        if trade_account is not None:
            cash = trade_account.get_cash()
            if cash + order.deal_amount * trade_price < max(
                order.deal_amount * trade_price * cost_ratio,
                self.min_cost,
            ):
                order.deal_amount = 0
            else:
                order.deal_amount = self.round_amount_by_trade_unit(order.deal_amount, order.factor)

        trade_val = order.deal_amount * trade_price
        trade_cost = order.deal_amount * trade_price * cost_ratio
        return trade_price, trade_val, trade_cost