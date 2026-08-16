"""Explicit holdings and cost-aware accounting, including both legs of a pair."""
from dataclasses import dataclass
from src.costs import CostModel
from src.decision.trade_expression import PairTradeExpression

@dataclass(frozen=True)
class TradeRecord:
    entry_date: object; exit_date: object; direction: str; gross_notional: float
    gross_pnl: float; costs: float; net_pnl: float; holding_days: int

def pair_trade_record(expression: PairTradeExpression, *, entry_date: object, exit_date: object, exit_target_price: float, exit_peer_price: float, costs: CostModel, holding_days: int) -> TradeRecord:
    gross=expression.marked_pnl(exit_target_price, exit_peer_price)
    total_cost=expression.transaction_cost(costs, True)+expression.transaction_cost(costs, False)
    return TradeRecord(entry_date, exit_date, "PAIR", expression.gross_notional(), gross, total_cost, gross-total_cost, holding_days)
