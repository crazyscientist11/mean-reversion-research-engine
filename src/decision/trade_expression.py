"""Mapping from a statistical pair residual to marked, two-leg holdings."""
from dataclasses import dataclass, field
from typing import Mapping
from src.costs import CostModel

@dataclass(frozen=True)
class PairTradeExpression:
    target_ticker: str; peer_ticker: str; beta: float; target_price: float; peer_price: float; notional: float; multiplier: float=1.0; metadata: Mapping[str,str]=field(default_factory=dict)
    def shares(self) -> dict[str,float]:
        unit=self.notional/(self.target_price+abs(self.beta)*self.peer_price)
        return {self.target_ticker: unit*self.multiplier, self.peer_ticker: -self.beta*unit*self.multiplier}
    def gross_notional(self) -> float: return sum(abs(v)*p for v,p in zip(self.shares().values(),[self.target_price,self.peer_price]))
    def marked_pnl(self, new_target_price:float,new_peer_price:float) -> float:
        s=self.shares(); return s[self.target_ticker]*(new_target_price-self.target_price)+s[self.peer_ticker]*(new_peer_price-self.peer_price)
    def transaction_cost(self,costs:CostModel,entering:bool)->float: return costs.trade_cost(self.gross_notional(),entering)
