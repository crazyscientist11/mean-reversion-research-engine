"""Explicit transaction and daily opportunity-cost assumptions."""
from dataclasses import dataclass

@dataclass(frozen=True)
class CostModel:
    fixed_entry_cost: float = 0.0
    fixed_exit_cost: float = 0.0
    bid_ask_bps: float = 0.0
    slippage_bps: float = 0.0
    commission: float = 0.0
    annual_opportunity_rate: float = 0.0
    trading_days_per_year: int = 252
    def daily_discount(self) -> float:
        if self.annual_opportunity_rate < 0 or self.trading_days_per_year < 1: raise ValueError("invalid opportunity-cost settings")
        return (1 + self.annual_opportunity_rate) ** (-1 / self.trading_days_per_year)
    def trade_cost(self, gross_notional: float, entering: bool) -> float:
        fixed = self.fixed_entry_cost if entering else self.fixed_exit_cost
        return fixed + self.commission + gross_notional * (self.bid_ask_bps + self.slippage_bps) / 10000
