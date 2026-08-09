"""Explicit boundary conventions and candidate first-passage research tables."""
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

class ResidualDirection(str, Enum):
    LONG_RESIDUAL = "LONG_RESIDUAL"   # exit above current state; stop below
    SHORT_RESIDUAL = "SHORT_RESIDUAL" # exit below current state; stop above

@dataclass(frozen=True)
class BoundaryDefinition:
    current_state: float; exit_boundary: float; stop_boundary: float; direction: ResidualDirection
    def validate(self) -> None:
        if self.direction is ResidualDirection.LONG_RESIDUAL and not self.stop_boundary < self.current_state < self.exit_boundary:
            raise ValueError("LONG_RESIDUAL requires stop < current_state < exit.")
        if self.direction is ResidualDirection.SHORT_RESIDUAL and not self.exit_boundary < self.current_state < self.stop_boundary:
            raise ValueError("SHORT_RESIDUAL requires exit < current_state < stop.")

def boundary_candidates(current_state: float, direction: ResidualDirection, exits: Iterable[float], stops: Iterable[float]) -> list[BoundaryDefinition]:
    results=[]
    for exit_boundary in exits:
        for stop_boundary in stops:
            candidate=BoundaryDefinition(current_state,exit_boundary,stop_boundary,direction); candidate.validate(); results.append(candidate)
    return results
