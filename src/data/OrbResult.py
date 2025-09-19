from dataclasses import dataclass
from typing import Optional
import datetime
import pandas as pd

@dataclass
class OrbResult:
    date: Optional[pd.Timestamp] = None
    current: Optional[float] = None
    isOpenRangeCompleted: bool = False
    isAbove: bool = False
    isBelow: bool = False
    high: Optional[float] = None
    low: Optional[float] = None
    open: Optional[float] = None
    close: Optional[float] = None
    breakout_age: int = -1          # Minuten seit letztem Ausbruch, -1 wenn keiner
    breakout_count: int = 0         # Wie viele Ausbrüche an diesem Tag

    # getter for range width percentage
    @property
    def range_width_pct(self) -> float:
        if not self.high or not self.low or self.high == 0 or self.low == 0:
            return 0.0
        return (self.high - self.low) / self.low * 100
