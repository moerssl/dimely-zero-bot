from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract, ContractDetails
from ibapi.ticktype import TickTypeEnum
from ibapi.common import *

from ibapi.utils import iswrapper
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone
from threading import Thread, RLock, Timer
import queue

class IbQueueWrapper(EWrapper):
    def __init__(self, market_data_dict, batch_size: int = 500, flush_interval: float = 0.1):
        super().__init__()
        self.optionsDataLock = RLock()
        self.req_ids = market_data_dict

        # DataFrames
        self.positions = pd.DataFrame(
            columns=["Account","Symbol","Quantity","Avg Cost","Id","Underlying","OptionType"]
        )
        self.market_data = pd.DataFrame(
            columns=["Symbol","DateTime","Price"]
        )
        with self.optionsDataLock:
            self.options_data = pd.DataFrame(
                columns=["Id","Symbol","Strike","dist","undPrice","Type","delta",
                         "bid","ask","time","bid_size","ask_size","last","Expiry",
                         "close","high","low","ConId","UnderConId",
                         "impliedVol","optPrice","pvDividend","gamma","vega","theta","delta_diff"]
            ).set_index('Id')

        self.contract_details = {}

        # Queue and writer thread for batched updates
        self._update_queue = queue.Queue()
        self._batch_size = batch_size
        self._flush_interval = flush_interval

        self._buffer = []

        self._writer = Thread(target=self._writer_loop, daemon=True)
        self._writer.start()

        self._flush_timer = Timer(2, self._flush)
        self._flush_timer.start()

    def _writer_loop(self):
        buffer = []
        while True:
            try:
                item = self._update_queue.get(timeout=self._flush_interval)
                self._buffer.append(item)
            except queue.Empty:
                pass
            if self._buffer and len(self._buffer) >= self._batch_size:
                self._flush()
                buffer.clear()

    def _flush(self):
        buffer = self._buffer
        with self.optionsDataLock:
            for df_name, row in buffer:
                if df_name == 'positions':
                    df = self.positions
                    if row['Id'] in df['Id'].values:
                        df.loc[df['Id']==row['Id'], list(row.keys())] = list(row.values())
                    else:
                        self.positions = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                elif df_name == 'market_data':
                    sym = row['Symbol']
                    self.market_data.loc[sym] = row
                elif df_name == 'options_data':
                    idx = row.pop('Id')
                    self.options_data.loc[idx, list(row.keys())] = list(row.values())

    # Helper to enqueue updates
    def _enqueue(self, df_name, row):
        self._update_queue.put((df_name, row))

    @iswrapper
    def position(self, account, contract, position, avgCost):
        if position != 0:
            new_row = {
                'Account': account,
                'Symbol': contract.localSymbol,
                'Quantity': position,
                'Avg Cost': avgCost,
                'Id': contract.conId,
                'Underlying': contract.symbol,
                'OptionType': contract.right
            }
            self._enqueue('positions', new_row)
        return super().position(account, contract, position, avgCost)

    @iswrapper
    def tickPrice(self, reqId, tickType, price, attrib):
        entry = self.req_ids.get(reqId, {})
        text = TickTypeEnum.toStr(tickType).lower()
        if 'symbol' in entry and text in ('last','close'):
            row = {'Symbol': entry['symbol'], 'DateTime': datetime.now(), 'Price': price}
            self._enqueue('market_data', row)
        elif 'id' in entry:
            row = {'Id': entry['id'], text: price, 'time': datetime.now(timezone.utc)}
            self._enqueue('options_data', row)
        return super().tickPrice(reqId, tickType, price, attrib)

    @iswrapper
    def tickSize(self, reqId, tickType, size):
        entry = self.req_ids.get(reqId, {})
        text = TickTypeEnum.toStr(tickType).lower()
        if 'id' in entry:
            row = {'Id': entry['id'], text: size, 'time': datetime.now(timezone.utc)}
            self._enqueue('options_data', row)
        return super().tickSize(reqId, tickType, size)

    @iswrapper
    def tickOptionComputation(self, reqId, tickType, tickAttrib,
                               impliedVol, delta, optPrice, pvDividend,
                               gamma, vega, theta, undPrice):
        entry = self.req_ids.get(reqId, {})
        if 'id' in entry:
            id_ = entry['id']
            with self.optionsDataLock:
                strike = self.options_data.loc[id_, 'Strike']
            dist = round(abs(strike-undPrice)/undPrice*100,2) if (undPrice is not None and strike is not None) else None
            row = {
                'Id': id_, 'impliedVol': impliedVol,
                'delta': round(delta,3) if delta is not None else delta,
                'optPrice': optPrice, 'pvDividend': pvDividend,
                'gamma': gamma, 'vega': vega, 'theta': theta,
                'undPrice': round(undPrice,2) if undPrice is not None else undPrice,
                'dist': dist, 'time': datetime.now(timezone.utc)
            }
            self._enqueue('options_data', row)
        return super().tickOptionComputation(reqId, tickType, tickAttrib,
                                             impliedVol, delta, optPrice,
                                             pvDividend, gamma, vega, theta, undPrice)

    @iswrapper
    def contractDetails(self, reqId, contractDetails):
        self.contract_details[reqId] = contractDetails.contract
        
        cd = contractDetails.contract
        row = {
            'Id': cd.conId, 
            'ConId': cd.conId, 
            'Symbol': cd.symbol,
            'Strike': cd.strike, 
            'Type': cd.right,
            'Expiry': cd.lastTradeDate, 
            'UnderConId': contractDetails.underConId
        }
        self._enqueue('options_data', row)
        return super().contractDetails(reqId, contractDetails)

    # Thread-safe read accessors
    def get_positions(self) -> pd.DataFrame:
        """Return a copy of positions."""
        with self.optionsDataLock:
            return self.positions.copy()

    def get_market_data(self) -> pd.DataFrame:
        """Return a copy of market data."""
        # market_data isn't locked but safe for read
        return self.market_data.copy()

    def get_options_data(self) -> pd.DataFrame:
        """Return a copy of options data."""
        with self.optionsDataLock:
            return self.options_data.copy()
