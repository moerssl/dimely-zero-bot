from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract, ContractDetails
from ibapi.ticktype import TickType, TickTypeEnum
from ibapi.order import Order

from ibapi.utils import iswrapper
from ibapi.common import *
import pandas as pd
import time
from datetime import datetime, timedelta
from threading import RLock

class IBWrapper(EWrapper):
    def __init__(self, market_data_dict):
        EWrapper.__init__(self)
        self.optionsDataLock = RLock()

        self.req_ids = market_data_dict
        self.positions = pd.DataFrame(columns=["Account", "Symbol", "Quantity", "Avg Cost", "Id", "Underlying", "OptionType"])
        self.market_data = pd.DataFrame(columns=["Symbol", "DateTime", "Price"])
        with self.optionsDataLock:
            self.options_data = pd.DataFrame(columns=[
                "Id", "Symbol", "Strike", "undPrice", "Type", "delta", "bid", "ask", "Expiry",
                "close", "last", "high", "low",
                "ConId", "UnderConId", "impliedVol", "optPrice", "pvDividend", "gamma", "vega", "theta", "delta_diff"
            ])
            self.options_data.set_index('Id', inplace=True)
        self.market_data.set_index('Symbol', inplace=True)


        self.contract_details = {}
    
    @iswrapper
    def position(self, account: str, contract: Contract, position: float, avgCost: float):
        if (position!= 0):
            print("Position.", "Account:", account, "Symbol:", contract.symbol, "Quantity:", position, "Avg cost:", avgCost)
            new_row = {"Account": account, "Symbol": contract.localSymbol, "Quantity": position, "Avg Cost": avgCost, "Id": contract.conId, "Underlying": contract.symbol, "OptionType": contract.right}    
            self.positions = pd.concat([self.positions, pd.DataFrame([new_row])], ignore_index=True)

    @iswrapper
    def positionEnd(self):
        
        # self.positions = pd.concat([self.positions, pd.DataFrame([new_row])], ignore_index=True)
        return super().positionEnd()
    
    @iswrapper
    def tickOptionComputation(self, reqId, tickType, tickAttrib, impliedVol, delta, optPrice, pvDividend, gamma, vega, theta, undPrice):
        entry = self.req_ids[reqId]
        text = TickTypeEnum.toStr(tickType).lower()
        if ("id" in entry):
            id = entry["id"]
            #print("Tick Price. Ticker Id:", reqId, "Id:", id, "Type:", text, "Price:", price)
            with self.optionsDataLock:

                self.options_data.loc[id, "impliedVol"] = impliedVol
                self.options_data.loc[id, "delta"] = delta
                self.options_data.loc[id, "optPrice"] = optPrice
                self.options_data.loc[id, "pvDividend"] = pvDividend
                self.options_data.loc[id, "gamma"] = gamma
                self.options_data.loc[id, "vega"] = vega
                self.options_data.loc[id, "theta"] = theta
                self.options_data.loc[id, "undPrice"] = undPrice

        return super().tickOptionComputation(reqId, tickType, tickAttrib, impliedVol, delta, optPrice, pvDividend, gamma, vega, theta, undPrice)
    
    @iswrapper
    def tickPrice(self, reqId: TickerId, tickType: TickType, price: float, attrib):
        entry = self.req_ids[reqId]
        text = TickTypeEnum.toStr(tickType).lower()

        if ("symbol" in entry):
            symbol = entry["symbol"]
            #print("Tick Price. Ticker Id:", reqId, "Symbol:", symbol, "Type:", text, "Price:", price)
            if (text == "last" or text == "close"):
                new_row = {"Symbol": symbol, "DateTime": datetime.now(), "Price": price}
                #self.market_data = pd.concat([self.market_data, pd.DataFrame([new_row])], ignore_index=True)
                self.market_data.loc[symbol] = new_row
        elif ("id" in entry):
            id = entry["id"]
            #print("Tick Price. Ticker Id:", reqId, "Id:", id, "Type:", text, "Price:", price)
            with self.optionsDataLock:
                self.options_data.loc[id, text] = price


    @iswrapper
    def contractDetails(self, reqId: int, contractDetails: ContractDetails):
        
        self.contract_details[reqId] = contractDetails.contract
        
        new_row = {            
            "ConId": contractDetails.contract.conId,
            "Symbol": contractDetails.contract.symbol,
            "Strike": contractDetails.contract.strike,
            "Type": contractDetails.contract.right,
            "Expiry": contractDetails.contract.lastTradeDate,
            "UnderConId": contractDetails.underConId,
            
        }
        with self.optionsDataLock:
            self.options_data.loc[contractDetails.contract.conId] = new_row

    def contractDetailsEnd(self, reqId):
        return super().contractDetailsEnd(reqId)
    
   