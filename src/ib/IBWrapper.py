from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract, ContractDetails
from ibapi.ticktype import TickType, TickTypeEnum
from ibapi.order import Order
from data.ExpectedValueTrader import ExpectedValueTrader

from ibapi.utils import iswrapper
from ibapi.common import *
import pandas as pd
import time
from datetime import date, datetime, timedelta, timezone
from threading import RLock
import numpy as np

from util.Logger import Logger

class IBWrapper(EWrapper):
    def __init__(self, market_data_dict):
        EWrapper.__init__(self)
        self.optionsDataLock = RLock()

        self.req_ids = market_data_dict
        self.positions = pd.DataFrame(columns=["Account", "Symbol", "Quantity", "Avg Cost", "Id", "Underlying", "OptionType"])
        self.market_data = pd.DataFrame(columns=["Symbol", "DateTime", "Price"])
        with self.optionsDataLock:
            self.options_data = pd.DataFrame(columns=[
                "Id", "Symbol", "Strike", "dist", "undPrice", "Type", "delta", "bid", "ask", "time", "bid_size", "ask_size", "last", "DTE", "Expiry",
                "close", "high", "low",
                "ConId", "UnderConId", "impliedVol", "optPrice", "pvDividend", "gamma", "vega", "theta", "delta_diff"
            ])
            self.options_data.set_index('Id', inplace=True)
            self.options_data['time'] = pd.to_datetime(self.options_data['time'], utc=True, errors='coerce')

        self.market_data.set_index('Symbol', inplace=False)


        self.contract_details = {}
        self.contractDetailsPending = False
        


    @iswrapper
    def position(self, account: str, contract: Contract, position: float, avgCost: float):
        if position != 0:
            print("Position.", "Account:", account, "Symbol:", contract.symbol, 
                "Quantity:", position, "Avg cost:", avgCost)
            
            # Create a new row as a dictionary with all DataFrame columns
            new_row = {
                "Account": account,
                "Symbol": contract.localSymbol,
                "Quantity": position,
                "Avg Cost": avgCost,
                "Id": contract.conId,
                "Underlying": contract.symbol,
                "OptionType": contract.right
            }

            # Check if the contract.conId already exists in the DataFrame
            if contract.conId in self.positions["Id"].values:
                # Update the existing row
                self.positions.loc[self.positions["Id"] == contract.conId, new_row.keys()] = list(new_row.values())
            else:
                # Add the new row if it doesn't exist
                self.positions = pd.concat([self.positions, pd.DataFrame([new_row])], ignore_index=True)




    
    @iswrapper
    def tickOptionComputation(self, reqId, tickType, tickAttrib, impliedVol, delta, optPrice, pvDividend, gamma, vega, theta, undPrice):
        if not reqId in self.req_ids:
            return super().tickOptionComputation(reqId, tickType, tickAttrib, impliedVol, delta, optPrice, pvDividend, gamma, vega, theta, undPrice)
        entry = self.req_ids[reqId]
        text = TickTypeEnum.toStr(tickType).lower()

        if ("id" in entry):
            id = entry["id"]
            #print("Tick Price. Ticker Id:", reqId, "Id:", id, "Type:", text, "Price:", price)
            with self.optionsDataLock:
                if id not in self.options_data.index:
                    return
                strike = self.options_data.loc[id, "Strike"]
                type = self.options_data.loc[id, "Type"]

                if (undPrice is not None and strike is not None):
                    distance = abs(strike - undPrice)
                    percentage = undPrice * 0.05

                    self.options_data.loc[id, "dist"] = round(distance / undPrice * 100,2) 

                    if (distance > percentage):
                        self.cancel_options_market_data({"Id": id})

                now = datetime.now(timezone.utc)
                last = self.options_data.loc[id, "time"]

                if last and not pd.isna(last) and (now - last).total_seconds() > 60:
                    self.cancel_options_market_data({"Id": id})

                


                self.options_data.loc[id, "impliedVol"] = impliedVol
                self.options_data.loc[id, "delta"] = round(delta,3) if delta is not None else delta
                self.options_data.loc[id, "optPrice"] = optPrice
                self.options_data.loc[id, "pvDividend"] = pvDividend
                self.options_data.loc[id, "gamma"] = gamma
                self.options_data.loc[id, "vega"] = vega
                self.options_data.loc[id, "theta"] = theta
                self.options_data.loc[id, "undPrice"] = round(undPrice,2) if undPrice is not None else undPrice
                self.options_data.loc[id, "time"] = datetime.now(timezone.utc)
                

        return super().tickOptionComputation(reqId, tickType, tickAttrib, impliedVol, delta, optPrice, pvDividend, gamma, vega, theta, undPrice)
    
    @iswrapper
    def tickPrice(self, reqId: TickerId, tickType: TickType, price: float, attrib):
        if not reqId in self.req_ids:
            return super().tickPrice(reqId, tickType, price, attrib)
        
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
                """
                if text.startswith("delayed_"):
                      delayed_text = text[len("delayed_"):]
                      val =  self.options_data.loc[id, delayed_text]
                      if not (val > 0):
                          text = delayed_text
                """
                self.options_data.loc[id, text] = price
                self.options_data.loc[id, "time"] = datetime.now(timezone.utc)


    @iswrapper
    def tickSize(self, reqId, tickType, size):
        if not reqId in self.req_ids:
            return super().tickSize(reqId, tickType, size)
        
        entry = self.req_ids[reqId]
        text = TickTypeEnum.toStr(tickType).lower()

        if ("id" in entry):
            id = entry["id"]
            #print("Tick Size. Ticker Id:", reqId, "Id:", id, "Type:", text, "Size:", size)
            with self.optionsDataLock:             
                if text.startswith("delayed_"):
                      delayed_text = text[len("delayed_"):]
                      val =  self.options_data.loc[id, delayed_text]
                      if not (val > 0):
                          text = delayed_text

                self.options_data.loc[id, text] = float(size)
                self.options_data.loc[id, "time"] = datetime.now(timezone.utc)

        return super().tickSize(reqId, tickType, size)
    def cleanPrices(self):
        # Check if the necessary columns are in the DataFrame
        required_columns = ['bid', 'ask', 'last', 'close']
        missing_columns = [col for col in required_columns if col not in self.options_data.columns]
        if missing_columns:
            return
        with self.optionsDataLock:
            # Update 'bid' and 'ask' without loops:
            # For each row, if bid (or ask) is >= 0, keep it; 
            # otherwise, if last is >= 0, use last; else, use close.
            self.options_data['bid'] = np.where(
                (self.options_data['bid'] >= 0),
                self.options_data['bid'],
                np.where((self.options_data['last'] >= 0), self.options_data['last'], self.options_data['close'])
            )
            self.options_data['ask'] = np.where(
                (self.options_data['ask'] >= 0),
                self.options_data['ask'],
                np.where((self.options_data['last'] >= 0), self.options_data['last'], self.options_data['close'])
            )

    @iswrapper
    def error(self, reqId, errorTime, errorCode, errorString, advancedOrderRejectJson=""):
        Logger.log(f"Error. Id: {reqId}, Time: {errorTime}, Code: {errorCode}, Msg: {errorString}")
        return super().error(reqId, errorTime, errorCode, errorString, advancedOrderRejectJson)
    
    def calculate_dte(self, date_string):
        # Convert string 'YYYYMMDD' to a date object
        target_date = datetime.strptime(date_string[0], "%Y%m%d").date()
        
        # Get today's date
        today = date.today()
        
        # Subtract the dates to get a timedelta object
        delta = target_date - today
        
        # Return the number of days
        return delta.days   
    @iswrapper
    def contractDetails(self, reqId: int, contractDetails: ContractDetails):
        
        self.contract_details[reqId] = contractDetails.contract
        expiry = self._normalize_expiry(contractDetails.contract.lastTradeDate),
        dte = self.calculate_dte(expiry)
        now_utc = datetime.now(timezone.utc)
        new_row = {            
            "ConId": contractDetails.contract.conId,
            "Symbol": contractDetails.contract.symbol,
            "Strike": contractDetails.contract.strike,
            "Type": contractDetails.contract.right,
            "DTE": dte,
            "Expiry": expiry[0],
            "UnderConId": contractDetails.underConId,
            "reqId": reqId,
            'time': now_utc
            
        }
        with self.optionsDataLock:
            self.options_data.loc[contractDetails.contract.conId] = new_row


    @staticmethod
    def _normalize_expiry(raw: str) -> str:
        """
        Normalises all known IBAPI lastTradeDate formats to clean YYYYMMDD.

        Known formats:
        "20250321"              → "20250321"   (already clean)
        "20250321 15:00:00"     → "20250321"   (strip time suffix)
        "20250321-15:00:00"     → "20250321"   (strip UTC dash notation)
        "202503"                → "20250331"   (contract month → last day of month)
        """
        if not raw:
            return raw

        raw = raw.strip()

        # Strip time suffix: "20250321 15:00:00" or "20250321-15:00:00"
        # Only strip if it starts with 8 digits (avoid mangling YYYYMM)
        if len(raw) > 8 and raw[:8].isdigit():
            raw = raw[:8]

        # Contract month YYYYMM → last calendar day of that month
        if len(raw) == 6 and raw.isdigit():
            dt = pd.to_datetime(raw, format="%Y%m") + pd.offsets.MonthEnd(0)
            return dt.strftime("%Y%m%d")

        return raw  # Already YYYYMMDD
    
   