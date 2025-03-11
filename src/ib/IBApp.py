
from typing import List, Tuple
import numpy as np
from ib.IBWrapper import IBWrapper
from ib.IBClient import IBClient
from ibapi.contract import Contract

from ibapi.order import Order
from ibapi.contract import Contract, ComboLeg
from ibapi.order import Order
from ibapi.order_condition import OrderCondition, PriceCondition

from ibapi.utils import iswrapper
from ibapi.common import *
import pandas as pd
import time
from datetime import datetime, timedelta
from logging import getLogger
from threading import Thread, Condition, RLock, Lock

from sklearn.linear_model import LinearRegression  
import joblib
import talib
import scipy.stats as stats
from scipy.stats import norm
from util.StrategyEnum import StrategyEnum
from display.CandleStickChart import CandlestickChart

logger = getLogger(__name__)
logger.setLevel("INFO")


class IBApp(IBWrapper, IBClient):
    def __init__(self, daysBack="4 D"):
        self.logger = getLogger(__name__)
        self.market_data_req_ids = {}
        self.option_id_req = {}
        self.orders: List[Tuple[int, Contract, Order]] = []
        self.apiOrders = {}
        self.actionLog = []
        self.next_order_id = None
        self.last_used_order_id = None
        IBWrapper.__init__(self, self.market_data_req_ids)
        IBClient.__init__(self, self)
        self.reqId = 0
        self.cond = Condition()
        self.apiOrderLock = RLock()
        self.candleLock = RLock()
        self.candleData = {}
        self.daily_predictions = pd.DataFrame(columns=["date", "symbol", "predicted_close", "actual_close", "target_hit"])
        self.models = {}  # Store models for each symbol
        self.chart = CandlestickChart("SPX")
        self.historicalDataFinished = {}
        self.daysBack = daysBack

        self._update_interval = 0.2  # Throttle updates to at most one every 200ms

        # Start a dedicated updater thread.
        self._data_lock = Lock()
        self._updater_thread = Thread(target=self._chart_updater, daemon=True)
        self._updater_thread.start()

    def nextReqId(self):
        self.reqId += 1
        return self.reqId
    
    def nextOrderId(self):
        with self.cond:
            self.reqIds(-1)
            # Wait for the next valid order ID
            while self.next_order_id is None or self.next_order_id == self.last_used_order_id:
                self.cond.wait()
            self.last_used_order_id = self.next_order_id
            self.next_order_id = None
        return self.last_used_order_id

    @iswrapper
    def nextValidId(self, orderId: int):
        self.addToActionLog("Next Valid ID: " + str(orderId))
        self.next_order_id = orderId

        # Notify the waiting thread
        with self.cond:
            self.cond.notify()

    def addToActionLog(self, action):
        self.actionLog.append((datetime.now(), action))

    def fetch_positions(self):
        self.reqPositions()

    def request_market_data(self, symbols):
        for i, symbol_type in enumerate(symbols):
            symbol, type, ex = symbol_type
            contract = Contract()
            contract.symbol = symbol
            contract.secType = type
            if (type != "IND"):
                contract.currency = "USD"
            contract.exchange = ex

            req = self.nextReqId()
            self.market_data_req_ids[req] = { "symbol": symbol }
            self.reqMktData(req, contract, "", False, False, [])
            time.sleep(0.1)  # To avoid pacing violations

    def cancel_options_market_data(self,options_data):
        if (options_data["Id"] not in self.option_id_req):
            return
        
        req = self.option_id_req[options_data["Id"]]
        self.cancelMktData(req)
    
    def request_options_market_data(self,options_data):
        if (options_data is None):
            return
        if (options_data["ConId"] in self.option_id_req):
            # print("options ID already subscribed data available", options_data["ConId"])
            #print(options_data)

            return
        
        req = self.nextReqId()
        
        self.market_data_req_ids[req] = { "id": options_data["ConId"] }
        self.option_id_req[options_data["ConId"]] = req    
        contract = Contract()
        contract.conId = options_data["ConId"]
        contract.exchange = "SMART"
        contract.secType = "OPT"

        self.addToActionLog("Requesting market data for " + options_data["Symbol"] + " " + options_data["Type"] + " " + str(options_data["Strike"]))

        self.reqMktData(req, contract, "", False, False, [])
        time.sleep(0.1)  # To avoid pacing violations


    def fetch_options_data(self, symbols):
        for i, symbol_type in enumerate(symbols):
            symbol, type,ex = symbol_type

            for right in ["C", "P"]:
                contract = Contract()
                contract.symbol = symbol
                contract.secType = "OPT"
                contract.currency = "USD"
                contract.exchange = "SMART"
                contract.lastTradeDateOrContractMonth = datetime.today().strftime("%Y%m%d")
                contract.right = right
                self.reqContractDetails(i, contract)
                time.sleep(0.1)  # To avoid pacing violations

    def find_iron_condor_legs(self, puts, calls, current_price, distance=5, wingspan=5):
        # Calculate target prices
        target_put_strike = current_price - distance
        target_call_strike = current_price + distance

        # Sort the DataFrames
        puts_sorted = puts.sort_values(by="Strike")
        calls_sorted = calls.sort_values(by="Strike")

        # Find the closest strike for puts
        if not puts_sorted.empty:
            puts_sorted['Diff'] = (puts_sorted["Strike"] - target_put_strike).abs()
            putClosestToFiveToUnderlying = puts_sorted.loc[puts_sorted['Diff'].idxmin()]
        else:
            putClosestToFiveToUnderlying = None

        # Find the closest strike for calls
        if not calls_sorted.empty:
            calls_sorted['Diff'] = (calls_sorted["Strike"] - target_call_strike).abs()
            callClosestToFiveToUnderlying = calls_sorted.loc[calls_sorted['Diff'].idxmin()]
        else:
            callClosestToFiveToUnderlying = None

        # Define short legs (assuming the strikes were found)
        short_put_strike = putClosestToFiveToUnderlying["Strike"] if putClosestToFiveToUnderlying is not None else None
        short_call_strike = callClosestToFiveToUnderlying["Strike"] if callClosestToFiveToUnderlying is not None else None

        # Calculate long legs
        long_put_wing_strike = short_put_strike - wingspan if short_put_strike is not None else None
        long_call_wing_strike = short_call_strike + wingspan if short_call_strike is not None else None

        


        # Find the rows for the target strikes
        short_put_row = self.find_closest_strike_row(puts, short_put_strike, 'put')
        long_put_wing_row = self.find_closest_strike_row(puts, long_put_wing_strike, 'put')
        short_call_row = self.find_closest_strike_row(calls, short_call_strike, 'call')
        long_call_wing_row = self.find_closest_strike_row(calls, long_call_wing_strike, 'call')


        # Combine the results into a dictionary
        iron_condor_rows = {
            "long_call": long_call_wing_row,
            "short_call": short_call_row,
            "short_put": short_put_row,
            "long_put": long_put_wing_row
        }
        
        self.request_options_market_data(long_call_wing_row)
        self.request_options_market_data(short_call_row)
        self.request_options_market_data(short_put_row)
        self.request_options_market_data(long_put_wing_row)

        return iron_condor_rows



    def construct_from_underlying(self, symbol, distance=5, wingspan=5):
        if (symbol not in self.market_data.index):
            return "No market data available for " + symbol
        
        row = self.market_data.loc[symbol]
        current_price = row["Price"]
        with self.optionsDataLock:
            optionsForSymbol = self.options_data[self.options_data["Symbol"] == symbol]
        puts = optionsForSymbol[optionsForSymbol["Type"] == "P"]
        calls = optionsForSymbol[optionsForSymbol["Type"] == "C"]

        return self.find_iron_condor_legs(puts, calls, current_price,distance, wingspan)


    def getTilesData(self):
        def forDisplay(rows):
            """
            This function takes a dictionary where each entry is a dataframe row
            and returns a pandas DataFrame.

            Parameters:
            rows (dict): A dictionary where keys are row names and values are dictionaries representing rows.

            Returns:
            pd.DataFrame: A pandas DataFrame created from the input dictionary.
            """

            if (rows is None or not hasattr(rows, "items")):
                return pd.DataFrame()
            
            # Filter out None values
            filtered_rows = {key: value for key, value in rows.items() if value is not None}

            # Converting dictionary of rows to a DataFrame
            if filtered_rows:
                df = pd.DataFrame.from_dict(filtered_rows, orient='index')
                return df
            else:
                
                return  pd.DataFrame()
            
        return [
            {
                "title": "Market Data",
                "content": self.market_data
            },
            {
                "title": "Positions",
                "content": self.positions
            },
            {
                "title": "SPX Trades",
                "content": forDisplay(self.construct_from_underlying("SPX", 15, 5))
            },
                        {
                "title": "Free To Trade?",
                "content": forDisplay(self.checkUnderlyingOptions()).reset_index()

            },
            {
                "title": "SPX 15 Delta Bear Call",
                "content": forDisplay(self.build_credit_spread("SPX", 0.10, "C"))
            },
            {
                "title": "SPX 15 Delta Bull Put",
                "content": forDisplay(self.build_credit_spread("SPX", 0.15, "P"))
            },
            {
                "title": "Candle Data",
                "content": self.candleData.get("SPX", pd.DataFrame()).iloc[::-1],
                "colspan": 3,
                "exclude_columns": ["date", "minutes_since_open", "remaining_intervals", "volume", "wap", "count"]
            },

            {
                "title": "Orders",
                "content": pd.DataFrame(self.orders)
            },
            {
                "title": "API Orders",
                "content": pd.DataFrame.from_dict(self.apiOrders, orient='index')
            },
            {
                "title": "Action Log",
                "content": pd.DataFrame(self.actionLog).sort_values(by=0, ascending=False)[1] if len(self.actionLog) > 0 else pd.DataFrame() 
            },
            {
            "title": "Options Data",
            "content": self.options_data# .dropna(subset=['delta'])
            },
            {
                "title": "Profitability",
                "content": self.calculate_signal_metrics("SPX")
            },
            {
                "title": "SPX Trades",
                "content": self.get_trades("SPX"),
                "colspan": 3
            }

            
        ]
        """
        {
            "title": "Options Data",
            "content": self.options_data.dropna(subset=['delta'])
        },
        """

    def send_iron_condor_order(self, symbol, distance=5, wingspan=5):
        rows = self.construct_from_underlying(symbol, distance, wingspan)
        self.place_combo_order(rows)

    def send_credit_spread_order(self, symbol, target_delta, type="C", wingspan=10):
        rows = self.build_credit_spread(symbol, target_delta, type, wingspan)
        self.place_combo_order(rows, 50, None, "CreditSpread-"+type)

    def create_order(self, action, quantity, orderType, price=None):
        order = Order()
        order.action = action
        order.totalQuantity = quantity
        order.orderType = orderType
        if price:
            order.lmtPrice = price
        order.transmit = False
        return order




    def place_combo_order(self, contract_rows, tp=None, sl=None, ref="IronCondor"):
        if(contract_rows is None):
            return
        
        # Access the first key-value pair from the dictionary
        first_key = next(iter(contract_rows))
        ref = "Bot-"+ref+"-"+contract_rows[first_key]["Symbol"]
        # Create Combo Contract
        combo_contract = Contract()
        combo_contract.symbol = contract_rows[first_key]["Symbol"]
        combo_contract.secType = "BAG"
        combo_contract.currency = "USD"
        combo_contract.exchange = "SMART"
        combo_contract.comboLegs = []

        for key, row in contract_rows.items():
            leg = ComboLeg()
            leg.conId = row["ConId"]
            leg.ratio = 1
            leg.action = "BUY" if "long" in key else "SELL"
            leg.exchange = "SMART"
            combo_contract.comboLegs.append(leg)

        # Calculate the limit price for the LMT order
        limit_price = 0
        for key, row in contract_rows.items():
            if "long" in key:
                limit_price += row["ask"]  # Ask price for long legs
            else:
                limit_price -= row["bid"]  # Bid price for short legs

        # Create a Limit Order (parent order)
        limit_order = self.create_order("BUY", 1, "LMT", limit_price)
        limit_order.orderRef = ref

        # Place the Combo Order (parent order)
        parent_order_id = self.nextOrderId()

        # Create a Stop Order (child order)
        """
        stop_price = 0
        for key, row in contract_rows.items():
            if "short" in key:
                stop_price += row["bid"]  # Add bid price for short legs
            else:
                stop_price -= row["ask"]  # Subtract ask price for long legs
        """
        def round_to_next_multiple(number, multiple=0.05):
            """
            Round the given number to the nearest multiple of 'multiple'.
            :param number: float, the number to be rounded
            :param multiple: float, the multiple to round to (default is 0.05)
            :return: float, the rounded number
            """
            return round(number / multiple) * multiple

        if (sl is None):
            stop_order = self.create_order("SELL", 1, "MKT")
            stop_order.orderRef = ref
            stop_order.parentId = parent_order_id  # Link to parent order

            # Add conditions for short legs being in-the-money
            for key, row in contract_rows.items():
                if "long" in key:
                    continue

                condition = PriceCondition()
                condition.conId = row["UnderConId"]
                condition.isConjunctionConnection = False
                condition.triggerMethod = 2

                if (condition.conId == 416904 or condition.conId == 13455763 or condition.conId == 137851301): #SPX, VIX, XSP
                    condition.exchange = 'CBOE'
                else:
                    condition.exchange = 'SMART'

                if "short_call" in key:
                    condition.isMore = True
                    condition.price = row["Strike"] + 0.5
                    stop_order.conditions.append(condition)
                elif "short_put" in key:
                    condition.isMore = False
                    condition.price = row["Strike"] - 0.5
                    stop_order.conditions.append(condition)
        else:
            stop_order = self.create_order("SELL", 1, "STP")
            stop_order.auxPrice = round_to_next_multiple(limit_price * (sl/100))
            stop_order.orderRef = ref
            stop_order.parentId = parent_order_id  # Link to parent order

        tpOrder = None
        if (tp is not None):
            tpOrder = self.create_order("SELL", 1, "LMT", round_to_next_multiple(limit_price * (tp/100)))
            tpOrder.orderRef = ref
            tpOrder.parentId = parent_order_id  # Link to parent order
        # Place the Stop Order (child order)
        # self.placeOrder(self.nextReqId(), combo_contract, stop_order)

        self.orders.append((parent_order_id, combo_contract, limit_order))
        self.placeOrder(parent_order_id, combo_contract, limit_order)

        if (tpOrder is not None):
            tpOrderId = self.nextOrderId()
            self.orders.append((tpOrderId, combo_contract, tpOrder))
            self.placeOrder(tpOrderId, combo_contract, tpOrder)


        stpLossId = self.nextOrderId()
        stop_order.transmit = True
        self.orders.append((stpLossId, combo_contract, stop_order))
        self.placeOrder(stpLossId, combo_contract, stop_order)

    @iswrapper
    def error(self, reqId, errorTime, errorCode, errorString, advancedOrderRejectJson=""):
        if errorCode not in [2104, 2106,2158]:
            print("Error. Id: ", reqId, "Time: ", errorTime, " Code: ", errorCode, " Msg: ", errorString)
            self.addToActionLog(" Code: " + str(errorCode) + " Msg: " + str(errorString) + " Id: " + str(reqId))
        return super().error(reqId, errorTime, errorCode, errorString, advancedOrderRejectJson)
    
    def filter_options_data(self, symbol, type="C"):
        with self.optionsDataLock:
            # Ensure the boolean Series and DataFrame have the same index
            mask = (self.options_data["Symbol"] == symbol) & (self.options_data["Type"] == type)
            aligned_mask = mask.reindex(self.options_data.index, fill_value=False)

            # Print indices for debugging
            print("self.options_data index:", self.options_data.index)
            print("aligned_mask index:", aligned_mask.index)

            # Use the aligned boolean Series for indexing
            df_symbol = self.options_data.loc[aligned_mask, :]
            return df_symbol

    def get_closest_delta_row(self, symbol, target_delta, type="C"):
        df_symbol = self.filter_options_data(symbol, type)
        # Ensure delta column exists, set default if not
        #df_symbol['delta_diff'] = np.abs(df_symbol.get('delta', pd.Series(0)).abs() - target_delta)
        df_symbol['delta_diff'] = np.where(df_symbol['delta'].notna(), np.abs(df_symbol['delta'].abs() - target_delta), np.nan)

        # Check if delta_diff exists and find the closest row
        if 'delta_diff' in df_symbol.columns and not df_symbol['delta_diff'].isnull().all():
            return df_symbol.loc[df_symbol['delta_diff'].idxmin()]
        else:
            return None
    
    def find_closest_strike_row(self, df, strike, option_type):
        """
        Helper function to find the closest row in a DataFrame based on the 'Strike' column.

        Parameters:
        df (pd.DataFrame): The DataFrame to search.
        strike (float or None): The strike value to search for.
        option_type (str): The type of option, either 'put' or 'call'.

        Returns:
        pd.Series or None: The found row as a pandas Series or None if not found.
        """
        if strike is None:
            return None

        if option_type == 'put' or option_type == 'P':
            # Find the closest strike less than or equal to the target strike
            matching_rows = df[df["Strike"] <= strike]
            if not matching_rows.empty:
                closest_row = matching_rows.loc[matching_rows["Strike"].idxmax()]
                return closest_row
        
        elif option_type == 'call' or option_type == 'C':
            # Find the closest strike greater than or equal to the target strike
            matching_rows = df[df["Strike"] >= strike]
            if not matching_rows.empty:
                closest_row = matching_rows.loc[matching_rows["Strike"].idxmin()]
                return closest_row

        return None
    
    def build_credit_spread(self, symbol, target_delta, type="C", wingspan=10):
        if (self.options_data.empty):
            return None
        df_symbol = self.filter_options_data(symbol, type)
        deltaRow = self.get_closest_delta_row(symbol, target_delta, type)
        if deltaRow is None:
            return None

        strike = deltaRow['Strike']

        wingrow = None
        if type == "C":
            wingrow = self.find_closest_strike_row(df_symbol, strike + wingspan, "C")
            self.request_options_market_data(wingrow)

            return {
                "long_call": wingrow,
                "short_call": deltaRow
            }
        elif type == "P":
            wingrow = self.find_closest_strike_row(df_symbol, strike - wingspan, "P")
            self.request_options_market_data(wingrow)

            return {
                "short_put": deltaRow,
                "long_put": wingrow
            }
    
    def subscribe_to_target_delta(self, symbol, target_delta, type="C"):
        closest_row = self.get_closest_delta_row(symbol, target_delta, type)
        df_symbol = self.options_data.loc[(self.options_data["Symbol"] == symbol) & (self.options_data["Type"] == type), :]

        if closest_row is None:
            return None
        
        closest_strike_row = None

        absDelta = np.abs(closest_row['delta'])
        strike = closest_row['Strike']

        if absDelta < target_delta:
            if type == "C":
                closest_strike_row = self.find_closest_strike_row(df_symbol, strike - 5, "C")
            elif type == "P":
                closest_strike_row = self.find_closest_strike_row(df_symbol, strike + 5, "P")
        elif absDelta > target_delta:
            if type == "C":
                closest_strike_row = self.find_closest_strike_row(df_symbol, strike + 5, "C")
            elif type == "P":
                closest_strike_row = self.find_closest_strike_row(df_symbol, strike - 5, "P")

        self.request_options_market_data(closest_strike_row)
        return closest_row
    
    def find_closest_strike_for_symbol(self, symbol, strike, type):
        df_symbol = self.filter_options_data(symbol, type)
        return self.find_closest_strike_row(df_symbol, strike, type)
    
    @iswrapper
    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
        with self.apiOrderLock:
            if orderId not in self.apiOrders:
                self.apiOrders[orderId] = {}

            self.apiOrders[orderId]["status"] = status
            self.apiOrders[orderId]["filled"] = filled
            self.apiOrders[orderId]["remaining"] = remaining
            self.apiOrders[orderId]["avgFillPrice"] = avgFillPrice
            self.apiOrders[orderId]["permId"] = permId
            self.apiOrders[orderId]["parentId"] = parentId
            self.apiOrders[orderId]["lastFillPrice"] = lastFillPrice
            self.apiOrders[orderId]["clientId"] = clientId
            self.apiOrders[orderId]["whyHeld"] = whyHeld
            self.apiOrders[orderId]["mktCapPrice"] = mktCapPrice

            self.addToActionLog("orderStatus "+ str(orderId) +" "+ str(status))

        return super().orderStatus(orderId, status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice)
    
    def checkUnderlyingOptions(self):
        symbol = "SPX"
        signal = None
        date = None
        put = None
        call = None
        if ("SPX" in self.candleData):
            df: pd.DataFrame = self.candleData[symbol]
            if ("final_signal" in df.columns):
                row = df.iloc[-1]
                signal = row["final_signal"]
                date = row["date"]
                put = row["put_strike"]
                call = row["call_strike"]
        return {
            "hasSpxCall": self.hasOrdersOrPositions(symbol, "C"),
            "hasSpxPut": self.hasOrdersOrPositions(symbol, "P"),
            "signal": signal,
            "date": date,
            "call": call,
            "put": put

        }
    
    @iswrapper
    def openOrder(self, orderId, contract, order: Order, orderState):
        self.addToActionLog("openOrder "+str(orderId)+" "+ str(order.orderId))

        with self.apiOrderLock:
            if orderId not in self.apiOrders:
                self.apiOrders[orderId] = {}

            self.apiOrders[orderId]["contract"] = contract
            self.apiOrders[orderId]["order"] = order
            self.apiOrders[orderId]["orderState"] = orderState
        return super().openOrder(orderId, contract, order, orderState)
    
    def check_open_orders(self, underlying, option_type):
        open_orders = []

        for orderId, order in self.apiOrders.items():
            contract: Contract = order.get("contract")

            if contract and contract.secType in ["OPT", "BAG"] and contract.symbol == underlying:
                if contract.secType == "OPT" and contract.right == option_type:
                    open_orders.append(order)
                elif contract.secType == "BAG":
                    for l in contract.comboLegs:
                        leg: ComboLeg = l
                        try:
                            existingOption = self.options_data.loc[leg.conId]
                            if existingOption["Symbol"] == underlying and existingOption["Type"] == option_type:
                                open_orders.append(order)
                        except KeyError:
                            pass  # Handle cases where leg.conId is not in self.options_data


                        """
                        if leg.symbol == underlying and leg.right == option_type:
                            open_orders.append(order)
                            break
                        """

        return open_orders

    def hasOrdersOrPositions(self, symbol, type=None):

        for (id,contract, order) in self.orders:
            if (type is not None):
                if (contract.symbol == symbol and contract.right == type):
                    return True
            else:
                if (contract.symbol == symbol):
                    return True
            
        for index, row in self.positions.iterrows():
            if (type is not None):
                if (row["Symbol"] == symbol and row["OptionType"] == type and row["Quantity"] != 0):
                    return True
            else:
                if (row["Symbol"] == symbol and row["Quantity"] != 0):
                    return True
                
        return len(self.check_open_orders(symbol,type)) > 0

    def reqHistoricalDataFor(self, symbol, type, exchange):
        contract = Contract()
        contract.symbol = symbol
        contract.secType = type
        contract.exchange = exchange
        
        contract.currency = "USD"

        reqId = self.nextReqId()

        self.market_data_req_ids[reqId] = { "symbol": symbol }
        self.historicalDataFinished[symbol] = False
        self.addToActionLog("Requesting historical data for " + symbol + " "+ str(reqId ))
        self.reqHistoricalData(reqId, contract, "", self.daysBack, "5 mins", "TRADES", 1, 1, True, [])
        time.sleep(0.1)  # To avoid pacing violations



    @iswrapper
    def historicalData(self, reqId, bar: BarData):
        self.logger.debug(f"Received historicalData for reqId: {reqId}")
        
        if reqId not in self.market_data_req_ids:
            self.logger.error(f"reqId {reqId} not found in market_data_req_ids")
            self.cancelHistoricalData(reqId)
            return

        symbol = self.market_data_req_ids[reqId]["symbol"]
        if symbol not in self.candleData:
            self.candleData[symbol] = pd.DataFrame(columns=["date", "datetime", "narrow_bands", "open", "close", "trend", "call_strike", "put_strike", "call_p", "put_p", "c_dist_p", "p_dist_p", "final_signal", "temp_signal", "high", "low",  "volume", "wap", "count"])

        bar_data = {
            "date": bar.date,
            "close": bar.close,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "volume": bar.volume,
            "wap": bar.wap,
            "count": bar.barCount
        }

        # Convert bar_data to a DataFrame with the same columns as self.candleData[symbol]
        bar_data_df = pd.DataFrame([bar_data], columns=self.candleData[symbol].columns)

        # Debugging output
        #self.logger.debug(f"bar_data_df: {bar_data_df}")
        #self.logger.debug(f"self.candleData[symbol].columns: {self.candleData[symbol].columns}")
        #self.logger.debug(f"self.candleData[symbol] before update: {self.candleData[symbol]}")

        existing_index = self.candleData[symbol][self.candleData[symbol]['date'] == bar.date].index
        if not existing_index.empty:
            self.candleData[symbol].loc[existing_index] = bar_data_df.values[0]
        else:
            self.candleData[symbol] = pd.concat([self.candleData[symbol], bar_data_df], ignore_index=True)

        # Debugging output
        #self.logger.debug(f"self.candleData[symbol] after update: {self.candleData[symbol]}")

        return super().historicalData(reqId, bar)

    def addIndicators(self, reqId):
        symbol = self.market_data_req_ids[reqId]["symbol"]

        with self.candleLock:
            # --- 1. Parse DateTime (Assuming Berlin Time) ---
            self.candleData[symbol]["datetime"] = pd.to_datetime(
                self.candleData[symbol]["date"], format="%Y%m%d %H:%M:%S %Z"
            )
            
            # --- 2. Technical Indicators Using TA-Lib ---
            # Convert the 'close' prices to a numpy array for TA-Lib
            close = self.candleData[symbol]["close"].values.astype(float)
            high = self.candleData[symbol]["high"].values.astype(float)
            low = self.candleData[symbol]["low"].values.astype(float)
            
            # Calculate SMA with period 5
            sma5 = talib.SMA(close, timeperiod=5)
            sma50 = talib.SMA(close, timeperiod=50)
            self.candleData[symbol]["SMA5"] = sma5.round(2)
            self.candleData[symbol]["SMA50"] = sma50.round(2)

            rsi = talib.RSI(close, timeperiod=14)
            self.candleData[symbol]["RSI"] = rsi.round(2)

            # Calculate ATR (absolute value)
            atr = talib.ATR(
                self.candleData[symbol]["high"].values.astype(float),
                self.candleData[symbol]["low"].values.astype(float),
                close,
                timeperiod=14
            )

            # Convert ATR to percentage of closing price
            atr_percent = (atr / close) * 100
            self.candleData[symbol]["ATR"] = atr.round(2)
            self.candleData[symbol]["ATR_percent"] = atr_percent

            self.candleData[symbol]["ATR_up"] = atr + close
            self.candleData[symbol]["ATR_down"] = close - atr


            macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            self.candleData[symbol]["MACD"] = macd
            self.candleData[symbol]["MACDSignal"] = macdsignal
            self.candleData[symbol]["MACDHist"] = macdhist


            
            # Calculate Bollinger Bands with period 20 and 2 standard deviations
            upperband, middleband, lowerband = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            self.candleData[symbol]["bb_up"] = upperband.round(2)
            self.candleData[symbol]["bb_mid"] = middleband.round(2)
            self.candleData[symbol]["bb_low"] = lowerband.round(2)

            isNarrow = self.checkNarrowBands(self.candleData[symbol])
            

            # Technical signal based on Bollinger Bands:
            # - If close > upper band: SELL PUT CREDIT SPREAD
            # - If close < lower band: SELL CALL CREDIT SPREAD
            # Otherwise: HOLD
            """
            tech_signal = np.where(
                (close > upperband) & (atr_percent < 0.3) & (rsi > 70),
                StrategyEnum.SellCall,  # Signal for selling call credit spread
                np.where(
                    (close < lowerband) & (atr_percent < 0.3) & (rsi < 30),
                    StrategyEnum.SellPut,  # Signal for selling put credit spread
                    np.where(
                        (atr_percent < 0.2) & (abs(close - middleband) < 0.25 * atr) & (rsi > 30) & (rsi < 70),
                        StrategyEnum.SellIronCondor,  # Signal for Iron Condor in low volatility, neutral RSI
                        StrategyEnum.Hold  # Default to holding if no conditions are met
                    )
                )
            )
            """
            # Modify Iron Condor condition to avoid tight Bollinger Bands
            iron_condor_condition = (
                (atr_percent < 0.2) &  # Low volatility
                (abs(close - middleband) < 0.25 * atr) &  # Price is near the middle, not too close to the bands
                (rsi > 40) & (rsi < 60)  # RSI in a neutral range (avoid extremes)
            )



            # Apply strategy logic with updated conditions
            tech_signal = np.where(
                isNarrow,  # Check if the isNarrow condition is True
                StrategyEnum.Hold,  # Set to Hold if isNarrow is True
                np.where(
                    (close > upperband) & (atr_percent < 0.3) & (rsi >= 66),
                    StrategyEnum.SellCall,  # Signal for selling call credit spread
                    np.where(
                        (close < lowerband) & (atr_percent < 0.3) & (rsi <= 33),
                        StrategyEnum.SellPut,  # Signal for selling put credit spread
                        np.where(
                            iron_condor_condition,  # Only signal Iron Condor when the market is range-bound
                            StrategyEnum.SellIronCondor,  # Confirm Iron Condor if conditions are met
                            StrategyEnum.Hold  # Default to holding if no conditions are met
                        )
                    )
                )
            )




            self.candleData[symbol]["tech_signal"] = tech_signal


            
            # Determine trend based on SMA5 (if available)
            trend = np.where(close > sma5, "UP", "DOWN")
            self.candleData[symbol]["trend"] = trend
            
            # --- 3. Probability-Based Indicators for SPX Spreads ---
            # Berlin trading hours: 15:30 (930 minutes) to 22:00 (1320 minutes); total = 390 minutes.
            # Calculate minutes since market open (15:30)
            self.candleData[symbol]["minutes_since_open"] = (
                self.candleData[symbol]["datetime"].dt.hour * 60 +
                self.candleData[symbol]["datetime"].dt.minute - 930
            ).clip(lower=0)
            
            # Calculate remaining minutes in the trading day
            remaining_minutes = (390 - self.candleData[symbol]["minutes_since_open"]).clip(lower=0)
            # Convert remaining minutes into number of 5-minute intervals (there are 78 intervals in 390 minutes)
            remaining_intervals = (remaining_minutes // 5)
            self.candleData[symbol]["remaining_intervals"] = remaining_intervals
            
            # Use available data (ideally today's candles) for volatility calculations.
            # Here we compute five-minute returns and their standard deviation in percentage terms.
            returns = self.candleData[symbol]["close"].pct_change()
            five_min_std = returns.std()
            if np.isnan(five_min_std) or five_min_std == 0:
                five_min_std = 0.001  # Minimal threshold
            
            # Scale five-minute volatility to daily volatility (using 78 intervals)
            daily_vol = five_min_std * np.sqrt(78)
            
            # Convert percentage volatility to absolute (dollar) volatility
            # using the current price. Note: 'current_price' is a series.
            current_price = self.candleData[symbol]["close"]
            daily_abs_std = current_price * daily_vol
            
            # Compute the remaining absolute volatility for the day
            remaining_std_dev = daily_abs_std * np.sqrt(remaining_intervals / 78)
            # Avoid too-small values in the denominator by ensuring a minimum volatility
            remaining_std_dev = np.maximum(remaining_std_dev, current_price * five_min_std)
            
            latestCallRow = self.get_closest_delta_row(symbol, 0.10, "C")
            latestPutRow = self.get_closest_delta_row(symbol, 0.15, "P")

            

            # Add the 'put_strike' and 'call_strike' columns if not already present
            if "put_strike" not in self.candleData[symbol]:
                self.candleData[symbol]["put_strike"] = np.nan
            if "call_strike" not in self.candleData[symbol]:
                self.candleData[symbol]["call_strike"] = np.nan

            if (latestPutRow is not None):
                put_strike_latest = latestPutRow["Strike"]
                # Set the most recent row's values for 'put_strike' and 'call_strike'
                self.candleData[symbol].iloc[-1, self.candleData[symbol].columns.get_loc("put_strike")] = put_strike_latest
            
            if (latestCallRow is not None):
                call_strike_latest = latestCallRow["Strike"]
                self.candleData[symbol].iloc[-1, self.candleData[symbol].columns.get_loc("call_strike")] = call_strike_latest
            
            # Calculate strikes
            fallback_put_strike_15_delta = current_price - 40
            fallback_call_strike_10_delta = current_price + 40

            # Round to nearest multiple of 5
            fallback_put_strike = np.floor(fallback_put_strike_15_delta / 5) * 5
            fallback_call_strike = np.ceil(fallback_call_strike_10_delta / 5) * 5

            # Calculate fallback values rounded to the nearest multiple of 5
            # fallback_put_strike = (current_price - remaining_std_dev * 0.8 ) // 5 * 5  # Round down
            # fallback_call_strike = np.ceil((current_price + remaining_std_dev * 0.8 ) / 5) * 5  # Round up

            # Fill NaN values with the rounded fallback values
            self.candleData[symbol]["put_strike"] = self.candleData[symbol]["put_strike"].fillna(fallback_put_strike)
            self.candleData[symbol]["call_strike"] = self.candleData[symbol]["call_strike"].fillna(fallback_call_strike)


            
            # Compute Z-scores using absolute differences (in dollars)
            z_scores_call = (self.candleData[symbol]["call_strike"] - current_price) / remaining_std_dev
            z_scores_put = (self.candleData[symbol]["put_strike"] - current_price) / remaining_std_dev
            
            # Clip extreme Z-scores to a reasonable range to avoid saturation of the CDF
            z_scores_call = np.clip(z_scores_call, -4, 4)
            z_scores_put = np.clip(z_scores_put, -4, 4)
            
            # Convert Z-scores to cumulative probabilities
            call_probabilities = norm.cdf(z_scores_call).round(4)      # Probability for call side
            put_probabilities = 1 - norm.cdf(z_scores_put).round(4)      # Probability for put side
            
            self.candleData[symbol]["call_p"] = call_probabilities
            self.candleData[symbol]["put_p"] = put_probabilities

            current_price = self.candleData[symbol]["close"]
            call_strike = self.candleData[symbol]["call_strike"]
            put_strike = self.candleData[symbol]["put_strike"]

            self.candleData[symbol]["c_dist_p"] = (( call_strike - close ) / close * 100).round(2)
            self.candleData[symbol]["p_dist_p"] = (( close - put_strike ) / close * 100).round(2)
            
            # --- 4. Combine the Signals ---
            # Here we combine the TA-Lib technical signal and the probability-based indicators.
            # For instance, if the technical indicator signals SELL PUT and the call probability > 0.8,
            # we issue a SELL PUT CREDIT SPREAD signal. Similarly for SELL CALL.


            final_signal = np.where(
                (tech_signal == StrategyEnum.SellPut) & (put_probabilities >= 0.8),
                StrategyEnum.SellPut,  # Confirm SellPut signal based on call probabilities
                np.where(
                    (tech_signal == StrategyEnum.SellCall) & (call_probabilities >= 0.8),
                    StrategyEnum.SellCall,  # Confirm SellCall signal based on put probabilities
                    np.where(
                        (tech_signal == StrategyEnum.SellIronCondor) & (call_probabilities > 0.85) & (put_probabilities > 0.85),
                        StrategyEnum.SellIronCondor,  # Confirm Iron Condor if both probabilities are high
                        StrategyEnum.Hold  # Default to holding if no conditions are met
                    )
                )
            )


            self.candleData[symbol]["temp_signal"] = final_signal

            # Create the trading_signal column
            def determine_trading_signal(prev, curr):
                if prev == StrategyEnum.SellPut and curr == StrategyEnum.Hold:
                    return StrategyEnum.SellPut
                elif prev == StrategyEnum.SellCall and curr == StrategyEnum.Hold:
                    return StrategyEnum.SellCall
                elif prev == StrategyEnum.SellIronCondor and curr == StrategyEnum.Hold:
                    return StrategyEnum.SellIronCondor
                return StrategyEnum.Hold

            self.candleData[symbol]["final_signal"] = self.candleData[symbol]["temp_signal"].shift(1).combine(self.candleData[symbol]["temp_signal"], determine_trading_signal)
            self.check_signal_profitability(self.candleData[symbol])

    def checkNarrowBands(self, df):
        # Assuming your dataframe is named `df` and contains `upper_band` and `lower_band` columns
        df['band_width'] = (df['bb_up'] - df['bb_low']).round(1)

        # Calculate descriptive statistics
        mean_width = df['band_width'].mean()
        std_width = df['band_width'].std()
        percentile_10 = df['band_width'].drop_duplicates().quantile(0.25)  # 10th percentile

        # Define a threshold for "too narrow" (you can choose based on your preference)
        threshold = percentile_10  # Example: Use 10th percentile as the threshold

        # Flag rows where the band width is below the threshold
        df['narrow_bands'] = df['band_width'] < threshold
        return df['narrow_bands']
        
    @iswrapper
    def historicalDataEnd(self, reqId, start: str, end: str):
        self.addToActionLog("Historical data request completed for reqId: " + str(reqId))
        with self.candleLock:
            symbol = self.market_data_req_ids[reqId]["symbol"]

            self.historicalDataFinished[symbol] = True

            self.addIndicators(reqId)
        return super().historicalDataEnd(reqId, start, end)

    @iswrapper
    def historicalDataUpdate(self, reqId, bar):

        try:
            with self.candleLock:
                self.historicalData(reqId, bar)
                self.addIndicators(reqId)
        except Exception as e:
            self.logger.error(f"Error in historicalDataUpdate: {e}")
            self.addToActionLog("Error in historicalDataUpdate: ")
            self.addToActionLog(str(e))
        return super().historicalDataUpdate(reqId, bar)


    def _chart_updater(self):
        self.addToActionLog("_chart_updater started")
        symbol = "SPX"
        while True:
            time.sleep(self._update_interval)
            with self._data_lock:
                self.addToActionLog("Checking Chart")
                if symbol not in self.candleData:
                    self.addToActionLog("Symbol not found; skipping update")
                    continue

                self.addToActionLog("Symbol found")
                data: pd.DataFrame = self.candleData[symbol]
                if data.empty:
                    self.addToActionLog("Empty CandleData; skipping update")
                    continue

                self.addToActionLog("Update Chart")
                # For testing, call update() directly rather than scheduling with after():
                self.chart.update(data)
            self.addToActionLog("DataLock Released")


    def plot_candlestick(self, symbol):

        if symbol not in self.candleData:
            return
        
        if symbol not in self.historicalDataFinished or not self.historicalDataFinished[symbol]:
            return

        data: pd.DataFrame = self.candleData[symbol]
        if (data.empty):
            return
        self.addToActionLog("Update Chart")
        self.chart.update(data)


    def calculate_signal_metrics(self, symbol):
        """
        Calculates the win rate for each signal and the breach depth if the strike was not profitable.
        
        Args:
            symbol (str): The symbol of the asset being analyzed.
        
        Returns:
            dict: A dictionary with win rates and average breach depths for each signal.
        """
        if symbol not in self.candleData:
            return
        
        df = self.candleData[symbol]
        metrics = {}

        # Ensure the required columns exist before calculations
        required_columns = ['SellCall_profit', 'SellPut_profit', 'SellIronCondor_profit']
        for col in required_columns:
            if col not in df.columns:
                df[col] = None  # Add missing columns with default None values

        def calc_metric(dataCol, targetCol):
        # Calculate SellCall metrics
            if dataCol in df.columns:
                # Filter rows with non-NaN values in SellCall_profit
                sell_call_data = df[df[dataCol].notna()]
                if not sell_call_data.empty:
                    # Count True and False values
                    true_count = (sell_call_data[dataCol] == True).sum()
                    false_count = (sell_call_data[dataCol] == False).sum()
                    
                    # Calculate win rate percentage
                    win_rate = (true_count / (true_count + false_count)) * 100 if (true_count + false_count) > 0 else 0
                    
                    # Add counts and win rate to metrics
                    metrics[targetCol] = {
                        "Win Rate (%)": win_rate,
                        "True Count": true_count,
                        "False Count": false_count
                    }
                else:
                    # No data for SellCall_profit
                    metrics[targetCol] = {
                        "Win Rate (%)": 0,
                        "True Count": 0,
                        "False Count": 0
                    }

        calc_metric("SellCall_profit", "SellCall")
        calc_metric("SellPut_profit", "SellPut")
        calc_metric("SellIronCondor_profit", "SellIronCondor")

        return pd.DataFrame.from_dict(metrics, orient="index").reset_index()


    def check_signal_profitability(self, df):
        """
        Checks if the signal was profitable by determining whether the relevant strikes were reached during the day.

        Args:
            df (pd.DataFrame): DataFrame containing columns such as 'final_signal', 'call_strike',
                            'put_strike', 'high', 'low', and other indicators.

        Returns:
            pd.DataFrame: DataFrame with new columns: 'SellCall_profit', 'SellPut_profit', 'SellIronCondor_profit'.
        """
        # Initialize profitability columns
        df['SellCall_profit'] = None
        df['SellPut_profit'] = None
        df['SellIronCondor_profit'] = None

        # Try 1
        # Identify unique days using 'remaining_intervals'
        df['day'] = (df['remaining_intervals'] < df['remaining_intervals'].shift(1)).cumsum()

        # Add 'day_high' and 'day_low' columns
        df['day_high'] = df.groupby('day')['high'].transform('max')
        df['day_low'] = df.groupby('day')['low'].transform('min')

       
        # Instead of reversing the entire DataFrame and risking misalignment, do it per group.
        df['day_high_remaining_2'] = df.groupby('day')['high'].transform(lambda x: x[::-1].cummax()[::-1])
        df['day_low_remaining_2'] = df.groupby('day')['low'].transform(lambda x: x[::-1].cummin()[::-1])

        # Try 2
        # Identify new day start
        df['new_day'] = df['remaining_intervals'] > df['remaining_intervals'].shift(1)

        # Forward-fill the new day to create groups
        df['day_group'] = df['new_day'].cumsum()

        # Calculate high and low for the remaining part of the day
        remaining_highs = []
        remaining_lows = []

        for i in range(len(df)):
            # Filter data from the current row to the end of the same day
            same_day_data = df[(df['day_group'] == df.loc[i, 'day_group']) & (df.index >= i)]
            remaining_highs.append(same_day_data['high'].max())
            remaining_lows.append(same_day_data['low'].min())

        df['day_high_remaining'] = remaining_highs
        df['day_low_remaining'] = remaining_lows



        # Vectorized calculation for each profitability column
        df['SellCall_profit'] = np.where(
            df['final_signal'] == StrategyEnum.SellCall, 
            df['call_strike'] >= df['day_high_remaining'], 
            None
        )

        df['SellPut_profit'] = np.where(
            df['final_signal'] == StrategyEnum.SellPut, 
            df['put_strike'] <= df['day_low_remaining'], 
            None
        )

        df['SellIronCondor_profit'] = np.where(
            df['final_signal'] == StrategyEnum.SellIronCondor, 
            (df['call_strike'] >= df['day_high_remaining']) & (df['put_strike'] <= df['day_low_remaining']), 
            None
        )


        return df

    def get_trades(self, symbol):
        if (symbol not in self.candleData):
            return 
        
        df: pd.DataFrame = self.candleData[symbol]

        if('final_signal' not in df.columns):
            return

        filtered = df[df["final_signal"] != StrategyEnum.Hold]
        # Dynamically select columns that exist in the DataFrame
        columns_to_return = [
            "datetime", "final_signal", "SellCall_profit", "SellPut_profit", "SellIronCondor_profit",
            "call_strike", "day_high_remaining", "put_strike", "day_low_remaining"
        ]
        existing_columns = [col for col in columns_to_return if col in filtered.columns]

        # Return the filtered DataFrame with existing columns
        return filtered[existing_columns].reset_index(drop=True)

    def candlesAsCsv(self, symbol):
        if (symbol not in self.candleData):
            return

        df = self.candleData[symbol]

        # Save the dataframe to a CSV file
        df.to_csv(symbol+'.csv', index=False)
