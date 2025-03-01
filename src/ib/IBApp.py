
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
from threading import Thread, Condition

logger = getLogger(__name__)
logger.setLevel("INFO")


class IBApp(IBWrapper, IBClient):
    def __init__(self):
        self.market_data_req_ids = {}
        self.option_id_req = {}
        self.orders = []
        self.actionLog = []
        self.next_order_id = None
        self.last_used_order_id = None
        IBWrapper.__init__(self, self.market_data_req_ids)
        IBClient.__init__(self, self)
        self.reqId = 0
        self.cond = Condition()

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
            print("No options data available")
            print(options_data)
            print("-----")
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
                print("No valid rows to display")
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
                "content": forDisplay(self.construct_from_underlying("SPX"))
            },
                        {
                "title": "QQQ Trades",
                "content": forDisplay(self.construct_from_underlying("QQQ",2,2))
            },
            {
                "title": "SPX 15 Delta Bear Call",
                "content": forDisplay(self.build_credit_spread("SPX", 0.15, "C"))
            },
            {
                "title": "SPX 15 Delta Bull Put",
                "content": forDisplay(self.build_credit_spread("SPX", 0.15, "P"))
            },

            {
                "title": "Orders",
                "content": pd.DataFrame(self.orders)
            },
            {
                "title": "Action Log",
                "content": pd.DataFrame(self.actionLog).sort_values(by=0, ascending=False) if len(self.actionLog) > 0 else pd.DataFrame() 
            },
            {
                "title": "Options Data",
                "content": self.options_data.dropna(subset=['delta'])
            }
        ]

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

    def hasOrdersOrPositions(self, symbol):
        for (id, contract, order) in self.orders:
            if (contract.symbol == symbol):
                return True
            
        for index, row in self.positions.iterrows():
            if (row["Symbol"] == symbol and row["Quantity"] != 0):
                return True


    def place_combo_order(self, contract_rows, tp=None, sl=None, ref="IronCondor"):
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
    
    def get_closest_delta_row(self, symbol, target_delta, type="C"):
        df_symbol = self.options_data[(self.options_data["Symbol"] == symbol) & (self.options_data["Type"] == type)]
        # Ensure delta column exists, set default if not
        df_symbol['delta_diff'] = np.abs(df_symbol.get('delta', pd.Series(0)).abs() - target_delta)
        
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
        df_symbol = self.options_data.loc[(self.options_data["Symbol"] == symbol) & (self.options_data["Type"] == type), :]

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