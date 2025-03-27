import os
from typing import List, Tuple
import numpy as np
from ib.IBWrapper import IBWrapper
from ib.IBClient import IBClient
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.contract import Contract, ComboLeg
from ibapi.order import Order
from ibapi.order_condition import OrderCondition, PriceCondition
from ibapi.order_state import OrderState

from ibapi.utils import iswrapper
from ibapi.common import *
import pandas as pd
import time
from datetime import datetime, timedelta
from logging import getLogger
from threading import Thread, Condition, RLock

from sklearn.linear_model import LinearRegression  
import joblib
import talib
import scipy.stats as stats
from scipy.stats import norm
from util.Chart import Chart
from util.StrategyEnum import StrategyEnum
from util.AdaptiveDataframeProcessor import AdaptiveDataframeProcessor
import util.TaAnalysisHelper as tah
from util.config import CONFIG

logger = getLogger(__name__)
# logger.setLevel("INFO")


class IBApp(IBWrapper, IBClient):
    def __init__(self, lookBackTime="8 D"):
        self.logger = getLogger(__name__)
        self.market_data_req_ids = {}
        self.market_data_prefix_ids = {}
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
        self.actionLock = RLock()

        self.predictLock = RLock()
        self.predctThread = None

        self.candleData = {}
        self.daily_predictions = pd.DataFrame(columns=["date", "symbol", "predicted_close", "actual_close", "target_hit"])
        self.models = {}  # Store models for each symbol

        self.chart_handler: Chart = None

        self.lookBackTime = lookBackTime
        self.mergePredictThread = None
        self.chartData = None

        self.marketDataPrefix = [
            ("TRADES", "", True),
            ("HISTORICAL_VOLATILITY", "HV_", False),
            ("OPTION_IMPLIED_VOLATILITY", "IV_", False),
        ]

        """
        self.loadCandle()
        distance = self.getDistanceToLastCandle("SPX")
        if (distance is not None and distance < 86000):
            self.lookBackTime = str(int(distance)) + " S"
        """

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
        with self.actionLock:
            self.actionLog.append((datetime.now(), action))
        print(datetime.now(), action)

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
                "title": "Free To Trade?",
                "content": forDisplay(self.checkUnderlyingOptions()).reset_index()

            },
            {
                "title": "SPX IC Trades",
                "content": forDisplay(self.construct_from_underlying("SPX", 5, 5))
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
                "title": "SPX 30/10 Bear Call",
                "content": forDisplay(self.build_credit_spread_dollar("SPX", 35, 10, "C"))
            },
            {
                "title": "SPX 30/10  Bull Put",
                "content": forDisplay(self.build_credit_spread_dollar("SPX", 35, 10, "P"))
            },
            
            {
                "title": "Candle Data",
                "content": self.candleData.get("SPX", pd.DataFrame()).iloc[::-1],
                "colspan": 3,
                "exclude_columns": ["minutes_since_open", "remaining_intervals", "volume", "wap", "count"]
            },

            {
                "title": "Orders",
                "content": pd.DataFrame(self.orders)
            },
            {
                "title": "Action Log",
                "content": pd.DataFrame(self.actionLog).sort_values(by=0, ascending=False)[1] if len(self.actionLog) > 0 else pd.DataFrame(),
                
            },
            {
                "title": "API Orders",
                "content": pd.DataFrame.from_dict(self.apiOrders, orient='index')
            },
            {
            "title": "Options Data",
            "content": self.options_data #.dropna(subset=['delta'])
            },
            {
                "title": "SPX Stats",
                "content": self.get_strategy_statistics(self.candleData.get("SPX", pd.DataFrame()))
            },
            {
                "title": "SPX Distances",
                "content": self.get_optimal_offsets(self.candleData.get("SPX", pd.DataFrame())),
                "colspan": 3
            }

            
        ]
    
        """
        {
            "title": "SPX RSI",
            "content": tah.calc_rsi_stuff(self.candleData.get("SPX", pd.DataFrame()), "RSI") if "SPX" in self.candleData else pd.DataFrame(),
            "colspan": 3
        }
        """
        """
        {
            "title": "Options Data",
            "content": self.options_data.dropna(subset=['delta'])
        },
        """
    def get_strategy_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes statistics for each strategy signal including wins, losses, and winrate.
        
        The criteria are:
        - For SellCall: win if call_strike > day_high_remaining_strike.
        - For SellPut: win if put_strike < day_low_remaining_strike.
        - For SellIronConor: win if (call_strike > day_high_remaining_strike AND put_strike < day_low_remaining_strike).
        
        Parameters:
        df (pd.DataFrame): Input DataFrame with required columns.
        
        Returns:
        pd.DataFrame: A DataFrame with columns 'signal', 'wins', 'losses', and 'winrate'
                    for each unique final_signal.
        """
        # Create a copy so that we don't modify the original DataFrame.
        df = df.copy()
        try:
            
            # Initialize a new column 'win' with a default value of False.
            df['win'] = False
            
            # Create boolean masks for each strategy using the enum values.
            mask_sell_call = df['final_signal'] == StrategyEnum.SellCall
            mask_sell_put = df['final_signal'] == StrategyEnum.SellPut
            mask_sell_iron = df['final_signal'] == StrategyEnum.SellIronCondor
            
            # For SellCall, a win when call_strike > day_high_remaining_strike.
            df.loc[mask_sell_call, 'win'] = df.loc[mask_sell_call, 'call_strike'] > df.loc[mask_sell_call, 'day_high_remaining_strike']
            
            # For SellPut, a win when put_strike < day_low_remaining_strike.
            df.loc[mask_sell_put, 'win'] = df.loc[mask_sell_put, 'put_strike'] < df.loc[mask_sell_put, 'day_low_remaining_strike']
            
            # For SellIronConor, a win when both conditions hold.
            cond_call = df.loc[mask_sell_iron, 'call_strike'] > df.loc[mask_sell_iron, 'day_high_remaining_strike']
            cond_put  = df.loc[mask_sell_iron, 'put_strike'] < df.loc[mask_sell_iron, 'day_low_remaining_strike']
            df.loc[mask_sell_iron, 'win'] = cond_call & cond_put
            
            # Group by the signal and aggregate stats: count trades, wins, then calculate losses and winrate.
            stats = df.groupby('final_signal', sort=False).agg(
                trades=('win', 'count'),
                wins=('win', 'sum')
            )
            
            stats['losses'] = stats['trades'] - stats['wins']
            stats['winrate'] = stats['wins'] / stats['trades']
            
            # Reset the index and rename final_signal column to signal.
            result = stats.reset_index().rename(columns={'final_signal': 'signal'})
            return result
        except Exception as e:
            print("Error in get_strategy_statistics:", e)
            return pd.DataFrame()

    def optimal_distance_vec(self, group: pd.DataFrame, col: str, desired_winrate=0.8) -> pd.Series:
        """
        Berechnet für eine Gruppe (z. B. alle Trades eines Signals) das minimale d (Strike‑Offset)
        und die damit erzielte Gewinnrate, sodass mindestens desired_winrate aller Trades (inklusive
        verlierender Fälle) als „gewinnend“ gelten würden, wenn der Optionsstrike bei close+d (bei Calls)
        bzw. close-d (bei Puts) gesetzt wird.
        
        Dabei wird angenommen, dass group[col] den Gewinnabstand (gap) enthält, z. B.:
          - bei SellCall: gap = day_high_remaining - close
          - bei SellPut:  gap = close - day_low_remaining
        
        Es werden die positiven (gültigen) gap‑Werte betrachtet. Ein Trade gewinnt, wenn gap < d.
        Wird das gewünschte Winrate-Ziel nicht erreicht, so wird das d zurückgegeben, bei dem
        die Gewinnquote maximal ist.
        
        Rückgabe:
            Eine pd.Series mit zwei Schlüsselwerten:
              'optimal_d' : berechneter minimaler Strike‑Offset
              'winrate'   : damit erreichte Gewinnrate
        """
        try:
            N = group.shape[0]
            # gaps_all kann auch negative Werte enthalten – wir betrachten nur die positiven als gültige Gewinne.
            gaps_all = group[col].dropna().values
            valid = gaps_all[gaps_all > 0]
            if valid.size == 0:
                return pd.Series({'optimal_d': np.nan, 'winrate': 0.0})
            # Kandidaten: eindeutige, positive gap‑Werte, aufsteigend sortiert.
            candidates = np.sort(np.unique(valid))
            
            optimal_d = np.nan
            achieved_winrate = 0.0
            for d in candidates:
                wins = np.sum(gaps_all < d)  # Zähle alle Trades, bei denen gap < d gilt
                winrate = wins / N
                # Sobald die gewünschte winrate erreicht ist, wählen wir diesen Kandidaten
                if winrate >= desired_winrate:
                    optimal_d = d
                    achieved_winrate = winrate
                    break
                if winrate > achieved_winrate:
                    achieved_winrate = winrate
                    optimal_d = d
            return pd.Series({'optimal_d': optimal_d, 'winrate': achieved_winrate})
        except Exception as e:
            self.addToActionLog(f"Error in optimal_distance_vec: {str(e)}")
            return pd.Series({'optimal_d': np.nan, 'winrate': 0.0})

    def get_optimal_offsets(self, df: pd.DataFrame, desired_winrate=0.8) -> pd.DataFrame:
        """
        Berechnet für jeden Signaltyp (und bei SellIronCondor je Leg separat) den optimalen Strike-Offset d
        (sodass mindestens desired_winrate an Trades gewinnen würden, wenn der Optionsstrike auf close+d (bei
        Calls) oder close-d (bei Puts) angesetzt würde) sowie die erreichte Gewinnrate.

        Annahmen:
          - Für SellCall: gap_call = day_high_remaining - close.
          - Für SellPut:  gap_put = close - day_low_remaining.
          - Für SellIronCondor wird beides separat berechnet (in den Spalten gap_call und gap_put).
        
        Rückgabe:
          Ein DataFrame mit den Spalten:
            signal, optimal_call_offset, achieved_call_winrate,
            optimal_put_offset, achieved_put_winrate.
          Bei Signalen, für die nur eine Seite relevant ist, wird die andere Spalte NaN.
        """
        try:
            df = df.copy()
            # Wandeln final_signal (falls Enum) in einen String um.
            if df['final_signal'].apply(lambda x: hasattr(x, "name")).all():
                df['final_signal'] = df['final_signal'].apply(lambda x: x.name)
            
            # Berechne die Gewinnabstände (gaps):
            df['gap_call'] = np.nan
            df['gap_put'] = np.nan
            
            # Für SellCall (Gewinnbedingung: close < day_high_remaining)
            mask_call = df['final_signal'] == StrategyEnum.SellCall.name
            df.loc[mask_call, 'gap_call'] = df.loc[mask_call, 'day_high_remaining'] - df.loc[mask_call, 'close']
            
            # Für SellPut (Gewinnbedingung: close > day_low_remaining)
            mask_put = df['final_signal'] == StrategyEnum.SellPut.name
            df.loc[mask_put, 'gap_put'] = df.loc[mask_put, 'close'] - df.loc[mask_put, 'day_low_remaining']
            
            # Für SellIronCondor: Verwende beide Metriken – beachte, dass nur gültige Trades betrachtet werden,
            # also die, bei denen der close zwischen day_low_remaining und day_high_remaining liegt.
            mask_iron = df['final_signal'] == StrategyEnum.SellIronCondor.name
            iron_win = (df.loc[mask_iron, 'close'] > df.loc[mask_iron, 'day_low_remaining']) & \
                       (df.loc[mask_iron, 'close'] < df.loc[mask_iron, 'day_high_remaining'])
            df.loc[mask_iron, 'gap_call'] = np.where(iron_win,
                                                     df.loc[mask_iron, 'day_high_remaining'] - df.loc[mask_iron, 'close'],
                                                     np.nan)
            df.loc[mask_iron, 'gap_put'] = np.where(iron_win,
                                                    df.loc[mask_iron, 'close'] - df.loc[mask_iron, 'day_low_remaining'],
                                                    np.nan)
            
            # Berechne den optimalen Offset für die Call-Seite pro Signal
            opt_call = df.groupby('final_signal', sort=False).apply(
                lambda g: self.optimal_distance_vec(g, 'gap_call', desired_winrate)
            ).reset_index()
            # Berechne den optimalen Offset für die Put-Seite pro Signal
            opt_put = df.groupby('final_signal', sort=False).apply(
                lambda g: self.optimal_distance_vec(g, 'gap_put', desired_winrate)
            ).reset_index()
            
            # Falls der GroupBy-Apply bereits die Schlüssel 'optimal_d' und 'winrate' als Spalten liefert,
            # können wir diese direkt verwenden. Wir fangen allerdings KeyErrors ab:
            try:
                _ = opt_call['optimal_d']
            except KeyError as e:
                self.addToActionLog(f"KeyError in opt_call: {str(e)}")
                opt_call['optimal_d'] = np.nan
                opt_call['winrate'] = np.nan
                
            try:
                _ = opt_put['optimal_d']
            except KeyError as e:
                self.addToActionLog(f"KeyError in opt_put: {str(e)}")
                opt_put['optimal_d'] = np.nan
                opt_put['winrate'] = np.nan
            
            # Wir benennen die Spalten um, sodass sie aussagekräftig sind:
            opt_call = opt_call.rename(columns={'optimal_d': 'optimal_call_offset', 'winrate': 'achieved_call_winrate'})
            opt_put = opt_put.rename(columns={'optimal_d': 'optimal_put_offset', 'winrate': 'achieved_put_winrate'})
            
            # Merge der Ergebnisse (outer Join, damit alle Signaltypen enthalten bleiben)
            result = pd.merge(opt_call, opt_put, on='final_signal', how='outer')
            result = result.rename(columns={'final_signal': 'signal'})
            
            return result
        except Exception as e:
            self.addToActionLog(f"Error in get_optimal_offsets: {str(e)}")
            return pd.DataFrame()
        
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
        try:
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
        except Exception as e:
            self.addToActionLog("Error placing order: " + str(e))

    @iswrapper
    def error(self, reqId, errorTime, errorCode, errorString, advancedOrderRejectJson=""):
        if errorCode not in [2104, 2106,2158]:
            print("Error. Id: ", reqId, "Time: ", errorTime, " Code: ", errorCode, " Msg: ", errorString)
            self.addToActionLog(" Msg: " + str(errorString) + " Code: " + str(errorCode) + " Id: " + str(reqId))
        return super().error(reqId, errorTime, errorCode, errorString, advancedOrderRejectJson)
    
    def filter_options_data(self, symbol, type="C"):
        with self.optionsDataLock:
            # Ensure the boolean Series and DataFrame have the same index
            mask = (self.options_data["Symbol"] == symbol) & (self.options_data["Type"] == type)
            aligned_mask = mask.reindex(self.options_data.index, fill_value=False)

            # Print indices for debugging
            #print("self.options_data index:", self.options_data.index)
            #print("aligned_mask index:", aligned_mask.index)

            # Use the aligned boolean Series for indexing
            df_symbol = self.options_data.loc[aligned_mask, :]
            return df_symbol

    def get_closest_delta_row(self, symbol, target_delta, type="C"):
        df_symbol = self.filter_options_data(symbol, type)
        # Ensure delta column exists, set default if not
        #df_symbol['delta_diff'] = np.abs(df_symbol.get('delta', pd.Series(0)).abs() - target_delta)
        """
        df_symbol['delta_diff'] = np.where(df_symbol['delta'].notna(), np.abs(df_symbol['delta'].abs() - target_delta), np.nan)
        """

        df_symbol.loc[:, 'delta_diff'] = np.where(
            df_symbol['delta'].notna(),
            np.abs(df_symbol['delta'].abs() - target_delta),
            np.nan
        )


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
    
    def build_credit_spread_dollar(self, symbol, distance, wingspan, type="C"):
        if (symbol not in self.market_data.index):
            return 
        row = self.market_data.loc[symbol]
        current_price = row["Price"]

        target_strike = current_price + distance
        if type == "P":
            target_strike = current_price - distance
        
        return self.find_credit_spread_legs(symbol, target_strike, type, wingspan)
    
    def find_credit_spread_legs(self, symbol, short_strike, type="C", wingspan=10):
        if (self.options_data.empty):
            return None
        df_symbol = self.filter_options_data(symbol, type)
        deltaRow = self.find_closest_strike_row(df_symbol, short_strike + wingspan, "C")
        self.request_options_market_data(deltaRow)
        
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

            if (status in ["Filled", "Cancelled", "Inactive"]):
                self.addToActionLog("orderStatus "+ str(orderId) +" "+ str(status) + " " + str(filled) + " " + str(remaining) + " " + str(avgFillPrice) + " " + str(permId) + " " + str(parentId) + " " + str(lastFillPrice) + " " + str(clientId) + " " + str(whyHeld) + " " + str(mktCapPrice))
                del self.apiOrders[orderId]
                if(orderId in self.orders):
                    self.remove_order_by_id(orderId)
        return super().orderStatus(orderId, status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice)
    
    def remove_order_by_id(self, target_number: int) -> None:
        self.orders = [order for order in self.orders if order[0] != target_number]

    @iswrapper
    def openOrder(self, orderId, contract, order: Order, orderState: OrderState):
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
            contract = order.get("contract")
            if contract and contract.secType in ["OPT", "BAG"] and contract.symbol == underlying:
                if contract.secType == "OPT" and contract.right == option_type:
                    open_orders.append(order)
                elif contract.secType == "BAG":
                    for leg in contract.comboLegs:
                        if leg.symbol == underlying and leg.right == option_type:
                            open_orders.append(order)
                            break

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

    def reqHistoricalDataFor(self, symbol, type, exchange, catchUp = False):
        contract = Contract()
        contract.symbol = symbol
        contract.secType = type
        contract.exchange = exchange
        
        contract.currency = "USD"



        for whatToShow, prefix, upToDate in self.marketDataPrefix:
            if (upToDate and catchUp):
                continue
            
            lookBackTime = self.lookBackTime
            if (catchUp):
                lookBackTime = "600 S"

            reqId = self.nextReqId()

            self.market_data_req_ids[reqId] = { "symbol": symbol, "prefix": prefix }
            self.addToActionLog("Requesting historical data for " + symbol + " "+ str(reqId ))
            print("Requesting historical data for " + whatToShow + " " + symbol + " "+ str(reqId ), lookBackTime)
            self.reqHistoricalData(reqId, contract, "", lookBackTime, "5 mins", whatToShow, 1, 2, upToDate, [])
        
            # self.reqHistoricalData(reqId, contract, "", self.lookBackTime, "5 mins", "TRADES", 1, 1, True, [])
            time.sleep(1)  # To avoid pacing violations



    @iswrapper
    def historicalData(self, reqId, bar: BarData):
        try:
            self.logger.debug(f"Received historicalData for reqId: {reqId}")
            


            if reqId not in self.market_data_req_ids:
                self.logger.error(f"reqId {reqId} not found in market_data_req_ids")
                self.cancelHistoricalData(reqId)
                return

            # Fetch symbol and prefix for the current request ID
            symbol = self.market_data_req_ids[reqId]["symbol"]
            prefix = self.market_data_req_ids[reqId]["prefix"]

            print(f"Received historicalData for reqId: {reqId} symbol: {symbol} prefix: {prefix}", end="\r")
            with self.candleLock:
                # Initialize DataFrame if the symbol is not yet present
                if symbol not in self.candleData:
                    # Generate all possible columns with and without prefixes
                    prefixes = [p[1] for p in self.marketDataPrefix]
                    base_columns = ["date", "open", "close", "high", "low", "volume", "wap", "count"]
                    columns = ["date", "open", "close", "trend", "call_strike", "put_strike", "call_p", "put_p", "c_dist_p", "p_dist_p", "final_signal", "tech_signal", "high", "low",  "volume", "wap", "count"]

                    for pre in prefixes:
                        for col in base_columns:
                            colName = f"{pre}{col}" if pre else col
                            if colName not in columns:
                                columns.append(colName)
                            
                    self.candleData[symbol] = pd.DataFrame(columns=columns)

                # Prepare bar data with the appropriate prefix
                bar_data = {
                    f"date": bar.date,
                    f"{prefix}close": bar.close,
                    f"{prefix}open": bar.open,
                    f"{prefix}high": bar.high,
                    f"{prefix}low": bar.low,
                    f"{prefix}volume": bar.volume,
                    f"{prefix}wap": bar.wap,
                    f"{prefix}count": bar.barCount
                }

                # Convert bar_data to a DataFrame
                relevant_columns = [f"{prefix}{col}" for col in ["close", "open", "high", "low", "volume", "wap", "count"]]
                relevant_columns.insert(0, "date")
                bar_data_df = pd.DataFrame([bar_data], columns=relevant_columns)

                # Debugging output
                self.logger.debug(f"bar_data_df: {bar_data_df}")
                self.logger.debug(f"self.candleData[symbol].columns: {self.candleData[symbol].columns}")
                self.logger.debug(f"self.candleData[symbol] before update: {self.candleData[symbol]}")

                # Check if the row already exists and update only the relevant columns
                existing_index = self.candleData[symbol][self.candleData[symbol]['date'] == bar.date].index

                if not existing_index.empty:
                    # Update explicitly using .at to avoid SettingWithCopyWarning
                    for col in relevant_columns:
                        self.candleData[symbol].at[existing_index[0], col] = bar_data_df[col].values[0]
                else:
                    # Add the new data to the DataFrame
                    self.candleData[symbol] = pd.concat([self.candleData[symbol], bar_data_df], ignore_index=True)
        except Exception as e:
            self.accountDownloadEnd("Error in historicalData: " + str(e))


    def addIndicators(self, reqId):
        symbol = self.market_data_req_ids[reqId]["symbol"]

        with self.candleLock:
            
            if symbol not in self.candleData:
                self.addToActionLog(f"No candle data available for symbol: {symbol}")
            else:
                self.candleData[symbol] = tah.addIndicatorsOn(self, self.candleData[symbol], symbol)
                

    def get_chart_data(self, mergePredictions = False):
        symbol = "SPX"
        with self.candleLock:
            if (symbol not in self.candleData):
                return pd.DataFrame()
            self.addToActionLog("Plot chart with data")
            return self.candleData[symbol]
        
    @iswrapper
    def historicalDataEnd(self, reqId, start: str, end: str):
        self.addToActionLog("Historical data request completed for reqId: " + str(reqId))
        
        self.addIndicators(reqId)
        self.saveCandle()
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


    def checkUnderlyingOptions(self):
        symbol = "SPX"
        signal = None
        date = None
        put = None
        call = None
        with self.candleLock:
            if ("SPX" in self.candleData):
                df: pd.DataFrame = self.candleData[symbol]
                if (len(df) > 1 and "final_signal" in df.columns):
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
        
    def saveCandle(self):
        with self.candleLock:
            for symbol, df in self.candleData.items():
                df.to_csv(f"{symbol}_candle.csv", index=False)

    def loadCandle(self):
            for symbol in ["SPX"]:
                
                filePath = f"{symbol}_candle.csv"
                # Check if the file exists before loading
                if os.path.exists(filePath):
                    with self.candleLock:
                        df = pd.read_csv(filePath)
                        self.addToActionLog(f"Loaded candle data for {symbol}")
                        self.candleData[symbol] = tah.addIndicatorsOn(self, df)
                        self.addToActionLog(f"Added indicators for {symbol}")


    def getDistanceToLastCandle(self, symbol):
        with self.candleLock:
            if (symbol in self.candleData):
                df: pd.DataFrame = self.candleData[symbol]
                if (len(df) > 1):
                    last = df.iloc[-1]
                    return (datetime.now() - pd.to_datetime(last["date"])).total_seconds()
            return None

