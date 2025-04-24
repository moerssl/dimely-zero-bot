from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Tuple
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract, ComboLeg
from ibapi.order import Order
from ibapi.order_state import OrderState
from ibapi.order_condition import OrderCondition, PriceCondition, TimeCondition
from ibapi.order import OrderComboLeg
from ibapi.tag_value import TagValue

from ibapi.utils import iswrapper
from threading import Thread, Condition, RLock
import pandas as pd
import math


class TwsOrderAdapter(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        EWrapper.__init__(self)
        self.req_ids = {}
        self.reqId = 0
        self.cond = Condition()
        self.actionLog = []
        self.next_order_id = None
        self.last_used_order_id = None

        self.orders: List[Tuple[int, Contract, Order]] = []
        self.actionLock = RLock()

        self.positions = {}
        self.apiOrders = {}
        self.orderContracts = {}
        self.lastContractsDataUpdatedAt = None




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

    def create_order(self, action, quantity, orderType, price=None):
        order = Order()
        order.action = action
        order.totalQuantity = quantity
        order.orderType = orderType
        if price:
            order.lmtPrice = price
        order.transmit = False
        return order

    def calcSpreadPrice(contract_rows, start=0.0):
        if contract_rows is None:
            return
        # Calculate the limit price for the LMT order
        limit_price = start
        for key, row in contract_rows.items():
            if row is not None:
                if "long" in key:
                    limit_price += row["ask"]  # Ask price for long legs
                else:
                    limit_price -= row["bid"]  # Bid price for short legs
        return limit_price
        
    def calcMaxRisk(contract_rows, premiums=None):
        if contract_rows is None:
            return

        # Initialize risks for calls and puts
        call_risk = 0.0
        put_risk = 0.0

        # Calculate call risk if present and values are not None
        if (
            "long_call" in contract_rows and contract_rows["long_call"] is not None and
            "short_call" in contract_rows and contract_rows["short_call"] is not None
        ):
            call_risk = contract_rows["long_call"]["Strike"] - contract_rows["short_call"]["Strike"]

        # Calculate put risk if present and values are not None
        if (
            "long_put" in contract_rows and contract_rows["long_put"] is not None and
            "short_put" in contract_rows and contract_rows["short_put"] is not None
        ):
            put_risk = contract_rows["short_put"]["Strike"] - contract_rows["long_put"]["Strike"]

        # Adjust for premiums
        if premiums is None:
            premiums = TwsOrderAdapter.calcSpreadPrice(contract_rows)

        # Maximum risk is the greater of call risk or put risk, minus premiums
        risk =  max(call_risk, put_risk) - abs(premiums)
        return round(risk, 2)

           
    def validateSpread(contract_rows):
        if contract_rows is None:
            return False
        
        for key, row in contract_rows.items():
            if row is None:
                return False
            
            # check if ask is present and is a number
            ask = row.get("ask", None)
            bid = row.get("bid", None)

            if ask is None or math.isnan(ask):
                return False
            
            if bid is None or math.isnan(bid):
                return False
            
        return True


    def place_combo_order(self, contract_rows, tp=None, sl=None, ref="IronCondor"):
        #start a new thread to place the order
        if (not TwsOrderAdapter.validateSpread(contract_rows)):
            self.addToActionLog("Invalid Spread, abort")
            return
        
        thread = Thread(target=self.place_combo_order_threaded, args=(contract_rows, tp, sl, ref))
        thread.start()


    def place_combo_order_threaded(self, contract_rows, tp=None, sl=None, ref="IronCondor"):
        try:
            if(contract_rows is None):
                return
            
            self.addToActionLog("Placing Order "+ ref)
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
                if "long" in key or "short" in key:
                    leg = ComboLeg()
                    leg.conId = row["ConId"]
                    leg.ratio = 1
                    leg.action = "BUY" if "long" in key else "SELL"
                    leg.exchange = "SMART"
                    combo_contract.comboLegs.append(leg)

            # Calculate the limit price for the LMT order
            """
            limit_price = 0.05
            for key, row in contract_rows.items():
                if "long" in key:
                    limit_price += row["ask"]  # Ask price for long legs
                else:
                    limit_price -= row["bid"]  # Bid price for short legs
            """
            limit_price = TwsOrderAdapter.calcSpreadPrice(contract_rows, 0.05)

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
            
            multiple = 0.01
            # change for SPX
            if (row["Symbol"] == "SPX"):
                multiple = 0.05

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
                stop_order.auxPrice = round_to_next_multiple(limit_price * (sl/100), multiple)
                stop_order.orderRef = ref
                stop_order.parentId = parent_order_id  # Link to parent order

            tpOrder = None
            if (tp is not None):
                tpOrder = self.create_order("SELL", 1, "LMT", round_to_next_multiple(limit_price * (tp/100), multiple))
                tpOrder.orderRef = ref
                tpOrder.parentId = parent_order_id  # Link to parent order
            # Place the Stop Order (child order)
            # self.placeOrder(self.nextReqId(), combo_contract, stop_order)

            limit_order = self.enhanceOrderWithUrgentPrio(limit_order)
            #limit_order = self.enhanceOrderToCancelInTime(limit_order)
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

    def addToActionLog(self, action):
        with self.actionLock:
            self.actionLog.append((datetime.now(), action))
        print(datetime.now(), action)

    @iswrapper
    def position(self, account: str, contract: Contract, position: Decimal, avgCost: float):
        if (position == 0):
            if (contract.conId in self.positions):
                del self.positions[contract.conId]
            return super().position(account, contract, position, avgCost)
        
        if (contract.conId not in self.positions):
            self.positions[contract.conId] = {}
        self.positions[contract.conId]["position"] = position
        self.positions[contract.conId]["avgCost"] = avgCost
        self.positions[contract.conId]["account"] = account
        self.positions[contract.conId]["contract"] = contract
        return super().position(account, contract, position, avgCost)
    
    @iswrapper
    def openOrder(self, orderId: int, contract: Contract, order: Order, orderState: OrderState):
        # Check if order has a combo contract
        if (len(contract.comboLegs) > 0):
            for leg in contract.comboLegs:
                if leg.conId not in self.orderContracts:
                    self.orderContracts[leg.conId] = {
                        "conId": leg.conId,                        
                    }
        else:
            self.orderContracts[contract.conId] = {
                "conId": contract.conId,
                "symbol": contract.symbol,
                "secType": contract.secType,
                "right": contract.right,
            }


                    


        orderDict = {
            "contractId": contract.conId,
            "symbol": contract.symbol,
            "secType": contract.secType,
            "right": contract.right,
            "status": orderState.status,
            "order": order,
            "contract": contract,
        }
    
        self.apiOrders[orderId] = orderDict
        return super().openOrder(orderId, contract, order, orderState)
    
    @iswrapper
    def openOrderEnd(self):
        self.lastContractsDataUpdatedAt = datetime.now()
    
    @iswrapper
    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
        if orderId in self.apiOrders:
            if (status in ["Filled", "Cancelled", "Inactive"]):
                apiOrder = self.apiOrders[orderId]
                # Remove any associated contracts from orderContracts
                for conId in apiOrder["contract"].comboLegs:
                    if conId.conId in self.orderContracts:
                        del self.orderContracts[conId.conId]
                if apiOrder["contract"].conId in self.orderContracts:
                    del self.orderContracts[apiOrder["contract"].conId]

                del self.apiOrders[orderId]


                return super().orderStatus(orderId, status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice)
            """
            contract, order, orderState = self.apiOrders[orderId]
            orderState.status = status
            """
            # Update the order status in the orders list
            self.apiOrders[orderId]["status"] = status
            # Update the order status in the orders list

        return super().orderStatus(orderId, status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice)
    
    def enhance_order_contracts_from_dataframe(self, contract_rows: pd.DataFrame):
        """
        Enhance order contracts with additional information from the DataFrame.
        :param contract_rows: DataFrame containing contract information.
        """
        ids = self.orderContracts.keys()

        # get the data for the ids from the DataFrame
        contract_rows = contract_rows[contract_rows["ConId"].isin(ids)]
                
        for key, row in contract_rows.iterrows():
            if row["ConId"] in self.orderContracts:
                self.orderContracts[row["ConId"]].update(row.to_dict())

    def is_room_for_new_positions(self, symbol, opt_type=None):
        dataframe = pd.DataFrame(self.orderContracts.values())
        if dataframe.empty:
            return True
        
        # Check if the DataFrame contains the 'Symbol' column
        if 'Symbol' not in dataframe.columns:
            return False
        
        # Check if opt_type is provided and Type column exists
        if opt_type:
            if 'Type' not in dataframe.columns:
                return False
            # Ensure all 'Type' values are valid (either 'P' or 'C')
            if not dataframe['Type'].isin(['P', 'C']).all():
                return False
            # Check for matching rows with both symbol and type
            filtered = dataframe[(dataframe['Symbol'] == symbol) & (dataframe['Type'] == opt_type)]
            return filtered.empty
        
        # Check for matching rows with only the symbol
        filtered = dataframe[dataframe['Symbol'] == symbol]
        return filtered.empty

    def enhanceOrderWithUrgentPrio(self, baseOrder: Order, priority: str = "Urgent"):
        baseOrder.algoStrategy = "Adaptive"
        baseOrder.algoParams = []
        baseOrder.algoParams.append(TagValue("adaptivePriority", priority))
        return baseOrder
    
    def enhanceOrderToCancelInTime(self, order: Order, minutes=3):
        # Set the order to be canceled if conditions are not met
        order.conditionsCancelOrder = True

        # Calculate the time 3 minutes from now
        cancel_time = datetime.now() + timedelta(minutes=minutes)
        # Format the time as required, e.g. "YYYYMMDD HH:MM:SS"
        time_str = cancel_time.strftime("%Y%m%d %H:%M:%S")

        # Create and set up the TimeCondition
        time_cond = TimeCondition()
        time_cond.time = time_str  # The order will auto-cancel once the current time exceeds this value
        order.conditions.append(time_cond)
        return order