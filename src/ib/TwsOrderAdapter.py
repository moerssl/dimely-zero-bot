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
import pytz
import time

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
    
    def nextOrderId(self, timeout=5):
        with self.cond:
            self.reqIds(-1)
            deadline = time.time() + timeout
            while self.next_order_id is None or self.next_order_id == self.last_used_order_id:
                remaining = deadline - time.time()
                if remaining <= 0:
                    # Timeout reached: return the current last used order id.
                    return self.next_order_id if self.next_order_id is not None else self.last_used_order_id
                self.cond.wait(remaining)
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
    def asNonGuaranteed(self, order: Order):
        order.smartComboRoutingParams = []
        order.smartComboRoutingParams.append(TagValue("NonGuaranteed", "1"))
        return order

    def create_trail_limit_order(self, lmt, row, percent=0.25, quantity=1):
        # Create a TRAIL LIMIT order.
        order = Order()
        order.action = "SELL"              # Sell to exit a long position or to short cover
        order.orderType = "TRAIL"      # Use a trailing stop limit order
        order.totalQuantity = quantity            # Set your desired quantity
        # For a TRAIL LIMIT order, we use lmtPriceOffset to define the offset (0.03)
        # Here, trailingPercent is irrelevant when using an offset, so set it to 0.
        order.trailStopPrice = round(lmt * (1 - percent),2)  # Set the stop price to be 0.5 less than the limit price
        
        order.auxPrice = 0.05
        order = self.asNonGuaranteed(order)

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
                elif "short" in key:
                    limit_price -= row["bid"]  # Bid price for short legs
        return limit_price
    
    def calcSpreadPriceInv(contract_rows, start=0.0):
        if contract_rows is None:
            return
        # Calculate the limit price for the LMT order
        limit_price = start
        for key, row in contract_rows.items():
            if row is not None:
                if "long" in key:
                    limit_price -= row["ask"]  # Ask price for long legs
                elif "short" in key:
                    limit_price += row["bid"]  # Bid price for short legs
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


    def place_combo_order(self, contract_rows, tp=None, sl=None, ref="IronCondor", touchDistance=0.5, amount=1, getOutOfMarket=True):
        #start a new thread to place the order
        if (not TwsOrderAdapter.validateSpread(contract_rows)):
            self.addToActionLog("Invalid Spread, abort")
            return
        
        thread = Thread(target=self.place_combo_order_threaded, args=(contract_rows, tp, sl, ref,touchDistance,amount, getOutOfMarket))
        thread.start()


    def place_combo_order_threaded(self, contract_rows, tp=None, sl=None, ref="IronCondor", touchDistance=0.5, amount=1, getOutOfMarket=True):
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
            multiple = 0.01
            makeItMarketable = 0.03

            # change for SPX
            if (row["Symbol"] == "SPX"):
                multiple = 0.05
                makeItMarketable = 0.05
            limit_price = TwsOrderAdapter.calcSpreadPrice(contract_rows, makeItMarketable)

            # Create a Limit Order (parent order)
            limit_order = self.create_order("BUY", amount, "LMT", limit_price)
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
            


            if (sl is None and touchDistance is not None):
                stop_order = self.create_order("SELL", amount, "MKT")
                stop_order.orderRef = ref
                stop_order.parentId = parent_order_id  # Link to parent order
                stop_order = self.asNonGuaranteed(stop_order)
                stop_order.conditions = []

                # Add conditions for short legs being in-the-money
                for key, row in contract_rows.items():
                    if "long" in key:
                        continue

                    condition = PriceCondition()
                    condition.conId = row["UnderConId"]
                    condition.isConjunctionConnection = False
                    # condition.triggerMethod = 2

                    if (condition.conId == 416904 or condition.conId == 13455763 or condition.conId == 137851301): #SPX, VIX, XSP
                        condition.exchange = 'CBOE'
                    else:
                        condition.exchange = 'SMART'

                    if "short_call" in key:
                        condition.isMore = True
                        condition.price = row["Strike"] + touchDistance
                        stop_order.conditions.append(condition)
                    elif "short_put" in key:
                        condition.isMore = False
                        condition.price = row["Strike"] - touchDistance
                        stop_order.conditions.append(condition)
            elif (sl is not None):
                stop_order = self.create_order("SELL", amount, "STP")
                stop_order.auxPrice = round_to_next_multiple(limit_price * (sl/100), multiple)
                stop_order.orderRef = ref
                stop_order.parentId = parent_order_id  # Link to parent order
                stop_order = self.asNonGuaranteed(stop_order)


            tpOrder = None
            if (tp is not None):
                price = min(-0.05, limit_price * (tp/100))
                tpOrder = self.create_order("SELL", amount, "LMT", round_to_next_multiple(price, multiple))
                tpOrder.orderRef = ref
                tpOrder.parentId = parent_order_id  # Link to parent order
            # Place the Stop Order (child order)
            # self.placeOrder(self.nextReqId(), combo_contract, stop_order)
            trailOrder = None
            try:
                contract = next(iter(contract_rows.values()))
                trailOrder: Order = self.create_trail_limit_order(limit_price, contract, quantity=amount)
                trailOrder.orderRef = ref
                trailOrder.parentId = parent_order_id  # Link to parent order
                trailOrder = self.asNonGuaranteed(trailOrder)
            except Exception as e:
                self.addToActionLog("Error creating trailing limit order: " + str(e))
                trailOrder = None

            outOfMarketOrder = None
            if (getOutOfMarket):
                outOfMarketOrder = self.create_order("SELL", amount, "MKT")
                outOfMarketOrder.orderRef = ref
                outOfMarketOrder.parentId = parent_order_id  # Link to parent order
                outOfMarketOrder = self.enhanceOrderToActionAtTime(outOfMarketOrder, 15, 45)
            limit_order = self.enhanceOrderWithUrgentPrio(limit_order)
            limit_order = self.enhanceOrderToCancelInTime(limit_order)

            if trailOrder is not None:
                trailOrder.transmit = True
            
            elif outOfMarketOrder is not None:
                outOfMarketOrder.transmit = True
            elif stop_order is not None:
                stop_order.transmit = True
            elif tpOrder is not None:
                tpOrder.transmit = True
            else:
                limit_order.transmit = True

            try:
                self.orders.append((parent_order_id, combo_contract, limit_order))
                self.placeOrder(parent_order_id, combo_contract, limit_order)
            except Exception as e:
                self.addToActionLog("Error placing parent order: " + str(e))
                return

            if (tpOrder is not None):
                try:
                    tpOrderId = self.nextOrderId()
                    self.orders.append((tpOrderId, combo_contract, tpOrder))
                    self.placeOrder(tpOrderId, combo_contract, tpOrder)
                except Exception as e:
                    self.addToActionLog("Error placing TP order: " + str(e))
                    tpOrder = None
                    

            if (stop_order is not None):
                try:
                    stpLossId = self.nextOrderId()
                    self.orders.append((stpLossId, combo_contract, stop_order))
                    self.placeOrder(stpLossId, combo_contract, stop_order)
                except Exception as e:
                    self.addToActionLog("Error placing Stop Loss order: " + str(e))
                    stop_order = None

            if (outOfMarketOrder is not None):
                try:
                    outOfMarketId = self.nextOrderId()
                    self.orders.append((outOfMarketId, combo_contract, outOfMarketOrder))
                    self.placeOrder(outOfMarketId, combo_contract, outOfMarketOrder)
                except Exception as e:
                    self.addToActionLog("Error placing Out of Market order: " + str(e))
                    outOfMarketOrder = None

            if (trailOrder is not None):
                try:
                    trailOrderId = self.nextOrderId()
                    self.orders.append((trailOrderId, combo_contract, trailOrder))
                    self.placeOrder(trailOrderId, combo_contract, trailOrder)
                except Exception as e:
                    self.addToActionLog("Error placing Trailing Limit order: " + str(e))
                    trailOrder = None

            # Check all remaining order. If there is no order with transmit=True, set transmit=True for the limit order
            # create iterable for orders
            orderList = [limit_order, stop_order, tpOrder, outOfMarketOrder, trailOrder]
            transmitFound = False
            for order in orderList:
                if order is not None and order.transmit:
                    transmitFound = True
                    break
            if not transmitFound:
                limit_order.transmit = True
                self.addToActionLog("No transmit order found, setting transmit=True for limit order")
                self.placeOrder(parent_order_id, combo_contract, limit_order)


        except Exception as e:
            self.addToActionLog("Error placing order: " + str(e))


    def place_combo_order_threaded_inv(self, contract_rows, tp=None, sl=None, ref="IronCondor", touchDistance=0.5, amount=1, getOutOfMarket=True):
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
            multiple = 0.01

            # change for SPX
            if (row["Symbol"] == "SPX"):
                multiple = 0.05
            limit_price = TwsOrderAdapter.calcSpreadPriceInv(contract_rows, multiple)

            # Create a Limit Order (parent order)
            limit_order = self.create_order("SELL", amount, "LMT", limit_price)
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
            


            if (sl is None and touchDistance is not None):
                stop_order = self.create_order("BUY", amount, "MKT")
                stop_order.orderRef = ref
                stop_order.parentId = parent_order_id  # Link to parent order
                stop_order = self.asNonGuaranteed(stop_order)

                # Add conditions for short legs being in-the-money
                for key, row in contract_rows.items():
                    if "long" in key:
                        continue

                    condition = PriceCondition()
                    condition.conId = row["UnderConId"]
                    condition.isConjunctionConnection = False
                    # condition.triggerMethod = 2

                    if (condition.conId == 416904 or condition.conId == 13455763 or condition.conId == 137851301): #SPX, VIX, XSP
                        condition.exchange = 'CBOE'
                    else:
                        condition.exchange = 'SMART'

                    if "short_call" in key:
                        condition.isMore = True
                        condition.price = row["Strike"] + touchDistance
                        stop_order.conditions.append(condition)
                    elif "short_put" in key:
                        condition.isMore = False
                        condition.price = row["Strike"] - touchDistance
                        stop_order.conditions.append(condition)
            elif (sl is not None):
                stop_order = self.create_order("BUY", amount, "STP")
                stop_order.auxPrice = round_to_next_multiple(limit_price * (sl/100), multiple)
                stop_order.orderRef = ref
                stop_order.parentId = parent_order_id  # Link to parent order
                stop_order = self.asNonGuaranteed(stop_order)


            tpOrder = None
            if (tp is not None):
                price = max(0.05, limit_price * (tp/100))
                tpOrder = self.create_order("BUY", amount, "LMT", round_to_next_multiple(price, multiple))
                tpOrder.orderRef = ref
                tpOrder.parentId = parent_order_id  # Link to parent order
            # Place the Stop Order (child order)
            # self.placeOrder(self.nextReqId(), combo_contract, stop_order)
            trailOrder = None
            try:
                contract = next(iter(contract_rows.values()))
                trailOrder: Order = self.create_trail_limit_order(limit_price, contract, quantity=amount)
                trailOrder.action = "BUY"  # Set action to BUY for trailing limit order
                trailOrder.orderRef = ref
                trailOrder.parentId = parent_order_id  # Link to parent order
                trailOrder = self.asNonGuaranteed(trailOrder)
            except Exception as e:
                self.addToActionLog("Error creating trailing limit order: " + str(e))
                trailOrder = None

            outOfMarketOrder = None
            if (getOutOfMarket):
                outOfMarketOrder = self.create_order("BUY", amount, "MKT")
                outOfMarketOrder.orderRef = ref
                outOfMarketOrder.parentId = parent_order_id  # Link to parent order
                outOfMarketOrder = self.enhanceOrderToActionAtTime(outOfMarketOrder, 15, 45)
            limit_order = self.enhanceOrderWithUrgentPrio(limit_order)
            limit_order = self.enhanceOrderToCancelInTime(limit_order)

            if trailOrder is not None:
                trailOrder.transmit = True
            
            elif outOfMarketOrder is not None:
                outOfMarketOrder.transmit = True
            elif stop_order is not None:
                stop_order.transmit = True
            elif tpOrder is not None:
                tpOrder.transmit = True
            else:
                limit_order.transmit = True

            self.orders.append((parent_order_id, combo_contract, limit_order))
            self.placeOrder(parent_order_id, combo_contract, limit_order)

            if (tpOrder is not None):
                tpOrderId = self.nextOrderId()
                self.orders.append((tpOrderId, combo_contract, tpOrder))
                self.placeOrder(tpOrderId, combo_contract, tpOrder)

            if (stop_order is not None):
                stpLossId = self.nextOrderId()
                self.orders.append((stpLossId, combo_contract, stop_order))
                self.placeOrder(stpLossId, combo_contract, stop_order)

            if (outOfMarketOrder is not None):
                outOfMarketId = self.nextOrderId()
                self.orders.append((outOfMarketId, combo_contract, outOfMarketOrder))
                self.placeOrder(outOfMarketId, combo_contract, outOfMarketOrder)
            if (trailOrder is not None):
                trailOrderId = self.nextOrderId()
                self.orders.append((trailOrderId, combo_contract, trailOrder))
                self.placeOrder(trailOrderId, combo_contract, trailOrder)

        except Exception as e:
            self.addToActionLog("Error placing order: " + str(e))



    def addToActionLog(self, action):
        with self.actionLock:
            self.actionLog.append((datetime.now(), action))
        print(datetime.now(), action)

    @iswrapper
    def error(self, reqId, errorTime, errorCode, errorString, advancedOrderRejectJson=""):
        if advancedOrderRejectJson != "":
            self.addToActionLog("Advanced Order Reject: " + advancedOrderRejectJson)
        self.addToActionLog("Error: " + str(errorCode) + " " + errorString)
        return super().error(reqId, errorTime, errorCode, errorString, advancedOrderRejectJson)

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

    def has_existing_order_contracts(self, contract_rows: dict):
        """
        Check if the order contracts already exist in the DataFrame.
        :param contract_rows: Dictionary containing contract information.
        :return: True if existing order contracts are found, False otherwise.
        """
        for key, row in contract_rows.items():
            if row["ConId"] in self.orderContracts:
                return True
        return False

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
    

    def enhanceOrderToCancelInTime(self, order: Order, minutes=1):
        # Set the order to be canceled if conditions are not met
        order.conditionsCancelOrder = True

        # Define the New York timezone (US/Eastern)
        ny_timezone = pytz.timezone("US/Eastern")

        # Calculate the time 3 minutes from now in UTC
        utc_now = datetime.now(pytz.utc)
        cancel_time = utc_now + timedelta(minutes=minutes)

        # Convert the calculated time to New York timezone
        cancel_time_in_ny = cancel_time.astimezone(ny_timezone)

        # Format the time as required, e.g. "YYYYMMDD HH:MM:SS US/Eastern"
        time_str = cancel_time_in_ny.strftime("%Y%m%d %H:%M:%S") + " US/Eastern"


        # Create and set up the TimeCondition
        time_cond = TimeCondition()
        time_cond.time = time_str  # The order will auto-cancel once the current time exceeds this value
        time_cond.isMore = True
        order.conditions.append(time_cond)
        return order
    
    def enhanceOrderToActionAtTime(self, order: Order, hour: int, minute: int, conditionsCancelOrder: bool = False):
        # Set the order cancellation condition based on the provided parameter
        order.conditionsCancelOrder = conditionsCancelOrder

        # Define the New York timezone (US/Eastern)
        ny_timezone = pytz.timezone("US/Eastern")

        # Get today's date in the New York timezone
        today_ny = datetime.now(ny_timezone).date()

        # Combine today's date with the provided hour and minute in New York timezone
        cancel_time_in_ny = datetime(today_ny.year, today_ny.month, today_ny.day, hour, minute, tzinfo=ny_timezone)

        # Convert the New York time to UTC
        cancel_time_utc = cancel_time_in_ny.astimezone(pytz.utc)

        # Format the time as required, e.g., "YYYYMMDD HH:MM:SS US/Eastern"
        time_str = cancel_time_in_ny.strftime("%Y%m%d %H:%M:%S") + " US/Eastern"

        # Create and set up the TimeCondition
        time_cond = TimeCondition()
        time_cond.time = time_str  # The order will auto-cancel once the current time exceeds this value
        time_cond.isMore = True
        order.conditions.append(time_cond)
        return order
