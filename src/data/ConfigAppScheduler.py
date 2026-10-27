import copy
from decimal import Decimal
from functools import partial
import math
from time import sleep
import time

import pandas as pd
from data.OrbResult import OrbResult
from data.BaseAppScheduler import BaseAppScheduler
from ib.TwsOrderAdapter import TwsOrderAdapter
from ib.IBApp import IBApp
from ibapi.contract import Contract
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.combining import OrTrigger
import json
import inspect
from util.Logger import Logger
from datetime import date, datetime, timedelta

from util.TelegramMessenger import send_telegram_message


 

def build_options_summary(df):
    # Helper: safe column check
    def has(col):
        return col in df.columns

    total_rows = len(df)

    bid_rows = df[df["bid"] > 0].shape[0] if has("bid") else None
    ask_rows = df[df["ask"] > 0].shape[0] if has("ask") else None

    # Rows where both bid and ask > 0
    if has("bid") and has("ask"):
        both_mask = (df["bid"] > 0) & (df["ask"] > 0)
        both_rows = df[both_mask].shape[0]
    else:
        both_mask = None
        both_rows = None

    # DTE range
    min_dte = df["DTE"].min() if has("DTE") else None
    max_dte = df["DTE"].max() if has("DTE") else None

    # Strike range for rows where bid & ask > 0
    if both_mask is not None and has("Strike"):
        valid_strikes = df.loc[both_mask, "Strike"]
        min_strike = valid_strikes.min() if not valid_strikes.empty else None
        max_strike = valid_strikes.max() if not valid_strikes.empty else None
    else:
        min_strike = None
        max_strike = None

    summary = (
        f"rows={total_rows}, "
        f"bid>0={bid_rows}, "
        f"ask>0={ask_rows}, "
        f"bid&ask>0={both_rows}, "
        f"minDTE={min_dte}, "
        f"maxDTE={max_dte}, "
        f"minStrike_bid&ask>0={min_strike}, "
        f"maxStrike_bid&ask>0={max_strike}"
    )

    return summary



class ConfigAppScheduler(BaseAppScheduler):
    def __init__(self, app, orderApp, historyApp, symbol="SPY", secondSymbol="QQQ", config_file="config.json"):
        """
        Initialize the ConfigAppScheduler with the given parameters.
        """
        super().__init__(app, orderApp, historyApp, symbol, secondSymbol)
        self.config_file = config_file
        self.app.useStandardTiles = False
        self.jobForTitleTime = None
        self.firstWheelAttemptTime = None
        self.load_config()

        self.currentDteWheelMinPrice = None

        
        self.ordersPlacedByOrderRef = {}

        Logger.log("Scheduler started")
        
    def increaseOrderPlaced(self, orderRef):
        self.ordersPlacedByOrderRef[orderRef] = self.ordersPlacedByOrderRef.get(orderRef, 0) + 1

    def getOrdersPlaced(self, orderRef):
        return self.ordersPlacedByOrderRef.get(orderRef, 0)
     
    def run(self):
        """
        fixedDistanceTrigger = CronTrigger(
            day_of_week='mon-fri', 
            hour=14, 
            minute="30-35", 
            second="0/30", 
            timezone=self.ny_tz 
        )
        self.scheduler.add_job(
            func=self.checkFixedDistanceIronCondor, 
            trigger=fixedDistanceTrigger,   
            id="checkFixedDistanceIronCondor", 

            misfire_grace_time=15)
        """
        for job in self.jobs:
            required_keys = []
            if job.get("multi") is None:
                required_keys = ["day_of_week", "hour", "minute"]

            required_keys += ["function", "id"]
            missing_keys = [key for key in required_keys if job.get(key) is None]


            if missing_keys:
                raise ValueError(f"Missing required keys in job '{job.get('id')}'. Required: {required_keys}, Missing: {missing_keys}")

            # check if "multi" is configured, if so create a multi trigger
            if job.get("multi") is not None:
                triggers = []
                # create a cron trigger for each entry in "multi"
                for multi_trigger in job.get("multi"):

                    trigger_args = {
                        "day_of_week": multi_trigger.get("day_of_week", job.get("day_of_week")),
                        "hour": multi_trigger.get("hour", job.get("hour")),
                        "minute": multi_trigger.get("minute", job.get("minute")),
                        "timezone": self.ny_tz
                    }

                    # If 'second' is provided, include it
                    if multi_trigger.get("second") is not None:
                        trigger_args["second"] = multi_trigger.get("second")

                    
                    triggers.append(CronTrigger(**trigger_args))
                    
                trigger = OrTrigger(triggers)
            else:
                # Build trigger arguments dynamically, excluding 'second' if missing
                trigger_args = {
                    "day_of_week": job.get("day_of_week"),
                    "hour": job.get("hour"),
                    "minute": job.get("minute"),
                    "timezone": self.ny_tz
                }

                # If 'second' is provided, include it
                if job.get("second") is not None:
                    trigger_args["second"] = job.get("second")

                trigger = CronTrigger(**trigger_args)

            # Retrieve function dynamically
            job_function = getattr(self, job.get("function"), None)
            if job_function:
                kwargs = job.get("kwargs", {})

                # Validate kwargs before scheduling
                if not self.validate_kwargs(job_function, kwargs):
                    return

                try:
                    if self.validate_kwargs(job_function, {"job_id": job.get("id")}):
                        kwargs["job_id"] = job.get("id")
                except ValueError:
                    self.app.addToActionLog(f"Invalid kwargs for job '{job.get('id')}'. Skipping job.")

                self.scheduler.add_job(
                    func=job_function,
                    trigger=trigger,
                    kwargs=kwargs,
                    id=job.get("id"),
                    misfire_grace_time=15
                )

                if job.get("terminalTitle", False):
                    self.jobForTitleTime = job.get("id")


                if job.get("display", True):
                    """
                    self.app.additionalTilesFuncs.append({
                        "function": lambda: self.printJobLegs(job_function, kwargs),
                        "title": job.get("title", job.get("id")),
                        "colspan": job.get("colspan", 4)
                    })
                    """

                    tile_func = partial(self.printJobLegs, job_function, copy.deepcopy(kwargs))

                    self.app.additionalTilesFuncs.append({
                        "function": tile_func,
                        "title": job.get("title", job.get("id")),
                        "colspan": job.get("colspan", 4)
                    })

        
        
        
        return super().run()
    
    def printJobLegs(self, function, kwargs):
        legs = None
        try:
            if function.__name__ == "shutdown":
                return "Shutdown function does not return legs"

            # Add "order=False" to kwargs if not present
            if self.validate_kwargs(function, {"order": False}) and "order" not in kwargs:
                kwargs["order"] = False

            legs = function(**kwargs)
            if legs is None:
                return "No legs found for the job"
            
            if not isinstance(legs, dict):
                return legs


            try:
                # Define the desired order of keys
                key_order = ["long_call", "short_call", "short_put", "long_put", "short_stock", "long_stock"]

                # Ensure all expected keys are present, assigning None if missing
                ordered_legs = {key: legs.get(key, None) for key in key_order}
                if len(ordered_legs) == 0:
                    return legs

                return IBApp.defineSpreadTile("", ordered_legs, 2, kwargs.get("tp", 1))
            except Exception as e:
                return legs
        except Exception as e:
            return legs


    
    def checkDeltaIronCondor(self, short_delta=0.4, long_delta=0.16, orderRef="DeltaWingIronCondor", maxSpreadDistance=0.1, order=True):
        """
        Check for Delta Iron Condor trades based on the specified deltas.
        """
        isWarmedUp = self.hasWarmedUp()
        hasRoomForNewPositions = self.orderApp.is_room_for_new_positions(self.symbol)
        
        callLegs = self.app.build_credit_spread_by_delta(self.symbol, short_delta, long_delta, "C", requiredOTM=False)
        putLegs = self.app.build_credit_spread_by_delta(self.symbol, short_delta, long_delta, "P", requiredOTM=False)

        # ensure that call legs are higher or at least equal to put legs
        if callLegs is not None and putLegs is not None:
            try:
                shortCall = callLegs.get("short_call")
                longCall = callLegs.get("long_call")
                shortPut = putLegs.get("short_put")
                longPut = putLegs.get("long_put")

                shortCallStrike = shortCall["Strike"] if shortCall is not None and "Strike" in shortCall else None
                longCallStrike = longCall["Strike"] if longCall is not None and "Strike" in longCall else None
                shortPutStrike = shortPut["Strike"] if shortPut is not None and "Strike" in shortPut else None
                longPutStrike = longPut["Strike"] if longPut is not None and "Strike" in longPut else None

                absShortCallDelta = abs(shortCall["delta"]) if shortCall is not None and "delta" in shortCall else None
                absLongCallDelta = abs(longCall["delta"]) if longCall is not None and "delta" in longCall else None
                absShortPutDelta = abs(shortPut["delta"]) if shortPut is not None and "delta" in shortPut else None
                absLongPutDelta = abs(longPut["delta"]) if longPut is not None and "delta" in longPut else None

                undPrice = shortCall["undPrice"]

                max_attempts = 3
                attempts = 0
                
                while shortCallStrike < shortPutStrike and attempts < max_attempts:
                    # compute credit‐spread width and mirror the OTM distance
                    callDistance = longCallStrike - shortCallStrike
                    otmDistance  = abs(shortCallStrike - undPrice)
                    desiredPutStrike = undPrice - otmDistance

                    # rebuild the put side
                    putLegs = self.app.find_credit_spread_legs(
                        self.symbol,
                        desiredPutStrike,
                        "P",
                        callDistance
                    )

                    # extract the new put strikes
                    shortPut = putLegs.get("short_put", {})
                    longPut  = putLegs.get("long_put", {})
                    shortPutStrike = shortPut.get("Strike")
                    longPutStrike  = longPut.get("Strike")

                    attempts += 1
                
                # if still inverted after retries, bail out
                if shortCallStrike < shortPutStrike:
                    self.app.addToActionLog(
                        f"Iron Condor inversion persists after {attempts} attempts "
                        f"(call {shortCallStrike} < put {shortPutStrike}), skipping"
                    )
                    return
            except Exception as e:
                self.app.addToActionLog(str(e))
                Logger.log(str(e))


        icLegs = {**(callLegs or {}), **(putLegs or {})}

        if not order:
            return icLegs
        
        if (not isWarmedUp or not hasRoomForNewPositions) and order:
            self.app.addToActionLog("DONT TRADE DELTA IRON CONDOR TRADE")
            return

        askPrice = TwsOrderAdapter.calcSpreadPrice(icLegs)
        bidPrice = TwsOrderAdapter.calcSpreadPriceOpposite(icLegs)

        absSpreadDistance = abs(abs(askPrice) - abs(bidPrice))

        if absSpreadDistance > maxSpreadDistance:
            self.app.addToActionLog(f"Spread distance {absSpreadDistance} is wide small, not placing order")
            send_telegram_message(f"Delta Iron Condor spread distance {absSpreadDistance} is wide small for {self.symbol}, not placing order")
            return
        self.orderApp.place_combo_order(icLegs, None, None, orderRef, getOutOfMarket=False, touchDistance=None)
        sleep(0.5)

    
    def checkDeltaTrade(self, type="C", delta=0.16, order=True):
        self.app.addToActionLog("CHECKING DELTA TRADE")
        minPrice = self.wing_span * 0.05
        if self.hasWarmedUp():
            if self.orderApp.is_room_for_new_positions(self.symbol, type):
                try:
                    self.app.addToActionLog("Room for Delta Call Trade, adding Orders")

                    legs = self.app.build_credit_spread(self.symbol, delta, type, self.wing_span, minPrice=minPrice)
                    if not order:
                        return legs
                    limit_price = TwsOrderAdapter.calcSpreadPrice(legs)
                    if limit_price is not None and abs(limit_price) >= minPrice:
                        self.orderApp.place_combo_order(legs, self.tp_percentage, 200, "SchedCallCredit")
                        sleep(0.5)
                except Exception as e:
                    self.app.addToActionLog(e)

    def checkSeparatedIronCondor(self, callDistance=None, putDistance=None, wingspan=None, tp=None, sl=None, touchDistance=None, order=True):
        STRIKE_COL = "Strike"
        
        if not self.hasWarmedUp():
            return "Awaiting warmup"
        
        if not self.orderApp.is_room_for_new_positions(self.symbol) and order:
            self.app.addToActionLog("No room for new positions, not placing order")
            if (order):
                Logger.log("No room for new positions, not placing order " + self.symbol)
            return
        
        if callDistance is None:
            callDistance = self.callDistance

        if putDistance is None:
            putDistance = self.putDistance

        if wingspan is None:
            wingspan = self.wing_span

        callLegs = self.app.build_credit_spread_dollar(self.symbol, callDistance, wingspan, "C")   # build_credit_spread(self.symbol, callDistance, "C", wingspan)
        putLegs = self.app.build_credit_spread_dollar(self.symbol, putDistance, wingspan, "P")

        # Check if long legs are further away from the underlying than short legs
        if callLegs is None or putLegs is None:
            if (order):
                Logger.log("No valid IC Legs found "+ self.symbol)
            return
        

        
        shortCallStrike = callLegs.get("short_call", {}).get(STRIKE_COL)
        longCallStrike = callLegs.get("long_call", {}).get(STRIKE_COL)
        if shortCallStrike > longCallStrike:
            self.app.addToActionLog("Short call strike is less than long call strike, not placing order")
            Logger.log("Short call strike is less than long call strike, not placing order")
            return
        
        shortPutStrike = putLegs.get("short_put", {}).get(STRIKE_COL)
        longPutStrike = putLegs.get("long_put", {}).get(STRIKE_COL)
        if shortPutStrike < longPutStrike:
            self.app.addToActionLog("Short put strike is greater than long put strike, not placing order")
            Logger.log("Short put strike is greater than long put strike, not placing order")
            return
        
        try:
            legs = {**callLegs, **putLegs}

            if not order:
                return legs
            Logger.log("Placing Separated Combo Order " + self.symbol)
            forceTouchOrder = sl is not None and touchDistance is not None
            self.orderApp.place_combo_order(legs, tp=tp, sl=sl, ref="SeparatedIronCondor", getOutOfMarket=False, touchDistance=touchDistance, forceTouchOrder=forceTouchOrder)
            sleep(0.5)
        except Exception as e:
            self.app.addToActionLog(e)
            Logger.log(e)
    
    def checkLongORB(self, orb=60, tp=150, sl=90, mid=0.1, order=True, withPut=True, withCalls=True):
        call = self.app.find_single_contract_by_mid_price(self.symbol, mid, "C") if withCalls else None
        put = self.app.find_single_contract_by_mid_price(self.symbol, mid, "P")  if withPut else None
        return self.checkLongContractORB(call, put, orb=orb, tp=tp, sl=sl, order=order)
    
    def checkLongOrbByDelta(self, orb=60, tp=110, sl=90, delta=0.4, order=True, withPut=True, withCalls=True):
        call = self.app.get_closest_delta_row(self.symbol, delta, "C") if withCalls else None
        put = self.app.get_closest_delta_row(self.symbol, delta, "P") if withPut else None

        try:
            if call is not None:
                self.app.request_options_market_data(call)
            if put is not None:
                self.app.request_options_market_data(put)
        except Exception as e:
            self.app.addToActionLog(f"Error requesting market data for options: {e}")
            Logger.log(f"Error requesting market data for options: {e}")


        return self.checkLongContractORB(call, put, orb=orb, tp=tp, sl=sl, order=order)

    def checkLongContractORB(self, call, put, orb=60, tp=150, sl=90, order=True):
        orb: OrbResult = self.app.find_orb_by_symbol(self.symbol, orb)
        aboveOrBelow = None
        completed = None
        if orb is not None:
            completed = orb.isOpenRangeCompleted
            if orb.isAbove:
                aboveOrBelow = "above"
            elif orb.isBelow:
                aboveOrBelow = "below"
            else:
                aboveOrBelow = "neither"

        if not order:
            return {
                "completed": completed,
                "aboveOrBelow": aboveOrBelow,
                "long_call": call,
                "long_put": put
            }
        if not self.hasWarmedUp():
            self.app.addToActionLog("Awaiting warmup for Long ORB trade")
            return
        
        if (self.orderApp.minutesPassedSinceLastPositionClosed() < 5 and order):
            self.app.addToActionLog("Not enough time has passed since the last position was closed, skipping order placement")
            return
        
        if not self.orderApp.is_room_for_new_positions(self.symbol):
            self.app.addToActionLog("No room for new positions, not placing order")
            return
        if orb is None or not orb.isOpenRangeCompleted:
            self.app.addToActionLog("Open Range not completed, not placing order")
            return
        
        if orb.breakout_age > 10:
            self.app.addToActionLog("Open Range breakout age is too high, not placing order")
            return

        if aboveOrBelow is None:
            self.app.addToActionLog("Open Range not above or below, not placing order")
            Logger.log("Open Range not above or below, not placing order " + self.symbol)
            return

        if orb.isAbove and call is not None:
            self.app.addToActionLog(f"Ordering Long Call {self.symbol} at {call['Strike']} with TP {tp}")
            Logger.log(f"Ordering Long Call {self.symbol} at {call['Strike']} with TP {tp}")
            self.orderApp.place_single_contract_order(call, tp=tp, sl=sl, orderRef="LongCall_ORB")
            sleep(0.5)            
        elif orb.isBelow and put is not None:
            self.app.addToActionLog(f"Ordering Long Put {self.symbol} at {put['Strike']} with TP {tp}")
            Logger.log(f"Ordering Long Put {self.symbol} at {put['Strike']} with TP {tp}")
            self.orderApp.place_single_contract_order(put, tp=tp, sl=sl, orderRef="LongPut_ORB")
            sleep(0.5)
        
                                                             
    def checkOrbCreditSpread(self, wingspan, tp=None, sl=None, orb=60, order=True, touchDistance=None, minRiskRewardPercentage=5.5, minRangeWidthPercentage=0.08, max_orb_age=10, job_id=None, putOnly=False, callOnly=False):
        try:
            """
            Check for Open Range Breakout Credit Spread trades.
            """
            orb: OrbResult = self.app.find_orb_by_symbol(self.symbol, orb)
            if orb is None or not orb.isOpenRangeCompleted:
                self.app.addToActionLog("Open Range not completed, not placing order")
                if order:
                    Logger.log("Open Range not completed, not placing order " + self.symbol)
                return
            
            if not self.hasWarmedUp():
                self.app.addToActionLog("Awaiting warmup for ORB Credit Spread trade")
                return
            
            if not self.orderApp.is_room_for_new_positions(self.symbol) and order:
                self.app.addToActionLog("No room for new positions, not placing order")
                return

            callLegs = None
            putlegs = None

            if not putOnly:
                callLegs = self.app.find_credit_spread_legs(self.symbol, orb.high, "C", wingspan) 

            if not callOnly:
                putlegs = self.app.find_credit_spread_legs(self.symbol, orb.low, "P", wingspan) 

            if not order:
                return {**(callLegs or {}), **(putlegs or {})}
            today = date.today()
            if orb.breakout_age > max_orb_age or not orb.isOpenRangeCompleted or orb.date != today:
                self.app.addToActionLog("Open Range breakout age is too high or not completed, not placing order")
                return
            
            if orb.range_width_pct < minRangeWidthPercentage:
                self.app.addToActionLog(f"Open Range width percentage {orb.range_width_pct:.2f}% is too low, not placing order")
                self.scheduler.remove_job(job_id) if job_id else None
                return

            legs = None
            if orb.isAbove:
                self.app.addToActionLog(f"Open Range is above, building call credit spread for {self.symbol}")
                legs = putlegs
            elif orb.isBelow:
                self.app.addToActionLog(f"Open Range is below, building put credit spread for {self.symbol}")
                legs = callLegs

            if legs is None:
                self.app.addToActionLog("No valid ORB Credit Spread legs found, not placing order")
                Logger.log("No valid ORB Credit Spread legs found, not placing order " + self.symbol)
                return
            
            # check if legs have different strie  
            shortLeg = legs.get("short_call") or legs.get("short_put")
            longLeg = legs.get("long_call") or legs.get("long_put")
            if shortLeg is None or longLeg is None:
                self.app.addToActionLog("No valid short or long leg found for ORB Credit Spread, not placing order")
                Logger.log("No valid short or long leg found for ORB Credit Spread, not placing order " + self.symbol)
                return
            if shortLeg["Strike"] == longLeg["Strike"]:
                self.app.addToActionLog("Short and long leg have the same strike, not placing order")
                Logger.log("Short and long leg have the same strike, not placing order " + self.symbol)
                return

            risk = TwsOrderAdapter.calcMaxRisk(legs)
            midPrice = abs(TwsOrderAdapter.calcSpreadPrice(legs))

            rewardRiskPercentage = midPrice / risk * 100 if risk != 0 else 0

            if rewardRiskPercentage < minRiskRewardPercentage:
                self.app.addToActionLog(f"Reward/Risk percentage {rewardRiskPercentage:.2f}% ({midPrice} / {risk}) is below minimum {minRiskRewardPercentage}%, not placing order")
                return
            
            Logger.log("Placing ORB Credit Spread Order " + self.symbol)
            self.orderApp.place_combo_order(legs, tp=tp, sl=sl, ref="OrbCreditSpread", getOutOfMarket=False, touchDistance=touchDistance, useMidPrice=True)
        except Exception as e:
            self.app.addToActionLog(f"Error in checkOrbCreditSpread: {e}")
            Logger.log(f"Error in checkOrbCreditSpread: {e}")
            return None

    def checkFixedDistanceIronCondor(self, distance=None, wingspan=None, tp=None, sl=None, touchDistance=None, order=True):
        """
        callLegs = self.app.build_credit_spread(self.secondarySymbol, 0.4, "C", 1)
        putLegs = self.app.build_credit_spread(self.secondarySymbol, 0.4, "P", 1)
        hasCallLegs = self.orderApp.has_existing_order_contracts(callLegs)
        hasPutLegs = self.orderApp.has_existing_order_contracts(putLegs)

        ironCondorLegs = {**callLegs, **putLegs}

        icPrice = TwsOrderAdapter.calcSpreadPrice(ironCondorLegs)

        if not hasCallLegs and not hasPutLegs and abs(icPrice) > 0.2:
            self.app.addToActionLog("Ordering 2155 Iron Condor "+ self.secondarySymbol)
            self.orderApp.place_combo_order(ironCondorLegs, None, 200, "IronCondor_IC_2155", getOutOfMarket=False)
            sleep(0.5)
        """
        if distance is None:
            distance = self.ic_wingspan

        if wingspan is None:
            wingspan = self.wing_span

        self.app.addToActionLog("Check IC "+ self.symbol)
        legs = self.app.construct_from_underlying(self.symbol, distance, wingspan)
        legsPrice = TwsOrderAdapter.calcSpreadPrice(legs)

        if not order:
            return legs

        elif self.hasWarmedUp() and self.orderApp.is_room_for_new_positions(self.symbol) and abs(legsPrice) > 0.2: 
            # check if touch needs to be forced
            forceTouchOrder = sl is not None and touchDistance is not None

            self.app.addToActionLog("Ordering Iron Condor")
            self.orderApp.place_combo_order(legs, tp, sl, "IronCondor", getOutOfMarket=False, touchDistance=touchDistance, forceTouchOrder=forceTouchOrder)
            sleep(0.5)
            
    def check03ReferenceLevel(self, orderRef="03_Reference", order=True):
        data = self.app.checkUnderlyingOptions(self.symbol)
        row = data.get("row")
        signal = data.get("ref03_signal", "WAIT")
        if (signal is None):
            signal = "WAIT"
        dte = 0
        if (row is not None):        
            val = row.get("ref03_dte", 0)
            dte = int(val) if pd.notna(val) else 0
            
        contracts = {}
        contracts["long_call"] = self.app.get_by_delta_or_lower_by_dte(
                symbol=self.symbol,
                target_delta=0.50,
                min_price=0.10,
                type="C",
                targetDTE=dte,
                exactDTE=True,
                closestDelta=True
            )
        contracts["long_put"] = self.app.get_by_delta_or_lower_by_dte(
                symbol=self.symbol,
                target_delta=0.50,
                min_price=0.10,
                type="P",
                targetDTE=dte,
                exactDTE=True,
                closestDelta=True
            )
        
        for c in contracts.values():
            if c is not None:
                self.app.request_options_market_data(c)
                
        if (dte != self.lastRequestedDte):
            self.goForNextDayOptionsData(dte - self.lastRequestedDte)
            
            

        if (order):
            pass
        else:
            
            
            if (signal in contracts.keys()):
                return { f"{signal}": contracts["signal"]}
            
            contracts[f"long_{signal}"] = pd.Series({
                "Symbol": signal
            })
            return contracts

    def checkAtrPutCreditSpread(self, wingSpan=50, tp=0.9, tickSize=0.05, orderRef="ATR", order=True):
        targetType = "P"

        atrTargetPrice = self.app.checkUnderlyingOptions(self.symbol).get("atrTarget", 0)
        
        if (not atrTargetPrice or not atrTargetPrice > 0):
            return
        
        shortContract = self.app.find_closest_strike_for_symbol(self.symbol, atrTargetPrice, targetType)
        longContract = self.app.find_closest_strike_for_symbol(self.symbol, atrTargetPrice - wingSpan, targetType)
        
        self.app.request_options_market_data(shortContract)
        self.app.request_options_market_data(longContract)
        
        contract_rows = {
                "short_put": shortContract,
                "long_put": longContract
            }
        
        if order:
            shortBid = shortContract.get("bid")
            shortAsk = shortContract.get("ask")
            longBid  = longContract.get("bid")
            longAsk  = longContract.get("ask")
            
            if (shortAsk > 0 and shortBid > 0 and longAsk > 0 and longBid > 0):
                # 1. Mittelkurse berechnen
                shortMid = (shortBid + shortAsk) / 2
                longMid = (longBid + longAsk) / 2
                
                # 2. Roh-Midprice des Spreads
                rawSpreadPrice = shortMid - longMid
                
                # 3. Auf die Tick-Size runden (z.B. auf 0.05 Schritte)
                spreadPrice = round(round(rawSpreadPrice / tickSize) * tickSize, 2) - self.getOrdersPlaced(orderRef) * tickSize
                if (spreadPrice > 0):
                    if not self.hasWarmedUp():
                        self.app.addToActionLog("Awaiting warmup for Wheel Trade")
                        return
                    
                    if self.orderApp.hasOptionPositionsForSymbol(self.symbol):
                        self.shutdown()
                    
                    if self.orderApp.has_existing_order_contracts(contract_rows):
                        return
                    
                    self.increaseOrderPlaced(orderRef)
                    self.orderApp.place_combo_order(contract_rows, lmt=spreadPrice, ref=orderRef, forceTouchOrder=False, getOutOfMarket=False, touchDistance=None)
                    
                    
        else:
            return contract_rows
        

    def checkWheelTrade(self, distance=0.2, zeroDteMinPrice=0.2, xDteMinPrice=0.25, order=True):
        putType = "P"
        callType = "C"

        # check for existing stock position
        stock_position = self.orderApp.getStockPositionsForSymbol(self.symbol)
        if stock_position is None:
            return
        
        if self.currentDteWheelMinPrice is None:
            self.currentDteWheelMinPrice = zeroDteMinPrice
        
        amountOfShares = stock_position.get("position", 0)
        amountOfContracts = abs(amountOfShares // 100)
        targetStrikeByPosition = stock_position.get("avgCost", None)
        underlyingPrice = self.app.checkUnderlyingOptions(self.symbol).get("price", 0)
        targetPrice = None
        targetType = None
        daysBetween = self.app.business_days_since_last_request()
        referenceDistance = underlyingPrice * (0.02 * daysBetween)

        if underlyingPrice is None or targetStrikeByPosition is None:
            return

        if amountOfShares <= 0: # short position, consider selling puts
            targetPrice = min(targetStrikeByPosition, underlyingPrice - distance)
            targetPrice = math.floor(targetPrice)
            targetType = putType

            if math.floor(underlyingPrice) - targetPrice > referenceDistance:
                oldTargetPrice = targetPrice
                targetPrice = math.floor(underlyingPrice - referenceDistance) 
                #self.app.addToActionLog(f"Target price for puts is too far from underlying price, adjusting to max 5 below. Underlying: {underlyingPrice}, Old Target: {oldTargetPrice} New Target: {targetPrice}")
                Logger.log(f"Target price for puts is too far from underlying price, adjusting to max {referenceDistance} below. Underlying: {underlyingPrice}, Old Target: {oldTargetPrice}, New Target: {targetPrice}")
            

        else: # long position, consider selling calls
            targetPrice = max(targetStrikeByPosition, underlyingPrice + distance)
            targetPrice = math.ceil(targetPrice)
            targetType = callType


            if targetPrice - math.ceil(underlyingPrice) > referenceDistance:
                oldTargetPrice = targetPrice                
                targetPrice = math.ceil(underlyingPrice + referenceDistance) 

                

                #self.app.addToActionLog(f"Target price for calls is too far from underlying price, adjusting to max 5 above. Underlying: {underlyingPrice}, Old Target: {oldTargetPrice} New Target: {targetPrice}")
                Logger.log(f"Target price for calls is too far from underlying price, adjusting to max  {referenceDistance} above. Underlying: {underlyingPrice}, Old Target: {oldTargetPrice}, New Target: {targetPrice}")

        if targetPrice is None or targetType is None:
            self.app.addToActionLog("No valid target price or type for Wheel Trade")
            return
        
        optionContract = self.app.find_closest_strike_for_symbol(self.symbol, targetPrice, targetType)
        if order:
            
            if not self.hasWarmedUp():
                self.app.addToActionLog("Awaiting warmup for Wheel Trade")
                return
            
            if self.firstWheelAttemptTime is None:
                self.firstWheelAttemptTime = datetime.now()
           
            if optionContract is None:
                self.app.addToActionLog("No valid option contract found for Wheel Trade")
                Logger.log(f"No valid option contract found for {self.symbol} {targetType} at {targetPrice}")
                if self.isWheelThresoldTimePassed():
                    send_telegram_message(f"No valid option contract found for {self.symbol} {targetType} at {targetPrice}, trying next day options data")
                    self.goForNextDayOptionsData()
                    self.firstWheelAttemptTime = None
                return
            
            midPrice = (optionContract.get("ask") + optionContract.get("bid")) / 2 if optionContract.get("ask") is not None and optionContract.get("bid") is not None else None
            
            if midPrice is None or midPrice < self.currentDteWheelMinPrice:
                self.app.addToActionLog(f"Mid price for {self.symbol} {targetType} at {targetPrice} is too low: {midPrice}, not placing order")
                Logger.log(f"Mid price for {self.symbol} {targetType} at {targetPrice} is too low: {midPrice}, not placing order")

                try:
                    # when no valid price is found within 5 minutes, try next day options data
                    
                    if self.isWheelThresoldTimePassed():
                        self.app.addToActionLog("No valid price found for Wheel Trade within 15 minutes, fetching next day options data")
                        send_telegram_message(f"Mid price for {self.symbol} {targetType} at {targetPrice} is too low: {midPrice}, trying next day options data")
                        self.currentDteWheelMinPrice = xDteMinPrice

                        self.goForNextDayOptionsData()
                        self.firstWheelAttemptTime = None
                except Exception as e:
                    Logger.log(f"Error while checking time for next day options data fetch: {e}")


                return
            
            if self.orderApp.hasPositionsForSymbol(self.symbol, targetType):
                self.app.addToActionLog(f"Existing position found for {self.symbol} {targetType}, not placing new Wheel Trade order")
                return
            
            self.app.addToActionLog(f"Placing Wheel Trade order for {self.symbol} at {targetPrice} with type {targetType}")
            if self.orderApp.has_existing_order_contracts({f"short_{'put' if targetType == putType else 'call'}": optionContract}):
                self.app.addToActionLog(f"Existing order found for {self.symbol} {targetType} at {targetPrice}, not placing duplicate order")
                return
            tp = None
            dte = optionContract.get("DTE", None)
            if (dte is not None and midPrice > 2 * xDteMinPrice):
                tp = math.floor(midPrice / 2)
                tp = max(tp, 0.05)
                
            self.orderApp.place_single_contract_order(optionContract, orderRef="WheelTrade", action="SELL", quantity=amountOfContracts, tp=tp)
            sleep(0.5)
        else:
            if targetType == putType:
                return {
                    "short_put": optionContract
                }
            elif targetType == callType:
                return {
                    "short_call": optionContract
                }
            return optionContract

        
    def check_and_place_wheel_trade(
        self,
        distance: float = 0.2,
        zeroDteMinPrice: float = 0.2,
        xDteMinPrice: float = 0.25,
        target_delta: float = 0.3,
        order: bool = True,
        next_day_days: int = 7,
        max_attempts_before_escalation: int = 3
    ):
        """
        - Exit immediately if there is no stock position (None or 0 shares).
        - Detect long (shares > 0) or short (shares < 0) and decide to sell CALLs (long) or PUTs (short).
        - Search for an option via get_by_delta_or_lower_by_dte.
        - If no option matches, escalate by requesting next-day options data and increase min-price threshold.
        - If an option is found, compute a robust midPrice (ask/bid/last fallback) and place a SELL order if criteria met.
        - Uses self.orderApp, self.app, Logger, send_telegram_message, and self.goForNextDayOptionsData as in your codebase.
        """

        putType = "P"
        callType = "C"

        # --- 1) Get and validate stock position (exit if none or zero) ---
        stock_position = self.orderApp.getStockPositionsForSymbol(self.symbol)
        if not stock_position:
            self.app.addToActionLog(f"No stock position found for {self.symbol}; exiting Wheel Trade")
            return "No stock position found"

        amountOfShares = int(stock_position.get("position", 0) or 0)
        if amountOfShares == 0:
            self.app.addToActionLog(f"No stock position (0 shares) for {self.symbol}; exiting Wheel Trade")
            return "No stock position found"

        # --- 2) Setup and decide target type/strike ---
        targetStrikeByPosition = stock_position.get("avgCost", None)
        underlyingPrice = self.app.checkUnderlyingOptions(self.symbol).get("price", None)

        if targetStrikeByPosition is None or underlyingPrice is None:
            self.app.addToActionLog(f"Missing price data for {self.symbol}: underlying={underlyingPrice}, avgCost={targetStrikeByPosition}")
            return "Missing price data"

        # Decide whether to sell puts (cover short) or calls (cover long)
        if amountOfShares < 0:
            # short stock position -> consider selling puts (to collect premium / roll)
            targetType = putType
            targetPrice = math.floor(targetStrikeByPosition)
        else:
            # long stock position -> consider selling calls (covered calls)
            targetType = callType
            targetPrice = math.ceil(targetStrikeByPosition)

        # Ensure min-price state exists
        if self.currentDteWheelMinPrice is None:
            self.currentDteWheelMinPrice = zeroDteMinPrice

        # --- 3) Attempt to find an option contract ---
        self.app.optionMarketDataRequestTargetStrike = targetPrice
        optionContract = self.app.get_by_delta_or_lower_by_dte(
            self.symbol,
            target_delta,
            self.currentDteWheelMinPrice,
            targetType,
            target_underlying_price=targetPrice
        )
        
        if optionContract is None:
            self.app.request_closest_strike(self.symbol, targetPrice, targetType)
            optionContract = self.app.find_closest_strike_by_dte(df=self.app.options_data, symbol=self.symbol, strike=targetPrice, option_type=targetType)

        # Normalize optionContract to dict if it's a pandas Series/row
        if optionContract is not None and hasattr(optionContract, "to_dict"):
            self.app.request_options_market_data(optionContract)
            optionContract = optionContract.to_dict()

        # --- 4) If no option found -> escalate (request next-day options) ---
        if optionContract is None:
            # increment attempt counter (create if missing)
            self.wheel_attempts = getattr(self, "wheel_attempts", 0) + 1
            self.app.addToActionLog(f"No option contract found for {self.symbol} {targetType} at {targetPrice} (attempt {self.wheel_attempts})")
            Logger.log(f"No option contract found for {self.symbol} {targetType} at {targetPrice} (attempt {self.wheel_attempts})")

            # If we've tried enough times, escalate to next-day options and raise min price
            if self.wheel_attempts >= max_attempts_before_escalation or self.isWheelThresoldTimePassed():
                if not self.orderApp.hasOptionPositionsForSymbol(self.symbol):
                    if order:
                        send_telegram_message(f"No valid option contract found for {self.symbol} {targetType} at {targetPrice}, requesting next-day options")
                        # raise the min price for xDTE attempts
                        self.currentDteWheelMinPrice = xDteMinPrice
                        self.goForNextDayOptionsData(daysToAdd=next_day_days, reset=True)
                        self.wheel_attempts = 0
                        self.firstWheelAttemptTime = None
                    return "Request next-day options data"
            return "No contract found that suits the search parameters"

        # Reset attempts on success
        self.wheel_attempts = 0

        # --- 5) Validate strike relation (avoid selling OTM in wrong direction) ---
        optionStrike = optionContract.get("Strike")
        if optionStrike is None:
            Logger.log(f"Option strike missing for {self.symbol} contract; skipping strike-range check")
        else:
            if (targetType == putType and optionStrike > targetStrikeByPosition) or (targetType == callType and optionStrike < targetStrikeByPosition):
                # Contract is on the wrong side of the position; escalate to next-day options
                self.app.addToActionLog("Contract out of strike range, requesting next-day options")
                send_telegram_message(f"{self.symbol} {targetType} contract out of strike range: found {optionStrike}, target {targetStrikeByPosition}")
                if not self.orderApp.hasOptionPositionsForSymbol(self.symbol) and order:
                    self.goForNextDayOptionsData(daysToAdd=next_day_days, reset=True)
                    return "Contract out of strike range; requested next-day options"

        # --- 6) Compute robust midPrice (ask/bid/last fallback) ---
        ask = optionContract.get("ask")
        bid = optionContract.get("bid")
        last = optionContract.get("last")
        if ask is not None and bid is not None:
            midPrice = (ask + bid) / 2.0
        elif ask is not None:
            midPrice = ask
        elif bid is not None:
            midPrice = bid
        elif last is not None:
            midPrice = last
        else:
            midPrice = None

        if midPrice is None or midPrice < self.currentDteWheelMinPrice or not midPrice > 0:
            self.app.addToActionLog(f"Mid price too low for {self.symbol} {targetType} at {targetPrice}: {midPrice}")

            # escalate to next-day options if threshold/time/attempts reached
            self.wheel_attempts = getattr(self, "wheel_attempts", 0) + 1
            if self.wheel_attempts >= max_attempts_before_escalation or self.isWheelThresoldTimePassed():
                if order and not self.orderApp.hasOptionPositionsForSymbol(self.symbol):
                    send_telegram_message(f"Mid price too low for {self.symbol} {targetType} at {targetPrice}: {midPrice}. Requesting next-day options.")
                    self.currentDteWheelMinPrice = xDteMinPrice
                    self.goForNextDayOptionsData(daysToAdd=next_day_days, reset=True)
                    self.wheel_attempts = 0
                    self.firstWheelAttemptTime = None
                    return "Request next-day options due to low price"
            return self.toReturnDict(optionContract, targetType)


        # --- 7) Ensure we have contracts to sell (quantity) ---
        amountOfContracts = abs(amountOfShares) // 100
        if amountOfContracts <= 0:
            self.app.addToActionLog(f"Insufficient shares to create option contracts for {self.symbol} (shares={amountOfShares})")
            return "Insufficient contracts"

        # --- 8) Check for existing positions or duplicate orders ---
        if self.orderApp.hasPositionsForSymbol(self.symbol, targetType):
            self.app.addToActionLog(f"Existing option position found for {self.symbol} {targetType}; not placing new order")
            return "Existing option position present"

        short_key = f"short_{'put' if targetType == putType else 'call'}"
        if self.orderApp.has_existing_order_contracts({short_key: optionContract}):
            self.app.addToActionLog(f"Existing order found for {self.symbol} {targetType} at {optionStrike}; skipping duplicate")
            return "Duplicate order exists"

        # --- 9) Place order (if requested) ---
        if order:
            if not self.hasWarmedUp():
                self.app.addToActionLog("Awaiting warmup for Wheel Trade")
                return "Awaiting warmup"
            
            if not midPrice > 0:
                return 

            # compute take-profit (tp) if daySteps configured
            tp = None
            if getattr(self, "daySteps", 0) != 0:
                tp = max(math.floor(midPrice / 2 * 100) / 100, 0.05)

            self.app.addToActionLog(f"Placing Wheel Trade SELL order for {self.symbol} {targetType} strike {optionStrike} qty {amountOfContracts} mid {midPrice}")
            self.orderApp.place_single_contract_order(
                optionContract,
                orderRef="WheelTrade",
                action="SELL",
                quantity=amountOfContracts,
                mid=midPrice,
                tp=tp
            )
            time.sleep(0.5)
            return "Order placed"

        # --- 10) If not ordering, return the contract info for inspection ---
        return self.toReturnDict(optionContract, targetType)

    def toReturnDict(self, optionContract, targetType):
        putType = "P"
        callType = "C"
        if targetType == putType:
            return {"short_put": optionContract}
        else:
            return {"short_call": optionContract}

    def checkDeltaWheelTrade(self, distance=0.2, zeroDteMinPrice=0.2, xDteMinPrice=0.25, order=True):
        putType = "P"
        callType = "C"

        # check for existing stock position
        stock_position = self.orderApp.getStockPositionsForSymbol(self.symbol)
        if stock_position is None:
            return "No stock position found"
        
        if self.currentDteWheelMinPrice is None:
            self.currentDteWheelMinPrice = zeroDteMinPrice
        
        amountOfShares = stock_position.get("position", 0)
        amountOfContracts = abs(amountOfShares // 100)
        targetStrikeByPosition = stock_position.get("avgCost", None)
        underlyingPrice = self.app.checkUnderlyingOptions(self.symbol).get("price", 0)
        targetPrice = None
        targetType = None
        daysBetween = self.app.business_days_since_last_request()
        

        if underlyingPrice is None or targetStrikeByPosition is None:
            return f"Prices not found underlying: {underlyingPrice}, targetStrikePrice {targetStrikeByPosition}"

        if amountOfShares <= 0: # short position, consider selling puts
            targetType = putType
            targetPrice = math.floor(targetStrikeByPosition)

        else: # long position, consider selling calls
            targetType = callType
            targetPrice = math.ceil(targetStrikeByPosition)
        
        self.app.optionMarketDataRequestTargetStrike = targetPrice
        optionContract = self.app.get_by_delta_or_lower_by_dte(self.symbol, 0.3, self.currentDteWheelMinPrice, targetType, target_underlying_price=targetPrice)
        optionContractTargetStrike = None
        if optionContract is not None:
            optionContractTargetStrike = optionContract.get("Strike")
        """
        if targetType == putType and optionContractTargetStrike > targetStrikeByPosition:
            optionContract = self.app.find_closest_strike_for_symbol(self.symbol, targetPrice, targetType)
        if targetType == callType and optionContractTargetStrike < targetStrikeByPosition:
        """
        if order:
            if self.app.contractDetailsPending:
                self.app.addToActionLog("Contract Details Pending")
                return "Contract Details Pending"
            
            summary = build_options_summary(self.app.options_data)
            runningRequests = f"{len(self.app.option_id_req)} options_id_req | {len(self.app.market_data_req_ids)} market_data_req_ids"
            self.app.addToActionLog(runningRequests)
            self.app.addToActionLog(summary)
            
            if not self.hasWarmedUp():
                self.app.addToActionLog("Awaiting warmup for Wheel Trade")
                return "Awaiting warmup for Wheel Trade"
            
            if self.firstWheelAttemptTime is None:
                self.firstWheelAttemptTime = datetime.now()
           
            if optionContract is None:
                self.app.addToActionLog(f"No valid option contract found for {self.symbol} {targetType} at {targetPrice}")
                
                Logger.log(f"No valid option contract found for {self.symbol} {targetType} at {targetPrice}")
                if self.isWheelThresoldTimePassed(1):
                    if not self.orderApp.hasOptionPositionsForSymbol(self.symbol):
                        send_telegram_message(f"No valid option contract found for {self.symbol} {targetType} at {targetPrice}, trying next day options data")
                        self.goForNextDayOptionsData(daysToAdd=7, reset=True)
                        self.firstWheelAttemptTime = None
                return
            

            if (targetType == putType and optionContractTargetStrike > targetStrikeByPosition) or (targetType == callType and optionContractTargetStrike < targetStrikeByPosition):
                
                if self.hasWarmedUp(0.5):
                    self.app.addToActionLog("Contract out of strike range, next day")
                    send_telegram_message(f"{targetType} Contract out of strike range, next day. found {optionContractTargetStrike}, target {targetStrikeByPosition}")
                    if not self.orderApp.hasOptionPositionsForSymbol(self.symbol):
                        self.goForNextDayOptionsData(daysToAdd=7, reset=True)

                        return "Try next day"

            midPrice = (optionContract.get("ask") + optionContract.get("bid")) / 2 if optionContract.get("ask") is not None and optionContract.get("bid") is not None else None
            
            if midPrice is None or midPrice < self.currentDteWheelMinPrice:
                self.app.addToActionLog(f"Mid price for {self.symbol} {targetType} at {targetPrice} is too low: {midPrice}, not placing order")
                Logger.log(f"Mid price for {self.symbol} {targetType} at {targetPrice} is too low: {midPrice}, not placing order")

                try:
                    # when no valid price is found within 5 minutes, try next day options data
                    
                    if self.isWheelThresoldTimePassed():
                        if not self.orderApp.hasOptionPositionsForSymbol(self.symbol):
                        
                            
                            self.app.addToActionLog("No valid price found for Wheel Trade within 15 minutes, fetching next day options data")
                            send_telegram_message(f"Mid price for {self.symbol} {targetType} at {targetPrice} is too low: {midPrice}, trying next day options data")
                            self.currentDteWheelMinPrice = xDteMinPrice

                            self.goForNextDayOptionsData(7, reset=True)
                            self.firstWheelAttemptTime = None
                except Exception as e:
                    Logger.log(f"Error while checking time for next day options data fetch: {e}")


                return
            
            if self.orderApp.hasPositionsForSymbol(self.symbol, targetType):
                self.app.addToActionLog(f"Existing position found for {self.symbol} {targetType}, not placing new Wheel Trade order")
                return
            
            self.app.addToActionLog(f"Placing Wheel Trade order for {self.symbol} at {targetPrice} with type {targetType}")
            if self.orderApp.has_existing_order_contracts({f"short_{'put' if targetType == putType else 'call'}": optionContract}):
                self.app.addToActionLog(f"Existing order found for {self.symbol} {targetType} at {targetPrice}, not placing duplicate order")
                return
            tp = None
            if (self.daySteps != 0):
                tp = max(math.floor(midPrice / 2 * 100) / 100, 0.05)
            self.orderApp.place_single_contract_order(optionContract, orderRef="WheelTrade", action="SELL", quantity=amountOfContracts, mid=midPrice, tp=tp)
            sleep(0.5)
        else:
            if optionContract is None:
                return "No contract found that suits the search parameters"
            
            
            if targetType == putType:
                return {
                    "short_put": optionContract
                }
            elif targetType == callType:
                return {
                    "short_call": optionContract
                }
            return optionContract or "Contract not found"
        
   
    def checkLongPositionTakeProfit(self, requiredDistance=0.005, displayAmountShares=-100, displayCostBasis=247.34, order=True):
        # check for existing stock position
        stock_position = self.orderApp.getStockPositionsForSymbol(self.symbol)
        hint = ""
        if stock_position is None:
            hint = "No stock position"
            if order:
                return hint
            
            dictContract = {
                self.symbol: self.symbol
            }
            amountOfShares = displayAmountShares
            costBasis = displayCostBasis
        else:
            contract: Contract = stock_position["contract"]
            dictContract = {
                "ConId": contract.conId,
                "Symbol": contract.symbol            
            }
        
            amountOfShares = float(stock_position.get("position", 0))
            costBasis = stock_position.get("avgCost", None)
        # underlyingPrice = self.app.checkUnderlyingOptions(self.symbol).get("price", 0)
        underlyingPrice = self.app.getPriceForSymbol(self.symbol)

        closingAction = "SELL"
        quantity = abs(amountOfShares)
        if amountOfShares < 0:
            closingAction = "BUY"

        if not costBasis:
            return "no avg price"
        
        if not underlyingPrice:
            return "no underlyng price"
        

        if amountOfShares > 0:
            # having shares, sell call
            currentDistance = underlyingPrice - costBasis 
        
        else:
            # short shares, sell put
            currentDistance = costBasis - underlyingPrice
        currentDistancePercent = currentDistance / underlyingPrice
        pnl = (underlyingPrice - costBasis) * amountOfShares
        doClosePosition = currentDistancePercent > requiredDistance
        
        if order:
            if self.orderApp.has_existing_order_contracts({"stock": dictContract}):
                return
            
            if (self.orderApp.hasOptionPositionsForSymbol(contract.symbol)):
                self.app.addToActionLog(f"Existing option positions found for {contract.symbol}, not placing order")
                return
            if not doClosePosition:
                self.app.addToActionLog(f"Required distance for {contract.symbol} not met, not placing stock order")
                return 
            self.orderApp.place_single_contract_order(dictContract, orderRef="PositionTakeProfit", action=closingAction, quantity=quantity, mid=underlyingPrice, doTransmit=True, outsideRTH=True)
        else:
            dictContract["mid"] = underlyingPrice
            dictContract["bid"] = underlyingPrice
            dictContract["ask"] = underlyingPrice
            dictContract["currentDistance"] = round(currentDistance,2)
            dictContract["cdP"] = round(currentDistancePercent*100,2)
            dictContract["reqDistance"] = requiredDistance
            dictContract["pnl"] = round(pnl)
            dictContract["closePosition"] =  doClosePosition
            
            dictContract["hint"] = hint
            return  {
                "long_stock":dictContract
            }

    def load_config(self):
        """Load job configuration from JSON file, filtered by symbol."""
        with open(self.config_file, 'r') as file:
            config = json.load(file)
        self.jobs = config.get("symbols", {}).get(self.symbol, [])

    def validate_kwargs(self, func, kwargs):
        """Ensure kwargs match the function's expected parameters."""
        if func:
            sig = inspect.signature(func)
            valid_params = set(sig.parameters.keys())

            # Check if all provided kwargs exist in function parameters
            for key in kwargs.keys():
                if key not in valid_params:
                    return False
            return True
        return False
    
    def get_jobs_dataframe(self):
        # check if a job is set for the title time
        if self.jobForTitleTime is not None:
            # check if a job for that id exists
            job = self.scheduler.get_job(self.jobForTitleTime)
            if job is not None:
                # determine how much time is left until the job is executed
                next_run_time: datetime = job.next_run_time
                if next_run_time is not None:
                    now = datetime.now(tz=next_run_time.tzinfo)  # Ensure 'now' matches the timezone
                    time_left = next_run_time - now
                    # format it in hours and minutes
                    hours, remainder = divmod(time_left.total_seconds(), 3600)

                    # set the title using os module
                    import os
                    # use HH:MM format
                    os.system(f'title {self.symbol} - {int(hours)}:{int(remainder // 60):02} left')
                

                

        return super().get_jobs_dataframe()
    
    def isWheelThresoldTimePassed(self, minutes=2):
        minutesInSeconds = 60 * minutes
        if self.firstWheelAttemptTime is None:
            self.firstWheelAttemptTime = datetime.now()
        return (datetime.now() - self.firstWheelAttemptTime).total_seconds() > minutesInSeconds
    
    def checkOptionPositionProfit(self, order=True, job_id=None):
      try:    
        positions = self.orderApp.getOptionPositionsForSymbol(self.symbol)
        contractPositions = []
        
        for pos in positions:
            shares = pos["position"]
            if shares == 0:
                continue

            contract: Contract = pos["contract"]
            multiplier = float(contract.multiplier)
            costBasis = float(pos.get("avgCost")) / multiplier 
            
            if costBasis < self.minPrice * 2 and len(positions) >= 1:
                
                #if order:
                #    jobs = self.scheduler.get_jobs()
                #    if job_id in jobs:
                #        self.scheduler.remove_job(job_id=job_id)
                self.shutdown()
                return "Skip take option profit. costbase too low"
            
            marketData = self.app.find_single_contract_by_id(contract.conId)
            bid = None
            ask = None
            pnl = None
            dte = self.calculate_dte(contract.lastTradeDateOrContractMonth)
            close = False
            should_close = False
            if marketData is not None:

                bid = marketData.get("bid")
                ask = marketData.get("ask")
                try:
                    if order:
                        self.app.request_options_market_data(marketData)
                except Exception as e:
                    self.app.addToActionLog(str(e))
            
                # --- Directional Logic Block ---
                orderAction = None
                if shares > 0:
                    if bid > 0:
                        # LONG POSITION logic
                        # We exit at the Bid (what buyers are offering)
                        pnl = (bid - costBasis) * multiplier * shares if bid else None
                        close = (bid >= costBasis * 1.5) if bid else False
                        orderAction = "SELL"

                else:
                    if ask > 0:
                        # SHORT POSITION logic
                        # We exit at the Ask (what sellers are demanding)
                        # We use abs(shares) to keep the PnL positive when in profit
                        orderAction = "BUY"
                        pnl = (costBasis - ask) * multiplier * float(abs(shares)) if ask else None
                        close = (ask <= costBasis * 0.5 and costBasis >= self.minPrice * 2) if ask else False
                # -------------------------------

                # Final safety: Only close if there's time left and we found a price
                should_close = close and dte > 0 and len(positions) == 1
                if should_close and order:
                    if (not self.orderApp.has_existing_order_contracts({"op": marketData})):
                        self.orderApp.place_single_contract_order(marketData, orderAction, abs(shares), orderRef="wheelOptionTakeProfit", doTransmit=True)
                        if not self.optionPositionCloseDataRequested:
                            self.optionPositionCloseDataRequested = True
                            send_telegram_message(f"Requesting more options data for existing position")
                            self.goForNextDayOptionsData(daysToAdd=7, reset=True)
            else:
                try:
                    if self.positionContractsRequestTime is None or datetime.now() - self.positionContractsRequestTime  > timedelta(minutes=2):
                        self.positionContractsRequestTime = datetime.now()
                        req = self.app.nextReqId()                
                        self.app.reqContractDetails(req, contract)

                except Exception as e:
                    self.app.addToActionLog(str(e))
                
            dictContract = {  
                "job": job_id,              
                "Symbol": contract.symbol,
                "Contracts": shares,
                "Type": contract.right,
                "DTE": dte,
                "Strike": round(contract.strike, 2),  
                "pnl": pnl, 
                "costBasis": costBasis,
                "bid": bid,
                "ask": ask,
                "close": should_close
            }
            
            contractPositions.append(dictContract)
        df = pd.DataFrame(contractPositions)
        if df.empty:
            if order:
                self.app.reqPositions()
            return df
        else:
            return df.sort_values("Strike", ascending=False)
      except Exception as e:
          self.app.addToActionLog(e)
 
    
