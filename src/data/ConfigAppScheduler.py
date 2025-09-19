import copy
from functools import partial
import math
from time import sleep
from data.OrbResult import OrbResult
from data.BaseAppScheduler import BaseAppScheduler
from ib.TwsOrderAdapter import TwsOrderAdapter
from ib.IBApp import IBApp
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.combining import OrTrigger
import json
import inspect
from util.Logger import Logger
from datetime import date, datetime

from util.TelegramMessenger import send_telegram_message


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

        Logger.log("Scheduler started")
     
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
                key_order = ["long_call", "short_call", "short_put", "long_put"]

                # Ensure all expected keys are present, assigning None if missing
                ordered_legs = {key: legs.get(key, None) for key in key_order}

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

                if (shortCallStrike < shortPutStrike):
                    callDistance = longCallStrike - shortCallStrike
                    otmDistance = shortCallStrike - undPrice

                    desiredPutStrike = undPrice - otmDistance

                    putLegs = self.app.find_credit_spread_legs(self.symbol, desiredPutStrike, "P", callDistance)
            except Exception as e:
                self.app.addToActionLog(str(e))
                Logger.log(str(e))


                    

            
            

        if callLegs is None or putLegs is None:
            self.app.addToActionLog("No valid legs found for Delta Iron Condor trade")
            return

        icLegs = {**callLegs, **putLegs}

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
        targetStrikeByPosition = stock_position.get("avgCost", 0)
        underlyingPrice = self.app.checkUnderlyingOptions(self.symbol).get("price", 0)
        targetPrice = None
        targetType = None

        if underlyingPrice is None or targetStrikeByPosition is None:
            return

        if amountOfShares <= 0: # short position, consider selling puts
            # check for existung put positions
            
            if self.orderApp.hasPositionsForSymbol(self.symbol, putType) and order:
                return
            
            targetPrice = min(targetStrikeByPosition, underlyingPrice - distance)
            targetPrice = math.floor(targetPrice)
            targetType = putType
            

        else: # long position, consider selling calls
            if self.orderApp.hasPositionsForSymbol(self.symbol, callType) and order:
                return
            
            targetPrice = max(targetStrikeByPosition, underlyingPrice + distance)
            targetPrice = math.ceil(targetPrice)
            targetType = callType

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
            
            self.app.addToActionLog(f"Placing Wheel Trade order for {self.symbol} at {targetPrice} with type {targetType}")
            if self.orderApp.has_existing_order_contracts({f"short_{'put' if targetType == putType else 'call'}": optionContract}):
                self.app.addToActionLog(f"Existing order found for {self.symbol} {targetType} at {targetPrice}, not placing duplicate order")
                return
            self.orderApp.place_single_contract_order(optionContract, orderRef="WheelTrade", action="SELL", quantity=amountOfContracts)
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
        
    def goForNextDayOptionsData(self):
        if self.hasWarmedUp():
            self.app.addToActionLog("Fetching next day options data")
            self.app.cancel_all_options_market_data()
            self.app.reset_options_data()
            self.reset_start_time()

            self.app.fetch_next_day_options_data()

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
        return (datetime.now() - self.firstWheelAttemptTime).total_seconds() > minutesInSeconds
    
