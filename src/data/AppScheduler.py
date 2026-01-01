from time import sleep
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from ib.IBApp import IBApp
from ib.TwsOrderAdapter import TwsOrderAdapter
from data.CandlePredictor import CandlePredictor

import pytz
from datetime import datetime, timedelta
from util.StrategyEnum import StrategyEnum
from util.time import inTzTime
import pandas as pd

import logging

# Get the APScheduler logger
# logging.getLogger('apscheduler').setLevel(logging.DEBUG)


class AppScheduler:
    def __init__(self, app: IBApp, orderApp: TwsOrderAdapter, historyApp: IBApp, symbol = "SPY", secondSymbol = "QQQ"):    
        self.app = app
        self.orderApp = orderApp
        self.historyApp = historyApp
        self.scheduler = BackgroundScheduler()        
        self.all_predictions_df = None
        self.ai_predictions = None

        self.symbol = symbol
        self.secondarySymbol = secondSymbol

        # Define New York timezone for the triggers.
        self.ny_tz = pytz.timezone("America/New_York")

        self.callDistance, self.putDistance, self.wing_span, self.target_premium, self.ic_wingspan, self.tp_percentage, self.sl_percentag, self.call_delta, self.put_delta = app.get_offset_configs(symbol)

        self.warmUpThreshold = timedelta(minutes=1)
        self.startTime = datetime.now()
        self.historySymbols = None
        self.minPrice = 0.2

        df = pd.DataFrame()
        if self.symbol in self.app.candleData:
            df = self.app.candleData[self.symbol]
        self.predictor: CandlePredictor = CandlePredictor(df)
        


    def hasWarmedUp(self):
        """
        Check if the scheduler has warmed up by comparing the current time with the start time.
        """
        warmUpTime = datetime.now() - self.startTime
        return warmUpTime >= self.warmUpThreshold
    

    def get_jobs_dataframe(self):
        """
        Returns a DataFrame containing all current jobs with their next runtime formatted as 'Day HH:MM:SS'.
        Handles edge cases where next_run_time is None.
        """
        jobs = self.scheduler.get_jobs()
        data = []
        
        for job in jobs:
            if job.next_run_time:
                day = job.next_run_time.strftime("%a")  # Short day name (e.g., Mon, Tue, Wed)
                time = job.next_run_time.strftime("%H:%M:%S")  # Time formatted as HH:MM:SS
                next_run = f"{day} {time}"
            else:
                next_run = "Not Scheduled"
            
            data.append({"Job ID": job.id, "Time": next_run})

        df = pd.DataFrame(data)
        
        # Ensure sorting works even when "Not Scheduled" is present by separating valid times
        df_valid = df[df["Time"] != "Not Scheduled"].sort_values(by="Time", ascending=True)
        df_invalid = df[df["Time"] == "Not Scheduled"]
        
        return pd.concat([df_valid, df_invalid], ignore_index=True)

    def run(self):
        """
        Start the scheduler and add jobs for checkDeltaTrade and checkMultiEntryIconCondor.
        """
        historyCatchUpTrigger = CronTrigger(minute="*/2", second="20", hour="9-16", timezone=self.ny_tz)
        self.scheduler.add_job(func=self.catch_up_history, trigger=historyCatchUpTrigger, id="catch_up_history")
        # Schedule checkDeltaTrade every minute between 9:45 and 15:30 New York time.
        delta_first_run = inTzTime(9,45)
        delta_last_run = inTzTime(15,15)
        delta_trade_trigger = CronTrigger(minute="*", hour="9-15", second="10", timezone=self.ny_tz, start_date=delta_first_run, end_date=delta_last_run)
        #self.scheduler.add_job(func=self.checkDeltaTrade, trigger=delta_trade_trigger, id="checkDeltaTrade")
        
        catch_up_trigger = CronTrigger(second=30, minute="*", hour="9-16", timezone=self.ny_tz, start_date=delta_first_run)
        self.scheduler.add_job(func=self.catch_up, trigger=catch_up_trigger, id="catch_up")

        # Schedule checkMultiEntryIconCondor every 30 minutes between 12:00 and 15:00 New York time.
        multi_first_run = inTzTime(12,0)
        multi_last_run = inTzTime(15,0)
        multi_entry_trigger = CronTrigger(minute="0,30", hour="12-14", timezone=self.ny_tz, start_date=multi_first_run, end_date=multi_last_run)
        #self.scheduler.add_job(func=self.checkMultiEntryIconCondor, trigger=multi_entry_trigger, id="checkMultiEntryIconCondor")

        lateIcTrigger = CronTrigger(hour="15", minute="55", timezone=self.ny_tz)
        #self.scheduler.add_job(self.checkLateIronCondor, trigger=lateIcTrigger, id="checkLateIronCondor")

        #secondaryTrigger = CronTrigger(hour="15", minute="0", second=14, day_of_week='tue,wed,thu,fri', timezone=self.ny_tz)
        #self.scheduler.add_job(self.checkSecondaryDeltaTrade, trigger=secondaryTrigger, id="checkSecondaryDeltaTrade", misfire_grace_time=15)

        taTradeTrigger = CronTrigger(minute="*", hour="9-15", second="*/20", timezone=self.ny_tz, start_date=delta_first_run, end_date=delta_last_run)
        self.scheduler.add_job(self.checkTechnicalEntry, trigger=taTradeTrigger, id="checkTechnicalEntry", misfire_grace_time=15)

        cooldownTrigger = CronTrigger(second="34", minute="*/5", hour="9-15", timezone=self.ny_tz, start_date=delta_first_run, end_date=delta_last_run)
        self.scheduler.add_job(self.cooldown, trigger=cooldownTrigger, id="cooldown", misfire_grace_time=15)

        predictionsTrigger = IntervalTrigger(seconds=30)
        self.scheduler.add_job(self.predict, trigger=predictionsTrigger, id="predict", misfire_grace_time=15)

        twoPMTrigger = CronTrigger(hour="14", minute="0", second="37", timezone=self.ny_tz)
        self.scheduler.add_job(self.checkTwoPmDeltaTrade, trigger=twoPMTrigger, id="checkDeltaTrade_2PM", misfire_grace_time=15)

        # Start the scheduler.
        self.scheduler.start()

    def catch_up(self):
        self.app.addToActionLog("CATCHING UP")
        self.orderApp.enhance_order_contracts_from_dataframe(self.app.options_data)
        #self.orderApp.reqAutoOpenOrders()
        sleep(0.5)
        self.orderApp.reqAllOpenOrders()
        sleep(0.5)

    def predict(self):
        """
        Predict the next candle using the CandlePredicter and log the result.
        """
        
        if self.hasWarmedUp():
            df = pd.DataFrame()
            if self.symbol in self.app.candleData:
                df = self.app.candleData[self.symbol]
            else:
                return
            try:
                self.predictor.update_dataframe(df, "replace")
                if (self.predictor.ai_model is None or 
                    not hasattr(self.predictor.ai_model, "estimators_")) and len(df) > 0:
                    self.app.addToActionLog("Train")

                    self.predictor.train_ai_model()
                else:
                    self.app.addToActionLog("Predicting")
                    self.ai_predictions = self.predictor.predict_day_extremes_ai()
                    self.all_predictions_df = self.predictor.predict_all_candles_ai()
                    self.predictor.evaluate_and_adjust_model()
                    self.predictor.save_model()

            except Exception as e:
                self.app.addToActionLog("Error in predict: " + str(e))
        elif (self.predictor.ai_model is None):
            try:
                self.app.addToActionLog("Loading model")
                self.predictor.load_model()
            except Exception as e:
                self.app.addToActionLog("Error in load_model: " + str(e))



    def printPredictions(self):
        return self.ai_predictions 
    
    def printEnhancedPredictions(self):
        # latest first, only chosen columns
        try:
            df = None
            if self.all_predictions_df is not None:

                df = self.all_predictions_df[["datetime", 
                                            "open", 
                                            "high", 
                                            "low", 
                                            "close", 
                                            "remaining_day_high", 
                                            "remaining_day_low"
                                            ]]
                df = df.sort_values(by="datetime", ascending=False)
                df = df.reset_index(drop=True)
            return df
        except Exception as e:
            self.app.addToActionLog("Error in printEnhancedPredictions: " + str(e))
            return None

    def checkLateIronCondor(self):
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

        self.app.addToActionLog("Check Late IC "+ self.symbol)
        legs = self.app.construct_from_underlying(self.symbol, self.ic_wingspan, self.ic_wingspan)
        legsPrice = TwsOrderAdapter.calcSpreadPrice(legs)

        if self.hasWarmedUp() and not self.orderApp.has_existing_order_contracts(legs) and abs(legsPrice) > 0.2: 
            self.app.addToActionLog("Ordering 2155 Iron Condor")
            self.orderApp.place_combo_order(legs, None, None, "IronCondor_2155", getOutOfMarket=False)
            sleep(0.5)

    def checkTechnicalEntry(self, symbol="SPY"):
        try:
            if  self.hasWarmedUp():
                self.app.addToActionLog("CHECKING TECHNICAL ENTRY")
                technicalAnalysis = self.historyApp.checkUnderlyingOptions(symbol)
                signal = technicalAnalysis.get("signal", StrategyEnum.Hold)
                if signal == StrategyEnum.Hold:
                    return
                
                callStrike = technicalAnalysis.get("call")
                putStrike = technicalAnalysis.get("put")
                minPrice = self.wing_span * 0.05
                
                # callLegs = self.app.find_credit_spread_legs(symbol, callStrike, "C", self.callDistance)
                # putLegs = self.app.find_credit_spread_legs(symbol, putStrike, "P", self.putDistance)
                callLegs = self.app.build_credit_spread(self.symbol, self.call_delta, "C", self.wing_span, minPrice)
                putLegs = self.app.build_credit_spread(self.symbol, self.put_delta, "P", self.wing_span, minPrice)


                callPrice = TwsOrderAdapter.calcSpreadPrice(callLegs)
                putPrice = TwsOrderAdapter.calcSpreadPrice(putLegs)

                if signal == StrategyEnum.SellCall and  self.orderApp.is_room_for_new_positions(self.symbol, "C") and abs(callPrice) >= self.minPrice:
                    self.app.addToActionLog("Ordering TA Call Credit Spread")
                    self.orderApp.place_combo_order(callLegs, self.tp_percentage, self.sl_percentag, "TACallCredit")
                    sleep(0.5)
                elif signal == StrategyEnum.SellPut and  self.orderApp.is_room_for_new_positions(self.symbol, "P") and abs(putPrice) >= self.minPrice:    
                    self.app.addToActionLog("Ordering TA Put Credit Spread")
                    self.orderApp.place_combo_order(putLegs, self.tp_percentage, self.sl_percentag, "TAPutCredit")
                    sleep(0.5)
        except Exception as e:
            self.app.addToActionLog("Error in checkTechnicalEntry: " + str(e))

            



    def checkMultiEntryIconCondor(self):
        self.app.addToActionLog("CHECKING MEIC")
        if self.hasWarmedUp():
            calls = self.app.build_credit_spread_by_premium(self.symbol, 0.8, "C", self.wing_span * 2)
            puts = self.app.build_credit_spread_by_premium(self.symbol, 0.8, "P", self.wing_span * 2)

            hasCallLegs = self.orderApp.has_existing_order_contracts(calls)
            hasPutLegs = self.orderApp.has_existing_order_contracts(puts)

            if not hasCallLegs:
                self.app.addToActionLog("Ordering MEIC Call")
                self.orderApp.place_combo_order(calls, self.tp_percentage, 200, "MEIC-CALL")
                sleep(0.5)

            if not hasPutLegs:
                self.app.addToActionLog("Ordering MEIC Put")
                self.orderApp.place_combo_order(puts, self.tp_percentage, 200, "MEIC-PUT")
                sleep(0.5)

    def checkDeltaTrade(self):
        self.app.addToActionLog("CHECKING DELTA TRADE")
        currentCandle = self.app.checkUnderlyingOptions(self.symbol)
        minPrice = self.minPrice
        BEARISH = "bearish"
        BULLISH = "bullish"
        if self.hasWarmedUp():
            if self.orderApp.is_room_for_new_positions(self.symbol, "C") and currentCandle["sentiment"] == BEARISH and currentCandle["prevSentiment"] == BEARISH:
                try:
                    self.app.addToActionLog("Room for Delta Call Trade, adding Orders")

                    callLegs = self.app.build_credit_spread(self.symbol, 0.16, "C", self.wing_span, minPrice)
                    limit_price = TwsOrderAdapter.calcSpreadPrice(callLegs)
                    if limit_price is not None and abs(limit_price) >= minPrice:
                        self.orderApp.place_combo_order(callLegs, self.tp_percentage, 200, "SchedCallCredit")
                        sleep(0.5)
                except Exception as e:
                    self.app.addToActionLog(e)

            if self.orderApp.is_room_for_new_positions(self.symbol, "P") and currentCandle["sentiment"] == BULLISH and currentCandle["prevSentiment"] == BULLISH:
                self.app.addToActionLog("Room for Delta Put Trade, adding Orders")
                putLegs = self.app.build_credit_spread(self.symbol, 0.16, "P", self.wing_span, minPrice)
                limit_price = TwsOrderAdapter.calcSpreadPrice(putLegs)
                
                if limit_price is not None and abs(limit_price) >= minPrice:
                    self.orderApp.place_combo_order(putLegs, self.tp_percentage, 200, "SchedPutCredit")
                    sleep(0.5)

    def checkDeltaTradeForSymbol(self, symbol, delta=0.25, wing_span=1, tp_percentage=None, sl_percentage=None, touch=None, minPrice=0.2):
        
        
        
        self.app.addToActionLog("CHECKING DELTA TRADE "+ symbol)        
        if self.hasWarmedUp():
            if self.orderApp.is_room_for_new_positions(symbol):
                try:
                    self.app.addToActionLog("Room for Delta Call Trade, adding Orders")

                    callLegs = self.app.build_credit_spread(symbol, delta, "C", wing_span)
                    putLegs = self.app.build_credit_spread(symbol, delta, "P", wing_span)

                    ironCondorLegs = {**callLegs, **putLegs}


                    limit_price = TwsOrderAdapter.calcSpreadPrice(ironCondorLegs)
                    if abs(limit_price) >= minPrice * 2:
                        self.orderApp.place_combo_order(ironCondorLegs, tp_percentage, sl_percentage, "SchedIronCondor", touch)
                        sleep(0.5)
                except Exception as e:
                    self.app.addToActionLog(e)

            if self.orderApp.is_room_for_new_positions(symbol, "P"):
                self.app.addToActionLog("Room for Delta Put Trade, adding Orders")
                limit_price = TwsOrderAdapter.calcSpreadPrice(putLegs)
                
                if abs(limit_price) >= minPrice:
                    self.orderApp.place_combo_order(putLegs, tp_percentage, sl_percentage, "SchedPutCredit", touch)
                    sleep(0.5)

    def checkTwoPmDeltaTrade(self):
        self.app.addToActionLog("CHECKING DELTA TRADE "+ self.symbol)
        symbol = self.symbol
        delta = 0.35
        wing_span = 5
        minPrice = 0.2
        tp_percentage = None
        sl_percentage = 200

        if self.hasWarmedUp():

            try:
                self.app.addToActionLog("Room for Delta Call Trade, adding Orders")

                callLegs = self.app.build_credit_spread(symbol, delta, "C", wing_span)
                putLegs = self.app.build_credit_spread(symbol, delta, "P", wing_span)

                ironCondorLegs = {**callLegs, **putLegs}


                limit_price = TwsOrderAdapter.calcSpreadPrice(ironCondorLegs)
                if not TwsOrderAdapter.has_existing_order_contracts(ironCondorLegs):
                    if abs(limit_price) >= minPrice * 2:
                        self.orderApp.place_combo_order(ironCondorLegs, tp_percentage, sl_percentage, "SchedIronCondor")
                        sleep(0.5)
            except Exception as e:
                self.app.addToActionLog(e)
            

    def checkSecondaryDeltaTrade(self):
        self.app.addToActionLog("CHECKING DELTA TRADE "+ self.secondarySymbol)
        if self.orderApp.is_room_for_new_positions(self.secondarySymbol) and self.hasWarmedUp(): 
            vixPrice = self.app.getPriceForSymbol("VIX")
            self.app.addToActionLog("Ordering 2100 Iron Condor")
            self.checkDeltaTradeForSymbol(self.secondarySymbol)

    def catch_up_history(self):
        if self.historySymbols:
            self.historyApp.reqHistoricalDataFor(*self.historySymbols, True)

    def cooldown(self):
        self.app.cancel_all_options_market_data()
        self.app.request_market_data()