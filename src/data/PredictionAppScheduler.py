import ctypes
import os
import sys
from time import sleep
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_ERROR, JobExecutionEvent
from ib.IBApp import IBApp
from ib.TwsOrderAdapter import TwsOrderAdapter
from data.CandlePredictor import CandlePredictor
from data.BaseAppScheduler import BaseAppScheduler

import pytz
from datetime import datetime, timedelta
from util.StrategyEnum import StrategyEnum
from util.time import inTzTime
import pandas as pd

import logging
from util.Logger import Logger

# Get the APScheduler logger
logging.getLogger('apscheduler').setLevel(logging.INFO)


class PredictionAppScheduler(BaseAppScheduler):
    def __init__(self, app: IBApp, orderApp: TwsOrderAdapter, historyApp: IBApp, symbol = "SPY", secondSymbol = "QQQ"):    
        super().__init__(app, orderApp, historyApp, symbol, secondSymbol)    
        self.all_predictions_df = None
        self.ai_predictions = None

        

        df = pd.DataFrame()
        if self.symbol in self.app.candleData:
            df = self.app.candleData[self.symbol]
        self.predictor: CandlePredictor = CandlePredictor(df)
   

    def run(self):
        """
        Start the scheduler and add jobs for checkDeltaTrade and checkMultiEntryIconCondor.
        """
        predictionsTrigger = IntervalTrigger(seconds=30)
        self.scheduler.add_job(self.predict, trigger=predictionsTrigger, id="predict", misfire_grace_time=15)
        return super().run()


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

