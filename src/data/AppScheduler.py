from time import sleep
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from ib.IBApp import IBApp
from ib.TwsOrderAdapter import TwsOrderAdapter

import pytz
from datetime import datetime, timedelta
from util.time import inTzTime

import logging

# Get the APScheduler logger
logging.getLogger('apscheduler').setLevel(logging.DEBUG)

class AppScheduler:
    def __init__(self, app: IBApp, orderApp: TwsOrderAdapter, symbol = "SPY"):
        self.app = app
        self.orderApp = orderApp
        self.scheduler = BackgroundScheduler()
        self.symbol = symbol

        # Define New York timezone for the triggers.
        self.ny_tz = pytz.timezone("America/New_York")

        self.callDistance, self.putDistance, self.wing_span, self.target_premium, self.ic_wingspan, self.tp_percentage, self.sl_percentag = app.get_offset_configs(symbol)

        self.warmUpThreshold = timedelta(minutes=2)
        self.startTime = datetime.now()

    def hasWarmedUp(self):
        """
        Check if the scheduler has warmed up by comparing the current time with the start time.
        """
        warmUpTime = datetime.now() - self.startTime
        self.app.addToActionLog(f"WarmUp Time: {warmUpTime}")
        return warmUpTime >= self.warmUpThreshold

    def run(self):
        """
        Start the scheduler and add jobs for checkDeltaTrade and checkMultiEntryIconCondor.
        """
        # Schedule checkDeltaTrade every minute between 9:45 and 15:30 New York time.
        delta_first_run = inTzTime(9,45)
        delta_last_run = inTzTime(15,0)
        delta_trade_trigger = CronTrigger(minute="*", hour="9-15", timezone=self.ny_tz, start_date=delta_first_run, end_date=delta_last_run)
        self.scheduler.add_job(func=self.checkDeltaTrade, trigger=delta_trade_trigger, id="checkDeltaTrade")

        # Schedule checkMultiEntryIconCondor every 30 minutes between 12:00 and 15:00 New York time.
        multi_first_run = inTzTime(12,0)
        multi_last_run = inTzTime(15,0)
        multi_entry_trigger = CronTrigger(minute="0,30", hour="12-14", timezone=self.ny_tz, start_date=multi_first_run, end_date=multi_last_run)
        self.scheduler.add_job(func=self.checkMultiEntryIconCondor, trigger=multi_entry_trigger, id="checkMultiEntryIconCondor")

        lateIcTrigger = CronTrigger(hour=15, minute=55, timezone=self.ny_tz)
        self.scheduler.add_job(self.checkLateIronCondor, trigger=lateIcTrigger, id="checkLateIronCondor")


        # Start the scheduler.
        self.scheduler.start()

    def checkLateIronCondor(self):
        self.app.addToActionLog("Check Late IC")
        if self.orderApp.is_room_for_new_positions(self.symbol) and self.hasWarmedUp(): 
            self.app.addToActionLog("Ordering SPX Iron Condor")
            legs = self.app.construct_from_underlying(self.symbol, self.ic_wingspan, self.ic_wingspan)
            self.orderApp.place_combo_order(legs, None, None, "IronCondor_2155")
        

    def checkMultiEntryIconCondor(self):
        self.app.addToActionLog("CHECKING MEIC")
        if self.orderApp.is_room_for_new_positions(self.symbol) and self.hasWarmedUp():
            self.app.addToActionLog("Open to trade MEIC, adding Orders")

    def checkDeltaTrade(self):
        self.app.addToActionLog("CHECKING DELTA TRADE")
        if self.hasWarmedUp():
            if self.orderApp.is_room_for_new_positions(self.symbol, "C"):
                try:
                    self.app.addToActionLog("Room for Delta Call Trade, adding Orders")

                    callLegs = self.app.build_credit_spread(self.symbol, 0.15, "C", self.wing_span)
                    self.orderApp.place_combo_order(callLegs, self.tp_percentage, None, "SchedCallCredit")

                    sleep(0.5)
                except Exception as e:
                    self.app.addToActionLog(e)

            if self.orderApp.is_room_for_new_positions(self.symbol, "P"):
                self.app.addToActionLog("Room for Delta Put Trade, adding Orders")
                putLegs = self.app.build_credit_spread(self.symbol, 0.15, "P", self.wing_span)
                self.orderApp.place_combo_order(putLegs, self.tp_percentage, None, "SchedPutCredit")
                sleep(0.5)

    
