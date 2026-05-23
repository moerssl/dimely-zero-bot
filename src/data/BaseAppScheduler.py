import ctypes
import os
from time import sleep
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.events import EVENT_JOB_ERROR, JobExecutionEvent
from ib.IBApp import IBApp
from ib.TwsOrderAdapter import TwsOrderAdapter
from ibapi.contract import Contract

import pytz
from datetime import datetime, timedelta, date
from util.time import inTzTime
import pandas as pd

import logging
from util.Logger import Logger

from util.TelegramMessenger import send_telegram_message

# Get the APScheduler logger
logging.getLogger('apscheduler').setLevel(logging.INFO)


class BaseAppScheduler:
    def __init__(self, app: IBApp, orderApp: TwsOrderAdapter, historyApp: IBApp, symbol = "SPY", secondSymbol = "QQQ"):    
        self.app = app
        self.orderApp = orderApp
        self.historyApp = historyApp
        self.scheduler = BackgroundScheduler()        


        self.symbol = symbol
        self.secondarySymbol = secondSymbol

        # Define New York timezone for the triggers.
        self.ny_tz = pytz.timezone("America/New_York")

        self.callDistance, self.putDistance, self.wing_span, self.target_premium, self.ic_wingspan, self.tp_percentage, self.sl_percentag, self.call_delta, self.put_delta = app.get_offset_configs(symbol)

        self.warmUpThreshold = timedelta(seconds=150)
        self.startTime = datetime.now()
        self.historySymbols = None
        self.minPrice = 0.2
        
        self.positionContractsRequestTime = None
        self.optionPositionCloseDataRequested = False
        
        self.scheduler.add_listener(self.handleError, EVENT_JOB_ERROR)
        self.itmOptions = set()

    def reset_start_time(self):
        self.startTime = datetime.now()

    def hasWarmedUp(self, multiplier=1):
        """
        Check if the scheduler has warmed up by comparing the current time with the start time.
        """
        warmUpTime = datetime.now() - self.startTime
        pricesReceived = self.app.has_prices_received()
        warmUpTimeReached = warmUpTime >= self.warmUpThreshold * multiplier
        return pricesReceived or warmUpTimeReached

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

        # onyl continue if we have jobs
        if not data:
            return pd.DataFrame(columns=["Job ID", "Time"])

        df = pd.DataFrame(data)
        
        # Ensure sorting works even when "Not Scheduled" is present by separating valid times
        df_valid = df[df["Time"] != "Not Scheduled"].sort_values(by="Time", ascending=True)
        df_invalid = df[df["Time"] == "Not Scheduled"]
        
        return pd.concat([df_valid, df_invalid], ignore_index=True)
    
    def checkShutdownJob(self):
        """
        Checks if there is a job already scheduled in future day that uses the shutdown function. 
        This indicated, that the shutdown time for today has already passed.
        If so, shut down the app.
        """
        jobs = self.scheduler.get_jobs()
        for job in jobs:
            if job.func == self.shutdown and job.next_run_time is not None:
                # check if next run is in the future and not today
                if job.next_run_time > datetime.now(self.ny_tz):
                    nextRunTime = job.next_run_time.date()
                    nextDay = datetime.now(self.ny_tz).date()
                    if job.next_run_time.date() > datetime.now(self.ny_tz).date():
                        self.app.addToActionLog("Shutdown job already scheduled for future day, shutting down now.")
                        self.shutdown()
                        return True

    def run(self):
        """
        Start the scheduler and add jobs for checkDeltaTrade and checkMultiEntryIconCondor.
        """
        delta_first_run = inTzTime(9,45)
        delta_last_run = inTzTime(15,15)
        
        catch_up_trigger = CronTrigger(second=30, minute="*", hour="9-16", timezone=self.ny_tz, start_date=delta_first_run)
        self.scheduler.add_job(func=self.catch_up, trigger=catch_up_trigger, id="catch_up")

        # cooldownTrigger = CronTrigger(second="34", minute="*/5", hour="9-15", timezone=self.ny_tz, start_date=delta_first_run, end_date=delta_last_run)
        start_time = datetime.now() + timedelta(seconds=30)
        cooldownTrigger = IntervalTrigger(minutes=2, start_date=start_time)
        
        self.scheduler.add_job(self.cooldown, trigger=cooldownTrigger, id="cooldown", misfire_grace_time=15)

        self.scheduler.start()

        self.checkShutdownJob()

    def catch_up(self):
        self.app.addToActionLog("CATCHING UP")
        self.orderApp.enhance_order_contracts_from_dataframe(self.app.options_data)
        #self.orderApp.reqAutoOpenOrders()
        sleep(0.5)
        self.orderApp.reqAllOpenOrders()
        sleep(0.5)

    def print_stock_positions(self):
        stock_position = self.orderApp.getStockPositionsForSymbol(self.symbol)
        if stock_position is not None:
            return stock_position
        return "No stock position found for " + self.symbol
    
    def calculate_dte(self, date_string):
        # Convert string 'YYYYMMDD' to a date object
        target_date = datetime.strptime(date_string, "%Y%m%d").date()
        
        # Get today's date
        today = date.today()
        
        # Subtract the dates to get a timedelta object
        delta = target_date - today
        
        # Return the number of days
        return delta.days   

        
    def print_short_option_itm_check(self):
        shortPositions = self.orderApp.getOptionPositionsForSymbol(self.symbol)
        underlyingPrice = self.app.getPriceForSymbol(self.symbol)
        contractPositions = []
        
        for pos in shortPositions:
            contract: Contract = pos["contract"]
            con_id = contract.conId
            
            dte = self.calculate_dte(contract.lastTradeDateOrContractMonth)
            dictContract = {                
                "Symbol": contract.symbol,
                "Contracts": pos["position"],
                "Type": contract.right,
                "DTE": dte,
                "Strike": round(contract.strike,2),                
                "Price": underlyingPrice,
            }
            
            if underlyingPrice:
                is_currently_itm = underlyingPrice < contract.strike if contract.right == "P" else underlyingPrice > contract.strike
                dictContract["ITM"] = is_currently_itm
                dictContract["room"] = round(underlyingPrice - contract.strike,2) if contract.right == "P" else round(contract.strike - underlyingPrice,2)

                option_desc = f"{contract.symbol} {contract.strike}{contract.right} (Exp: {contract.lastTradeDateOrContractMonth})"
                
                if is_currently_itm and con_id not in self.itmOptions:
                    # Transition: OTM -> ITM
                    self.itmOptions.add(con_id)
                    send_telegram_message(f"🚨 ALERT: {option_desc} moved IN-THE-MONEY (ITM).")
                    
                elif not is_currently_itm and con_id in self.itmOptions:
                    # Transition: ITM -> OTM
                    self.itmOptions.remove(con_id)
                    send_telegram_message(f"✅ RELIEF: {option_desc} is now OUT-OF-THE-MONEY (OTM).")
            
            contractPositions.append(dictContract)
        df = pd.DataFrame(contractPositions)
        if df.empty:
            return df
        else:
            return df.sort_values("Strike", ascending=False)
    
    
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

    def catch_up_history(self):
        if self.historySymbols:
            self.historyApp.reqHistoricalDataFor(*self.historySymbols, True)

    def cooldown(self):
        Logger.log("Cancelling option market data")
        self.app.addToActionLog("Cancelling option market data")
        try:
            self.app.cancel_all_options_market_data()
        except Exception as e:
            Logger.log(str(e))
        sleep(5)
        Logger.log("Request market data")
        self.app.addToActionLog("Request market data")
        self.app.request_market_data()
        self.app.request_all_options_data()

    def shutdown(self):
        pass 
        import inspect
        caller = inspect.stack()[1]
        module = inspect.getmodule(caller.frame)
        caller_name = caller.function
        filename = caller.filename
        lineno = caller.lineno

        info = f"Shutdown called from {caller_name} in {module.__name__} at {filename}:{lineno}"
        print(info)
        self.app.addToActionLog(info)
        

        self._close_console_window()
        os._exit(0)

    def _close_console_window(self):
        """Ask Windows to close this process’s console window."""
        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if hwnd:
            # WM_CLOSE == 0x0010
            ctypes.windll.user32.PostMessageW(hwnd, 0x0010, 0, 0)

    def handleError(self, event: JobExecutionEvent):
        try:
            if event.exception:
                job = self.scheduler.get_job(event.job_id)
                args = job.args if job else []
                kwargs = job.kwargs if job else {}
                job_name = job.name if job else event.job_id
                func = job.func if job else 'Unknown'

                # Build the error log message
                error_msg = (
                    f"❌ Error in job '{job_name}': {event.exception}\n"
                    f"🔧 Function: {func}\n"
                    f"📦 Args: {args}, Kwargs: {kwargs}\n"
                    f"📄 Traceback:\n{event.traceback}"
                )

                # Log to your app and your logger
                self.app.addToActionLog(error_msg)
                Logger.log(error_msg)
        except Exception as e:
            self.app.addToActionLog(f"⚠️ Error in error handler: {e}")

    def goForNextDayOptionsData(self, daysToAdd=1, reset=True):
        if self.hasWarmedUp():
            self.app.addToActionLog(f"Fetching next {daysToAdd} days options data")
            self.reset_start_time()
            
            target_date = datetime.today() + timedelta(days=daysToAdd)
            # 1. Reset erst NACHDEM neue ContractDetails angekommen sind
            self.app.contractDetailsPending = True
            
            if reset:
                self.app.cancel_all_options_market_data()
                try:
                    self.app.reset_options_data()
                except Exception as error:
                    self.app.addToActionLog("Error resetting optionbs data")
            
            
            try:
                self.app.fetch_next_days_options_data(daysToAdd)
            except Exception as error:
                self.app.addToActionLog("error fetchiung next day")
            self.daySteps += 1