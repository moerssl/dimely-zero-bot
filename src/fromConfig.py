import datetime
import subprocess
from util.Logger import Logger
import threading
import asyncio
from time import sleep
import pandas as pd
import curses
from wakepy import keep

# Import your IBApp and related modules.
from ib.IBApp import IBApp
from ib.TwsOrderAdapter import TwsOrderAdapter
from display.tiled import  display_data_tiled
from display.dash_app import start_dash_app


# Import the predictor class.
from data.ConfigAppScheduler import ConfigAppScheduler

import os
import sys
import psutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--symbol", default="qqq")
parser.add_argument("-t", "--time", default="8 D")
parser.add_argument("-c", "--candle-size", default="5 mins")

args = parser.parse_args()

symbolParam = args.symbol
timeParam = args.time
candleSizeParam = args.candle_size



import os

# Change the terminal title
os.system("echo -ne '\033]0;"+ symbolParam +"\007'")


portMap = {
    "IWM": 8051,
    "QQQ": 8052,
    "SPY": 8053,
    "SPX": 8054,
    "XSP": 8055
}




PID_FILE = symbolParam+'.pid'
Logger.set_log_prefix(symbolParam)

def is_already_running():
    """Check if another instance is running."""
    if os.path.exists(PID_FILE):
        with open(PID_FILE, "r") as f:
            pid = int(f.read().strip())
        if psutil.pid_exists(pid):  # Cross-platform method to check process existence
            return True
    return False

async def main():
    # Initialize IBApp and connect.
    app = IBApp()
    historyApp = IBApp(lookBackTime=timeParam, candleSize=candleSizeParam)
    orderApp = TwsOrderAdapter()
    focusSymbol = (symbolParam, "STK", "SMART")
    if symbolParam.lower() == "spx" or symbolParam.lower() == "xsp":
        focusSymbol = (symbolParam.upper(), "IND", "CBOE")
    
    thirdSymbol = ("SPY", "STK", "SMART")
    vix = ("VIX", "IND", "CBOE")

    focusContractSymbol, _, _  = focusSymbol
    

    scheduler = ConfigAppScheduler(app, orderApp, historyApp,symbol=focusContractSymbol)
    orderApp.actionLog = app.actionLog
    startClientId = stock_symbol_to_client_id(symbolParam, max_id=9999)
    def next_client_id():
        nonlocal startClientId
        client_id = startClientId
        startClientId += 1
        return client_id
    
    def req():
        stock_position = orderApp.getStockPositionsForSymbol(focusContractSymbol)
        app.setMarketDataType()

        if stock_position:
            targetStrikeByPosition = stock_position.get("avgCost", None)
            if targetStrikeByPosition:
                app.optionMarketDataRequestTargetStrike = targetStrikeByPosition
                
        today = datetime.date.today()

        # Friday is index 4 (Monday = 0, ..., Sunday = 6)
        days_ahead = (4 - today.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7  # Moves to next week if today is Friday

        next_friday = today + datetime.timedelta(days=days_ahead)

        
        app.request_market_data(symbols)
        if (symbolParam.lower() == "spx"):
            app.fetch_options_data(optionSymbols, next_friday, ["P"])
        else:
            app.fetch_options_data(optionSymbols)
        app.request_all_options_data()
        
    orderApp.onPositionEnd(req)
    

    historyCandles = {}
    scheduler.historySymbols = focusSymbol
    app.focusContract = focusSymbol
    historyApp.focusContract = focusSymbol

    app.candleData = historyCandles
    historyApp.candleData = historyCandles

    app.additionalTilesFuncs.append({
        "content": f"http://localhost:{portMap.get(symbolParam.upper(), 8050)}",
        "title": "Port for web app", 
        "colspan": 1
        })
    app.additionalTilesFuncs.append(scheduler.get_jobs_dataframe)
    app.additionalTilesFuncs.append(orderApp.minutesPassedSinceLastPositionClosed)
    app.additionalTilesFuncs.append(scheduler.print_stock_positions)
    app.additionalTilesFuncs.append({
        "function": scheduler.print_short_option_itm_check,
        "colspan": 2,
        "title": "Options"
        })
    app.additionalTilesFuncs.append({
        "content": (lambda d: d.iloc[(d["delta"] - 50).abs().argsort()] if "delta" in d.columns else d)(scheduler.app.options_data),        
        "colspan": 2,
        "title": "Options II"
        })
    print("Connecting to TWS/IB Gateway...")
    twsPort = 7497
    app.connect("127.0.0.1", twsPort, clientId=next_client_id())  # Use your connection parameters.
    orderApp.connect("127.0.0.1", twsPort, clientId=next_client_id())  # Use your connection parameters.
    historyApp.connect("127.0.0.1", twsPort, clientId=next_client_id())  # Use your connection parameters.
    
    def run_loop(client):
        client.run()
    api_thread = threading.Thread(target=run_loop, daemon=True, args=(app,))
    api_thread.start()
    sleep(0.1)
    order_thread = threading.Thread(target=run_loop, daemon=True, args=(orderApp,))
    order_thread.start()
    sleep(0.1)
    history_thread = threading.Thread(target=run_loop, daemon=True, args=(historyApp,))
    history_thread.start()
    
    dash_thread = threading.Thread(target=start_dash_app, args=(historyApp,orderApp,app,portMap.get(symbolParam.upper(), 8050)), daemon=True)
    dash_thread.start()

    scheduler.run()

    def start_curses_thread(app: IBApp, orderApp: TwsOrderAdapter):
        curses.wrapper(display_data_tiled, app, orderApp, historyApp, focusSymbol)
        app.disconnect()

    curses_thread = threading.Thread(target=start_curses_thread, args=(app,orderApp,), daemon=True)
    curses_thread.start()

    #app.setMarketDataType()

    #app.reqMarketDataType(4)

    print("Fetching positions and orders...")
    app.fetch_positions()
    app.reqAllOpenOrders()
    orderApp.reqPositions()
    orderApp.reqAllOpenOrders()
    app.additionalTilesFuncs.append({
        "function": lambda: orderApp.is_room_for_new_positions(symbolParam),
        "title": "Room for New Positions " + symbolParam,
    })
    app.additionalTilesFuncs.append({
        "function": lambda: orderApp.get_all_orders(),
        "title": "Orders " + symbolParam,
    })
    tomorrow = pd.Timestamp.now() + pd.Timedelta(days=1)
    
    symbols = [focusSymbol,thirdSymbol]
    optionSymbols = [focusSymbol]

    # historyApp.reqHistoricalDataFor(*vix, False, "2 D", "1 day")
    historyApp.reqHistoricalDataFor(*focusSymbol)


    
    # Keep the event loop alive so the scheduler's executor
    # remains active.
    while True:
        await asyncio.sleep(1)




def stock_symbol_to_client_id(symbol: str, max_id: int = 9999) -> int:
    """
    Converts a stock symbol into a numeric client ID without requiring external libraries.
    
    Args:
        symbol (str): The stock symbol.
        max_id (int): The maximum possible client ID to keep values within a range.

    Returns:
        int: A numeric client ID derived from the stock symbol.
    """
    return abs(hash(symbol)) % max_id + 1  # Ensures ID is within [1, max_id]

if __name__ == "__main__":
    # Run single-instance check only when executed directly
    if is_already_running():
        print("Another instance is already running. Exiting.")
        sys.exit(1)

    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))



    with keep.running():
        try:
            asyncio.run(main())
        finally:
            os.remove(PID_FILE)  # Cleanup on exit
            sys.exit(0)