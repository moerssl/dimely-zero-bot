import threading
import asyncio
from time import sleep
import pandas as pd
import curses
from wakepy import keep

# Import your IBApp and related modules.
from ib.IBApp import IBApp
from ib.TwsOrderAdapter import TwsOrderAdapter
from display.tiled import display_data_tiled
from display.dash_app import start_dash_app
from util.Chart import Chart

# Import the predictor class.
from data.ConfigAppScheduler import ConfigAppScheduler

import os
import sys
import psutil
symbolParam = sys.argv[1] if len(sys.argv) > 1 else "qqq"
PID_FILE = symbolParam+'.pid'

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
    historyApp = IBApp(lookBackTime="8 D")
    orderApp = TwsOrderAdapter()
    focusSymbol = (symbolParam, "STK", "SMART")
    secondarySymbol = ("SPX", "IND", "CBOE")
    thirdSymbol = ("SPY", "STK", "SMART")
    vix = ("VIX", "IND", "CBOE")

    focusContractSymbol, _, _  = focusSymbol
    secondaryContractSymbol, _, _ = secondarySymbol

    scheduler = ConfigAppScheduler(app, orderApp, historyApp,symbol=focusContractSymbol, secondSymbol=secondarySymbol[0])
    orderApp.actionLog = app.actionLog
    startClientId = 9
    def next_client_id():
        nonlocal startClientId
        client_id = startClientId
        startClientId += 1
        return client_id

    historyCandles = {}
    scheduler.historySymbols = focusSymbol
    app.focusContract = focusSymbol
    historyApp.focusContract = focusSymbol

    app.candleData = historyCandles
    historyApp.candleData = historyCandles
    app.additionalTilesFuncs.append({
        "function": scheduler.printEnhancedPredictions,
        "title": "Enhanced Predictions",
        "colspan": 3
    })
    app.additionalTilesFuncs.append(scheduler.get_jobs_dataframe)
    app.additionalTilesFuncs.append(scheduler.printPredictions)
    print("Connecting to TWS/IB Gateway...")
    app.connect("127.0.0.1", 7497, clientId=next_client_id())  # Use your connection parameters.
    orderApp.connect("127.0.0.1", 7497, clientId=next_client_id())  # Use your connection parameters.
    historyApp.connect("127.0.0.1", 7497, clientId=next_client_id())  # Use your connection parameters.
    
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
    
    dash_thread = threading.Thread(target=start_dash_app, args=(historyApp,orderApp,8051), daemon=True)
    dash_thread.start()

    scheduler.run()

    def start_curses_thread(app: IBApp, orderApp: TwsOrderAdapter):
        curses.wrapper(display_data_tiled, app, orderApp, historyApp, focusSymbol)
        app.disconnect()

    curses_thread = threading.Thread(target=start_curses_thread, args=(app,orderApp,), daemon=True)
    curses_thread.start()

    #app.setMarketDataType()

    print("Fetching positions and orders...")
    app.fetch_positions()
    app.reqAllOpenOrders()
    tomorrow = pd.Timestamp.now() + pd.Timedelta(days=1)
    
    symbols = [focusSymbol,vix,thirdSymbol]
    optionSymbols = [focusSymbol]
    app.request_market_data(symbols)
    app.fetch_options_data(optionSymbols)
    historyApp.reqHistoricalDataFor(*vix, False, "2 D", "1 day")
    historyApp.reqHistoricalDataFor(*focusSymbol)
    
    # Keep the event loop alive so the scheduler's executor
    # remains active.
    while True:
        await asyncio.sleep(10)

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