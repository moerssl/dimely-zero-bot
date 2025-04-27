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
from data.AppScheduler import AppScheduler

async def main():
    # Initialize IBApp and connect.
    app = IBApp()
    historyApp = IBApp(lookBackTime="8 D")
    orderApp = TwsOrderAdapter()
    scheduler = AppScheduler(app, orderApp)
    orderApp.actionLog = app.actionLog
    focusSymbol = ("SPY", "STK", "SMART")
    secondarySymbol = ("SPX", "IND", "CBOE")
    thirdSymbol = ("QQQ", "STK", "SMART")
    vix = ("VIX", "IND", "CBOE")

    historyCandles = {}

    app.candleData = historyCandles
    historyApp.candleData = historyCandles
    print("Connecting to TWS/IB Gateway...")
    app.connect("127.0.0.1", 7497, clientId=1)  # Use your connection parameters.
    orderApp.connect("127.0.0.1", 7497, clientId=2)  # Use your connection parameters.
    historyApp.connect("127.0.0.1", 7497, clientId=3)  # Use your connection parameters.
    
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
    
    dash_thread = threading.Thread(target=start_dash_app, args=(historyApp,orderApp,), daemon=True)
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
    
    symbols = [focusSymbol,vix,thirdSymbol]
    optionSymbols = [focusSymbol,thirdSymbol]
    app.request_market_data(symbols)
    app.fetch_options_data(optionSymbols)
    historyApp.reqHistoricalDataFor(*vix, False, "2 D", "1 day")
    historyApp.reqHistoricalDataFor(*focusSymbol)
    
    # Keep the event loop alive so the scheduler's executor
    # remains active.
    while True:
        await asyncio.sleep(10)

if __name__ == "__main__":
    with keep.running():
        asyncio.run(main())