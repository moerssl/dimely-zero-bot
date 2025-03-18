import threading
import multiprocessing
from time import sleep
from ib.IBApp import IBApp
import curses
from display.tiled import display_data, display_data_old, display_data_tiled
from display.dash_app import start_dash_app
from util.Chart import Chart

def start_chart(chart, app):
    chart.start(app.get_chart_data)

def main():
    app = IBApp()
    print("Connecting to TWS/IB Gateway...")
    app.connect("127.0.0.1", 7497, clientId=1)  # Update with your TWS/IB Gateway connection details
    def run_loop():
        app.run()



    # Start the client loop in a separate thread
    api_thread = threading.Thread(target=run_loop)
    api_thread.start()    

    sleep(1)

    # app.reqMarketDataType(4)

    print("Fetching positions...")
    app.fetch_positions()
    app.reqAllOpenOrders()

    symbols = [
        ("VIX","IND","CBOE"), 
        ("SPX","IND","CBOE"), 
    ]
    app.request_market_data(symbols)
    app.fetch_options_data(symbols)

    app.reqHistoricalDataFor("SPX","IND","CBOE")

    #data = app.getTilesData()
    #print(data)
    chart = Chart()
    chart.initialize()
   


    # Start the Dash charting app in a separate thread
    dash_thread = threading.Thread(target=start_dash_app, args=(app,), daemon=True)
    dash_thread.start()

    # Start the display thread
    curses.wrapper(display_data_tiled, app)

    app.disconnect()

if __name__ == "__main__":
    main()
    