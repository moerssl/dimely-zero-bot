import threading
from time import sleep
from ib.IBApp import IBApp
import curses
from display.tiled import display_data, display_data_old, display_data_tiled



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

    symbols = [("SPX","IND","CBOE"), ("VIX","IND","CBOE"), ("QQQ","STK","SMART"), ("IWM","STK","SMART")]
    app.request_market_data(symbols)

    app.fetch_options_data(symbols)

    data = app.getTilesData()
    print(data)


    # Start the display thread
    curses.wrapper(display_data_tiled, app)

    app.disconnect()

if __name__ == "__main__":
    main()
    