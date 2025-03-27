import threading
import multiprocessing
from time import sleep
from ib.IBApp import IBApp
import asyncio

from display.tiled import display_data, display_data_old, display_data_tiled
from display.dash_app import start_dash_app
from util.Chart import Chart

def start_chart(chart, app):
    chart.start(app.get_chart_data)

async def main():
    app = IBApp()
    data = app.get_chart_data()
    print(data)
    
    start_dash_app(app)
    

if __name__ == "__main__":
    asyncio.run(main())
    