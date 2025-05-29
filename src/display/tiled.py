import curses


import curses
import datetime
from time import sleep

import pandas as pd
import numpy as np
import talib

import schedule

from ib.IBApp import IBApp
from ib.TwsOrderAdapter import TwsOrderAdapter
from util.config import CONFIG

pd.options.display.float_format = '{:.2f}'.format

def initialize_curses(screen):
    curses.curs_set(0)
    screen.nodelay(True)
    screen.timeout(500)
    # Turn off automatic echoing of keys to the screen
    curses.noecho()
    # React to keys instantly, without requiring the Enter key
    curses.cbreak()
    # Enable special keys, such as arrow keys
    screen.keypad(True)

def catch_up(app, historyAdapter, orderAdapter, focusSymbol):
    try:
        app.addToActionLog("Catching up...")
        #app.setMarketDataType()
        #app.fetch_positions()
        #sleep(0.5)
        historyAdapter.reqHistoricalDataFor(*focusSymbol, True)

        sleep(0.5)
        orderAdapter.reqAllOpenOrders()
        sleep(0.5)
    except Exception as e:
        app.addToActionLog("Error in catch_up: " + str(e))

def saveCandles(historyAdapter, app):
    try:
        historyAdapter.saveCandle()
    except Exception as e:
        app.addToActionLog("Error in saveCandles: " + str(e))

def enchance_order_data(orderApp: TwsOrderAdapter, app: IBApp, symbol):
    try:
        if (app.lastContractsDataUpdatedAt is None or 
            orderApp.lastContractsDataUpdatedAt is None or
            orderApp.lastContractsDataUpdatedAt != app.lastContractsDataUpdatedAt):
            orderApp.enhance_order_contracts_from_dataframe(app.options_data)
            now = datetime.datetime.now()
            orderApp.lastContractsDataUpdatedAt = now
            app.lastContractsDataUpdatedAt = now
    except Exception as e:
        app.addToActionLog("Error in enchance_order_data: " + str(e))



def display_data_tiled(screen, app: IBApp, orderAdapter: TwsOrderAdapter, historyAdapter: IBApp, focusSymbol: tuple):
    initialize_curses(screen)
    symbol, secType, exchange = focusSymbol
    # get the offsets from the config
    callDistance, putDistance, wing_span, target_premium, ic_wingspan, tp_percentage, sl = app.get_offset_configs(symbol)
    

    # schedule.every(150).seconds.do(catch_up, app, historyAdapter, orderAdapter, focusSymbol)    
    # schedule.every(5).minutes.do(saveCandles, historyAdapter, app)

    # catch_up(app, historyAdapter, orderAdapter, focusSymbol)

    display_data = True
    while True:
        schedule.run_pending()
        # Clear the screen
        screen.clear()
        # Get screen dimensions
        height, width = screen.getmaxyx()

        #app.cleanPrices()

        if (display_data):
            data: list = app.getTilesData(symbol)
            freeToTrade = {
                "title": "Free to Trade?",
                "content": [{
                    "SPY CALL": orderAdapter.is_room_for_new_positions("SPY", "C"),
                    "SPY PUT ": orderAdapter.is_room_for_new_positions("SPY", "P"),
                    "SPY IC ": orderAdapter.is_room_for_new_positions("SPY"),
                }]
            }
            

            #data.append(freeToTrade)

            # Determine number of columns using your heuristic.
            num_tiles = len(data)
            num_columns = int(num_tiles ** 0.5) or 1  # ensure at least one column

            # Build rows with a simple algorithm that respects each tile's colspan.
            rows = []
            current_row = []
            current_width = 0
            for tile in data:
                colspan = tile.get("colspan", 1)
                # Safety: ensure colspan does not exceed num_columns.
                if colspan > num_columns:
                    colspan = num_columns
                # If the current row cannot accommodate this tile, start a new row.
                if current_width + colspan > num_columns:
                    rows.append(current_row)
                    current_row = []
                    current_width = 0
                # Add the tile to the row and update the width.
                # It might be useful to store the effective colspan back in the tile.
                tile["colspan"] = colspan  
                current_row.append(tile)
                current_width += colspan
            if current_row:
                rows.append(current_row)

            num_rows = len(rows)
            tile_height = height // num_rows
            tile_width_unit = width // num_columns

            current_y = 0
            for row in rows:
                current_x = 0
                for tile in row:
                    colspan = tile.get("colspan", 1)
                    # Calculate the actual width based on the colspan.
                    tile_width = tile_width_unit * colspan
                    
                    # Draw the title
                    screen.addstr(current_y, current_x, tile["title"], curses.A_BOLD)
                    
                    # Draw the content
                    content = tile["content"]
                    exclude_columns = tile.get("exclude_columns", [])
                    if isinstance(content, list):
                        for i, line in enumerate(content):
                            if i < tile_height - 1:  # leave a row for the title
                                screen.addstr(current_y + i + 1, current_x, str(line)[:tile_width-1])
                    elif isinstance(content, dict):
                        # If content is a dictionary, format it as a string.
                        formatted_content = "\n".join([f"{k}: {v}" for k, v in content.items()])
                        for i, line in enumerate(formatted_content.split('\n')):
                            if i < tile_height - 1:
                                screen.addstr(current_y + i + 1, current_x, line[:tile_width-1])
                    else:
                        # If content has a "columns" attribute, assume it's a DataFrame.
                        if hasattr(content, "columns"):
                            
                            
                            available_width = tile_width - 1
                            selected_cols = []
                            total_width = 0
                            # Determine which DataFrame columns fit in the tile.
                            for col in content.columns:
                                col_header = str(col)
                                col_width = len(col_header)
                                if col_header in exclude_columns:
                                    continue
                                # Look at a few cell values (up to available rows) to decide column width.
                                for cell in content[col].head(tile_height - 2):
                                    cell_str = str(cell)
                                    if len(cell_str) > col_width:
                                        col_width = len(cell_str)
                                col_width += 1  # add a space for padding
                                if total_width + col_width <= available_width:
                                    selected_cols.append((col, col_width))
                                    total_width += col_width
                                else:
                                    break
                            # Print header row for the DataFrame.
                            if selected_cols:
                                header_line = ""
                                for col, width in selected_cols:
                                    header_line += str(col).ljust(width)
                                screen.addstr(current_y + 1, current_x, header_line[:tile_width-1])
                            # Print each DataFrame row that fits.
                            for i, row_data in enumerate(content.itertuples(index=False, name=None)):
                                if i >= tile_height - 2:
                                    break
                                row_line = ""
                                for col, width in selected_cols:
                                    idx = content.columns.get_loc(col)
                                    cell_val = str(row_data[idx]) 
                                    row_line += cell_val.ljust(width)


                                screen.addstr(current_y + i + 2, current_x, row_line[:tile_width-1])
                    
                        else:
                            for i, line in enumerate(str(content).split('\n')):
                                if i < tile_height - 1:
                                    screen.addstr(current_y + i + 1, current_x, line[:tile_width-1])
                                
                    # Move current_x by the tile's width.
                    current_x += tile_width
                # Move to the next row.
                current_y += tile_height

        else:
            # Get screen dimensions
            height, width = screen.getmaxyx()

            # Get the DataFrame
            df = app.candleData.get(symbol, pd.DataFrame())

            # Determine the number of rows that can fit on the screen
            rows_to_display = height-2

            # Slice the DataFrame to only include the most recent rows
            recent_data = df.tail(rows_to_display)

            # Convert the sliced DataFrame to a string for display
            dataframeToDisplay = str(recent_data)

            # Display the DataFrame on the screen
            for i, line in enumerate(dataframeToDisplay.split('\n')):
                if i < height:
                    screen.addstr(i, 0, line[:width])

        try:
            # putDistance = CONFIG["default_strike_offset_put"]
            # callDistance = CONFIG["default_strike_offset_call"]
            def credit_by_premium(type):
                try:
                    app.addToActionLog("credit_by_premium")
                    legs = app.build_credit_spread_by_premium(symbol, target_premium, type)
                    orderAdapter.place_combo_order(legs, tp_percentage, 200, type+"-Credit")
                except Exception as e:
                    app.addToActionLog("Error placing order: " + str(e))

            # Refresh the screen
            screen.refresh()
            key = screen.getch()

            # Check for escape key press
            if key == 27:  # Escape key
                #app.save_models()
                break

            if key == ord('V'):
                display_data = not display_data

            if key == ord('P'):
                app.addToActionLog("P key pressed")
                legs = app.build_credit_spread(symbol, 0.15, "P", wing_span)
                orderAdapter.place_combo_order(legs, tp_percentage, None, "PutCredit")
                

            if key == ord('C'):
                app.addToActionLog("C key pressed")
                legs = app.build_credit_spread(symbol, 0.15, "C", wing_span)
                orderAdapter.place_combo_order(legs, tp_percentage, None, "CallCredit")
            if key == ord('I'):
                app.addToActionLog("i key pressed")
                legs = app.construct_from_underlying(symbol, ic_wingspan, ic_wingspan)
                orderAdapter.place_combo_order(legs, tp_percentage, None, "IronCondor")

            if key == ord("E"):
                app.addToActionLog("E key pressen")
                legs: dict = app.evTrader.find_best_ev_credit_spreads(symbol, 10,1-(20/100))
                orderAdapter.place_combo_order(legs, 20, 200, "EV_IronCondor")
                strikeString = ""
                for key, val in legs.items():
                    strikeString += str(val.get("Strike"))
                    strikeString += " | "

                app.addToActionLog(strikeString)

            if key == ord('1'):
                app.addToActionLog("1 key pressed")
                legs = app.build_credit_spread_dollar(symbol, callDistance, wing_span, "C")
                orderAdapter.place_combo_order(legs, tp_percentage, None, "CallCredit")
            if key == ord('2'):
                app.addToActionLog("2 key pressed")
                legs = app.build_credit_spread_dollar(symbol, putDistance, wing_span, "P")
                
                orderAdapter.place_combo_order(legs, tp_percentage, None, "PutCredit")
            if key == ord('M'):
                app.addToActionLog("m key pressed")
                credit_by_premium("C")
                sleep(0.5)
                credit_by_premium("P")
            if key == ord('3'):
                credit_by_premium("C")
            if key == ord('4'):
                credit_by_premium("P")

            
            if key == curses.KEY_UP:
                if (symbol == "SPX"):
                    CONFIG["default_strike_offset_put"] += 5
                else:
                    CONFIG["offsets"][symbol]["put"] += 1
                app.addIndicatorsFor(symbol)
            elif key == curses.KEY_DOWN:
                if (symbol == "SPX"):
                    CONFIG["default_strike_offset_put"] -= 5
                else:
                    CONFIG["offsets"][symbol]["put"] -= 1
                app.addIndicatorsFor(symbol)
            elif key == curses.KEY_LEFT:
                if (symbol == "SPX"):
                    CONFIG["default_strike_offset_call"] -= 5
                else:
                    CONFIG["offsets"][symbol]["call"] -= 1
                app.addIndicatorsFor(symbol)
            elif key == curses.KEY_RIGHT:
                if (symbol == "SPX"):
                    CONFIG["default_strike_offset_call"] += 5
                else:
                    CONFIG["offsets"][symbol]["call"] += 1
                app.addIndicatorsFor(symbol)
            elif key == curses.KEY_PPAGE: #Page Up                
                if (symbol == "SPX"):
                    CONFIG["default_strike_offset"] += 5
                else:
                    CONFIG["offsets"][symbol]["late_ic_windspan"] += 1
                app.addIndicatorsFor(symbol)
            elif key == curses.KEY_NPAGE: #Page Down
                if (symbol == "SPX"):
                    CONFIG["default_strike_offset"] -= 5
                else:
                    CONFIG["offsets"][symbol]["late_ic_windspan"] -= 1
                app.addIndicatorsFor(symbol)

            
        except Exception as e:
            app.addToActionLog(f"Error: {e}")
