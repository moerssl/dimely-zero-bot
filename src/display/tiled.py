import curses


import curses
import datetime

import pandas as pd
import talib

from ib.IBApp import IBApp

def display_data_tiled(screen, app: IBApp):
    curses.curs_set(0)
    screen.nodelay(True)
    screen.timeout(500)

    display_data = True
    while True:
        # Clear the screen
        screen.clear()
        # Get screen dimensions
        height, width = screen.getmaxyx()

        if (display_data):
            data = app.getTilesData()

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
            df = app.candleData.get("SPX", pd.DataFrame())

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


        # Refresh the screen
        screen.refresh()
        key = screen.getch()

        # Check for escape key press
        if key == 27:  # Escape key
            #app.save_models()
            break

        if key == ord('v'):
            display_data = not display_data

        if key == ord('p'):
            app.addToActionLog("P key pressed")
            app.send_credit_spread_order("SPX", 0.15, "P")

        if key == ord('c'):
            app.addToActionLog("C key pressed")
            app.send_credit_spread_order("SPX", 0.15, "C")

        """
        target_time = "21:55"
        target_close_time = "22:00"
        target_time_obj = datetime.datetime.strptime(target_time, "%H:%M").time()
        target_close_time_obj = datetime.datetime.strptime(target_close_time, "%H:%M").time()

        current_time = datetime.datetime.now().time()

        if current_time >= target_time_obj and current_time <= target_close_time_obj:
            app.addToActionLog("Target time reached")

            if not app.hasOrdersOrPositions("SPX"):
                app.addToActionLog("Ordering SPX Iron Condor")
                app.send_iron_condor_order("SPX", 15, 5)

            if not app.hasOrdersOrPositions("QQQ"):
                app.addToActionLog("Ordering QQQ Iron Condor")
                app.send_iron_condor_order("QQQ", 2, 2)
        """


def display_data_old(screen, app):
    curses.curs_set(0)
    screen.nodelay(True)
    screen.timeout(500)
    

    while True:
        # Clear the screen
        #screen.clear()

        data = app.getTilesData()

        print(data)
        # Get screen dimensions
        height, width = screen.getmaxyx()
        third_width = width // len(data)

        for j, tile in enumerate(data):
            screen.addstr(0, j * third_width, tile["title"], curses.A_BOLD)
            positions = str(tile["content"])
            for i, line in enumerate(positions.split('\n')):
                if i < height:
                    screen.addstr(i + 1, j * third_width, line[:third_width])
                    



        # Refresh the screen
        screen.refresh()

        # Check for escape key press
        if screen.getch() == 27:  # Escape key
            break

def display_data_old2(screen, app):
    curses.curs_set(0)
    screen.nodelay(True)
    screen.timeout(500)

    while True:
        # Clear the screen
        screen.clear()

        # Get screen dimensions
        height, width = screen.getmaxyx()
        third_width = width // 3

        # Print positions in the first third
        positions = str(app.positions)
        for i, line in enumerate(positions.split('\n')):
            if i < height:
                screen.addstr(i, 0, line[:third_width])

        # Print options data in the second third
        options_data = str(app.construct_from_underlying("SPX"))
        for i, line in enumerate(options_data.split('\n')):
            if i < height:
                screen.addstr(i, third_width, line[:third_width])

        # Print market data in the third third
        market_data = str(app.market_data)
        for i, line in enumerate(market_data.split('\n')):
            if i < height:
                screen.addstr(i, third_width * 2, line[:third_width])

        # Refresh the screen
        screen.refresh()

        # Check for escape key press
        if screen.getch() == 27:  # Escape key
            break 
def display_data(screen, app):
    curses.curs_set(0)  # Hide cursor

    tiles_data = app.getTilesData()
    num_tiles = len(tiles_data)
    height, width = screen.getmaxyx()
    tile_width = width // num_tiles

    for index, tile in enumerate(tiles_data):
        tile_win = curses.newwin(height, tile_width, 0, index * tile_width)
        tile_win.box()
        tile_win.addstr(1, 1, tile["title"], curses.A_BOLD)
        
        # Truncate content if it doesn't fit
        content = str(tile["content"])[:height * tile_width - 4]
        content_lines = [content[i:i + tile_width - 2] for i in range(0, len(content), tile_width - 2)]
        
        for i, line in enumerate(content_lines, start=2):
            if i >= height - 2:
                break
            tile_win.addstr(i, 1, line)
        
        tile_win.refresh()

        # Check for escape key press
        if screen.getch() == 27:  # Escape key
            break