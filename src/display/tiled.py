import curses


import curses
import datetime

from ib.IBApp import IBApp

def display_data_tiled(screen, app: IBApp):
    curses.curs_set(0)
    screen.nodelay(True)
    screen.timeout(500)

    while True:
        # Clear the screen
        screen.clear()

        data = app.getTilesData()

        # Get screen dimensions
        height, width = screen.getmaxyx()

        num_tiles = len(data)
        num_rows = (num_tiles + 1) // 2
        tile_height = height // num_rows
        tile_width = width // 2

        for j, tile in enumerate(data):
            row = j // 2
            col = j % 2
            x = col * tile_width
            y = row * tile_height
            screen.addstr(y, x, tile["title"], curses.A_BOLD)
            content = tile["content"]
            if (isinstance(content, list)):
                for i, line in enumerate(content):
                    lineStr = str(line)
                    if i < tile_height - 1:  # Adjust to leave space for the title
                        screen.addstr(y + i + 1, x, lineStr[:tile_width])
            else:
                positions = str(tile["content"])
                
                for i, line in enumerate(positions.split('\n')):
                    if i < tile_height - 1:  # Adjust to leave space for the title
                        screen.addstr(y + i + 1, x, line[:tile_width])

        # Refresh the screen
        screen.refresh()

        # Check for escape key press
        if screen.getch() == 27:  # Escape key
            break

        if (screen.getch() == ord('p')):
            app.addToActionLog("P key pressed")
            app.send_credit_spread_order("SPX", 0.15, "P")

        if (screen.getch() == ord('c')):
            app.addToActionLog("C key pressed")
            app.send_credit_spread_order("SPX", 0.15, "C")
            
        target_time = "21:55"
        target_close_time = "22:00"
        target_time_obj = datetime.datetime.strptime(target_time, "%H:%M").time()
        target_close_time_obj = datetime.datetime.strptime(target_close_time, "%H:%M").time()

        current_time = datetime.datetime.now().time()
    
        if current_time >= target_time_obj and current_time <= target_close_time_obj:
            app.addToActionLog("Target time reached")

            if (not app.hasOrdersOrPositions("SPX")):
                app.addToActionLog("Ordering SPX Iron Condor")
                app.send_iron_condor_order("SPX")


            if (not app.hasOrdersOrPositions("QQQ")):
                app.addToActionLog("Ordering QQQ Iron Condor")
                app.send_iron_condor_order("QQQ", 2, 2)


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