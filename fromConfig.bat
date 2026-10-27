REM start cmd /k ".venv\Scripts\activate &&python src\fromConfig.py QQQ"
REM start cmd /k ".venv\Scripts\activate &&python src\fromConfig.py SPY"
start "" .venv\Scripts\python.exe src\fromConfig.py -s SPX -t "1 Y" -c "1 week"
start "" .venv\Scripts\python.exe src\fromConfig.py -s IWM
start "" .venv\Scripts\python.exe src\fromConfig.py -s SPY
REM start /min cmd /c ".venv\Scripts\activate && python src\fromConfig.py GLW"
REM start /min cmd /c ".venv\Scripts\activate && python src\fromConfig.py XSP"
REM start cmd /c ".venv\Scripts\activate && python src\fromConfig.py SPY"

