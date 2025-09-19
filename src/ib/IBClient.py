from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.utils import iswrapper
from ibapi.common import *
import pandas as pd
import time
from datetime import datetime, timedelta

class IBClient(EClient):
    def __init__(self, wrapper):
        EClient.__init__(self, wrapper)
        self.port = None
        self.host = None
        self.clientId = None

    def connect(self, host, port, clientId):
        self.host = host if self.host is None else self.host
        self.port = port     if self.port is None else self.port
        self.clientId = clientId if self.clientId is None else self.clientId
        
        return super().connect(host, port, clientId)
