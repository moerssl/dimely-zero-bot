from threading import Thread
from ib.TwsOrderAdapter import TwsOrderAdapter


def main():
    orderApp = TwsOrderAdapter()

    orderApp.connect("127.0.0.1", 7497, clientId=1)
    orderApp.reqIds(-1)
    orderApp.reqAllOpenOrders()
    orderApp.reqAutoOpenOrders(True)
    orderApp.reqPositions()
    orderThread = Thread(target=orderApp.run, daemon=True)
    orderThread.start()

if (__name__ == "__main__"):
    main()