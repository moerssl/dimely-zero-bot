from flask import Flask, jsonify, url_for
from flask import request
import pandas as pd
import traceback

class Api:
    def set_cors_headers(self, response):
        """Set CORS headers for the response."""
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response
    
    def __init__(self, server: Flask, app, orderApp, config_file: str):
        self.server = server
        self.app = app
        self.config_file = config_file

        # allow corss for all routes
        self.server.after_request(self.set_cors_headers)

    

        @self.server.route("/api/all", methods=["GET"])
        def getAll():
            """Endpoint to fetch all data."""
            chart_data: pd.DataFrame = self.app.get_chart_data(mergePredictions=True)
            if isinstance(chart_data, pd.DataFrame):
                chart_data = chart_data.iloc[::-1]
                return chart_data.to_html(index=False)
            
            
        @self.server.route("/api/candle", methods=["GET"])
        def getCandleFb():
            return self.getCandle("SPY")

        @self.server.route("/api/candle/<string:symbol>", methods=["GET"])
        def getCandle(symbol):
            """Endpoint to fetch chart data for a given symbol."""
            # Load the columns configuration from the file
            columns_to_display = self._read_config()

            chart_data: pd.DataFrame = self.app.get_chart_data(mergePredictions=True)
            if isinstance(chart_data, pd.DataFrame):
                # Validate that the specified columns exist in the DataFrame
                missing_columns = [col for col in columns_to_display if col not in chart_data.columns]
                if missing_columns:
                    return jsonify({"error": f"Columns {missing_columns} not found in data"}), 400
                
                chart_data = chart_data[columns_to_display]  # Filter to the specified columns
                reversed_data = chart_data.iloc[::-1]  # Reverse the DataFrame
                return reversed_data.to_html(index=False)  # Convert DataFrame to HTML table
            else:
                return jsonify({"error": "No data available"}), 400

        @self.server.route("/api/time", methods=["GET"])
        def getTime():
            """Endpoint to fetch the current time."""
            current_time = pd.Timestamp.now()
            return jsonify({"current_time": current_time.strftime("%Y-%m-%d %H:%M:%S")})
            
        @self.server.route("/api/options", methods=["GET"])
        def getOptions():
            data: pd.DataFrame = self.app.options_data
            if isinstance(data, pd.DataFrame):
                # Calculate otm distance for non-NaN undPrice for call (C) and put (P)
                data["otm"] = float('-inf')
                data.loc[(data["Type"] == "P") & (data["undPrice"].notna()), "otm"] = data["undPrice"] - data["Strike"]
                data.loc[(data["Type"] == "C") & (data["undPrice"].notna()), "otm"] = data["Strike"] - data["undPrice"]

                # Sort non-NaN undPrice by otm ascending, NaN undPrice last
                sorted_data = data[data["otm"] > -5].sort_values(by=["otm"], ascending=True, na_position='last')
                if (len(sorted_data) > 0):
                    return sorted_data.to_html(index=False)
                else:
                    return data.to_html(index=False)
            else:
                return jsonify({"error": "No data available"}), 400

        @self.server.route("/api/positions", methods=["GET"])
        def getPositions():
            """Endpoint to fetch positions."""
            data = orderApp.positions
            df = pd.DataFrame(data.values())
            return df.to_html(index=False)

        @self.server.route("/api/orders", methods=["GET"])
        def getOrders():
            """Endpoint to fetch orders."""
            data: dict = orderApp.apiOrders

            df = pd.DataFrame(data.values())
            return self.toJsonOrHtml(df)
        
        @self.server.route("/api/orders/contracts", methods=["GET"])
        def getOrderContracts():
            """Endpoint to fetch orders."""
            data: dict = orderApp.orderContracts

            df = pd.DataFrame(data.values())
            return self.toJsonOrHtml(df)
        
        @self.server.route('/routes', methods=['GET'])
        def show_routes():
            try:
                routes_html = "<h1>Available Routes</h1><ul>"
                errors = []
                for rule in self.server.url_map.iter_rules():
                    if "GET" in rule.methods:  # Only include routes accessible via GET method
                        try:
                            route_url = url_for(rule.endpoint)  # Generate a URL for the route
                            routes_html += f'<li><a href="{route_url}">{route_url}</a></li>'
                        except Exception as e:
                            errors.append(f"Error generating URL for {rule.endpoint}: {e}")
                            continue
                if errors:
                    routes_html += "<h2>Errors:</h2><ul>"
                    for error in errors:
                        routes_html += f"<li>{error}</li>"
                    routes_html += "</ul>"
                routes_html += "</ul>"
                return routes_html
            except Exception as e:
                return f"Error generating routes: {e}", 500
            
        @self.server.errorhandler(Exception)
        def handle_exception(e):
            """Handle all uncaught exceptions and return a detailed error message."""
            response = {
                "error": str(e),  # Brief error message
                "details": traceback.format_exc(),  # Full traceback for debugging
            }
            return jsonify(response), 500

    def toJsonOrHtml(self, data: pd.DataFrame) -> str:
        # read accept header from request with html default
        accept_header = request.headers.get("Accept", "text/html")

        if "application/json" in accept_header:
            return data.to_json(orient="records")
        elif "text/html" in accept_header:
            return data.to_html(index=False)
        else:
            return data.to_csv(index=False)
        

    def _read_config(self) -> list:
        """Reads the columns configuration from the config file."""
        try:
            with open(self.config_file, "r") as file:
                columns = file.read().splitlines()
                return [col.strip() for col in columns if col.strip()]  # Remove empty lines and whitespace
        except Exception as e:
            print(f"Error reading config file: {e}")
            return []  # Fallback to an empty list
