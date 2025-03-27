from flask import Flask, jsonify
import pandas as pd

class Api:
    def __init__(self, server: Flask, app, config_file: str):
        self.server = server
        self.app = app
        self.config_file = config_file

        @self.server.route("/api/all", methods=["GET"])
        def getAll():
            """Endpoint to fetch all data."""
            chart_data: pd.DataFrame = self.app.get_chart_data(mergePredictions=True)
            if isinstance(chart_data, pd.DataFrame):
                chart_data = chart_data.iloc[::-1]
                return chart_data.to_html(index=False)

        @self.server.route("/api/candle", methods=["GET"])
        def getCandle():
            """Endpoint to fetch chart data."""
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

    def _read_config(self) -> list:
        """Reads the columns configuration from the config file."""
        try:
            with open(self.config_file, "r") as file:
                columns = file.read().splitlines()
                return [col.strip() for col in columns if col.strip()]  # Remove empty lines and whitespace
        except Exception as e:
            print(f"Error reading config file: {e}")
            return []  # Fallback to an empty list
