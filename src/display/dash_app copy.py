from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import importlib  # To dynamically reload the chart config
import display.chart_config as chart_config  # Import your external chart configuration file
from util.StrategyEnum import StrategyEnum
from display.Api import Api
import pandas as pd
from ib.IBApp import IBApp
import threading

def start_dash_app(app: IBApp, orderApp):
    dash_app = Dash(__name__)
    flaskApi = Api(dash_app.server, app, orderApp, "candle-config.txt")

    def load_chart_config():
        importlib.reload(chart_config)  # Dynamically reload the chart config
        return chart_config.get_chart_config()

    dash_app.layout = html.Div(
        style={"height": "100vh", "width": "100vw", "margin": "0", "padding": "0"},
        children=[
            dcc.Graph(
                id="candlestick-chart",
                style={"height": "100%", "width": "100%"},
                config={"scrollZoom": True}  # Enable scroll zoom
            ),
            dcc.Interval(
                id="interval-component",
                interval=10000,  # Refresh every 1 second
                n_intervals=0
            )
        ]
    )

    @dash_app.callback(
        Output("candlestick-chart", "figure"),
        [Input("interval-component", "n_intervals")],
        [State("candlestick-chart", "relayoutData")]  # Preserve zoom and layout
    )
    def update_chart(n, relayout_data):
        fig = go.Figure()

        try:
            config = load_chart_config()  # Load chart config dynamically
            data: pd.DataFrame = app.get_chart_data()


            if not data.empty and "datetime" in data.columns:
                # We assume the "datetime" column is there to create the string version
                data = data.tail(400).reset_index(drop=True)
                shapes = []

                # Convert datetime to string for a categorical x-axis
                data["datetime_str"] = data["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

                # Define a mapping of StrategyEnum values to colors
                signal_color_map = {
                    StrategyEnum.SellPut: "red",
                    StrategyEnum.SellCall: "green",
                    StrategyEnum.SellIronCondor: "purple"
                }

                y_domain_start = 0
                for subplot_idx, subplot in enumerate(config["subplots"], start=1):
                    y_domain_end = y_domain_start + subplot["height"]

                    # Add horizontal line to separate subplots (skip the first subplot)
                    if subplot_idx > 1:  
                        shapes.append(dict(
                            type="line",
                            x0=0,
                            x1=1,
                            y0=y_domain_start,
                            y1=y_domain_start,
                            xref="paper",
                            yref="paper",
                            line=dict(color="black", width=1)
                        ))

                    for trace in subplot["traces"]:
                        try:
                            trace_type = trace.get("type", "")
                            # Prepare the required columns based on trace type.
                            if trace_type == "candlestick":
                                x_col = trace["columns"]["x"] + "_str"  # using the converted datetime string column
                                open_col = trace["columns"]["open"]
                                high_col = trace["columns"]["high"]
                                low_col = trace["columns"]["low"]
                                close_col = trace["columns"]["close"]
                                required_cols = [x_col, open_col, high_col, low_col, close_col]
                                if not all(col in data.columns for col in required_cols):
                                    print(f"Skipping candlestick trace '{trace['name']}' because some columns are missing.")
                                    continue

                                fig.add_trace(go.Candlestick(
                                    x=data[x_col],
                                    open=data[open_col],
                                    high=data[high_col],
                                    low=data[low_col],
                                    close=data[close_col],
                                    name=trace["name"],
                                    yaxis=f"y{subplot_idx}"
                                ))

                            elif trace_type == "line":
                                x_col = trace["columns"]["x"] + "_str"
                                y_col = trace["columns"]["y"]
                                required_cols = [x_col, y_col]
                                if not all(col in data.columns for col in required_cols):
                                    print(f"Skipping line trace '{trace['name']}' because some columns are missing.")
                                    continue

                                fig.add_trace(go.Scatter(
                                    x=data[x_col],
                                    y=data[y_col],
                                    mode="lines",
                                    line=dict(
                                        color=trace["style"]["color"],
                                        width=trace["style"]["width"],
                                        dash=trace["style"].get("dash", "solid")
                                    ),
                                    name=trace["name"],
                                    yaxis=f"y{subplot_idx}"
                                ))

                            elif trace_type == "bar":
                                x_col = trace["columns"]["x"] + "_str"
                                y_col = trace["columns"]["y"]
                                required_cols = [x_col, y_col]
                                if not all(col in data.columns for col in required_cols):
                                    print(f"Skipping bar trace '{trace['name']}' because some columns are missing.")
                                    continue

                                fig.add_trace(go.Bar(
                                    x=data[x_col],
                                    y=data[y_col],
                                    marker=dict(color=trace["style"]["color"]),
                                    name=trace["name"],
                                    yaxis=f"y{subplot_idx}"
                                ))


                        except Exception as e:
                            print(f"Error adding trace '{trace.get('name', 'unknown')}':", e)

                        try:
                            lines = subplot.get("lines", [])
                            for line in lines:
                                val = line.get("val", None)
                                style = line.get("style", {})
                                if val is not None:
                                    fig.add_shape(
                                        dict(
                                            type="line",
                                            x0=0,
                                            x1=1,
                                            y0=val,
                                            y1=val,
                                            line=dict(
                                                color=line["style"]["color"],
                                                width=line["style"]["width"],
                                                dash=line["style"].get("dash", "solid")
                                            ),
                                            yref=f"y{subplot_idx}",
                                            xref="paper"
                                        )
                                    )
                        except Exception as e:
                            print(f"Error adding line to subplot '{subplot.get('name', 'unknown')}':", e)
                            app.addToActionLog(f"Error adding line to subplot '{subplot.get('name', 'unknown')}': {e}")
                    # Update the subplot's y-axis configuration.
                    fig.update_layout({
                        f"yaxis{subplot_idx}": dict(
                            title=subplot["name"],
                            domain=[y_domain_start, y_domain_end],
                            autorange=True,
                            showgrid=True,
                            gridwidth=1,
                            gridcolor="black"
                        )
                    })
                    y_domain_start = y_domain_end

                # Add background shapes for signals.
                for i, row in data.iterrows():
                    def drawSignal(col="final_signal", opacity=0.2):
                        signal_value = row[col]
                        color = signal_color_map.get(signal_value, None)
                        if color:
                            x0 = row["datetime_str"]
                            x1 = data.loc[i + 1, "datetime_str"] if i + 1 < len(data) else row["datetime_str"]
                            shapes.append(dict(
                                type="rect",
                                xref="x",
                                yref="paper",
                                x0=x0,
                                x1=x1,
                                y0=0,
                                y1=1,
                                opacity=opacity,
                                fillcolor=color,
                                layer="below",
                                line=dict(width=0)
                            ))

                    drawSignal(opacity=0.3)
                    drawSignal("tech_signal", 0.1)

                    if ("remaining_intervals" in data.columns) and (row["remaining_intervals"] > 0):
                        if i > 0 and row["remaining_intervals"] > data.loc[i - 1, "remaining_intervals"]:
                            shapes.append(dict(
                                type="line",
                                x0=row["datetime_str"],
                                x1=row["datetime_str"],
                                y0=0,
                                y1=1,
                                xref="x",
                                yref="paper",
                                line=dict(
                                    color="orange",
                                    width=2,
                                    dash="dash"
                                )
                            ))

                # Display the latest row's date and close value.
                latest_row = data.iloc[-1]
                latest_date = latest_row["datetime_str"]
                latest_close = latest_row["close"]
                current_request_count = n

                fig.add_annotation(
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=1.05,
                    text=f"Latest: {latest_date}, Close: {latest_close}, Request Count: {current_request_count}",
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    align="center"
                )

                # Configure the x- and y-axes as well as the layout.
                fig.update_layout(
                    shapes=shapes,
                    title=config["layout"]["title"],
                    height=config["layout"]["height"],
                    legend_title=config["layout"]["legend_title"],
                    xaxis=dict(
                        title="Time",
                        type="category",
                        range=[data["datetime_str"].iloc[0], data["datetime_str"].iloc[-1]],
                        showgrid=False,
                        gridwidth=1,
                        gridcolor="black"
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor="black"
                    ),
                    plot_bgcolor="white",
                    paper_bgcolor="white"
                )

                # Enable crosshairs on the main figure
                fig.update_layout(
                    hovermode="x unified",
                    xaxis=dict(
                        showspikes=True,
                        spikemode="across",
                        spikecolor="gray",
                        spikethickness=1,
                        spikedash="dot"
                    ),
                    yaxis=dict(
                        showspikes=True,
                        spikemode="across",
                        spikecolor="gray",
                        spikethickness=1,
                        spikedash="dot"
                    )
                )

                # If the user has zoomed or adjusted the x-axis, preserve that range.
                if relayout_data and "xaxis.range[0]" in relayout_data and "xaxis.range[1]" in relayout_data:
                    fig.update_layout(
                        xaxis={"range": [relayout_data["xaxis.range[0]"], relayout_data["xaxis.range[1]"]]}
                    )
                    fig.update_yaxes(autorange=True)
        except Exception as e:
            print("Error in update_chart callback:", e)
        
        # If something goes wrong or no valid trace is drawn, return an empty figure.
        return fig

    threading.Thread(target=lambda: dash_app.run_server(debug=False, use_reloader=False), daemon=True).start()
