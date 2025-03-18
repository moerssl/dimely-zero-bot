from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import importlib  # To dynamically reload the chart config
import display.chart_config as chart_config  # Import your external chart configuration file
from util.StrategyEnum import StrategyEnum
from display.Api import Api
import pandas as pd
from ib.IBApp import IBApp

def start_dash_app(app: IBApp):
    dash_app = Dash(__name__)
    flaskApi = Api(dash_app.server, app, "candle-config.txt")

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
                interval=1000,  # Refresh every 1 second
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
        config = load_chart_config()  # Load chart config dynamically
        data: pd.DataFrame = app.get_chart_data()


        if not data.empty and "datetime" in data.columns:
            shapes = []

            # Convert datetime to string for a categorical x-axis
            data["datetime_str"] = data["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

            # Define a mapping of StrategyEnum values to colors
            signal_color_map = {
                StrategyEnum.SellPut: "red",
                StrategyEnum.SellCall: "green",
                StrategyEnum.SellIronCondor: "purple"
            }

            fig = go.Figure()

            # Add traces and subplots
            y_domain_start = 0
            for subplot_idx, subplot in enumerate(config["subplots"], start=1):
                y_domain_end = y_domain_start + subplot["height"]

                # Add horizontal line to separate subplots
                if subplot_idx > 1:  # Skip the first subplot
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
                        trace_type = trace["type"]
                        if trace_type == "candlestick":
                            fig.add_trace(go.Candlestick(
                                x=data[trace["columns"]["x"] + "_str"],  # Use string datetime
                                open=data[trace["columns"]["open"]],
                                high=data[trace["columns"]["high"]],
                                low=data[trace["columns"]["low"]],
                                close=data[trace["columns"]["close"]],
                                name=trace["name"],
                                yaxis=f"y{subplot_idx}"
                            ))
                        elif trace_type == "line":
                            fig.add_trace(go.Scatter(
                                x=data[trace["columns"]["x"] + "_str"],
                                y=data[trace["columns"]["y"]],
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
                            fig.add_trace(go.Bar(
                                x=data[trace["columns"]["x"] + "_str"],
                                y=data[trace["columns"]["y"]],
                                marker=dict(color=trace["style"]["color"]),
                                name=trace["name"],
                                yaxis=f"y{subplot_idx}"
                            ))
                    except Exception as e:
                        print(e)

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

            # Add background shapes for signals
            for i, row in data.iterrows():
                def drawSignal(col = "final_signal", opacity = 0.2):
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

                drawSignal()
                drawSignal("tech_signal", 0.1)

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

            # Display the latest row's date and close value
            latest_row = data.iloc[-1]
            latest_date = latest_row["datetime_str"]
            latest_close = latest_row["close"]
            current_request_count = n  # This increments with every interval-triggered request

            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=0.5,  # Centered at the top of the chart
                y=1.05,
                text=f"Latest: {latest_date}, Close: {latest_close}, Request Count: {current_request_count}",
                showarrow=False,
                font=dict(size=12, color="black"),
                align="center"
            )

            # Dynamically adjust x-axis
            fig.update_layout(
                shapes=shapes,  # Add subplot-separating lines
                title=config["layout"]["title"],
                height=config["layout"]["height"],
                legend_title=config["layout"]["legend_title"],
                xaxis=dict(
                    title="Time",
                    type="category",
                    range=[data["datetime_str"].iloc[0], data["datetime_str"].iloc[-1]],
                    showgrid=False,  # Enable x-axis grid
                    gridwidth=1,
                    gridcolor="black"
                ),
                yaxis=dict(
                    showgrid=True,  # Enable y-axis grid
                    gridwidth=1,
                    gridcolor="black"
                ),
                plot_bgcolor="white",  # Remove plot background
                paper_bgcolor="white"  # Remove paper background
                
            )


            if relayout_data and "xaxis.range[0]" in relayout_data and "xaxis.range[1]" in relayout_data:
                fig.update_layout(
                    xaxis={"range": [relayout_data["xaxis.range[0]"], relayout_data["xaxis.range[1]"]]}
                )
                fig.update_yaxes(autorange=True)

            return fig

        return go.Figure()

    dash_app.run_server(debug=False, use_reloader=False)
