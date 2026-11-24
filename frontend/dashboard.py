# ===========================
# 📦 IMPORTS
# ===========================
import os
import sys
import base64
import io
from pathlib import Path

from dash import Dash, dcc, html, Input, Output, State, ctx, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import dash
from geopy.distance import geodesic

# the model lives in ../backend/predict.py; call it directly (no separate API server)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from predict import run_prediction, CSV_FOLDER  # noqa: E402

SAMPLE_2024 = CSV_FOLDER / "A23A_2024.xlsx"
SAMPLE_2025 = CSV_FOLDER / "A23A_2025.xlsx"


# ===========================
# ⚙️ CONSTANTS & APP CONFIG
# ===========================
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], title="Iceberg Drift Forecasting")
server = app.server  # used by gunicorn when hosted

DEFAULT_LAT = 24.3257
DEFAULT_LON = 6.3658
DEFAULT_ZOOM = 1.66
DEFAULT_MAP_STYLE = "open-street-map"
DEFAULT_TITLE = "📍 Iceberg UI Project"


# ===========================
# 🗺️ Empty Map
# ===========================
def default_map():
    fig = go.Figure(go.Scattermap())
    fig.update_layout(
        map=dict(
            style=DEFAULT_MAP_STYLE,
            zoom=DEFAULT_ZOOM,
            center=dict(lat=DEFAULT_LAT, lon=DEFAULT_LON)
        ),
        margin=dict(r=0, t=0, l=0, b=0)
    )
    return fig


# ===========================
# 🧱 APP LAYOUT
# ===========================
app.layout = html.Div(
    style={"backgroundColor": "#425c76", "color" : "white", "padding": "20px"},
    children=[
        html.H2(DEFAULT_TITLE, style={"textAlign": "center"}),
        html.P(
            "Forecasts iceberg drift with LSTM-IDRIFTNET, a physics-informed LSTM that combines past "
            "positions, iceberg area, wind (u10, v10) and ocean currents (uo, vo). "
            "Upload 2024 and 2025 tracks, or try the sample data below. NSF iHARP REU 2025.",
            style={"textAlign": "center", "maxWidth": "900px", "margin": "0 auto 20px auto"}
        ),

        # ----- Sample Data Section -----
        html.Div(
            style={
                "backgroundColor": "#2c3e50",
                "padding": "15px 20px",
                "borderRadius": "10px",
                "maxWidth": "1000px",
                "margin": "0 auto 20px auto",
            },
            children=[
                html.H4("🧪 Try It With Sample Data", style={"marginBottom": "8px"}),
                html.P(
                    "Weekly observed positions of iceberg A23a for 2024 and 2025, with wind and ocean-current "
                    "values at each position. Run the model on them in one click, or download them to see "
                    "the file format the upload expects.",
                    style={"marginBottom": "8px"}
                ),
                html.P(
                    [html.B("Required columns: "),
                     "Date, Latitude, Longitude, Area, u10, v10, uo, vo"],
                    style={"fontSize": "0.9em", "marginBottom": "12px"}
                ),
                dbc.Button("▶ Run With Sample Data", id="sample-button", color="success", className="me-2"),
                dbc.Button("⬇ 2024 Sample (.xlsx)", id="download-2024-btn", color="info", outline=True, className="me-2"),
                dbc.Button("⬇ 2025 Sample (.xlsx)", id="download-2025-btn", color="info", outline=True),
                dcc.Download(id="download-2024"),
                dcc.Download(id="download-2025"),
            ]
        ),

        # ----- Upload and Dropdown Controls -----
        dbc.Card(
            dbc.Row([
                dbc.Col(
                    dbc.ButtonGroup([
                        dcc.Upload(
                            id='upload-2024',
                            children=dbc.Button("Upload 2024 XLSX File", color="primary", className="mx-2"),
                            multiple=False,
                            accept='.xlsx'
                        ),
                        dcc.Upload(
                            id='upload-2025',
                            children=dbc.Button("Upload 2025 XLSX File", color="primary", className="mx-2"),
                            multiple=False,
                            accept='.xlsx'
                        ),
                        dbc.Button("Run ML Prediction", id="submit-button", color="primary", className="mx-2")
                    ]),
                    width="auto"
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id='visualization-type',
                        options=[
                            {'label': '2D Map', 'value': '2d'},
                            {'label': '2D Slider', 'value': '2d_slider'},
                            {'label': 'Animated Map', 'value': 'animated_map'}
                        ],
                        value='2d',
                        clearable=False,
                        style={'width': '500px'}
                    ),
                    width="auto"
                )
            ], justify="center", className="g-2", style={"maxWidth": "1000px", "margin": "auto"}),
           
            style={
                "marginBottom": "20px",
                "padding": "15px",
                "backgroundColor": "transparent",
                "boxShadow": "none",
                "outline": "none",
                "border": "none"
            }
        ),

        html.Div(id="upload-status", style={"textAlign": "center", "marginBottom": "10px", "fontSize": "0.9em"}),

        # ----- Toggle Checkbox -----
        html.Div(
            dbc.Checkbox(
                id="diff-toggle",
                label="Show Point Differences",
                value=False,
                style={"marginRight": "20px"}
            ),
            style={"display": "flex", "justifyContent": "flex-end", "marginBottom": "20px"}
        ),

        # ----- Main Content: Info Panel + Map -----
        html.Div(
            style={"display": "flex", "flexDirection": "row", "gap": "30px"},
            children=[
                # Left panel with metrics
                html.Div(
                    style={
                        "width": "30%",
                        "display": "flex",
                        "flexDirection": "column",
                        
                    },
                    children=[
                        html.Div(
                            style={
                                "backgroundColor": "#2c3e50",
                                "padding": "15px",
                                "borderRadius": "10px",
                                
                            },
                            children=[
                                html.H4("Prediction Info", style={"textAlign": "center"}),
                                html.P("ADE (deg):", style={"marginBottom": "5px"}),
                                html.Div(id="ade-deg", style={"fontWeight": "bold"}),
                                html.P("FDE (deg):", style={"marginBottom": "5px"}),
                                html.Div(id="fde-deg", style={"fontWeight": "bold"}),
                                html.P("ADE (m):", style={"marginBottom": "5px"}),
                                html.Div(id="ade-m", style={"fontWeight": "bold"}),
                                html.P("FDE (m):", style={"marginBottom": "5px"}),
                                html.Div(id="fde-m", style={"fontWeight": "bold"})
                            ]
                        ),
                    ]
                ),

                # Right panel: Map, Slider, Summary Table
                html.Div(
                    style={
                        "flex": "1",
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "20px"
                    },
                    children=[
                        dcc.Loading(
                            dcc.Graph(id='map-output', figure=default_map(), style={"height": "500px"}),
                            type="dot"
                        ),
                        html.Div(
                            id="slider-container",
                            style={'display': 'none'},
                            children=[
                                dcc.Slider(
                                    id='date-slider',
                                    min=0,
                                    max=0,
                                    step=1,
                                    value=0,
                                )
                            ]
                        ),
                        html.Div(
                            id="summary-table-container",
                            style={"marginTop": "10px"}
                        )
                    ]
                )
            ]
        ),

        # ----- Hidden Stores & Errors -----
        dcc.Store(id="stored-df"),
        html.Div(id='error-text', className="mt-2")
    ]
)


# ===========================
# 🔁 CALLBACK: Upload + Predict + Map Rendering
# ===========================
@app.callback(
    Output('map-output', 'figure'),
    Output('error-text', 'children'),
    Output('slider-container', 'style'),
    Output('date-slider', 'max'),
    Output('date-slider', 'marks'),
    Output("ade-deg", "children"),
    Output("fde-deg", "children"),
    Output("ade-m", "children"),
    Output("fde-m", "children"),
    Output('stored-df', 'data'),
    Input('submit-button', 'n_clicks'),
    Input('sample-button', 'n_clicks'),
    Input('visualization-type', 'value'),
    State('diff-toggle', 'value'),
    State('upload-2024', 'contents'),
    State('upload-2025', 'contents'),
    State('stored-df', 'data'),
    State("ade-deg", "children"),
    State("fde-deg", "children"),
    State("ade-m", "children"),
    State("fde-m", "children"),
)
def process_upload(n_clicks, sample_clicks, viz_type, diff_toggle, content_2024, content_2025,
                   stored_df, cur_ade_deg, cur_fde_deg, cur_ade_m, cur_fde_m):
    trigger = ctx.triggered_id
    if trigger not in ("submit-button", "sample-button", "visualization-type"):
        return default_map(), "", {'display': 'none'}, 0, {},"N/A","N/A","N/A","N/A", None

    # convert base64 upload content to an in-memory file
    def to_file(content):
        content_type, content_string = content.split(',')
        return io.BytesIO(base64.b64decode(content_string))

    try:
        if trigger == "visualization-type":
            # view changed: re-draw the last results without re-running the model
            if not stored_df:
                raise dash.exceptions.PreventUpdate
            df_prev = pd.DataFrame(stored_df).reset_index(drop=True)
            response_data = {
                "ADE (deg)": cur_ade_deg, "FDE (deg)": cur_fde_deg,
                "ADE (m)": cur_ade_m, "FDE (m)": cur_fde_m,
                "trajectory": df_prev[["Date", "GT_Lat", "GT_Lon", "Pred_Lat", "Pred_Lon"]].to_dict("records"),
            }
        else:
            if trigger == "sample-button":
                file_2024, file_2025 = SAMPLE_2024, SAMPLE_2025
            else:
                if not content_2024 or not content_2025:
                    return default_map(), "upload both Excel files, or use the sample data above.", {'display': 'none'}, 0, {}, "N/A","N/A","N/A","N/A", None
                file_2024, file_2025 = to_file(content_2024), to_file(content_2025)

            # run the LSTM-IDRIFTNET model directly
            try:
                response_data = run_prediction(file_2024, file_2025)
            except ValueError as e:
                return default_map(), f"data error: {e}", {'display': 'none'}, 0, {}, "N/A","N/A","N/A","N/A", None

        df_pred = pd.DataFrame(response_data["trajectory"])

        ade_deg = response_data.get("ADE (deg)", "N/A")
        fde_deg = response_data.get("FDE (deg)", "N/A")
        ade_m = response_data.get("ADE (m)", "N/A")
        fde_m = response_data.get("FDE (m)", "N/A")
 
        # ----- 2d map -----
        if viz_type == "2d":
            fig = go.Figure()
            fig.add_trace(go.Scattermap(
                lat=df_pred["GT_Lat"],
                lon=df_pred["GT_Lon"],
                mode="markers+lines",
                name="Ground Truth",
                marker=dict(color="blue")
            ))
            fig.add_trace(go.Scattermap(
                lat=df_pred["Pred_Lat"],
                lon=df_pred["Pred_Lon"],
                mode="markers+lines",
                name="Predicted",
                marker=dict(color="red")
            ))
            if diff_toggle:
                for i, row in df_pred.iterrows():
                    fig.add_trace(go.Scattermap(
                        lat=[row["GT_Lat"], row["Pred_Lat"]],
                        lon=[row["GT_Lon"], row["Pred_Lon"]],
                        mode="lines",
                        line=dict(color="orange", width=2),
                        showlegend=False,
                        hoverinfo="text",
                        text=f"Day {i+1} error"
            ))
            fig.update_layout(
                map=dict(
                    style=DEFAULT_MAP_STYLE,
                    zoom=4,
                    center=dict(lat=df_pred["GT_Lat"].mean(), lon= df_pred["GT_Lon"].mean())),
                margin=dict(r=0, t=0, l=0, b=0)
            )
            return fig, "✅ 2d prediction complete!", {'display': 'none'}, 0, {},ade_deg, fde_deg, ade_m, fde_m, df_pred.to_dict()
            
        # ----- 2d slider -----
        elif viz_type == "2d_slider":
            # convert the 'Date' column to just the date (no time info)
            df_pred["Date"] = pd.to_datetime(df_pred["Date"]).dt.date

            # get all unique dates from the data, sorted in order
            unique_dates = sorted(df_pred["Date"].unique())

            # start with the first date as the default
            selected_date = unique_dates[0]

            # keep only rows up to and including the selected date
            df_slice = df_pred[df_pred["Date"] <= selected_date]

            # create the initial figure
            fig = go.Figure()

            # add ground truth points as blue line
            fig.add_trace(go.Scattermap(
                lat=df_slice["GT_Lat"],
                lon=df_slice["GT_Lon"],
                name="Ground Truth",
                mode="markers+lines",
                marker=dict(color="blue")
            ))

            # add predicted points as red line
            fig.add_trace(go.Scattermap(
                lat=df_slice["Pred_Lat"],
                lon=df_slice["Pred_Lon"],
                name="Predicted",
                mode="markers+lines",
                marker=dict(color="red")
            ))

            # set map style, zoom, center, and other layout stuff
            fig.update_layout(
                map=dict(
                    style=DEFAULT_MAP_STYLE,
                    zoom=4,
                    center=dict(lat=df_pred["GT_Lat"].mean(), lon=df_pred["GT_Lon"].mean())
                ),
                margin=dict(r=0, t=0, l=0, b=0),
                title=f"Date: {selected_date}",
                meta={"dates": list(unique_dates), "full_df": df_pred.to_dict()}
            )


            # return the figure, a message, make the slider visible,
            # set the slider max index, and map each slider index to a readable date
            return fig, "slider complete", {'display': 'block'}, len(unique_dates)-1, {i: str(date) for i, date in enumerate(unique_dates)}, ade_deg, fde_deg, ade_m, fde_m, df_pred.to_dict()

        # ----- animated map -----
        elif viz_type == 'animated_map':
            # convert Date column to datetime objects
            df_pred["Date"] = pd.to_datetime(df_pred["Date"])

            # make sure rows are sorted by date for animation to play chronologically
            df_pred = df_pred.sort_values("Date")

            frames = []  # list of frames for the animation

            # loop through each unique date to create a "frame" up to that point
            for date in df_pred["Date"].unique():
                # slice the data to include all rows up to and including the current date
                sliced = df_pred[df_pred["Date"] <= date]

                # create an animation frame with ground truth and predicted points
                frames.append(go.Frame(
                    name=str(date.date()),  # this frame will be labeled by the current date
                    data=[
                        go.Scattermap(
                            lat=sliced["GT_Lat"], 
                            lon=sliced["GT_Lon"], 
                            mode="markers+lines", 
                            marker=dict(color="blue"), 
                            name="Ground Truth"
                        ),
                        go.Scattermap(
                            lat=sliced["Pred_Lat"], 
                            lon=sliced["Pred_Lon"], 
                            mode="markers+lines", 
                            marker=dict(color="red"), 
                            name="Predicted"
                        )
                    ]
                ))

            # initialize the figure with the first data point (first date)
            fig = go.Figure(
                data=[
                    go.Scattermap(
                        lat=[df_pred["GT_Lat"].iloc[0]], 
                        lon=[df_pred["GT_Lon"].iloc[0]], 
                        mode="markers+lines", 
                        marker=dict(color="blue"), 
                        name="Ground Truth"
                    ),
                    go.Scattermap(
                        lat=[df_pred["Pred_Lat"].iloc[0]], 
                        lon=[df_pred["Pred_Lon"].iloc[0]], 
                        mode="markers+lines", 
                        marker=dict(color="red"), 
                        name="Predicted"
                    )
                ],
                layout=go.Layout(
                    map=dict(
                    style=DEFAULT_MAP_STYLE,
                    zoom=4,
                    center=dict(lat=df_pred["GT_Lat"].mean(), lon=df_pred["GT_Lon"].mean())
                ),
                margin=dict(r=0, t=0, l=0, b=0),
                
                    # play/pause controls
                    updatemenus=[{
                        "buttons": [
                            {
                                "args": [None, {
                                    "frame": {"duration": 500, "redraw": True},  # 500ms between frames
                                    "fromcurrent": True
                                }],
                                "label": "▶️ Play",
                                "method": "animate"
                            },
                            {
                                "args": [[None], {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate"
                                }],
                                "label": "⏸ Pause",
                                "method": "animate"
                            }
                        ],
                        "type": "buttons"
                    }]
                ),
                frames=frames  # pass all animation frames to the figure
            )

            # return the animated figure, no error message, slider hidden
            return fig, "🎞️ animated map rendered!", {'display': 'none'}, 0, {}, ade_deg, fde_deg, ade_m, fde_m, df_pred.to_dict()
                # fallback
        else:
            return default_map(), "", {'display': 'none'}, 0, {}, 'N/A', 'N/A', 'N/A', 'N/A', None

    except dash.exceptions.PreventUpdate:
        raise
    except Exception as e:
        return default_map(), f"frontend error: {str(e)}", {'display': 'none'}, 0, {}, 'N/A', 'N/A', 'N/A', 'N/A', None 

# ===========================
# 🔁 CALLBACK: Update Slider Map
# ===========================
@app.callback(
    Output('map-output', 'figure', allow_duplicate=True),
    Input('date-slider', 'value'),
    State('map-output', 'figure'),
    State('visualization-type', 'value'),
    prevent_initial_call=True
)
def update_2d_slider_map(slider_value, current_fig, viz_type):
    # if we're not in 2d slider mode or no figure yet, stop
    if viz_type != '2d_slider' or not current_fig:
        raise dash.exceptions.PreventUpdate

    # get list of dates and pick the one based on slider position
    dates = pd.to_datetime(current_fig["layout"]["meta"]["dates"]).date
    selected_date = dates[slider_value]

    # recreate dataframe from saved data in the current figure
    df_pred = pd.DataFrame(current_fig["layout"]["meta"]["full_df"])
    df_pred["Date"] = pd.to_datetime(df_pred["Date"]).dt.date

    # keep only rows up to the selected date
    df_slice = df_pred[df_pred["Date"] <= selected_date]

    # build new map figure with updated data
    fig = go.Figure()
    fig.add_trace(go.Scattermap(
        lat=df_slice["GT_Lat"],
        lon=df_slice["GT_Lon"],
        name="Ground Truth",
        mode="markers+lines",
        marker=dict(color="blue")
    ))
    fig.add_trace(go.Scattermap(
        lat=df_slice["Pred_Lat"],
        lon=df_slice["Pred_Lon"],
        name="Predicted",
        mode="markers+lines",
        marker=dict(color="red")
    ))

    # update layout with map styling and store data again
    
    fig.update_layout(
        map=dict(
            style=DEFAULT_MAP_STYLE,
            zoom=4,
            center=dict(lat=df_pred["GT_Lat"].mean(), lon=df_pred["GT_Lon"].mean())
        ),
        margin=dict(r=0, t=0, l=0, b=0),
        title=f"Date: {selected_date}",
        meta={"dates": list(dates), "full_df": df_pred.to_dict()}
            )

    return fig

# ===========================
# 🔁 CALLBACK: Toggle Difference Lines
# ===========================
@app.callback(
    Output('map-output', 'figure', allow_duplicate=True),
    Input('diff-toggle', 'value'),
    State('map-output', 'figure'),
    State('stored-df', 'data'),
    State('visualization-type', 'value'),
    prevent_initial_call=True
)
def update_diff_lines(toggle, fig, df_dict, viz_type):
    if fig is None or df_dict is None:
        raise dash.exceptions.PreventUpdate

    df = pd.DataFrame(df_dict)

    # Remove any existing orange diff lines
    fig['data'] = [trace for trace in fig['data'] if not (
        trace.get('line', {}).get('color') == 'orange' and trace.get('mode') == 'lines')]

    if not toggle:
        return fig

    # For 2d_slider mode, only show lines up to selected date
    if viz_type == '2d_slider' and "title" in fig["layout"]:
        selected_date_str = fig["layout"]["title"]["text"].replace("Date: ", "").strip()
        try:
            selected_date = pd.to_datetime(selected_date_str).date()
            df["Date"] = pd.to_datetime(df["Date"]).dt.date
            df = df[df["Date"] <= selected_date]
        except Exception:
            pass  # fallback to drawing all if something goes wrong

    
    for idx, (_, row) in enumerate(df.iterrows()):
        fig['data'].append(
            go.Scattermap(
                lat=[row["GT_Lat"], row["Pred_Lat"]],
                lon=[row["GT_Lon"], row["Pred_Lon"]],
                mode="lines",
                line=dict(color="orange", width=2),
                showlegend=False,
                hoverinfo="text",
                text=f"Day {idx + 1}"
            )
        )

    return fig


# ===========================
# 📐 HELPER: Compute Geodesic Distances
# ===========================
def compute_distance_errors(df):

    # Compute geodesic distance
    df["Distance_Error_m"] = df.apply(
        lambda row: geodesic(
            (row["GT_Lat"], row["GT_Lon"]),
            (row["Pred_Lat"], row["Pred_Lon"])
        ).meters,
        axis=1
    )
    return df

# ===========================
# 📊 HELPER: Generate Summary Table
# ===========================
def generate_summary_table(df):
    df_display = pd.DataFrame({
        "Date": pd.to_datetime(df["Date"]).dt.date,
        "Ground Truth (Lat, Lon)": df.apply(lambda row: f"({row['GT_Lat']:.2f}, {row['GT_Lon']:.2f})", axis=1),
        "Predicted (Lat, Lon)": df.apply(lambda row: f"({row['Pred_Lat']:.2f}, {row['Pred_Lon']:.2f})", axis=1),
        "Distance (m)": df["Distance_Error_m"].round(2)
    })

    return dash_table.DataTable(
        data=df_display.to_dict("records"),
        columns=[{"name": col, "id": col} for col in df_display.columns],
        
        style_cell={
            "backgroundColor": "#425c76",
            "color": "white",
            "fontFamily": "monospace",
            "padding": "10px",
            "textAlign": "left"
        },
        style_header={
            "backgroundColor": "#2c3e50",
            "fontWeight": "bold",
            "textAlign": "left"
        },
        style_data_conditional=[
            {
                "if": {"row_index": "odd"},
                "backgroundColor": "#3d5770"
            }
        ],
        style_as_list_view=True
    )

# ===========================
# 🔁 CALLBACK: Update Summary Table
# ===========================
@app.callback(
    Output("summary-table-container", "children"),
    Input("stored-df", "data")
)
def update_summary_table(df_dict):
    if df_dict is None:
        raise dash.exceptions.PreventUpdate

    df = pd.DataFrame(df_dict)
    
    df = compute_distance_errors(df)
    return generate_summary_table(df)


# ===========================
# 🔁 CALLBACKS: Sample Downloads + Upload Status
# ===========================
@app.callback(Output("download-2024", "data"), Input("download-2024-btn", "n_clicks"), prevent_initial_call=True)
def download_sample_2024(_):
    return dcc.send_file(str(SAMPLE_2024))


@app.callback(Output("download-2025", "data"), Input("download-2025-btn", "n_clicks"), prevent_initial_call=True)
def download_sample_2025(_):
    return dcc.send_file(str(SAMPLE_2025))


@app.callback(
    Output("date-slider", "value", allow_duplicate=True),
    Input("visualization-type", "value"),
    Input("stored-df", "data"),
    prevent_initial_call=True,
)
def reset_slider(_viz, _data):
    # new view or new results: start the slider at the first date
    return 0


@app.callback(
    Output("upload-status", "children"),
    Input("upload-2024", "filename"),
    Input("upload-2025", "filename"),
)
def show_upload_status(name_2024, name_2025):
    if not name_2024 and not name_2025:
        return ""
    return f"2024 file: {name_2024 or '—'}   |   2025 file: {name_2025 or '—'}"


# ----- run app -----
if __name__ == '__main__':
    app.run(debug=os.environ.get("DASH_DEBUG", "1") == "1", port=int(os.environ.get("PORT", 8050)))
