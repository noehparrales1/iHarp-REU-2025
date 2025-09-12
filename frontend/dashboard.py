# ===========================
# 📦 IMPORTS
# ===========================
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import base64
import io
import requests
import dash
from geopy.distance import geodesic
import dash_table


# ===========================
# ⚙️ CONSTANTS & APP CONFIG
# ===========================
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

DEFAULT_LAT = 24.3257
DEFAULT_LON = 6.3658
DEFAULT_ZOOM = 1.66
DEFAULT_MAPBOX_STYLE = "open-street-map"
DEFAULT_TITLE = "📍 Iceberg UI Project"


# ===========================
# 🗺️ Empty Map
# ===========================
def default_map():
    fig = go.Figure(go.Scattermapbox())
    fig.update_layout(
        mapbox=dict(
            style=DEFAULT_MAPBOX_STYLE,
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
    State('diff-toggle', 'value'),
    State('upload-2024', 'contents'),
    State('upload-2024', 'filename'),
    State('upload-2025', 'contents'),
    State('upload-2025', 'filename'),
    State('visualization-type', 'value')
)


def process_upload(n_clicks, diff_toggle, content_2024, name_2024, content_2025, name_2025, viz_type):
    if not n_clicks:
        return default_map(), "", {'display': 'none'}, 0, {},"N/A","N/A","N/A","N/A", None
    if not content_2024 or not content_2025:
        return default_map(), "upload both Excel files.", {'display': 'none'}, 0, {}, "N/A","N/A","N/A","N/A", None

    # convert base64 content to file object
    def to_file(content, filename):
        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)
        return (filename, io.BytesIO(decoded), 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    #mime type 
    files = {
        'file_2024': to_file(content_2024, name_2024),
        'file_2025': to_file(content_2025, name_2025),
    }

    try:
        # send files to backend flask api
        res = requests.post("http://127.0.0.1:5000/predict", files=files)
        if res.status_code != 200:
            return default_map(), f"server error: {res.text}", {'display': 'none'}, 0, {}, "N/A","N/A","N/A","N/A", None

        response_data = res.json()
        df_pred = pd.DataFrame(response_data["trajectory"])

        ade_deg = response_data.get("ADE (deg)", "N/A")
        fde_deg = response_data.get("FDE (deg)", "N/A")
        ade_m = response_data.get("ADE (m)", "N/A")
        fde_m = response_data.get("FDE (m)", "N/A")
 
        # ----- 2d map -----
        if viz_type == "2d":
            fig = go.Figure()
            fig.add_trace(go.Scattermapbox(
                lat=df_pred["GT_Lat"],
                lon=df_pred["GT_Lon"],
                mode="markers+lines",
                name="Ground Truth",
                marker=dict(color="blue")
            ))
            fig.add_trace(go.Scattermapbox(
                lat=df_pred["Pred_Lat"],
                lon=df_pred["Pred_Lon"],
                mode="markers+lines",
                name="Predicted",
                marker=dict(color="red")
            ))
            if diff_toggle:
                for i, row in df_pred.iterrows():
                    fig.add_trace(go.Scattermapbox(
                        lat=[row["GT_Lat"], row["Pred_Lat"]],
                        lon=[row["GT_Lon"], row["Pred_Lon"]],
                        mode="lines",
                        line=dict(color="orange", width=2),
                        showlegend=False,
                        hoverinfo="text",
                        text=f"Day {i+1} error"
            ))
            fig.update_layout(
                mapbox=dict(
                    style=DEFAULT_MAPBOX_STYLE,
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
            fig.add_trace(go.Scattermapbox(
                lat=df_slice["GT_Lat"],
                lon=df_slice["GT_Lon"],
                name="Ground Truth",
                mode="markers+lines",
                marker=dict(color="blue")
            ))

            # add predicted points as red line
            fig.add_trace(go.Scattermapbox(
                lat=df_slice["Pred_Lat"],
                lon=df_slice["Pred_Lon"],
                name="Predicted",
                mode="markers+lines",
                marker=dict(color="red")
            ))

            # set map style, zoom, center, and other layout stuff
            fig.update_layout(
                mapbox=dict(
                    style=DEFAULT_MAPBOX_STYLE,
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
                        go.Scattermapbox(
                            lat=sliced["GT_Lat"], 
                            lon=sliced["GT_Lon"], 
                            mode="markers+lines", 
                            marker=dict(color="blue"), 
                            name="Ground Truth"
                        ),
                        go.Scattermapbox(
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
                    go.Scattermapbox(
                        lat=[df_pred["GT_Lat"].iloc[0]], 
                        lon=[df_pred["GT_Lon"].iloc[0]], 
                        mode="markers+lines", 
                        marker=dict(color="blue"), 
                        name="Ground Truth"
                    ),
                    go.Scattermapbox(
                        lat=[df_pred["Pred_Lat"].iloc[0]], 
                        lon=[df_pred["Pred_Lon"].iloc[0]], 
                        mode="markers+lines", 
                        marker=dict(color="red"), 
                        name="Predicted"
                    )
                ],
                layout=go.Layout(
                    mapbox=dict(
                    style=DEFAULT_MAPBOX_STYLE,
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
    fig.add_trace(go.Scattermapbox(
        lat=df_slice["GT_Lat"],
        lon=df_slice["GT_Lon"],
        name="Ground Truth",
        mode="markers+lines",
        marker=dict(color="blue")
    ))
    fig.add_trace(go.Scattermapbox(
        lat=df_slice["Pred_Lat"],
        lon=df_slice["Pred_Lon"],
        name="Predicted",
        mode="markers+lines",
        marker=dict(color="red")
    ))

    # update layout with map styling and store data again
    
    fig.update_layout(
        mapbox=dict(
            style=DEFAULT_MAPBOX_STYLE,
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
            go.Scattermapbox(
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


# ----- run app -----
if __name__ == '__main__':
    app.run(debug=True)
