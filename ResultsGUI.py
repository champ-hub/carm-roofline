import os
import csv
import platform
import datetime
import logging

#Third Party Libraries
#Run: pip install dash dash-bootstrap-components plotly numpy pandas diskcache
#To get all of the Libraries
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import diskcache
import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash import Input, Output, State, html, dcc, no_update, DiskcacheManager, callback_context

#Local Python Scripts
import run
import PMU_AI_Calculator
import DBI_AI_Calculator
import CodeInject

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)

CONFIG_FILE = "./config/auto_config/config.txt"
pathway = './Results/Roofline'

#Filter the list of files to only include CSV files
if os.path.exists(pathway):
    csv_files = [f for f in os.listdir(pathway) if f.endswith('.csv')]
else:
    csv_files = []
#Extract machine names from filenames
machine_names = [file.split('_')[0] for file in csv_files]

#Determine CPU architecture
CPU_Type = platform.machine()
if CPU_Type == "x86_64":
    isa_options = [
        {"label": "AVX512", "value": "avx512"},
        {"label": "AVX2", "value": "avx2"},
        {"label": "SSE", "value": "sse"},
        {"label": "Scalar", "value": "scalar"}
    ]
elif CPU_Type == "aarch64":
    isa_options = [
        {"label": "NEON", "value": "neon"},
        {"label": "Scalar", "value": "scalar"}
    ]
elif CPU_Type == "riscv64":
    isa_options = [
        {"label": "RVV", "value": "rvv"},
        {"label": "Scalar", "value": "scalar"}
    ]
else:
    isa_options = []

def read_library_path(tag):
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as file:
            for line in file:
                if line.strip() == "":
                    continue
                parts = line.strip().split("=")
                if len(parts) == 2:
                    key, value = parts
                    if key == tag:
                        return value
    return None

def write_library_path(tag, path):
    with open(CONFIG_FILE, "a") as file:
        file.write(f"{tag}={path}\n")

def read_csv_file(file_path):
    data_list = []
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        machine_name = header[1]
        l1_size = int(header[3])
        l2_size = int(header[5])
        l3_size = int(header[7])

        header2 = next(reader)
        for row in reader:
            if not row or not ''.join(row).strip():
                continue
            data = {}
            data['Date'] = row[0]
            data['ISA'] = row[1]
            data['Precision'] = row[2]
            data['Threads'] = int(row[3])
            data['Loads'] = int(row[4])
            data['Stores'] = int(row[5])
            data['Interleaved'] = row[6]
            data['DRAMBytes'] = int(row[7])
            data['FPInst'] = row[8]
            data['L1'] = float(row[9])
            data['L2'] = float(row[11])
            data['L3'] = float(row[13])
            data['DRAM'] = float(row[15])
            data['FP'] = float(row[17])
            data['FP_FMA'] = float(row[19])
            data_list.append(data)

    return machine_name, l1_size, l2_size, l3_size, data_list

def read_application_csv_file(file_path):
    if not os.path.exists(file_path):
        print("Application file does not exist:", file_path)
        return False

    data_list = []
    try:
        with open(file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader, None)

            if header is None:
                print("File is empty:", file_path)
                return False
            
            for row in reader:
                if row:
                    data = {
                        'Date': row[0],
                        'Method': row[1],
                        'Name': row[2],
                        'ISA': row[3],
                        'Precision': row[4],
                        'Threads': row[5],
                        'AI': float(row[6]),
                        'GFLOPS': float(row[7]),
                        'Bandwidth': float(row[8]),
                        'Time': float(row[9])
                    }
                    data_list.append(data)

    except Exception as e:
        print("Failed to read the file:", file_path, "Error:", e)
        return False
    return data_list if data_list else False

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True, background_callback_manager=background_callback_manager)

sidebar = dbc.Offcanvas(
    html.Div([
        dbc.Button("Run CARM Benchmarks", id="button-CARM", className="mb-2", style={'width': '100%'}),
        html.P("CARM Benchmarks Configuration", className="mb-2", 
            style={
                'color': 'white',
                'textAlign': 'center',
                'fontSize': '20px'
            }),
        dbc.Card([
            dbc.CardBody([
                dbc.Input(id="input-name", placeholder="Machine Name", className="mb-2"),
                html.P("Machine Cache Sizes per Core (Kb):", className="mb-1", style={'color': 'white'}),
                dbc.Row([
                    dbc.Col(dbc.Input(id="input-l1", placeholder="L1"), width=3),
                    dbc.Col(dbc.Input(id="input-l2", placeholder="L2"), width=4),
                    dbc.Col(dbc.Input(id="input-l3", placeholder="L3 Total Size"), width=5),
                ], className="mb-2"),
                html.P("Thread Counts to Benchmark:", className="mb-1", style={'color': 'white'}),
                dbc.Input(id="input-threads", placeholder="1 2 4 8 16 32 64...", className="mb-2"),
                dbc.Checklist(
                    options=[{"label": "Interleave Threads (NUMA)", "value": "interleaved"}],
                    id="checkbox-interleaved",
                    inline=True,
                    className="mb-2",
                    style={'color': 'white'},
                ),
                html.P("ISA Extensions to Benchmark:", className="mb-1", style={'color': 'white'}),
                dbc.Checklist(
                    options=isa_options,
                    id="checkbox-isa",
                    inline=True,
                    className="mb-2",
                    style={'color': 'white'},
                ),
                html.P("Precisions to Benchmark:", className="mb-1", style={'color': 'white'}),
                dbc.Checklist(
                    options=[{"label": "DP", "value": "dp"}, {"label": "SP", "value": "sp"}],
                    id="checkbox-precision",
                    inline=True,
                    className="mb-2",
                    style={'color': 'white'},
                ),
                html.P("Load/Store Ratio Configuration:", className="mb-1", style={'color': 'white'}),
                dbc.Input(id="input-ldst", placeholder="Custom Load/Store Ratio", className="mb-2"),
                dbc.Checklist(
                    options=[
                        {"label": "Only Loads", "value": "only_ld"},
                        {"label": "Only Stores", "value": "only_st"}
                    ],
                    value=[],
                    id="checklist-only_ldst",
                    inline=True,
                    className="mb-2",
                    style={'color': 'white'},
                ),
                html.P("DRAM Test Size Configuration:", className="mb-1", style={'color': 'white'}),
                dbc.Row([
                    dbc.Col(dbc.Input(id="input-dram", placeholder="Custom Size (Kb)"), width=7),
                    dbc.Col(dbc.Checkbox(id="checkbox-dram", label="Auto_Adjust"), width=5, style={'color': 'white'},),
                ], className="mb-2"),
            ])
        ], className="mb-3", style={'backgroundColor': '#6c757d'}),
        dbc.Button("Run Application Analysis", id="app-analysis-button", className="mb-2", style={'width': '100%'}),
        dbc.Button("Run Application ROI Code Injection", id="app-inject-button", className="mb-2", style={'width': '100%'}),
        dbc.Button("Stop Benchmark/Analysis", id="cancel-button", className="mb-2", style={'width': '100%'}),
    ], style={'backgroundColor': '#1a1a1a'}),
    id="offcanvas",
    title=html.H5("CARM Tool Functions", style={'color': 'white', 'fontsize': '30px'}),
    is_open=False,
    style={'backgroundColor': '#1a1a1a'},
)

#Main app layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(
            dbc.Button(
                html.Img(src="/assets/menu_icon.png", height="30px"),
                id="open-offcanvas",
                n_clicks=0,
                className="btn-sm",
                style={'border': 'none', 'background': 'transparent', 'padding': '0', 'margin': '0'}
            ),
            width="auto",
            style={'padding-right': '5px', 'padding-left': '5'}
        ),
        dbc.Col(
            dcc.Dropdown(
                id='filename',
                options=[{'label': machine_name, 'value': os.path.join(pathway, file)} for machine_name, file in zip(machine_names, csv_files)],
                multi=False,
                placeholder="Select Machine Results..."
            ),
            width=True
        ),
    ],
    align="center", 
    style={'margin-top': '1px'}
    ),
    dbc.Row([
        dbc.Col([
            html.Div(id='additional-dropdowns', style={'margin-top': '10px'}),
            html.Div(id='additional-dropdowns2'),
            html.Div(id='application-dropdown-container'),
            dcc.Graph(id='graphs', style={'display': 'none'})
            ])
        ]),
    dbc.Row([
        dbc.Col([
            html.Div(
                [
                    html.Span("⬆", style={'fontSize': '24px', 'color': '#6c757d', 'marginRight': '10px'}),
                    html.Span("Select a Machine to View CARM Results", style={'fontSize': '20px', 'fontWeight': 'bold', 'color': '#6c757d'}),
                    html.Span("⬆", style={'fontSize': '24px', 'color': '#6c757d', 'marginLeft': '10px'})
                ],
                id="initial-text",
                style={'textAlign': 'center', 'marginTop': '10px'}
            ),
            html.A(
                html.Img(
                    src="/assets/CHAMP_logo.svg",
                    id="initial-image",
                    style={'width': '99%', 'display': 'block', 'background': 'transparent', 'marginLeft': '40px'}
                ),
                href="https://github.com/champ-hub",
                target="_blank"
            ),
            
        ], width=10, style={'backgroundColor': '#e9ecef', 'textAlign': 'center'})
    ], id="initial-content", justify="center", style={'backgroundColor': '#e9ecef', 'textAlign': 'center'}),
    dcc.Store(id='machine-selected', data=False),
    sidebar,
    html.Div(id="invisible-output", style={"display": "none"}),  #Invisible Div to satisfy callback output
    html.Div(id="invisible-output2", style={"display": "none"}),  #Invisible Div to satisfy callback output
    html.Div(id="invisible-output3", style={"display": "none"}),  #Invisible Div to satisfy callback output
    html.Div(id="invisible-output-final", style={"display": "none"}),  #Invisible Div to satisfy callback output
    html.Div(id="file-path-valid", style={"display": "none"}), #Invisible Div to satisfy callback output
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Application To Profile", style={'text-align': 'center', 'color': 'white'}), style={'backgroundColor': '#6c757d'}),
        dbc.ModalBody([
            dbc.Input(id="machine-name-app", placeholder="Machine Name", className="mb-2"),
            html.P("Application Analysis Method", className="mb-1", style={'color': 'black', 'text-align': 'center'}),
            dbc.Card(
                dbc.CardBody(
                    html.Div([
                        dbc.Checklist(
                            options=[
                                {"label": "DBI", "value": "dbi"},
                                {"label": "DBI (ROI)", "value": "dbi_roi"},
                                {"label": "PMU (ROI)", "value": "pmu_roi"}
                            ],
                            value=[],
                            id="checklist-pmu-dbi",
                            inline=True,
                            className="mb-2",
                            style={'color': 'black'},
                        )
                    ],
                    className="d-flex flex-column align-items-center"
                    )
                ),
                style={'width': '350px', 'height': '60px', 'margin': '0px auto', 'padding': '0px', 'text-align': 'center'},
                className="mb-3"
            ),
            html.P("Application Specification", className="mb-1", style={'color': 'black', 'text-align': 'center'}),
            dbc.Card(
                dbc.CardBody(
                    html.Div(
                        [
                            dbc.Input(id="file-path-input", placeholder="Enter executable file path", type="text", className="mb-2"),
                            dbc.Input(id="text-input", placeholder="Enter executable arguments", type="text",className="mb-2"),
                        ],
                        className="d-flex flex-column align-items-center"
                    )
                ),
                style={'width': '100%', 'height': '120px', 'margin': '0px auto', 'padding': '0px', 'text-align': 'center'},
                className="mb-3"
            ),
            html.P("Application Source Code must be Injected to Profile Region of Interest", className="mb-1", style={'color': 'black', 'text-align': 'center', 'fontSize': '14px'}),                
        ],
        style={'backgroundColor': '#e9ecef'},
        id="modal-body"
        ),
        dbc.ModalFooter([
                dbc.Button("Run Application", id="submit-button", className="ms-auto", n_clicks=0, style={'margin-right': 'auto'}),
                dbc.Button("Close", id="close-modal-button", className="me-auto", n_clicks=0, style={'margin-left': 'auto'})
        ],
        className="w-100 d-flex",
        style={'backgroundColor': '#6c757d'}
        ),
    ],
    id="modal-profile",
    is_open=False,
    ),
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Enter Path to DynamoRIO Folder", style={'text-align': 'center', 'color': 'white'}), style={'backgroundColor': '#6c757d'}),
        dbc.ModalBody(
            [
                dbc.Input(id="library-path-input", placeholder="Enter the DynamoRIO path here...", type="text")
            ],
            id="library-modal-body",
            style={'backgroundColor': '#e9ecef'},
        ),
        dbc.ModalFooter(
            dbc.Button("Submit Path", id="submit-library-path", className="ms-auto", n_clicks=0),style={'backgroundColor': '#6c757d'},
        )
    ],
    id="library-modal",
    is_open=False
    ),
    dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Application To Inject ROI Code", style={'text-align': 'center', 'color': 'white'}), style={'backgroundColor': '#6c757d'}),
            dbc.ModalBody([
                html.P("Application Analysis Method", className="mb-1", style={'color': 'black', 'text-align': 'center'}),
                dbc.Card(
                    dbc.CardBody(
                        html.Div([
                            dbc.Checklist(
                                options=[
                                    {"label": "DBI", "value": "dbi"},
                                    {"label": "PMU", "value": "pmu"}
                                ],
                                value=[],
                                id="checklist-inject-pmu-dbi",
                                inline=True,
                                className="mb-2",
                                style={'color': 'black'},
                            )
                        ],
                        className="d-flex flex-column align-items-center"
                        )
                    ),
                    style={'width': '200px', 'height': '60px', 'margin': '0px auto', 'padding': '0px', 'text-align': 'center'},
                    className="mb-3"
                ),
                html.P("Application Specification", className="mb-1", style={'color': 'black', 'text-align': 'center'}),
                dbc.Card(
                    dbc.CardBody(
                        html.Div([
                            dbc.Input(id="file-path-input-inject", placeholder="Enter source code file path", type="text", className="mb-0"),
                            html.Div(id="file-path-error", style={'color': 'red', 'textAlign': 'center', 'marginTop': '6px', 'marginBottom': '0px', 'fontSize': '15px'}),
                            ],
                            className="d-flex flex-column align-items-center"
                        )
                    ),
                    style={'width': '100%', 'minHeight': '60px', 'margin': '0px auto', 'padding': '0px', 'text-align': 'center'},
                    className="mb-3"
                        ),
                html.P("Region of Interest Flags Should be Present in Source Code", className="mb-1", style={'color': 'black', 'text-align': 'center'}),
                dbc.Card(
                            dbc.CardBody([
                                        dbc.Row(
                                                [
                                                    dbc.Col(html.P("Region Start Flag:", className="mb-1", style={'color': 'black', 'text-align': 'right'}), width=6, align="center"),
                                                    dbc.Col(dbc.Badge("//CARM ROI START", color="#6c757d", className="me-1"), width=6, align="center"),
                                                ],
                                                className="align-items-center"
                                            ),
                                        dbc.Row(
                                                [
                                                    dbc.Col(html.P("Region End Flag:", className="mb-0", style={'color': 'black', 'text-align': 'right'}), width=6, align="center"),
                                                    dbc.Col(dbc.Badge("//CARM ROI END", color="#6c757d", className="mb-0"), width=6, align="center"),
                                                ],
                                                className="align-items-center"
                                            ),
                                        ]),
                            style={'width': '100%', 'height': '80px', 'margin': '0px auto', 'padding': '0px', 'text-align': 'center'},
                            className="mb-1"
                        ),
            
                    dbc.Card(
                                dbc.CardBody(
                                    [

                                                    dbc.Checklist(
                                                        options=[
                                                            {"label": "Create new injected source file", "value": "True"}
                                                        ],
                                                        value=[],
                                                        id="new-file-checklist",
                                                        inline=True,
                                                        className="mb-0",
                                                        style={'color': 'black', "text-align": "center"},
                                                    ),
                                        
                                    ],
                                    className="d-flex align-items-center justify-content-center",
                                ),
                                style={'width': '100%', 'height': '60%', 'margin': '0', 'padding': '0px', 'text-align': 'center'}
                            )    
            ],
            style={'backgroundColor': '#e9ecef'}
            ),
            dbc.ModalFooter(
                [
                    dbc.Button("Inject Code", id="submit-button-inject", className="ms-auto", n_clicks=0, style={'margin-right': 'auto'}),
                    dbc.Button("Close", id="close-modal-button-inject", className="me-auto", n_clicks=0, style={'margin-left': 'auto'}),
                ],
                className="w-100 d-flex",
                style={'backgroundColor': '#6c757d'}
            ),
    ],
    id="modal-inject",
    is_open=False,
    ),
],
fluid=True,
className="p-3",
style={'backgroundColor': '#e9ecef'}
)
#Callback to update the machine-selected state
@app.callback(
    Output('machine-selected', 'data'),
    Input('filename', 'value')
)
def update_machine_selected(filename):
    if filename:
        return True
    return False

#Callback to control the visibility of the initial image and text
@app.callback(
    [Output('initial-image', 'style'),
     Output('initial-text', 'style')],
    Input('machine-selected', 'data')
)
def toggle_initial_content(machine_selected):
    if machine_selected:
        return {'display': 'none'}, {'display': 'none'}
    return {'width': '99%', 'display': 'block', 'background': 'transparent', 'marginLeft': '40px'}, {'text-align': 'center', 'margin-top': '10px'}

#Toggle visibility of the sidebar
@app.callback(
    Output("offcanvas", "is_open"),
    [Input("open-offcanvas", "n_clicks")],
    [State("offcanvas", "is_open")]
)
def toggle_offcanvas(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("modal-profile", "is_open"),
    [Input("app-analysis-button", "n_clicks"),
     Input("close-modal-button", "n_clicks"),
     Input("submit-button", "n_clicks"),
     Input("invisible-output2", "children"),  #Output from file path validation
     Input("invisible-output3", "children")],  #Output from library modal
    [State("modal-profile", "is_open")],
    
)
def toggle_modal(open_clicks, close_clicks, submit_clicks, file_path_output, library_action, is_open):
    ctx = callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == "close-modal-button":
        return False
    elif trigger_id == "submit-button":
        #Check for error messages or requirements to keep the modal open
        parts = file_path_output.split()
        if "Error" in parts[0]:
            return True  #Keep the modal open if there's an error or further input needed
        return False  #Close if all conditions are okay and no further input is needed
    elif trigger_id == "app-analysis-button":
        return not is_open  #Toggle the normal way if opening

    return is_open


#Library Path Submission
@app.callback(
    [
        Output("library-modal", "is_open"),
        Output("library-modal-body", "children"),
        Output("invisible-output3", "children")
    ],
    [
        Input("submit-button", "n_clicks"),
        Input("submit-library-path", "n_clicks"),
        Input("invisible-output2", "children"),
    ],
    [
        State("checklist-pmu-dbi", "value"),
        State("file-path-input", "value"),
        State("text-input", "value"),
        State("library-path-input", "value"),
        State("library-modal", "is_open")
    ],
    prevent_initial_call=True,
)
def handle_submit_library(submit_clicks, submit_library_clicks, file_path_output, checklist_values, file_path, exec_arguments, library_path, is_library_modal_open):
    triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0] if callback_context.triggered else 'No clicks yet'
    if "Error" not in file_path_output:
        parts = file_path_output.split()

        checklist_values = parts[3]
        if triggered_id == "submit-button":
            if "dbi" in checklist_values:
                Dyno_path = read_library_path("DYNO")
                if not Dyno_path:
                    #Return state indicating library input is needed
                    return True, dbc.Input(id="library-path-input", placeholder="Enter the DynamoRIO path here...", type="text"), "need_input"
            return False, dbc.Input(id="library-path-input", placeholder="Enter the DynamoRIO path here...", type="text"), "Library path saved"

        elif triggered_id == "submit-library-path":
            if library_path and DBI_AI_Calculator.check_client_exists(library_path):
                write_library_path("DYNO", library_path)
                return False, dbc.Input(id="library-path-input", placeholder="Enter the DynamoRIO path here...", type="text"), "Library path saved"
            else:
                return True, [
                    dbc.Input(id="library-path-input", placeholder="Enter the DynamoRIO path here...", type="text"),
                    html.Div("Invalid path, try again", style={'color': 'red', 'margin-top': '10px'})
                ], "need_input"

    return no_update, dbc.Input(id="library-path-input", placeholder="Enter the DynamoRIO path here...", type="text"), no_update

#Main profile submission
@app.callback(
    [
        Output("modal-body", "children"),
        Output("invisible-output2", "children")  #To carry error or success messages
    ],
    Input("submit-button", "n_clicks"),
    [
        State("checklist-pmu-dbi", "value"),
        State("machine-name-app", "value"),
        State("file-path-input", "value"),
        State("text-input", "value"),
    ],
    prevent_initial_call=True,
)
def handle_submit(n_clicks, checklist_values, machine_name, file_path, exec_arguments):
    if not n_clicks:
        return modal_content, "No action taken here"

    error_message = None
    if not os.path.isfile(file_path):
        error_message = "The specified file was not found."

    modal_content = [
        dbc.Input(id="machine-name-app", placeholder="Machine Name", value=machine_name, className="mb-2"),
        html.P("Application Analysis Method", className="mb-1", style={'color': 'black', 'text-align': 'center'}),
        dbc.Card(
            dbc.CardBody(
                dbc.Checklist(
                    options=[
                        {"label": "DBI", "value": "dbi"},
                        {"label": "DBI (ROI)", "value": "dbi_roi"},
                        {"label": "PMU (ROI)", "value": "pmu_roi"}
                    ],
                    value=checklist_values,
                    id="checklist-pmu-dbi",
                    inline=True,
                    className="mb-2",
                    style={'color': 'black'}
                ),
                className="d-flex flex-column align-items-center"
            ),
            style={'width': '350px', 'height': '60px', 'margin': '0px auto', 'padding': '0px', 'text-align': 'center'},
            className="mb-3"
        ),
        html.P("Application Specification", className="mb-1", style={'color': 'black', 'text-align': 'center'}),
        dbc.Card(
            dbc.CardBody(
                [
                    dbc.Input(id="file-path-input", placeholder="Enter executable file path", type="text", value=file_path, className="mb-2"),
                    dbc.Input(id="text-input", placeholder="Enter executable arguments", type="text", value=exec_arguments, className="mb-2"),
                    html.Div(error_message, style={'color': 'red'}) if error_message else None
                ],
                className="d-flex flex-column align-items-center"
            ),
            style={'width': '100%', 'height': '140px', 'margin': '0px auto', 'padding': '0px', 'text-align': 'center'},
            className="mb-3"
        ),
        html.P("Application Source Code must be Injected to Profile Region of Interest", className="mb-1", style={'color': 'black', 'text-align': 'center', 'fontSize': '14px'}),
    ]

    if error_message:
        return modal_content, f"Error: {error_message}"

    modal_content_clear = [
        dbc.Input(id="machine-name-app", placeholder="Machine Name", className="mb-2"),
        html.P("Application Analysis Method", className="mb-1", style={'color': 'black', 'text-align': 'center'}),
        dbc.Card(
            dbc.CardBody(
                dbc.Checklist(
                    options=[
                        {"label": "DBI", "value": "dbi"},
                        {"label": "DBI (ROI)", "value": "dbi_roi"},
                        {"label": "PMU (ROI)", "value": "pmu_roi"}
                    ],
                    value=[],
                    id="checklist-pmu-dbi",
                    inline=True,
                    className="mb-2",
                    style={'color': 'black'}
                ),
                className="d-flex flex-column align-items-center"
            ),
            style={'width': '350px', 'height': '60px', 'margin': '0px auto', 'padding': '0px', 'text-align': 'center'},
            className="mb-3"
        ),
        html.P("Application Specification", className="mb-1", style={'color': 'black', 'text-align': 'center'}),
        dbc.Card(
            dbc.CardBody(
                [
                    dbc.Input(id="file-path-input", placeholder="Enter executable file path", type="text", className="mb-2"),
                    dbc.Input(id="text-input", placeholder="Enter executable arguments", type="text", className="mb-2"),
                ],
                className="d-flex flex-column align-items-center"
            ),
            style={'width': '100%', 'height': '120px', 'margin': '0px auto', 'padding': '0px', 'text-align': 'center'},
            className="mb-3"
        ),
        html.P("Application Source Code must be Injected to Profile Region of Interest", className="mb-1", style={'color': 'black', 'text-align': 'center', 'fontSize': '14px'}),
    ]
    if machine_name == None and exec_arguments == None:
        data = "Processing complete " + file_path + " " + str(checklist_values) + " " + "unnamed"
    if not machine_name == None and exec_arguments == None:
        data = "Processing complete " + file_path + " " + str(checklist_values) + " " + machine_name
    if machine_name == None and not exec_arguments == None:
        data = "Processing complete " + file_path + " " + str(checklist_values) + " " + "unnamed" + " " + exec_arguments
    if not machine_name == None and not exec_arguments == None:
        data = "Processing complete " + file_path + " " + str(checklist_values) + " " + machine_name + " " + exec_arguments

    return modal_content_clear, data

@app.callback(
    Output("invisible-output-final", "children"),
    [
        Input("invisible-output2", "children"),  #Output from file path validation
        Input("invisible-output3", "children")   #Output from library path validation
    ],
    [
        State("machine-name-app", "value"),
        State("file-path-input", "value"),
        State("text-input", "value"),
        State("checklist-pmu-dbi", "value"),
        State("modal-profile", "is_open"),
        State("library-modal", "is_open")
    ],
    prevent_initial_call=True,
    background=True,
    running=[
        (Output("app-analysis-button", "disabled"), True, False),
        (Output("cancel-button", "disabled"), False, True)
    ],
    cancel=[Input("cancel-button", "n_clicks")]
)
def execute_profiling(file_path_status, library_path_status, machine_name, file_path, arguments, checklist_values, profile_modal_open, library_modal_open):

    if "Processing complete" in file_path_status and "Library path saved" in library_path_status:
        parts = file_path_status.split()


        file_path = parts[2]
        checklist_values = parts[3]
        machine_name = parts[4]

        #Joining the rest as execution arguments if any
        exec_arguments = ' '.join(parts[5:]) if len(parts) > 5 else None
        try:
            exec_arguments_list = exec_arguments.split()
            if str(checklist_values) == str(['dbi']) or str(checklist_values) == str(['dbi_roi']):
                method = "DR"
                Dyno_path = read_library_path("DYNO")
                if str(checklist_values) == str(['dbi']):
                    exec_time = DBI_AI_Calculator.runApplication(0, file_path, exec_arguments_list)
                    
                    DBI_AI_Calculator.runDynamoRIO(Dyno_path, 0, file_path, exec_arguments_list)
                    
                if str(checklist_values) == str(['dbi_roi']):
                    method += "-ROI"
                    exec_time = DBI_AI_Calculator.runApplication(1, file_path, exec_arguments_list)
                    DBI_AI_Calculator.runDynamoRIO(Dyno_path, 1, file_path, exec_arguments_list)
                
                if CPU_Type == "x86_64":
                    fp_ops, memory_bytes, integer_ops = DBI_AI_Calculator.analyseDynamoRIOx86()
                    DBI_AI_Calculator.printDynamoRIOx86()
                elif CPU_Type == "aarch64":
                    fp_ops, memory_bytes, integer_ops = DBI_AI_Calculator.analyseDynamoRIOARM()
                    DBI_AI_Calculator.printDynamoRIOARM()
                else:
                    print("No opcode analysis support on this architecture.")

                time_taken_seconds = float (exec_time / 1e9)

                flops = fp_ops/time_taken_seconds

                gflops = flops / 1e9

                ai = float(fp_ops/memory_bytes)
                bandwidth = float((memory_bytes * 8) / exec_time)

                print("\nTotal FP operations:", fp_ops)
                print("Total memory bytes:", memory_bytes)
                print("Total integer operations:", integer_ops)
                print("\nExecution time (seconds):", time_taken_seconds)
                print("GFLOPS:", gflops)
                print("Arithmetic Intensity:", ai)

                ct = datetime.datetime.now()
                date = ct.strftime('%Y-%m-%d %H:%M:%S')
    

                DBI_AI_Calculator.update_csv(machine_name, file_path, gflops, ai, bandwidth, time_taken_seconds, "", date, None, None, None, method)
            if str(checklist_values) == str(['pmu_roi']):
                total_time_nsec, total_mem, total_sp, total_dp, thread_count = PMU_AI_Calculator.runPAPI(file_path, exec_arguments_list)
                total_fp = total_sp + total_dp

                sp_ratio = float (total_sp / total_fp)
                dp_ratio = float (total_dp / total_fp)

                memory_bytes = total_mem * (sp_ratio * 4 + dp_ratio * 8)

                if dp_ratio > 0.9:
                    precision = "dp"
                elif sp_ratio > 0.9:
                    precision = "sp"
                else:
                    precision = "mixed"

                

                ai = float (total_fp / memory_bytes)

                gflops = float(total_fp / total_time_nsec)
                bandwidth = float((memory_bytes * 8) / total_time_nsec)

                print("\nTotal FLOP Count:", total_fp)
                print("SP FLOP Ratio: " + str(sp_ratio) + " DP FLOP Ration: " + str(dp_ratio))
                print("Calculated Total Memory Bytes:", memory_bytes)
                print("Simple AI:", float(total_fp / total_mem))
                print("Complete AI:", ai)
                print("Threads Used:", thread_count)


                print("Execution Time (seconds):" + str(float(total_time_nsec / 1e9)))
                print("GFLOPS: " + str(gflops))
                print("Bandwidth (Gbps): " + str(bandwidth))

                ct = datetime.datetime.now()

                date = ct.strftime('%Y-%m-%d %H:%M:%S')
                PMU_AI_Calculator.update_csv(machine_name, file_path, gflops, ai, bandwidth, float(total_time_nsec / 1e9), "", date, None, precision, thread_count)
            return f"Script executed successfully:"
        except Exception as e:
            return f"Failed to execute script: {str(e)}"
    return "Waiting for conditions to execute script"


@app.callback(
    Output("checklist-pmu-dbi", "value"),
    [Input("checklist-pmu-dbi", "value")],
    [State("checklist-pmu-dbi", "value")]
)
def toggle_checklist_pmu_dbi(current_values, previous_values):
    ctx = dash.callback_context

    #Check if the callback was triggered by a user action
    if not ctx.triggered:
        return dash.no_update

    #Identify the property that triggered the callback
    trigger_id, trigger_prop = ctx.triggered[0]['prop_id'].split('.')

    if trigger_prop == 'children':
        return current_values  #Keep the initial preset values


    if not current_values:  #No selection is made (all deselected)
        return []

    #If more than one checkbox is selected
    if len(current_values) > 1:
        latest_selected = list(set(current_values) - set(previous_values))
        return latest_selected if latest_selected else current_values[-1:]  #Keep the latest selected one

    return current_values
@app.callback(
    Output("checklist-inject-pmu-dbi", "value"),
    [Input("checklist-inject-pmu-dbi", "value")],
    [State("checklist-inject-pmu-dbi", "value")]
)
def toggle_checklist(current_values, previous_values):
    ctx = dash.callback_context
    if not ctx.triggered:
        return []
    trigger_value = ctx.triggered[0]["value"]

    if not current_values:
        return []
    if len(current_values) > 1:
        latest_selected = list(set(current_values) - set(previous_values))
        return latest_selected if latest_selected else current_values[-1:]
    return current_values

@app.callback(
    Output("checklist-only_ldst", "value"),
    [Input("checklist-only_ldst", "value")],
    [State("checklist-only_ldst", "value")]
)
def toggle_checklist(current_values, previous_values):
    ctx = dash.callback_context
    if not ctx.triggered:
        return []
    trigger_value = ctx.triggered[0]["value"]

    if not current_values:  #All checkboxes are deselected
        return []
    if len(current_values) > 1:
        latest_selected = list(set(current_values) - set(previous_values))
        return latest_selected if latest_selected else current_values[-1:]
    return current_values

@app.callback(
    Output("submit-button-inject", "disabled"),
    [
        Input("checklist-inject-pmu-dbi", "value"),
        Input("file-path-input-inject", "value")
    ]
)
def update_button_status_inject(pmu_dbi_values, file_path):
    if pmu_dbi_values and file_path:
        return False
    return True

@app.callback(
    Output("submit-button", "disabled"),
    [
        Input("checklist-pmu-dbi", "value"),
        Input("file-path-input", "value")
    ]
)
def update_button_status(pmu_dbi_values, file_path):
    if pmu_dbi_values and file_path:
        return False
    return True


@app.callback(
    Output("modal-inject", "is_open"),
    [Input("app-inject-button", "n_clicks"),
     Input("close-modal-button-inject", "n_clicks"),
     Input("submit-button-inject", "n_clicks"),
     Input("file-path-valid", "children")],
    [State("modal-inject", "is_open")]
)
def toggle_modal_inject(open_clicks, close_clicks, submit_clicks, file_path_valid, is_open):
    ctx = callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == "close-modal-button-inject":
        return False
    elif trigger_id == "submit-button-inject":
        #Closing if the file path is valid
        if file_path_valid:
            return False
        return True
    elif trigger_id == "app-inject-button":
        return not is_open

    return is_open


@app.callback(
    [Output("file-path-valid", "children"),
     Output("file-path-error", "children"),
     ],
    [Input("submit-button-inject", "n_clicks"),
     Input("checklist-inject-pmu-dbi", "value"),
     Input("new-file-checklist", "value"),],
    [State("file-path-input-inject", "value")],
    prevent_initial_call=True
)
def check_file_path(n_clicks, pmu_dbi, new_file, file_path):
    if not file_path:
        return False, ""

    if not os.path.isfile(file_path) or (not file_path.endswith('.c') and not file_path.endswith('.cpp')):
        return False, "The specified file was not found or is not a C/C++ source file."
    injected = CodeInject.inject_code(file_path, pmu_dbi, new_file)

    if not injected:
        return False, "Region of Interest Flags not Found."
    return True, ""



@app.callback(
    Output("invisible-output", "children"),
    Input("button-CARM", "n_clicks"),
    [State("input-name", "value"),
     State("input-l1", "value"),
     State("input-l2", "value"),
     State("input-l3", "value"),
     State("input-threads", "value"),
     State("checkbox-interleaved", "value"),
     State("checkbox-isa", "value"),
     State("checkbox-precision", "value"),
     State("input-ldst", "value"),
     State("checklist-only_ldst", "value"),
     State("input-dram", "value"),
     State("checkbox-dram", "value")],
    prevent_initial_call=True,
    background=True,
    running=[
        (Output("button-CARM", "disabled"), True, False),
        (Output("cancel-button", "disabled"), False, True)
    ],
    cancel=[Input("cancel-button", "n_clicks")]
)
def execute_script1(n, name, l1_size, l2_size, l3_size, thread_set, interleaved, isa_set, precision_set, ldst_ratio, only_ldst, dram_bytes, dram_auto):
    if n is None:
        raise PreventUpdate
    num_ld = 2
    num_st = 1

    if name == None:
        name = "unnamed"
    if l1_size == None:
        l1_size = 0
    if l2_size == None:
        l2_size = 0
    if l3_size == None:
        l3_size = 0
    
    if thread_set == None or thread_set == "":
        thread_set = [1]
    else:
        thread_set = [int(thread) for thread in thread_set.split()]
    
    if isa_set == None:
        isa_set = ["auto"]
    
    if precision_set == None:
        precision_set = ["dp"]

    if ldst_ratio == None:
        ldst_ratio = 0
    else:
        num_ld = int(ldst_ratio)
        num_st = 1
    
    if dram_bytes == None:
        dram_bytes = 524288

    if not only_ldst == None:
        if 'only_ld' in only_ldst:
            num_ld = 1
            num_st = 0
        elif 'only_st' in only_ldst:
            num_st = 1
            num_ld = 0
    else:
        num_ld = 0
        num_st = 0

    freq = 0
    inst = "add"
    num_ops = 32768
    test_type = "roofline"
    verbose = 3
    set_freq = 0
    measure_freq = 0
    VLEN = 1
    tl1 = 2
    tl2 = 2
    plot = 0
    
    try:
        run.run_roofline(name, freq, l1_size, l2_size, l3_size, inst, isa_set, precision_set, num_ld, num_st, thread_set, interleaved, num_ops, int(dram_bytes), dram_auto, test_type, verbose, set_freq, measure_freq, VLEN, tl1, tl2, plot)
    except Exception as e:
        print("Task was interrupted:", e)
    return no_update

@app.callback(
    Output('additional-dropdowns', 'children'),
    [Input('filename', 'value')]
)
def update_additional_dropdowns(selected_file):
    if not selected_file:
        return html.Div([])

    _, _, _, _, data_list = read_csv_file(selected_file)
    df = pd.DataFrame(data_list)
    
    if df.empty:
        return html.Div([])

    fields = ['ISA', 'Precision', 'Threads', 'Loads', 'Stores', 'Interleaved', 'DRAM Bytes', 'FP Inst', 'Date']
    dropdowns = []

    for field in fields:
        if field == "Date":
            unique_values = sorted(df[field.replace(" ", "")].unique(), reverse=True)  
        else:
            unique_values = sorted(df[field.replace(" ", "")].unique())  
            
        options = [{'label': value, 'value': value} for value in unique_values]

        if field == "Date":
            width = '275px'
        elif field == "ISA":
            width = '230px'
        else: 
            width = '161px'

        dropdown_style = {'display': 'inline-block', 'width': width, 'margin': '5px'}
        dropdown = html.Div([
            dcc.Dropdown(
                id=f'{field.lower().replace(" ", "")}-dynamic-dropdown',
                placeholder=field,
                options=options,
                multi=False
            )
        ], 
        style=dropdown_style)
        
        dropdowns.append(dropdown)
    
    return dbc.Card(
    dbc.CardBody([
        html.Div([
            html.Div("CARM Results 1:", style={'height': '100%', 'margin-right': '10px', 'align-self': 'center'}),
            html.Div(dropdowns, style={'height': '130%', 'display': 'flex', 'justify-content': 'center'})
        ], style={'display': 'flex', 'align-items': 'center', 'height': '130%'})
    ]),
    style={'height': '60px', 'margin': '0px auto 10px auto', 'padding': '0px', 'text-align': 'center'},
)

@app.callback(
    Output('additional-dropdowns2', 'children'),
    [Input('filename', 'value')]
)
def update_additional_dropdowns2(selected_file):
    if not selected_file:
        return html.Div([])

    _, _, _, _, data_list = read_csv_file(selected_file)
    df = pd.DataFrame(data_list)
    
    if df.empty:
        return html.Div([])

    fields = ['ISA', 'Precision', 'Threads', 'Loads', 'Stores', 'Interleaved', 'DRAM Bytes', 'FP Inst', 'Date']
    dropdowns = []

    for field in fields:
        if field == "Date":
            unique_values = sorted(df[field.replace(" ", "")].unique(), reverse=True)  
        else:
            unique_values = sorted(df[field.replace(" ", "")].unique())  
        options = [{'label': value, 'value': value} for value in unique_values]
        if field == "Date":
            width = '275px'
        elif field == "ISA":
            width = '230px'
        else: 
            width = '161px'

        dropdown_style = {'display': 'inline-block', 'width': width, 'margin': '5px'}
        dropdown = html.Div([
            dcc.Dropdown(
                id=f'{field.lower().replace(" ", "")}-dynamic-dropdown2',
                placeholder=field,
                options=options,
                multi=False
            )
        ], style=dropdown_style)
        
        dropdowns.append(dropdown)
    
    return dbc.Card(
    dbc.CardBody([
        html.Div([
            html.Div("CARM Results 2:", style={'height': '100%', 'margin-right': '10px', 'align-self': 'center'}),
            html.Div(dropdowns, style={'height': '130%', 'display': 'flex', 'justify-content': 'center'})
        ], style={'display': 'flex', 'align-items': 'center', 'height': '130%'})
    ]),
    style={'height': '60px', 'margin': '0px auto', 'padding': '0px', 'text-align': 'center'},
)

@app.callback(
    [
        Output('application-dropdown-container', 'children'),
        Output('application-dropdown-container', 'style')
    ],
    [Input('filename', 'value')]
)
def update_application_dropdown(selected_file):
    if not selected_file:
        return [[], {'display': 'none'}]

 
    application_file_path = selected_file.replace("Roofline", "Applications")
    #Read the CSV file and extract data
    data_list = read_application_csv_file(application_file_path)
    
    if not data_list:
        df = None
        #No data found or invalid file, hide the dropdown
        return [html.Div(id="application-dropdown", style={"display": "none"}), {'display': 'none'}]

    #Data found
    df = pd.DataFrame(data_list)
 
    options = [{
    'label': f"{row['Name']} ({row['Method']}) - {row['Date']} ({' '.join(filter(None, [row['ISA'], row['Precision'], (str(row['Threads']) + ' Threads' if row['Threads'] else None)]))}{' |' if any([row['ISA'], row['Precision'], row['Threads']]) else ''} AI: {row['AI']} Gflops: {row['GFLOPS']})",
    'value': f"{row['Name']}_{row['Date']}_{row['ISA']}_{row['Precision']}_{row['Threads']}_{row['AI']}_{row['GFLOPS']}"
} for index, row in df.iloc[::-1].iterrows()]

    dropdown = dcc.Dropdown(
        id='application-dropdown',
        options=options,
        multi=True,
        placeholder="Select Applications to plot..."
    )

    dropdown_container = html.Div([
        dropdown
    ], style={'width': '100%', 'padding': '10px 0'})

    return [dropdown_container, {'display': 'block'}]


@app.callback(
    Output("isa-dynamic-dropdown", "options"),
    Input("precision-dynamic-dropdown", "value"),
    Input("threads-dynamic-dropdown", "value"),
    Input("loads-dynamic-dropdown", "value"),
    Input("stores-dynamic-dropdown", "value"),
    Input("interleaved-dynamic-dropdown", "value"),
    Input("drambytes-dynamic-dropdown", "value"),
    Input("fpinst-dynamic-dropdown", "value"),
    Input("date-dynamic-dropdown", "value"),
    Input("filename", "value"),
    prevent_initial_call=True
)
def chained_callback_ISA(Precision, Threads, Loads, Stores, Interleaved, DRAMBytes, FPInst, Date, selected_file):
    if not selected_file:
        return html.Div([])
    #Read the CSV file and extract data
    _, _, _, _, data_list = read_csv_file(selected_file)
    df = pd.DataFrame(data_list)

    query_conditions = []
    if Precision:
        query_conditions.append(f"Precision == @Precision")
    if Threads:
        query_conditions.append(f"Threads == @Threads")
    if Loads:
        query_conditions.append(f"Loads == @Loads")
    if Stores:
        query_conditions.append(f"Stores == @Stores")
    if Interleaved:
        query_conditions.append(f"Interleaved == @Interleaved")
    if DRAMBytes:
        query_conditions.append(f"DRAMBytes == @DRAMBytes")
    if FPInst:
        query_conditions.append(f"FPInst == @FPInst")
    if Date:
        query_conditions.append(f"Date == @Date")

    if query_conditions:
        query = " and ".join(query_conditions)
        df = df.query(query)

    return [{'label': precision, 'value': precision} for precision in sorted(df['ISA'].unique())]

@app.callback(
    Output("precision-dynamic-dropdown", "options"),
    Input("isa-dynamic-dropdown", "value"),
    Input("threads-dynamic-dropdown", "value"),
    Input("loads-dynamic-dropdown", "value"),
    Input("stores-dynamic-dropdown", "value"),
    Input("interleaved-dynamic-dropdown", "value"),
    Input("drambytes-dynamic-dropdown", "value"),
    Input("fpinst-dynamic-dropdown", "value"),
    Input("date-dynamic-dropdown", "value"),
    Input("filename", "value"),
    prevent_initial_call=True
)
def chained_callback_Precision(ISA, Threads, Loads, Stores, Interleaved, DRAMBytes, FPInst, Date, selected_file):
    if not selected_file:
        return []

    #Read the CSV file and extract data
    _, _, _, _, data_list = read_csv_file(selected_file)
    df = pd.DataFrame(data_list)
    if df.empty:
        return []

    query_conditions = []
    if ISA:
        query_conditions.append(f"ISA == @ISA")
    if Threads:
        query_conditions.append(f"Threads == @Threads")
    if Loads:
        query_conditions.append(f"Loads == @Loads")
    if Stores:
        query_conditions.append(f"Stores == @Stores")
    if Interleaved:
        query_conditions.append(f"Interleaved == @Interleaved")
    if DRAMBytes:
        query_conditions.append(f"DRAMBytes == @DRAMBytes")
    if FPInst:
        query_conditions.append(f"FPInst == @FPInst")
    if Date:
        query_conditions.append(f"Date == @Date")

    if query_conditions:
        query = " and ".join(query_conditions)
        df = df.query(query)

    return [{'label': precision, 'value': precision} for precision in sorted(df['Precision'].unique())]

@app.callback(
    Output("threads-dynamic-dropdown", "options"),
    Input("isa-dynamic-dropdown", "value"),
    Input("precision-dynamic-dropdown", "value"),
    Input("loads-dynamic-dropdown", "value"),
    Input("stores-dynamic-dropdown", "value"),
    Input("interleaved-dynamic-dropdown", "value"),
    Input("drambytes-dynamic-dropdown", "value"),
    Input("fpinst-dynamic-dropdown", "value"),
    Input("date-dynamic-dropdown", "value"),
    Input("filename", "value"),
    prevent_initial_call=True
)
def chained_callback_Threads(ISA, Precision, Loads, Stores, Interleaved, DRAMBytes, FPInst, Date, selected_file):
    if not selected_file:
        return html.Div([])
    #Read the CSV file and extract data
    _, _, _, _, data_list = read_csv_file(selected_file)
    df = pd.DataFrame(data_list)

    query_conditions = []
    if ISA:
        query_conditions.append(f"ISA == @ISA")
    if Precision:
        query_conditions.append(f"Precision == @Precision")
    if Loads:
        query_conditions.append(f"Loads == @Loads")
    if Stores:
        query_conditions.append(f"Stores == @Stores")
    if Interleaved:
        query_conditions.append(f"Interleaved == @Interleaved")
    if DRAMBytes:
        query_conditions.append(f"DRAMBytes == @DRAMBytes")
    if FPInst:
        query_conditions.append(f"FPInst == @FPInst")
    if Date:
        query_conditions.append(f"Date == @Date")

    if query_conditions:
        query = " and ".join(query_conditions)
        df = df.query(query)

    return [{'label': precision, 'value': precision} for precision in sorted(df['Threads'].unique())]

@app.callback(
    Output("loads-dynamic-dropdown", "options"),
    Input("isa-dynamic-dropdown", "value"),
    Input("precision-dynamic-dropdown", "value"),
    Input("threads-dynamic-dropdown", "value"),
    Input("stores-dynamic-dropdown", "value"),
    Input("interleaved-dynamic-dropdown", "value"),
    Input("drambytes-dynamic-dropdown", "value"),
    Input("fpinst-dynamic-dropdown", "value"),
    Input("date-dynamic-dropdown", "value"),
    Input("filename", "value"),
    prevent_initial_call=True
)
def chained_callback_Loads(ISA, Precision, Threads, Stores, Interleaved, DRAMBytes, FPInst, Date, selected_file):
    if not selected_file:
        return html.Div([])
    #Read the CSV file and extract data
    _, _, _, _, data_list = read_csv_file(selected_file)
    df = pd.DataFrame(data_list)

    query_conditions = []
    if ISA:
        query_conditions.append(f"ISA == @ISA")
    if Precision:
        query_conditions.append(f"Precision == @Precision")
    if Threads:
        query_conditions.append(f"Threads == @Threads")
    if Stores:
        query_conditions.append(f"Stores == @Stores")
    if Interleaved:
        query_conditions.append(f"Interleaved == @Interleaved")
    if DRAMBytes:
        query_conditions.append(f"DRAMBytes == @DRAMBytes")
    if FPInst:
        query_conditions.append(f"FPInst == @FPInst")
    if Date:
        query_conditions.append(f"Date == @Date")

    if query_conditions:
        query = " and ".join(query_conditions)
        df = df.query(query)

    return [{'label': precision, 'value': precision} for precision in sorted(df['Loads'].unique())]

@app.callback(
    Output("stores-dynamic-dropdown", "options"),
    Input("isa-dynamic-dropdown", "value"),
    Input("precision-dynamic-dropdown", "value"),
    Input("threads-dynamic-dropdown", "value"),
    Input("loads-dynamic-dropdown", "value"),
    Input("interleaved-dynamic-dropdown", "value"),
    Input("drambytes-dynamic-dropdown", "value"),
    Input("fpinst-dynamic-dropdown", "value"),
    Input("date-dynamic-dropdown", "value"),
    Input("filename", "value"),
    prevent_initial_call=True
)
def chained_callback_Stores(ISA, Precision, Threads, Loads, Interleaved, DRAMBytes, FPInst, Date, selected_file):
    if not selected_file:
        return html.Div([])
    #Read the CSV file and extract data
    _, _, _, _, data_list = read_csv_file(selected_file)
    df = pd.DataFrame(data_list)

    query_conditions = []
    if ISA:
        query_conditions.append(f"ISA == @ISA")
    if Precision:
        query_conditions.append(f"Precision == @Precision")
    if Threads:
        query_conditions.append(f"Threads == @Threads")
    if Loads:
        query_conditions.append(f"Loads == @Loads")
    if Interleaved:
        query_conditions.append(f"Interleaved == @Interleaved")
    if DRAMBytes:
        query_conditions.append(f"DRAMBytes == @DRAMBytes")
    if FPInst:
        query_conditions.append(f"FPInst == @FPInst")
    if Date:
        query_conditions.append(f"Date == @Date")

    if query_conditions:
        query = " and ".join(query_conditions)
        df = df.query(query)

    return [{'label': precision, 'value': precision} for precision in sorted(df['Stores'].unique())]

@app.callback(
    Output("interleaved-dynamic-dropdown", "options"),
    Input("isa-dynamic-dropdown", "value"),
    Input("precision-dynamic-dropdown", "value"),
    Input("threads-dynamic-dropdown", "value"),
    Input("loads-dynamic-dropdown", "value"),
    Input("stores-dynamic-dropdown", "value"),
    Input("drambytes-dynamic-dropdown", "value"),
    Input("fpinst-dynamic-dropdown", "value"),
    Input("date-dynamic-dropdown", "value"),
    Input("filename", "value"),
    prevent_initial_call=True
)
def chained_callback_Interleaved(ISA, Precision, Threads, Loads, Stores, DRAMBytes, FPInst, Date, selected_file):
    if not selected_file:
        return html.Div([])
    #Read the CSV file and extract data
    _, _, _, _, data_list = read_csv_file(selected_file)
    df = pd.DataFrame(data_list)

    query_conditions = []
    if ISA:
        query_conditions.append(f"ISA == @ISA")
    if Precision:
        query_conditions.append(f"Precision == @Precision")
    if Threads:
        query_conditions.append(f"Threads == @Threads")
    if Loads:
        query_conditions.append(f"Loads == @Loads")
    if Stores:
        query_conditions.append(f"Stores == @Stores")
    if DRAMBytes:
        query_conditions.append(f"DRAMBytes == @DRAMBytes")
    if FPInst:
        query_conditions.append(f"FPInst == @FPInst")
    if Date:
        query_conditions.append(f"Date == @Date")

    if query_conditions:
        query = " and ".join(query_conditions)
        df = df.query(query)

    return [{'label': precision, 'value': precision} for precision in sorted(df['Interleaved'].unique())]

@app.callback(
    Output("drambytes-dynamic-dropdown", "options"),
    Input("isa-dynamic-dropdown", "value"),
    Input("precision-dynamic-dropdown", "value"),
    Input("threads-dynamic-dropdown", "value"),
    Input("loads-dynamic-dropdown", "value"),
    Input("stores-dynamic-dropdown", "value"),
    Input("interleaved-dynamic-dropdown", "value"),
    Input("fpinst-dynamic-dropdown", "value"),
    Input("date-dynamic-dropdown", "value"),
    Input("filename", "value"),
    prevent_initial_call=True
)
def chained_callback_DRAMBytes(ISA, Precision, Threads, Loads, Stores, Interleaved, FPInst, Date, selected_file):
    if not selected_file:
        return html.Div([])
    #Read the CSV file and extract data
    _, _, _, _, data_list = read_csv_file(selected_file)
    df = pd.DataFrame(data_list)

    query_conditions = []
    if ISA:
        query_conditions.append(f"ISA == @ISA")
    if Precision:
        query_conditions.append(f"Precision == @Precision")
    if Threads:
        query_conditions.append(f"Threads == @Threads")
    if Loads:
        query_conditions.append(f"Loads == @Loads")
    if Stores:
        query_conditions.append(f"Stores == @Stores")
    if Interleaved:
        query_conditions.append(f"Interleaved == @Interleaved")
    if FPInst:
        query_conditions.append(f"FPInst == @FPInst")
    if Date:
        query_conditions.append(f"Date == @Date")

    if query_conditions:
        query = " and ".join(query_conditions)
        df = df.query(query)

    return [{'label': precision, 'value': precision} for precision in sorted(df['DRAMBytes'].unique())]

@app.callback(
    Output("fpinst-dynamic-dropdown", "options"),
    Input("isa-dynamic-dropdown", "value"),
    Input("precision-dynamic-dropdown", "value"),
    Input("threads-dynamic-dropdown", "value"),
    Input("loads-dynamic-dropdown", "value"),
    Input("stores-dynamic-dropdown", "value"),
    Input("interleaved-dynamic-dropdown", "value"),
    Input("drambytes-dynamic-dropdown", "value"),
    Input("date-dynamic-dropdown", "value"),
    Input("filename", "value"),
    prevent_initial_call=True
)
def chained_callback_FPInst(ISA, Precision, Threads, Loads, Stores, Interleaved, DRAMBytes, Date, selected_file):
    if not selected_file:
        return html.Div([])
    #Read the CSV file and extract data
    _, _, _, _, data_list = read_csv_file(selected_file)
    df = pd.DataFrame(data_list)

    query_conditions = []
    if ISA:
        query_conditions.append(f"ISA == @ISA")
    if Precision:
        query_conditions.append(f"Precision == @Precision")
    if Threads:
        query_conditions.append(f"Threads == @Threads")
    if Loads:
        query_conditions.append(f"Loads == @Loads")
    if Stores:
        query_conditions.append(f"Stores == @Stores")
    if Interleaved:
        query_conditions.append(f"Interleaved == @Interleaved")
    if DRAMBytes:
        query_conditions.append(f"DRAMBytes == @DRAMBytes")
    if Date:
        query_conditions.append(f"Date == @Date")

    if query_conditions:
        query = " and ".join(query_conditions)
        df = df.query(query)

    return [{'label': precision, 'value': precision} for precision in sorted(df['FPInst'].unique())]

@app.callback(
    Output("date-dynamic-dropdown", "options"),
    Input("isa-dynamic-dropdown", "value"),
    Input("precision-dynamic-dropdown", "value"),
    Input("threads-dynamic-dropdown", "value"),
    Input("loads-dynamic-dropdown", "value"),
    Input("stores-dynamic-dropdown", "value"),
    Input("interleaved-dynamic-dropdown", "value"),
    Input("drambytes-dynamic-dropdown", "value"),
    Input("fpinst-dynamic-dropdown", "value"),
    Input("filename", "value"),
    prevent_initial_call=True
)
def chained_callback_Date(ISA, Precision, Threads, Loads, Stores, Interleaved, DRAMBytes, FPInst, selected_file):
    if not selected_file:
        return html.Div([])
    #Read the CSV file and extract data
    _, _, _, _, data_list = read_csv_file(selected_file)
    df = pd.DataFrame(data_list)

    query_conditions = []
    if ISA:
        query_conditions.append(f"ISA == @ISA")
    if Precision:
        query_conditions.append(f"Precision == @Precision")
    if Threads:
        query_conditions.append(f"Threads == @Threads")
    if Loads:
        query_conditions.append(f"Loads == @Loads")
    if Stores:
        query_conditions.append(f"Stores == @Stores")
    if Interleaved:
        query_conditions.append(f"Interleaved == @Interleaved")
    if DRAMBytes:
        query_conditions.append(f"DRAMBytes == @DRAMBytes")
    if FPInst:
        query_conditions.append(f"FPInst == @FPInst")

    if query_conditions:
        query = " and ".join(query_conditions)
        df = df.query(query)

    return [{'label': precision, 'value': precision} for precision in sorted(df['Date'].unique(), reverse=True)]

@app.callback(
    Output("isa-dynamic-dropdown2", "options"),
    Input("precision-dynamic-dropdown2", "value"),
    Input("threads-dynamic-dropdown2", "value"),
    Input("loads-dynamic-dropdown2", "value"),
    Input("stores-dynamic-dropdown2", "value"),
    Input("interleaved-dynamic-dropdown2", "value"),
    Input("drambytes-dynamic-dropdown2", "value"),
    Input("fpinst-dynamic-dropdown2", "value"),
    Input("date-dynamic-dropdown2", "value"),
    Input("filename", "value"),
    prevent_initial_call=True
)
def chained_callback_ISA2(Precision, Threads, Loads, Stores, Interleaved, DRAMBytes, FPInst, Date, selected_file):
    if not selected_file:
        return html.Div([])
    #Read the CSV file and extract data
    _, _, _, _, data_list = read_csv_file(selected_file)
    df = pd.DataFrame(data_list)

    query_conditions = []
    if Precision:
        query_conditions.append(f"Precision == @Precision")
    if Threads:
        query_conditions.append(f"Threads == @Threads")
    if Loads:
        query_conditions.append(f"Loads == @Loads")
    if Stores:
        query_conditions.append(f"Stores == @Stores")
    if Interleaved:
        query_conditions.append(f"Interleaved == @Interleaved")
    if DRAMBytes:
        query_conditions.append(f"DRAMBytes == @DRAMBytes")
    if FPInst:
        query_conditions.append(f"FPInst == @FPInst")
    if Date:
        query_conditions.append(f"Date == @Date")

    if query_conditions:
        query = " and ".join(query_conditions)
        df = df.query(query)

    return [{'label': precision, 'value': precision} for precision in sorted(df['ISA'].unique())]

@app.callback(
    Output("precision-dynamic-dropdown2", "options"),
    Input("isa-dynamic-dropdown2", "value"),
    Input("threads-dynamic-dropdown2", "value"),
    Input("loads-dynamic-dropdown2", "value"),
    Input("stores-dynamic-dropdown2", "value"),
    Input("interleaved-dynamic-dropdown2", "value"),
    Input("drambytes-dynamic-dropdown2", "value"),
    Input("fpinst-dynamic-dropdown2", "value"),
    Input("date-dynamic-dropdown2", "value"),
    Input("filename", "value"),
    prevent_initial_call=True
)
def chained_callback_Precision2(ISA, Threads, Loads, Stores, Interleaved, DRAMBytes, FPInst, Date, selected_file):
    if not selected_file:
        return []

    #Read the CSV file and extract data
    _, _, _, _, data_list = read_csv_file(selected_file)
    df = pd.DataFrame(data_list)
    if df.empty:
        return []

    query_conditions = []
    if ISA:
        query_conditions.append(f"ISA == @ISA")
    if Threads:
        query_conditions.append(f"Threads == @Threads")
    if Loads:
        query_conditions.append(f"Loads == @Loads")
    if Stores:
        query_conditions.append(f"Stores == @Stores")
    if Interleaved:
        query_conditions.append(f"Interleaved == @Interleaved")
    if DRAMBytes:
        query_conditions.append(f"DRAMBytes == @DRAMBytes")
    if FPInst:
        query_conditions.append(f"FPInst == @FPInst")
    if Date:
        query_conditions.append(f"Date == @Date")

    if query_conditions:
        query = " and ".join(query_conditions)
        df = df.query(query)

    return [{'label': precision, 'value': precision} for precision in sorted(df['Precision'].unique())]

@app.callback(
    Output("threads-dynamic-dropdown2", "options"),
    Input("isa-dynamic-dropdown2", "value"),
    Input("precision-dynamic-dropdown2", "value"),
    Input("loads-dynamic-dropdown2", "value"),
    Input("stores-dynamic-dropdown2", "value"),
    Input("interleaved-dynamic-dropdown2", "value"),
    Input("drambytes-dynamic-dropdown2", "value"),
    Input("fpinst-dynamic-dropdown2", "value"),
    Input("date-dynamic-dropdown2", "value"),
    Input("filename", "value"),
    prevent_initial_call=True
)
def chained_callback_Threads2(ISA, Precision, Loads, Stores, Interleaved, DRAMBytes, FPInst, Date, selected_file):
    if not selected_file:
        return html.Div([])
    #Read the CSV file and extract data
    _, _, _, _, data_list = read_csv_file(selected_file)
    df = pd.DataFrame(data_list)

    query_conditions = []
    if ISA:
        query_conditions.append(f"ISA == @ISA")
    if Precision:
        query_conditions.append(f"Precision == @Precision")
    if Loads:
        query_conditions.append(f"Loads == @Loads")
    if Stores:
        query_conditions.append(f"Stores == @Stores")
    if Interleaved:
        query_conditions.append(f"Interleaved == @Interleaved")
    if DRAMBytes:
        query_conditions.append(f"DRAMBytes == @DRAMBytes")
    if FPInst:
        query_conditions.append(f"FPInst == @FPInst")
    if Date:
        query_conditions.append(f"Date == @Date")

    if query_conditions:
        query = " and ".join(query_conditions)
        df = df.query(query)

    return [{'label': precision, 'value': precision} for precision in sorted(df['Threads'].unique())]

@app.callback(
    Output("loads-dynamic-dropdown2", "options"),
    Input("isa-dynamic-dropdown2", "value"),
    Input("precision-dynamic-dropdown2", "value"),
    Input("threads-dynamic-dropdown2", "value"),
    Input("stores-dynamic-dropdown2", "value"),
    Input("interleaved-dynamic-dropdown2", "value"),
    Input("drambytes-dynamic-dropdown2", "value"),
    Input("fpinst-dynamic-dropdown2", "value"),
    Input("date-dynamic-dropdown2", "value"),
    Input("filename", "value"),
    prevent_initial_call=True
)
def chained_callback_Loads2(ISA, Precision, Threads, Stores, Interleaved, DRAMBytes, FPInst, Date, selected_file):
    if not selected_file:
        return html.Div([])
    #Read the CSV file and extract data
    _, _, _, _, data_list = read_csv_file(selected_file)
    df = pd.DataFrame(data_list)

    query_conditions = []
    if ISA:
        query_conditions.append(f"ISA == @ISA")
    if Precision:
        query_conditions.append(f"Precision == @Precision")
    if Threads:
        query_conditions.append(f"Threads == @Threads")
    if Stores:
        query_conditions.append(f"Stores == @Stores")
    if Interleaved:
        query_conditions.append(f"Interleaved == @Interleaved")
    if DRAMBytes:
        query_conditions.append(f"DRAMBytes == @DRAMBytes")
    if FPInst:
        query_conditions.append(f"FPInst == @FPInst")
    if Date:
        query_conditions.append(f"Date == @Date")

    if query_conditions:
        query = " and ".join(query_conditions)
        df = df.query(query)

    return [{'label': precision, 'value': precision} for precision in sorted(df['Loads'].unique())]

@app.callback(
    Output("stores-dynamic-dropdown2", "options"),
    Input("isa-dynamic-dropdown2", "value"),
    Input("precision-dynamic-dropdown2", "value"),
    Input("threads-dynamic-dropdown2", "value"),
    Input("loads-dynamic-dropdown2", "value"),
    Input("interleaved-dynamic-dropdown2", "value"),
    Input("drambytes-dynamic-dropdown2", "value"),
    Input("fpinst-dynamic-dropdown2", "value"),
    Input("date-dynamic-dropdown2", "value"),
    Input("filename", "value"),
    prevent_initial_call=True
)
def chained_callback_Stores2(ISA, Precision, Threads, Loads, Interleaved, DRAMBytes, FPInst, Date, selected_file):
    if not selected_file:
        return html.Div([])
    #Read the CSV file and extract data
    _, _, _, _, data_list = read_csv_file(selected_file)
    df = pd.DataFrame(data_list)

    query_conditions = []
    if ISA:
        query_conditions.append(f"ISA == @ISA")
    if Precision:
        query_conditions.append(f"Precision == @Precision")
    if Threads:
        query_conditions.append(f"Threads == @Threads")
    if Loads:
        query_conditions.append(f"Loads == @Loads")
    if Interleaved:
        query_conditions.append(f"Interleaved == @Interleaved")
    if DRAMBytes:
        query_conditions.append(f"DRAMBytes == @DRAMBytes")
    if FPInst:
        query_conditions.append(f"FPInst == @FPInst")
    if Date:
        query_conditions.append(f"Date == @Date")

    if query_conditions:
        query = " and ".join(query_conditions)
        df = df.query(query)

    return [{'label': precision, 'value': precision} for precision in sorted(df['Stores'].unique())]

@app.callback(
    Output("interleaved-dynamic-dropdown2", "options"),
    Input("isa-dynamic-dropdown2", "value"),
    Input("precision-dynamic-dropdown2", "value"),
    Input("threads-dynamic-dropdown2", "value"),
    Input("loads-dynamic-dropdown2", "value"),
    Input("stores-dynamic-dropdown2", "value"),
    Input("drambytes-dynamic-dropdown2", "value"),
    Input("fpinst-dynamic-dropdown2", "value"),
    Input("date-dynamic-dropdown2", "value"),
    Input("filename", "value"),
    prevent_initial_call=True
)
def chained_callback_Interleaved2(ISA, Precision, Threads, Loads, Stores, DRAMBytes, FPInst, Date, selected_file):
    if not selected_file:
        return html.Div([])
    #Read the CSV file and extract data
    _, _, _, _, data_list = read_csv_file(selected_file)
    df = pd.DataFrame(data_list)

    query_conditions = []
    if ISA:
        query_conditions.append(f"ISA == @ISA")
    if Precision:
        query_conditions.append(f"Precision == @Precision")
    if Threads:
        query_conditions.append(f"Threads == @Threads")
    if Loads:
        query_conditions.append(f"Loads == @Loads")
    if Stores:
        query_conditions.append(f"Stores == @Stores")
    if DRAMBytes:
        query_conditions.append(f"DRAMBytes == @DRAMBytes")
    if FPInst:
        query_conditions.append(f"FPInst == @FPInst")
    if Date:
        query_conditions.append(f"Date == @Date")

    if query_conditions:
        query = " and ".join(query_conditions)
        df = df.query(query)

    return [{'label': precision, 'value': precision} for precision in sorted(df['Interleaved'].unique())]

@app.callback(
    Output("drambytes-dynamic-dropdown2", "options"),
    Input("isa-dynamic-dropdown2", "value"),
    Input("precision-dynamic-dropdown2", "value"),
    Input("threads-dynamic-dropdown2", "value"),
    Input("loads-dynamic-dropdown2", "value"),
    Input("stores-dynamic-dropdown2", "value"),
    Input("interleaved-dynamic-dropdown2", "value"),
    Input("fpinst-dynamic-dropdown2", "value"),
    Input("date-dynamic-dropdown2", "value"),
    Input("filename", "value"),
    prevent_initial_call=True
)
def chained_callback_DRAMBytes2(ISA, Precision, Threads, Loads, Stores, Interleaved, FPInst, Date, selected_file):
    if not selected_file:
        return html.Div([])
    #Read the CSV file and extract data
    _, _, _, _, data_list = read_csv_file(selected_file)
    df = pd.DataFrame(data_list)

    query_conditions = []
    if ISA:
        query_conditions.append(f"ISA == @ISA")
    if Precision:
        query_conditions.append(f"Precision == @Precision")
    if Threads:
        query_conditions.append(f"Threads == @Threads")
    if Loads:
        query_conditions.append(f"Loads == @Loads")
    if Stores:
        query_conditions.append(f"Stores == @Stores")
    if Interleaved:
        query_conditions.append(f"Interleaved == @Interleaved")
    if FPInst:
        query_conditions.append(f"FPInst == @FPInst")
    if Date:
        query_conditions.append(f"Date == @Date")

    if query_conditions:
        query = " and ".join(query_conditions)
        df = df.query(query)

    return [{'label': precision, 'value': precision} for precision in sorted(df['DRAMBytes'].unique())]

@app.callback(
    Output("fpinst-dynamic-dropdown2", "options"),
    Input("isa-dynamic-dropdown2", "value"),
    Input("precision-dynamic-dropdown2", "value"),
    Input("threads-dynamic-dropdown2", "value"),
    Input("loads-dynamic-dropdown2", "value"),
    Input("stores-dynamic-dropdown2", "value"),
    Input("interleaved-dynamic-dropdown2", "value"),
    Input("drambytes-dynamic-dropdown2", "value"),
    Input("date-dynamic-dropdown2", "value"),
    Input("filename", "value"),
    prevent_initial_call=True
)
def chained_callback_FPInst2(ISA, Precision, Threads, Loads, Stores, Interleaved, DRAMBytes, Date, selected_file):
    if not selected_file:
        return html.Div([])
    #Read the CSV file and extract data
    _, _, _, _, data_list = read_csv_file(selected_file)
    df = pd.DataFrame(data_list)

    query_conditions = []
    if ISA:
        query_conditions.append(f"ISA == @ISA")
    if Precision:
        query_conditions.append(f"Precision == @Precision")
    if Threads:
        query_conditions.append(f"Threads == @Threads")
    if Loads:
        query_conditions.append(f"Loads == @Loads")
    if Stores:
        query_conditions.append(f"Stores == @Stores")
    if Interleaved:
        query_conditions.append(f"Interleaved == @Interleaved")
    if DRAMBytes:
        query_conditions.append(f"DRAMBytes == @DRAMBytes")
    if Date:
        query_conditions.append(f"Date == @Date")

    if query_conditions:
        query = " and ".join(query_conditions)
        df = df.query(query)

    return [{'label': precision, 'value': precision} for precision in sorted(df['FPInst'].unique())]

@app.callback(
    Output("date-dynamic-dropdown2", "options"),
    Input("isa-dynamic-dropdown2", "value"),
    Input("precision-dynamic-dropdown2", "value"),
    Input("threads-dynamic-dropdown2", "value"),
    Input("loads-dynamic-dropdown2", "value"),
    Input("stores-dynamic-dropdown2", "value"),
    Input("interleaved-dynamic-dropdown2", "value"),
    Input("drambytes-dynamic-dropdown2", "value"),
    Input("fpinst-dynamic-dropdown2", "value"),
    Input("filename", "value"),
    prevent_initial_call=True
)
def chained_callback_Date2(ISA, Precision, Threads, Loads, Stores, Interleaved, DRAMBytes, FPInst, selected_file):
    if not selected_file:
        return html.Div([])
    #Read the CSV file and extract data
    _, _, _, _, data_list = read_csv_file(selected_file)
    df = pd.DataFrame(data_list)

    query_conditions = []
    if ISA:
        query_conditions.append(f"ISA == @ISA")
    if Precision:
        query_conditions.append(f"Precision == @Precision")
    if Threads:
        query_conditions.append(f"Threads == @Threads")
    if Loads:
        query_conditions.append(f"Loads == @Loads")
    if Stores:
        query_conditions.append(f"Stores == @Stores")
    if Interleaved:
        query_conditions.append(f"Interleaved == @Interleaved")
    if DRAMBytes:
        query_conditions.append(f"DRAMBytes == @DRAMBytes")
    if FPInst:
        query_conditions.append(f"FPInst == @FPInst")

    if query_conditions:
        query = " and ".join(query_conditions)
        df = df.query(query)

    return [{'label': precision, 'value': precision} for precision in sorted(df['Date'].unique(), reverse=True)]


@app.callback(
    [
    Output(component_id='graphs', component_property='figure'),
    Output('graphs', 'style'),
    ],
    [
    Input("isa-dynamic-dropdown", "value"),
    Input("precision-dynamic-dropdown", "value"),
    Input("threads-dynamic-dropdown", "value"),
    Input("loads-dynamic-dropdown", "value"),
    Input("stores-dynamic-dropdown", "value"),
    Input("interleaved-dynamic-dropdown", "value"),
    Input("drambytes-dynamic-dropdown", "value"),
    Input("fpinst-dynamic-dropdown", "value"),
    Input("date-dynamic-dropdown", "value"),
    Input("isa-dynamic-dropdown2", "value"),
    Input("precision-dynamic-dropdown2", "value"),
    Input("threads-dynamic-dropdown2", "value"),
    Input("loads-dynamic-dropdown2", "value"),
    Input("stores-dynamic-dropdown2", "value"),
    Input("interleaved-dynamic-dropdown2", "value"),
    Input("drambytes-dynamic-dropdown2", "value"),
    Input("fpinst-dynamic-dropdown2", "value"),
    Input("date-dynamic-dropdown2", "value"),
    Input("filename", "value"),
    Input("application-dropdown", "value")
    ],
    prevent_initial_call=True
)
def analysis(ISA, Precision, Threads, Loads, Stores, Interleaved, DRAMBytes, FPInst, Date,
             ISA2, Precision2, Threads2, Loads2, Stores2, Interleaved2, DRAMBytes2, FPInst2, Date2, selected_file, selected_applications):
    if not selected_file:
        return go.Figure(), {'display': 'none'}

    #Read data and create DataFrame
    _, _, _, _, data_list = read_csv_file(selected_file)
    df = pd.DataFrame(data_list)


    #Get queries for both sets of inputs
    query1 = construct_query(ISA, Precision, Threads, Loads, Stores, Interleaved, DRAMBytes, FPInst, Date)
    query2 = construct_query(ISA2, Precision2, Threads2, Loads2, Stores2, Interleaved2, DRAMBytes2, FPInst2, Date2)

    #Filter data based on queries
    filtered_df1 = df.query(query1) if query1 else df
    filtered_df2 = df.query(query2) if query2 and any([ISA2, Precision2, Threads2, Loads2, Stores2, Interleaved2, DRAMBytes2, FPInst2, Date2]) else pd.DataFrame()


    #Generate traces for both datasets
    figure = go.Figure()
    if not filtered_df1.empty:
        values1 = filtered_df1.iloc[-1][['L1', 'L2', 'L3', 'DRAM', 'FP', 'FP_FMA', 'FPInst']].tolist()
        figure.add_traces(plot_roofline(values1, ''))
    if not filtered_df2.empty and not query2 == None:
        values2 = filtered_df2.iloc[-1][['L1', 'L2', 'L3', 'DRAM', 'FP', 'FP_FMA', 'FPInst']].tolist()
        figure.add_traces(plot_roofline(values2, '2'))

    #Plot the selected application as a dot
    if selected_applications:
        for selected_application in selected_applications:
            print(selected_application)
            #parts = selected_application.split('_')
            parts = selected_application.rsplit('_', 6)
            if len(parts) >= 7:
                name, date, isa, precision, threads, ai, gflops = parts[0], parts[1], parts[2], parts[3], parts[4], float(parts[5]), float(parts[6])
                #Add trace for each application
                figure.add_trace(go.Scatter(
                    x=[ai], 
                    y=[gflops], 
                    mode='markers', 
                    name=f'{name} ({date})', 
                    marker=dict(size=10)
                ))

    figure.update_layout(
    title={
        'text': 'Cache Aware Roofline Model',
        'y': 0.95,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {
            'family': "Arial",
            'size': 20,
            'color': "black"
        }
    },
    xaxis={
        'title': 'AI (Arithmetic Intensity)',
        'type': 'log',
        'dtick': '0.30102999566',
        'range': [np.log(0.18), np.log(11.2)],
        'title_standoff': 0,
        'automargin': True
    },
    yaxis={
        'title': 'Performance (GFLOP/s)',
        'type': 'log',
        'dtick': '0.30102999566',
        'range': [np.log(0.25), np.log(25)],
        'title_standoff': 0,
        'automargin': True
    },
    legend={
        'font': {'size': 12},
        'orientation': 'h',
        'x': 0.5,
        'y': -0.1,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    font={'size': 18},
    showlegend=True,
    height=675,
    width=1900,
    margin=dict(t=60, l=80, b=20, r=40),
    plot_bgcolor='#e9ecef',
    paper_bgcolor='#e9ecef'
)

    return figure, {'display': 'block', 'width': '100%'}


def construct_query(ISA, Precision, Threads, Loads, Stores, Interleaved, DRAMBytes, FPInst, Date):
    query_parts = []
    if ISA:
        query_parts.append(f"ISA == '{ISA}'")
    if Precision:
        query_parts.append(f"Precision == '{Precision}'")
    if Threads:
        query_parts.append(f"Threads == {Threads}")
    if Loads:
        query_parts.append(f"Loads == {Loads}")
    if Stores:
        query_parts.append(f"Stores == {Stores}")
    if Interleaved:
        query_parts.append(f"Interleaved == '{Interleaved}'")
    if DRAMBytes:
        query_parts.append(f"DRAMBytes == {DRAMBytes}")
    if FPInst:
        query_parts.append(f"FPInst == '{FPInst}'")
    if Date:
        query_parts.append(f"Date == '{Date}'")

    return " and ".join(query_parts) if query_parts else None
    
def plot_roofline(values, name_suffix):
    ai = np.linspace(0.00390625, 256, num=200000)
    traces = []
    cache_levels = ['L1', 'L2', 'L3', 'DRAM']
    if name_suffix == "":
        colors = ['black', 'black', 'black', 'black']
        color_inst = 'black'
    else:
        colors = ['red', 'red', 'red', 'red']
        color_inst = 'red'
    linestyles = ['solid', 'solid', 'dash', 'dot']

    for cache_level, color, linestyle in zip(cache_levels, colors, linestyles):
        if values[cache_levels.index(cache_level)] > 0:
            y_values = run.carm_eq(ai, values[cache_levels.index(cache_level)], values[5])
            trace = go.Scatter(
                x=ai, y=y_values,
                mode='lines',
                line=dict(color=color, dash=linestyle),
                name=f'{cache_level}'
            )
            traces.append(trace)

    for i in range(4):
        if values[i]:
            top_roof = values[i]
            break

    y_values = run.carm_eq(ai, top_roof, values[4])
    trace_inst = go.Scatter(
        x=ai, y=y_values,
        mode='lines',
        line=dict(color=color_inst, dash="dashdot"),
        name=values[6]
    )
    traces.append(trace_inst)
    
    return traces

if __name__ == '__main__':
    app.run_server(debug=False)
