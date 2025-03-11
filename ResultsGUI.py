import os
import csv
import platform
import datetime
import logging
import math
import copy
from copy import deepcopy

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
from dash import Input, Output, State, html, ALL, dcc, no_update, DiskcacheManager, callback_context
import dash_daq as daq

#Local Python Scripts
import run
import PMU_AI_Calculator
import DBI_AI_Calculator
import GUI_utils as gut
import utils as ut

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)


pathway = './carm_results/roofline'
lines_origin = {}
lines_origin2 = {}

#Default graph values
DEFAULTS = {
    'graph-width': 1900,
    'graph-height': 690,
    'line-size': 3,
    'dot-size': 10,
    'title-size': 20,
    'axis-size': 20,
    'legend-size': 13,
    'tick-size': 18,
    'annotation-size': 12,
    'tooltip-size': 14,
}

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
        {"label": "SVE", "value": "sve"},
        {"label": "NEON", "value": "neon"},
        {"label": "Scalar", "value": "scalar"}
    ]
elif CPU_Type == "riscv64":
    isa_options = [
        {"label": "RVV1.0", "value": "rvv1.0"},
        {"label": "RVV0.7", "value": "rvv0.7"},
        {"label": "Scalar", "value": "scalar"}
    ]
else:
    isa_options = []

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
        dbc.Button("Stop Benchmark/Analysis", id="cancel-button", className="mb-2", style={'width': '100%'}),
    ], style={'backgroundColor': '#1a1a1a'}),
    id="offcanvas",
    title=html.H5("CARM Tool Functions", style={'color': 'white', 'fontsize': '30px'}),
    is_open=False,
    style={'backgroundColor': '#1a1a1a'},
)

sidebar2 = dbc.Offcanvas(
    html.Div([
        html.P("Graph Customization", className="mb-2", 
            style={
                'color': 'white',
                'textAlign': 'center',
                'fontSize': '20px'
            }),
        dbc.Card(
        dbc.CardBody([
            html.Div([
                    dbc.Label("Use Exponent Notation", html_for="exponent-switch", 
                            style={"marginRight": "40px"}),
                    dbc.Switch(
                        id="exponent-switch",
                        label="",
                        value=True
                    )
                    ],
                    style={"display": "flex", "alignItems": "center"}
                ),
                html.Div([
                    dbc.Label("Show Lines Legend", html_for="line-legend-switch", 
                            style={"marginRight": "70px"}),
                    dbc.Switch(
                        id="line-legend-switch",
                        label="",
                        value=True
                    )
                    ],
                    style={"display": "flex", "alignItems": "left"}
                ),
                html.Div([
                    dbc.Label("Detailed Lines Legend", html_for="detail-legend-switch", 
                            style={"marginRight": "48px"}),
                    dbc.Switch(
                        id="detail-legend-switch",
                        label="",
                        value=False
                    )
                    ],
                    style={"display": "flex", "alignItems": "left"}
                ),
        
        ]),
        style={'backgroundColor': 'white', "alignItems": "center"},
        className="mb-2"
    ),
    dbc.Accordion([
            dbc.AccordionItem([
                dbc.Row(
                [
                    html.P("Graph Width:", className="mb-1 text-center", style={'color': 'black', 'margin-right': '10px', 'flex': '1'}),
                    html.P("Graph Height:", className="mb-1 text-center", style={'color': 'black', 'margin-right': '10px', 'flex': '1'}),
                ], justify="center", align="center"),
                dbc.Row(
                [
                    dcc.Input(id='graph-width', type='number', className="mb-2 text-center", min=10, max=5000,step=10, value=DEFAULTS['graph-width'], style={'width': 120, 'margin-right': '55px'}),
                    dcc.Input(id='graph-height', type='number', className="mb-2 text-center", min=10, max=5000,step=10, value=DEFAULTS['graph-height'], style={'width': 120,  'margin-right': '5px'}),
                ], justify="center", align="center"),
                dbc.Row(
                [
                    html.P("Lines Width:", className="mb-1 text-center", style={'color': 'black', 'margin-right': '10px', 'flex': '1'}),
                    html.P("Dots Size:", className="mb-1 text-center", style={'color': 'black', 'margin-right': '10px', 'flex': '1'}),
                ], justify="center", align="center"),
                dbc.Row(
                [
                    dcc.Input(id='line-size', type='number', min=1, className="mb-2", max=100,step=1, value=DEFAULTS['line-size'], style={'width': 120, 'margin-right': '55px'}),
                    dcc.Input(id='dot-size', type='number', min=1, className="mb-2", max=100,step=1, value=DEFAULTS['dot-size'], style={'width': 120, 'margin-right': '5px'}),
                ], justify="center", align="center"),

                dbc.Row(
                [
                    html.P("Title Font:", className="mb-1 text-center", style={'color': 'black', 'margin-right': '10px', 'flex': '1'}),
                    html.P("Axis Font:", className="mb-1 text-center", style={'color': 'black', 'margin-right': '10px', 'flex': '1'}),
                ], justify="center", align="center"),
                dbc.Row(
                [
                    dcc.Input(id='title-size', type='number', className="mb-2", min=1, max=100,step=1, value=DEFAULTS['title-size'], style={'width': 120, 'margin-right': '55px'}),    
                    dcc.Input(id='axis-size', type='number', className="mb-2", min=1, max=100,step=1, value=DEFAULTS['axis-size'], style={'width': 120, 'margin-right': '5px'}),
                ], justify="center", align="center"),
                dbc.Row(
                [
                    html.P("Legend Font:", className="mb-1 text-center", style={'color': 'black', 'margin-right': '10px', 'flex': '1'}),
                    html.P("Ticks Font:", className="mb-1 text-center", style={'color': 'black', 'margin-right': '10px', 'flex': '1'}),
                ], justify="center", align="center"),
                dbc.Row(
                [
                    dcc.Input(id='legend-size', type='number', className="mb-2", min=1, max=100,step=1, value=DEFAULTS['legend-size'], style={'width': 120, 'margin-right': '55px'}),
                    dcc.Input(id='tick-size', type='number', className="mb-2", min=1, max=100,step=1, value=DEFAULTS['tick-size'], style={'width': 120, 'margin-right': '5px'}),
                ], justify="center", align="center"),
                dbc.Row(
                [
                    html.P("Annotations:", className="mb-1 text-center", style={'color': 'black', 'margin-right': '10px', 'flex': '1'}),
                    html.P("Tooltip Font:", className="mb-1 text-center", style={'color': 'black', 'margin-right': '10px', 'flex': '1'}),
                ], justify="center", align="center"),
                dbc.Row(
                [
                    dcc.Input(id='annotation-size', type='number', className="mb-3", min=1, max=50,step=1, value=DEFAULTS['annotation-size'], style={'width': 120, 'margin-right': '55px'}),    
                    dcc.Input(id='tooltip-size', type='number', className="mb-3", min=1, max=50,step=1, value=DEFAULTS['tooltip-size'], style={'width': 120, 'margin-right': '5px'}),
                ], justify="center", align="center"),
                dbc.Button("Reset Changes", id="button-CARM-reset", className="mb-1", style={'width': '100%'}, n_clicks=1),
                        
            ], title="Change Graph/Line/Font Sizes", style={"alignItems": "center"}
            ),

        ],id='font-accordion', start_collapsed=False, always_open=True, flush=True, style={'backgroundColor': '#1a1a1a', "alignItems": "center"}, className="mb-2"
        ),
        dbc.Button("Edit Graph Text", id="button-CARM-edit", className="mb-2", style={'width': '100%'}, n_clicks=1),
        html.P("Notations Configuration", className="mb-2", 
            style={
                'color': 'white',
                'textAlign': 'center',
                'fontSize': '20px'
            }),
        html.Div([
            dbc.Accordion([], id='annotation-accordion', start_collapsed=True, always_open=True, flush=True, style={'backgroundColor': '#1a1a1a'})
        ], id='angle-inputs-container', style={'marginBottom': '15px'}),
        dbc.Button("Create Annotation", id="create-annotation-button", className="mb-2", style={'width': '100%'}),
        dbc.Button("Disable Annotations", id="disable-annotation-button", className="mb-2", style={'width': '100%'}),
    ], style={'backgroundColor': '#1a1a1a'}),
    id="offcanvas2",
    title=html.H5("Graph Options", style={'color': 'white', 'fontsize': '30px'}),
    is_open=False,
    placement= "end",
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
            style={'padding-right': '5px', 'padding-left': '5px'}
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
        dbc.Col(
            dbc.Button(
                html.Img(src="/assets/CARM_icon.svg", height="30px"),
                id="open-offcanvas2",
                n_clicks=0,
                className="btn-sm",
                style={'border': 'none', 'background': 'transparent', 'padding': '0', 'margin': '0', 'display': 'none'}
            ),
            width="auto",
            style={'padding-right': '5px', 'padding-left': '0px'}
        ),
    ],
    align="center", 
    style={'margin-top': '1px'}
    ),
    html.Div([
    dbc.Row([
        dbc.Col([
            html.Div(id='additional-dropdowns', style={'margin-top': '10px'}),
            html.Div(id='additional-dropdowns2'),
            html.Div(id='application-dropdown-container'),
            dcc.Graph(id='graphs', style={'display': 'block'}, config={'toImageButtonOptions': {'format': 'svg', 'filename': 'CARM_Tool'},
                    'editable': False,
                    'displaylogo': False,
                    'edits': {
                        'annotationPosition': True,
                    }
                }), 
            ])
        ]),
    ],
        id="slider-components",
        style={"display": "none"}
    ),
    html.Div(id='graph-size-data', style={'whiteSpace': 'pre-wrap', 'display': 'none'}),
    html.Div(id='graph-size-update', style={'whiteSpace': 'pre-wrap', 'display': 'none'}),
    dcc.Store(id='store-dimensions'),
    dcc.Store(id='graph-lines'),
    dcc.Store(id='graph-lines2'),
    dcc.Store(id='graph-values'),
    dcc.Store(id='graph-values2'),
    dcc.Store(id='graph-isa'),
    dcc.Store(id='graph-xrange'),
    dcc.Store(id='graph-yrange'),
    dcc.Store(id='change-annon'),
    dcc.Store(id='clicked-point-index', data=-1),
    dcc.Store(id="clicked-trace-index", data=-1),
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
                    style={'width': '99%', 'height': '92%','display': 'block', 'background': 'transparent', 'marginLeft': '40px'}
                ),
                href="https://github.com/champ-hub",
                target="_blank"
            ),
            
        ], width=10, style={'backgroundColor': '#e9ecef', 'textAlign': 'center'})
    ], id="initial-content", justify="center", style={'backgroundColor': '#e9ecef', 'textAlign': 'center'}),
    dcc.Store(id='machine-selected', data=False),
    sidebar,
    sidebar2,
    html.Div(id="invisible-output", style={"display": "none"}),
    html.Div(id="invisible-output2", style={"display": "none"}),
    html.Div(id="invisible-output3", style={"display": "none"}),
    html.Div(id="invisible-output-final", style={"display": "none"}),
    html.Div(id="file-path-valid", style={"display": "none"}),
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
        dbc.ModalHeader(dbc.ModalTitle("Edit Point Style", style={'text-align': 'center', 'color': 'white'}), style={'backgroundColor': '#6c757d'}),
        dbc.ModalBody([
            daq.ColorPicker(
                label=' ',
                id="dot-color-picker",
                value={"hex": "#0000FF"},  #default blue
            ),
            html.Hr(),
            dbc.Col([
                html.Div([
                    html.Label("Size:", style={"marginRight": "10px"}),  
                    dcc.Input(
                        id="dot-size-input",
                        type="number",
                        value=10,
                        min=1,
                        max=40,
                        step=1,
                        style={
                            "marginRight": "20px",
                            "width": "45px"
                        },
                    ),
                    html.Label("Shape:", style={"marginRight": "10px"}),
                    dcc.Dropdown(
                        id="dot-symbol-dropdown",
                        options=[
                            {"label": "Circle", "value": "circle"},
                            {"label": "Square", "value": "square"},
                            {"label": "Diamond", "value": "diamond"},
                            {"label": "Cross", "value": "cross"},
                            {"label": "X", "value": "x"},
                            {"label": "Triangle-Up", "value": "triangle-up"},
                            {"label": "Triangle-Down", "value": "triangle-down"},
                        ],
                        value="circle",
                        style={"width": "170px"}
                    ),
                    ],
                    style={
                        "display": "flex",
                        "alignItems": "center"
                    }
                )
                ],
                width="auto",
            ),
            ],
            style={'backgroundColor': '#e9ecef'},
            id="modal-body",
        ),
        dbc.ModalFooter([
            dbc.Button("Submit", id="dot-submit-button", className="ms-auto", n_clicks=0, style={'margin-right': 'auto'}),
            dbc.Button("Close", id="close-dot-modal", className="me-auto", n_clicks=0, style={'margin-left': 'auto'}),
            ],
            className="w-100 d-flex",
            style={'backgroundColor': '#6c757d'}
        ),
        ],
        id="point-edit-modal",
        is_open=False,
        style={"width": "auto", "centered": "true"},
    ),
    dbc.Modal([
        dbc.ModalHeader("Create a New Annotation"),
        dbc.ModalBody([
            dbc.Label("Annotation Text"),
            dbc.Input(type="text", id="annotation-text-input", placeholder="Enter annotation text"),
        ]),
        dbc.ModalFooter(
            dbc.Button("Submit", id="submit-annotation", className="ms-auto", n_clicks=0)
        ),
        ],
        id="annotation-modal",
        is_open=False,
    ),
    dcc.Store(id='annotations-store', data=[]),
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Enter Path to DynamoRIO Folder", style={'text-align': 'center', 'color': 'white'}), style={'backgroundColor': '#6c757d'}),
        dbc.ModalBody([
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
                    dbc.Row([
                        dbc.Col(html.P("Region Start Flag:", className="mb-1", style={'color': 'black', 'text-align': 'right'}), width=6, align="center"),
                        dbc.Col(dbc.Badge("//CARM ROI START", color="#6c757d", className="me-1"), width=6, align="center"),
                        ],
                        className="align-items-center"
                    ),
                    dbc.Row([
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
                dbc.CardBody([
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
        dbc.ModalFooter([
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

@app.callback(
    Output('graph-width', 'value'),
    Output('graph-height', 'value'),
    Output('line-size', 'value'),
    Output('dot-size', 'value'),
    Output('title-size', 'value'),
    Output('axis-size', 'value'),
    Output('legend-size', 'value'),
    Output('tick-size', 'value'),
    Output('annotation-size', 'value'),
    Output('tooltip-size', 'value'),
    Input('button-CARM-reset', 'n_clicks'),
    prevent_initial_call=True,
)
def reset_inputs(n_clicks):
    if n_clicks is None:
        raise PreventUpdate

    return (
        DEFAULTS['graph-width'],
        DEFAULTS['graph-height'],
        DEFAULTS['line-size'],
        DEFAULTS['dot-size'],
        DEFAULTS['title-size'],
        DEFAULTS['axis-size'],
        DEFAULTS['legend-size'],
        DEFAULTS['tick-size'],
        DEFAULTS['annotation-size'],
        DEFAULTS['tooltip-size'],
    )

@app.callback(
    Output("slider-components", "style"),
     Input('graph-lines', 'data'),
     Input("filename", "value"),
)
def toggle_components(lines, selected_file):
    #If no file is selected, hide the components.
    #Otherwise, show them.
    if not selected_file:
        return {'display': 'none'}
    else:
        return {'display': 'block'}
    
@app.callback(
    Output("open-offcanvas2", "style"),
    Input("graph-lines", "data"),
    Input("filename", "value"),
)
def toggle_sidebar_button(graph_lines, selected_file):
    #If no file is selected, hide the sidebar button
    if not selected_file:
        return {'border': 'none', 'background': 'transparent', 'padding': '0', 'margin': '0', 'display': 'none'}
    #Otherwise, show the sidebar button
    return {'border': 'none', 'background': 'transparent', 'padding': '0', 'margin': '0', 'display': 'block'}
    
@app.callback(
    [Output("point-edit-modal", "is_open"),
     Output("clicked-trace-index", "data"),
     Output("clicked-point-index", "data")],
    [
     Input("graphs", "clickData"),
     Input("close-dot-modal", "n_clicks"),
     Input("dot-submit-button", "n_clicks")
    ],
    [State("point-edit-modal", "is_open")]
)
def open_modal_on_click(click_data, close_clicks, submit_clicks, is_open):
    #If user clicks Close or Submit, close the modal & reset index.
    #If user clicks a point, open the modal & set that point's index.
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger_id in ["dot-submit-button", "close-dot-modal"]:
        return False, -1, -1

    #If a point was clicked in the graph
    if click_data:
        trace_idx = click_data["points"][0]["curveNumber"]
        point_idx = click_data["points"][0]["pointIndex"]
        return True, trace_idx, point_idx

    return is_open, -1, -1

#Callback to modify the point style directly in the figure
@app.callback(
    [Output('graphs', 'figure', allow_duplicate=True),
     Output('point-edit-modal', 'is_open', allow_duplicate=True)],
    [Input('dot-submit-button', 'n_clicks')],
    [
        State('dot-color-picker', 'value'),
        State('dot-size-input', 'value'),
        State('dot-symbol-dropdown', 'value'),
        State('clicked-trace-index', 'data'),
        State('clicked-point-index', 'data'),
        State('graphs', 'figure'),
    ],
    prevent_initial_call=True
)
def update_point_style(n_submit, chosen_color, chosen_size, chosen_symbol, trace_idx, point_idx, current_fig):

    if not n_submit:
        raise PreventUpdate
    if trace_idx < 0 or point_idx < 0:
        raise PreventUpdate

    trace_data = current_fig["data"][trace_idx]
    markers = trace_data["marker"]

    x_vals = trace_data["x"]
    n_points = len(x_vals) if x_vals else 0

    if n_points == 0:
        raise PreventUpdate
    
    #Convert color, size, symbol to lists if needed
    color_array = ut.ensure_list(markers, "color", "blue", n_points)
    size_array = ut.ensure_list(markers, "size", 10, n_points)
    symbol_array = ut.ensure_list(markers, "symbol", "circle", n_points)

    #Update the single clicked point
    color_array[point_idx] = chosen_color["hex"]
    size_array[point_idx] = chosen_size
    symbol_array[point_idx] = chosen_symbol

    #Write them back to the trace
    trace_data["marker"]["color"] = color_array
    trace_data["marker"]["size"] = size_array
    trace_data["marker"]["symbol"] = symbol_array

    return current_fig, False

# Callback to open the modal
@app.callback(
    Output("annotation-modal", "is_open"),
    [Input("create-annotation-button", "n_clicks"), 
     Input("submit-annotation", "n_clicks")],
    [State("annotation-modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

@app.callback(
    [Output('graphs', 'figure', allow_duplicate=True),
     Output('annotations-store', 'data')],
    [Input('submit-annotation', 'n_clicks')],
    [State('annotation-text-input', 'value'),
     State('graphs', 'figure'),
     State('annotations-store', 'data')],
    prevent_initial_call=True
)
def add_annotations(n_clicks, text, figure, annotations):
    ctx = callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if not trigger_id in ["graphs", "annotation-store"]:
        if n_clicks and text:
            #Initial annotation position
            x = math.log10(1)
            y = math.log10(1)

            #Create the new annotation
            new_annotation = {
                'x': x,
                'y': y,
                'xref': 'x',
                'yref': 'y',
                'text': text,
                'showarrow': False,
                'bgcolor': "white",
                'bordercolor': 'black',
                'borderwidth': 1,
            }

            #Append the new annotation to the existing annotations in the figure
            if 'annotations' in figure['layout']:
                figure['layout']['annotations'].append(new_annotation)
            else:
                figure['layout']['annotations'] = [new_annotation]

            #Update the annotations store
            if annotations is None:
                annotations = []
            annotations.append(new_annotation)
            return figure, annotations

        raise PreventUpdate
    else:
        raise PreventUpdate

@app.callback(
    [Output("graphs", "figure", allow_duplicate=True), 
     Output("disable-annotation-button", "children"),
     Output('change-annon', 'data', allow_duplicate=True),],
    [Input("disable-annotation-button", "n_clicks"),
     Input("disable-annotation-button", "children")],
    [State("graphs", "figure")],
    prevent_initial_call=True
)
def toggle_annotations(n_clicks, button_text, current_fig):
    if not n_clicks:
        return current_fig, "Disable Annotations"
    
    #If figure has annotations, remove them
    if "annotations" in current_fig["layout"] and current_fig["layout"]["annotations"]:
        current_fig["layout"]["annotations"] = []

    if button_text == "Disable Annotations":
        button_text = "Enable Annotations"
    else:
        #If no annotations, do nothing to the figure.
        button_text = "Disable Annotations"

    return current_fig, button_text, 1

@app.callback(
    Output('annotation-accordion', 'children'),
    Input('graphs', 'figure'),
    prevent_initial_call=True
)
def generate_angle_inputs(graph):
    if not graph or 'annotations' not in graph['layout']:
        return []  #No annotations, return no inputs

    annotations = graph['layout']['annotations']
    
    group_suffixes = ['_1', '_2']
    grouped_annotations = {suffix: [] for suffix in group_suffixes}
    ungrouped_annotations = []
    accordion_items = []

    #Group annotations based on the suffix
    for i, ann in enumerate(annotations):
        name = ann.get('name', '')
        matched = False
        for suffix in group_suffixes:
            if name.endswith(suffix):
                grouped_annotations[suffix].append((i, ann))
                matched = True
                break
        if not matched:
            #Collect annotations without the specified suffixes
            ungrouped_annotations.append((i, ann))

    for suffix, anns in grouped_annotations.items():
        if not anns:
            continue
        #Create cards for each annotation in the group
        cards = []
        for i, ann in anns:
            card = dbc.Card(
                [
                    dbc.CardHeader(
                        f"{ann.get('text')}",
                        style={
                            'color': 'white',
                            'fontWeight': 'bold',
                            'margin': '0px',
                            'padding': '2px 0px 0px 2px'
                        }
                    ),
                    dbc.CardBody(
                        [
                            dbc.Row(
                                [
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Div(
                                                        "Plot:",
                                                        style={
                                                            'color': 'white',
                                                            'marginRight': '10px',
                                                            'alignSelf': 'center'
                                                        }
                                                    ),
                                                    dbc.Checkbox(
                                                        id={'type': 'annotation-enable', 'index': i},
                                                        className="mb-0",
                                                        style={'alignSelf': 'center'},
                                                        value=ann.get('opacity', 1) == 1
                                                    ),
                                                ],
                                                style={'display': 'flex', 'alignItems': 'center'}
                                            ),
                                            html.Div(
                                                [
                                                    html.Div(
                                                        "Angle:",
                                                        style={
                                                            'color': 'white',
                                                            'marginRight': '10px',
                                                            'marginLeft': '30px',
                                                            'alignSelf': 'center'
                                                        }
                                                    ),
                                                    dbc.Input(
                                                        type="number",
                                                        placeholder="Angle",
                                                        value=round(ann.get('textangle', 0)),
                                                        id={'type': 'angle-input', 'index': i},
                                                        style={'width': '80px', 'height': '25px'}
                                                    ),
                                                ],
                                                style={
                                                    'display': 'flex',
                                                    'alignItems': 'center',
                                                    'marginRight': '30px'
                                                }
                                            ),
                                        ],
                                        style={
                                            'display': 'flex',
                                            'alignItems': 'center',
                                            'justifyContent': 'flex-start'
                                        }
                                    ),
                                ],
                                className="mb-0",
                                align="center"
                            ),
                        ],
                        style={'margin': '0px', 'padding': '0px 0px 2px 2px'}
                    ),
                ],
                className="mb-1",
                style={
                    'margin': '0px',
                    'padding': '0px 0px 2px 2px',
                    'backgroundColor': '#6c757d',
                    'Color': ann.get('bordercolor')
                }
            )
            cards.append(card)

        group_title = f"CARM Results {suffix[-1]}"
        accordion_item = dbc.AccordionItem(
            title=group_title,
            children=cards,
            item_id=f"group_{suffix}",
        )
        accordion_items.append(accordion_item)

    if ungrouped_annotations:
        cards = []
        for i, ann in ungrouped_annotations:
            card = dbc.Card(
                [
                    dbc.CardHeader(
                        f"{ann.get('text')}",
                        style={
                            'color': 'white',
                            'fontWeight': 'bold',
                            'margin': '0px',
                            'padding': '2px 0px 0px 2px'
                        }
                    ),
                    dbc.CardBody(
                        [
                            dbc.Row(
                                [
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Div(
                                                        "Plot:",
                                                        style={
                                                            'color': 'white',
                                                            'marginRight': '10px',
                                                            'alignSelf': 'center'
                                                        }
                                                    ),
                                                    dbc.Checkbox(
                                                        id={'type': 'annotation-enable', 'index': i},
                                                        className="mb-0",
                                                        style={'alignSelf': 'center'},
                                                        value=ann.get('opacity', 1) == 1
                                                    ),
                                                ],
                                                style={'display': 'flex', 'alignItems': 'center'}
                                            ),
                                            html.Div(
                                                [
                                                    html.Div(
                                                        "Angle:",
                                                        style={
                                                            'color': 'white',
                                                            'marginRight': '10px',
                                                            'marginLeft': '30px',
                                                            'alignSelf': 'center'
                                                        }
                                                    ),
                                                    dbc.Input(
                                                        type="number",
                                                        placeholder="Angle",
                                                        value=round(ann.get('textangle', 0)),
                                                        id={'type': 'angle-input', 'index': i},
                                                        style={'width': '80px', 'height': '25px'}
                                                    ),
                                                ],
                                                style={
                                                    'display': 'flex',
                                                    'alignItems': 'center',
                                                    'marginRight': '30px'
                                                }
                                            ),
                                        ],
                                        style={
                                            'display': 'flex',
                                            'alignItems': 'center',
                                            'justifyContent': 'flex-start'
                                        }
                                    ),
                                ],
                                className="mb-0",
                                align="center"
                            ),
                        ],
                        style={'margin': '0px', 'padding': '0px 0px 2px 2px'}
                    ),
                ],
                className="mb-1",
                style={
                    'margin': '0px',
                    'padding': '0px 0px 2px 2px',
                    'backgroundColor': '#6c757d',
                    'Color': ann.get('bordercolor')
                }
            )
            cards.append(card)

        accordion_item = dbc.AccordionItem(
            title='Custom Annotations',
            children=cards,
            item_id='other_annotations'
        )
        accordion_items.append(accordion_item)

    return accordion_items



@app.callback(
    Output('graphs', 'figure', allow_duplicate=True),
    [Input({'type': 'annotation-enable', 'index': ALL}, 'value')],
    [State('graphs', 'figure')],
    prevent_initial_call=True
)
def update_annotations_visibility(checkbox_values, figure):
    fig = go.Figure(figure)
    annotations = fig['layout']['annotations']

    if annotations:
        for i, ann in enumerate(annotations):
            if i < len(checkbox_values):
                if checkbox_values[i]:
                    ann['opacity'] = 1  # Visible
                else:
                    ann['opacity'] = 0  # Hidden
        fig.update_layout(annotations=annotations)

    return fig


@app.callback(
    Output('graphs', 'figure', allow_duplicate=True),
    [Input('annotation-size', 'value')],
    [State('graphs', 'figure')],
    prevent_initial_call=True
)
def update_annotations_font(font_size, figure):
    fig = go.Figure(figure)
    annotations = fig['layout']['annotations']

    if annotations:
        for i, ann in enumerate(annotations):
            ann['font']['size'] = font_size
        fig.update_layout(annotations=annotations)

    return fig

@app.callback(
    Output('graphs', 'figure', allow_duplicate=True),
    [Input({'type': 'angle-input', 'index': ALL}, 'value')],
    State('graphs', 'figure'),
    prevent_initial_call=True
)
def update_annotation_angles(input_angles, figure):
    #Callback to control annotations angle individually
    if not figure or not input_angles:
        raise dash.exceptions.PreventUpdate

    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    annotations = figure.get('layout', {}).get('annotations', [])

    for i, angle in enumerate(input_angles):
        if i < len(annotations):
            annotations[i]['textangle'] = angle
    
    new_figure = deepcopy(figure)
    new_figure['layout']['annotations'] = annotations

    return new_figure

#Toggle visibility of the sidebar
@app.callback(
    Output("offcanvas2", "is_open"),
    [Input("open-offcanvas2", "n_clicks")],
    [State("offcanvas2", "is_open")]
)
def toggle_offcanvas2(n, is_open):
    if n:
        return not is_open
    return is_open

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
    return {'width': '99%', 'height': '92%', 'display': 'block', 'background': 'transparent', 'marginLeft': '40px'}, {'text-align': 'center', 'margin-top': '10px'}

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
    [Output('graphs', 'figure', allow_duplicate=True),
     Output('graphs', 'config', allow_duplicate=True),
     Output('button-CARM-edit', 'children'),
     ],
    [Input('button-CARM-edit', 'n_clicks')],
    [State('graphs', 'figure'),
     State('graphs', 'config')],
     prevent_initial_call=True
)
def toggle_editable(n_clicks, figure, config):
    new_figure = copy.deepcopy(figure)
    if n_clicks % 2 == 0:
        config['editable'] = True
        return new_figure, config, "Save Text Changes"
    else:
        config['editable'] = False
        return new_figure, config, "Edit Graph Text"

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
                Dyno_path = ut.read_library_path("DYNO")
                if not Dyno_path:
                    #Return state indicating library input is needed
                    return True, dbc.Input(id="library-path-input", placeholder="Enter the DynamoRIO path here...", type="text"), "need_input"
            return False, dbc.Input(id="library-path-input", placeholder="Enter the DynamoRIO path here...", type="text"), "Library path saved"

        elif triggered_id == "submit-library-path":
            if library_path and DBI_AI_Calculator.check_client_exists(library_path):
                ut.write_library_path("DYNO", library_path)
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
            if exec_arguments != None:
                exec_arguments_list = exec_arguments.split()
            else:
                exec_arguments_list = None
            if str(checklist_values) == str(['dbi']) or str(checklist_values) == str(['dbi_roi']):
                method = "DR"
                Dyno_path = ut.read_library_path("DYNO")
                DBI_AI_Calculator.check_client_exists(Dyno_path)
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

                print("\n---------DBI RESULTS-----------")
                print("Total FP operations:", fp_ops)
                print("Total memory bytes:", memory_bytes)
                print("Total integer operations:", integer_ops)

                print("\nExecution time (seconds):", time_taken_seconds)
                print("GFLOP/s:", gflops)
                print("Bandwidth (GB/s): " + str(bandwidth))
                print("Arithmetic Intensity:", ai)
                print("------------------------------\n")

                ct = datetime.datetime.now()
                date = ct.strftime('%Y-%m-%d %H:%M:%S')
    

                DBI_AI_Calculator.update_csv(machine_name, file_path, gflops, ai, bandwidth, time_taken_seconds, "", date, None, None, None, method, 1, 1)
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

                print("\n---------PMU RESULTS-----------")
                print("Total FP Operations:", total_fp)
                print("Total Memory Bytes:", memory_bytes)
                #print("Simple AI:", float(total_fp / total_mem))
                print("SP FLOP Ratio: " + str(sp_ratio) + " DP FLOP Ration: " + str(dp_ratio))
                print("Threads Used:", thread_count)

                print("\nExecution Time (seconds):" + str(float(total_time_nsec / 1e9)))
                print("GFLOP/s: " + str(gflops))
                print("Bandwidth (GB/s): " + str(bandwidth))
                print("Arithmetic Intensity:", ai)
                print("------------------------------\n")

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
        run.run_roofline(name, freq, l1_size, l2_size, l3_size, inst, isa_set, precision_set, num_ld, num_st, thread_set, interleaved, num_ops, int(dram_bytes), dram_auto, test_type, verbose, set_freq, measure_freq, VLEN, tl1, tl2, plot, 1, "./Results")
    except Exception as e:
        print("Task was interrupted:", e)
    return no_update

@app.callback(
    Output('additional-dropdowns', 'children'),
    [Input('filename', 'value')]
)
def update_additional_dropdowns(selected_file):
    if selected_file:
        _, _, _, _, data_list = gut.read_csv_file(selected_file)
        df = pd.DataFrame(data_list)

    fields = ['ISA', 'Precision', 'Threads', 'Loads', 'Stores', 'Interleaved', 'DRAM Bytes', 'FP Inst', 'Date']
    dropdowns = []

    for field in fields:
        if selected_file:
            if not df.empty:
                if field == "Date":
                    unique_values = sorted(df[field.replace(" ", "")].unique(), reverse=True)  
                else:
                    unique_values = sorted(df[field.replace(" ", "")].unique())  
                options = [{'label': value, 'value': value} for value in unique_values]
                display = "flex"
            else:
                options = {}
                display = "none"
        else:
            options = {}
            display = "none"

        if field == "Date":
            width = 250
        elif field == "ISA":
            width = 200
        else:
            width = 160
        
        dropdowns.append(
            html.Div(
                dcc.Dropdown(
                    id=f'{field.lower().replace(" ", "")}-dynamic-dropdown',
                    placeholder=field,
                    options=options,
                    multi=False
                ),
                style = {
                'flex': '1 0 auto',
                'minWidth': width,
                'margin': '5px',
            }
            )
        )
    
    return dbc.Card(
        dbc.CardBody([
            html.Div(
                [
                    html.Div(
                        "CARM Results 1:", 
                        style={
                            'marginRight': '10px', 
                            'alignSelf': 'center', 
                            'fontWeight': 'bold',
                            'minWidth': '125px', 
                        }
                    ),
                    html.Div(
                        dropdowns, 
                        style={
                            'display': 'flex',
                            'width': '100%',
                            'justifyContent': 'space-between',
                            'alignItems': 'center',
                        }
                    )
                ],
                style={
                    'display': 'flex',
                    'alignItems': 'center',
                    'margin': '-10px auto auto auto'
                }
            )
        ]),
        style={
            'margin': '0px auto 10px auto',
            'padding': '0px',
            'textAlign': 'center',
            'display': 'flex',
            'height': '60px'
        }
    )


@app.callback(
    Output('additional-dropdowns2', 'children'),
    [Input('filename', 'value')]
)
def update_additional_dropdowns2(selected_file):
    if selected_file:
        _, _, _, _, data_list = gut.read_csv_file(selected_file)
        df = pd.DataFrame(data_list)

    fields = ['ISA', 'Precision', 'Threads', 'Loads', 'Stores', 'Interleaved', 'DRAM Bytes', 'FP Inst', 'Date']
    dropdowns = []

    for field in fields:
        if selected_file:
            if not df.empty:
                if field == "Date":
                    unique_values = sorted(df[field.replace(" ", "")].unique(), reverse=True)  
                else:
                    unique_values = sorted(df[field.replace(" ", "")].unique())  
                options = [{'label': value, 'value': value} for value in unique_values]
                display = "flex"
            else:
                options = {}
                display = "none"
        else:
            options = {}
            display = "none"

        if field == "Date":
            width = 250
        elif field == "ISA":
            width = 200
        else:
            width = 160
        
        dropdowns.append(
            html.Div(
                dcc.Dropdown(
                    id=f'{field.lower().replace(" ", "")}-dynamic-dropdown2',
                    placeholder=field,
                    options=options,
                    multi=False
                ),
                style = {
                'flex': '1 0 auto',
                'minWidth': width,
                'margin': '5px',
            }
            )
        )
    
    return dbc.Card(
        dbc.CardBody([
            html.Div(
                [
                    html.Div(
                        "CARM Results 2:", 
                        style={
                            'marginRight': '10px', 
                            'alignSelf': 'center', 
                            'fontWeight': 'bold',
                            'color': 'red',
                            'minWidth': '125px', 
                        }
                    ),
                    html.Div(
                        dropdowns, 
                        style={
                            'display': 'flex',
                            'width': '100%',
                            'justifyContent': 'space-between',
                            'alignItems': 'center',
                        }
                    )
                ],
                style={
                    'display': 'flex',
                    'alignItems': 'center',
                    'margin': '-10px auto auto auto'
                }
            )
        ]),
        style={
            'margin': '0px auto 10px auto',
            'padding': '0px',
            'textAlign': 'center',
            'display': 'flex',
            'height': '60px'
        }
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
        return [html.Div(id="application-dropdown", style={"display": "none"}), {'display': 'none'}]

 
    application_file_path = selected_file.replace("roofline", "applications")
    #Read the CSV file and extract data
    data_list = gut.read_application_csv_file(application_file_path)
    
    if not data_list:
        df = None
        #No data found or invalid file, hide the dropdown
        return [html.Div(id="application-dropdown", style={"display": "none"}), {'display': 'none'}]

    #Data found
    df = pd.DataFrame(data_list)
 
    options = [{
    'label': f"{row['Name']} ({row['Method']}) - {row['Date']} ({' '.join(filter(None, [row['ISA'], row['Precision'], ((str(row['Threads']) + ' Threads') if (row['Threads'] or row['Threads'] > 0) else None)]))}{' |' if any([row['ISA'], row['Precision'], row['Threads']]) else ''} AI: {row['AI']} Gflops: {row['GFLOPS']} Duration: {np.format_float_positional(row['Time'], trim='-')})",
    'value': f"{row['Name']}  {row['Method']}  {row['Date']}  {row['ISA']}  {row['Precision']}  {row['Threads']}  {row['AI']}  {row['GFLOPS']}  {np.format_float_positional(row['Time'], trim='-')}  {index}"
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
    _, _, _, _, data_list = gut.read_csv_file(selected_file)
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
    _, _, _, _, data_list = gut.read_csv_file(selected_file)
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
    _, _, _, _, data_list = gut.read_csv_file(selected_file)
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
    _, _, _, _, data_list = gut.read_csv_file(selected_file)
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
    _, _, _, _, data_list = gut.read_csv_file(selected_file)
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
    _, _, _, _, data_list = gut.read_csv_file(selected_file)
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
    _, _, _, _, data_list = gut.read_csv_file(selected_file)
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
    _, _, _, _, data_list = gut.read_csv_file(selected_file)
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
    _, _, _, _, data_list = gut.read_csv_file(selected_file)
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
    _, _, _, _, data_list = gut.read_csv_file(selected_file)
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
    _, _, _, _, data_list = gut.read_csv_file(selected_file)
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
    _, _, _, _, data_list = gut.read_csv_file(selected_file)
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
    _, _, _, _, data_list = gut.read_csv_file(selected_file)
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
    _, _, _, _, data_list = gut.read_csv_file(selected_file)
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
    _, _, _, _, data_list = gut.read_csv_file(selected_file)
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
    _, _, _, _, data_list = gut.read_csv_file(selected_file)
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
    _, _, _, _, data_list = gut.read_csv_file(selected_file)
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
    _, _, _, _, data_list = gut.read_csv_file(selected_file)
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
    Output('graph-size-update', 'children'),
    Output('graph-lines', 'data'),
    Output('graph-lines2', 'data'),
    Output('graph-values', 'data'),
    Output('graph-values2', 'data'),
    Output('graph-isa', 'data'),
    Output('graph-xrange', 'data'),
    Output('graph-yrange', 'data'),
    Output('change-annon', 'data'),
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
    Input("application-dropdown", "value"),
    Input('exponent-switch', 'value'),
    Input('line-legend-switch', 'value'),
    Input('detail-legend-switch', 'value'),
    Input('line-size', 'value'),
    Input('title-size', 'value'),
    Input('axis-size', 'value'),
    Input('tick-size', 'value'),
    Input('tooltip-size', 'value'),
    Input('legend-size', 'value'),
    Input('dot-size', 'value'),
    Input('graph-width', 'value'),
    Input('graph-height', 'value'),
    ],
    State('graphs', 'figure'),
    prevent_initial_call=True
)
def analysis(ISA, Precision, Threads, Loads, Stores, Interleaved, DRAMBytes, FPInst, Date,
             ISA2, Precision2, Threads2, Loads2, Stores2, Interleaved2, DRAMBytes2, FPInst2, Date2, selected_file, selected_applications, exponant, line_legend, line_legend_detailed, line_size, title_size, axis_size, tick_size, tooltip_size, legend_size, dot_size, g_width, g_height, figure):
    
    global lines_origin
    global lines_origin2
    
    if not selected_file:
        return go.Figure(),  "", None, None, None, None, None, [0, 0], [0, 0], None

    #Get trigger-id
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    annotations = {}
    if not figure is None:
        fig = go.Figure(figure)
        annotations = fig['layout']['annotations']

    if trigger_id not in ['graphs']:
        figure = go.Figure()

    #Read data and create DataFrame
    _, _, _, _, data_list = gut.read_csv_file(selected_file)
    df = pd.DataFrame(data_list)
    top_flops = 0
    top_flops2 = 0
    smallest_ai = 1000
    smallest_gflops = 1000
    smallest_gflops2 = 1000
    change_annotation = 0


    #Get queries for both sets of inputs
    query1 = construct_query(ISA, Precision, Threads, Loads, Stores, Interleaved, DRAMBytes, FPInst, Date)
    query2 = construct_query(ISA2, Precision2, Threads2, Loads2, Stores2, Interleaved2, DRAMBytes2, FPInst2, Date2)

    #Filter data based on queries
    filtered_df1 = df.query(query1) if query1 else df
    filtered_df2 = df.query(query2) if query2 and any([ISA2, Precision2, Threads2, Loads2, Stores2, Interleaved2, DRAMBytes2, FPInst2, Date2]) else pd.DataFrame()
    
    #Plot the selected application as a dot
    if selected_applications:
        for selected_application in selected_applications:
            parts = selected_application.rsplit('  ', 9)
            if len(parts) >= 9:
                name, method, date, isa, precision, threads, ai, gflops, time, index_start = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], float(parts[6]), float(parts[7]), parts[8], int(parts[9])
                #Add trace for each application
                smallest_gflops = min(smallest_gflops, gflops)
                smallest_ai = min (smallest_ai, ai)

                figure.add_trace(go.Scatter(
                        x=[ai], 
                        y=[gflops], 
                        mode='markers',
                        text=[f'{name} | {threads} threads | Duration: {time} </b><br> ({date})'],
                        name=f'{name}',
                        marker=dict(size=dot_size),
                        hovertemplate='<b>%{text} (AI: %{x}, GFLOP/s: %{y})<br><extra></extra>',
                    ))
    if trigger_id not in ['graphs']:
        figure.update_layout(
            hoverlabel={
                    'font_size': tooltip_size,
                },
            title={
                'text': 'Cache Aware Roofline Model',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {
                    'family': "Arial",
                    'size': title_size,
                    'color': "black"
                }
            },
            xaxis={
                'title':{
                    'text': 'Arithmetic Intensity (flop/byte)',
                    'font': {
                        'family': "Arial",
                        'size': axis_size,
                        'color': "black"
                    },
                },
                'type': 'log',
                'dtick': '0.30102999566',
                'title_standoff': 0,
                'automargin': True,
                'tickfont_size': tick_size,
            },
            yaxis={
                'title':{
                    'text': 'Performance (GFLOP/s)',
                    'font': {
                        'family': "Arial",
                        'size': axis_size,
                        'color': "black"
                    },
                },
                'type': 'log',
                'dtick': '0.30102999566',
                'title_standoff': 0,
                'automargin': True,
                'tickfont_size': tick_size,
            },
            legend={
                'font': {'size': legend_size},
                'orientation': 'h',
                'x': 0.5,
                'y': 0,
                'xanchor': 'center',
                'yanchor': 'bottom',
                'yref': 'container',
            },
            showlegend=True,
            height=g_height,
            width=g_width,
            margin=dict(t=60, l=80, b=20, r=40),
            plot_bgcolor='#e9ecef',
            paper_bgcolor='#e9ecef',
            clickmode='event',
        )
        figure.update_xaxes(showspikes=True)
        figure.update_yaxes(showspikes=True)

    if not filtered_df1.empty:
        values1 = filtered_df1.iloc[-1][['L1', 'L2', 'L3', 'DRAM', 'FP', 'FP_FMA', 'FPInst']].tolist()
        ISA = filtered_df1.iloc[-1][['ISA']].tolist()
        lines = gut.calculate_roofline(values1, smallest_ai/5)
        if lines != lines_origin and len(lines_origin) > 0 and trigger_id != 'interval-component':
            change_annotation = 1
            annotations = {}
        lines_origin = lines
        top_flops = lines['L1']['ridge'][1]
        smallest_gflops_aux = min(
            line['start'][1] for line in lines.values()
            if 'start' in line and isinstance(line['start'], (list, tuple)) and len(line['start']) > 1
        )
        smallest_gflops = min(smallest_gflops_aux, smallest_gflops)
        #If its just a zoom we dont plot the lines again, just re-calculate the angles for the annotations
        if not trigger_id in ['graphs', 'interval-component']:
            figure.add_traces(gut.plot_roofline(values1, lines, '', ISA[0], line_legend, int(line_size), line_legend_detailed))
        
    lines2 = {}
    values2 = []
    if not filtered_df2.empty and not query2 == None:
        values2 = filtered_df2.iloc[-1][['L1', 'L2', 'L3', 'DRAM', 'FP', 'FP_FMA', 'FPInst']].tolist()
        ISA.append(filtered_df2.iloc[-1][['ISA']].tolist()[0])
        lines2 = gut.calculate_roofline(values2, smallest_ai/5)

        if lines2 != lines_origin2 and trigger_id != 'interval-component':
            change_annotation = 1
            annotations = {}
        lines_origin2 = lines2

        top_flops2 = lines2['L1']['ridge'][1]
        smallest_gflops2 = min(
            line['start'][1] for line in lines2.values()
            if 'start' in line and isinstance(line['start'], (list, tuple)) and len(line['start']) > 1
        )
        if not trigger_id in ['graphs', 'interval-component']:
            figure.add_traces(gut.plot_roofline(values2, lines2, '2', ISA[1], line_legend, int(line_size), line_legend_detailed))
    else:
        if lines2 != lines_origin2 and trigger_id != 'interval-component':
            change_annotation = 1
            annotations = {} 
        lines_origin2 = lines2
    
    #Grab the axis range of the plot, after its reset or not
    xaxis_range = figure.layout.xaxis.range
    if xaxis_range:
        x_min_angle = 10**xaxis_range[0]
        x_max_angle = 10**xaxis_range[1]
    else:
        x_min_angle = min(0.00390625, smallest_ai/5)
        x_max_angle = 256

    #Extract the current y-axis range if available, otherwise use the data's min/max
    yaxis_range = figure.layout.yaxis.range
    
    if yaxis_range:
        y_min_angle = 10**yaxis_range[0]
        y_max_angle = 10**yaxis_range[1]
    else:
        y_min_angle = min(smallest_gflops,smallest_gflops2)*0.5
        y_max_angle = max(top_flops2,top_flops)*2

    if exponant:
        xaxis_range = figure.layout.xaxis.range
        x_min = min(0.00390625, smallest_ai/5)
        x_max = 256

        #Extract the current y-axis range if available, otherwise use the data's min/max
        yaxis_range = figure.layout.yaxis.range
        y_min = min(smallest_gflops,smallest_gflops2)
        y_max = max(top_flops2,top_flops)*1.3

        x_tickvals, x_ticktext = ut.make_power_of_two_ticks(x_min, x_max)
        y_tickvals, y_ticktext = ut.make_power_of_two_ticks(y_min, y_max)

        #Update axes to show 2^X notation
        figure.update_xaxes(tickmode='array', tickvals=x_tickvals, ticktext=x_ticktext)
        figure.update_yaxes(tickmode='array', tickvals=y_tickvals, ticktext=y_ticktext)
    else:
        #Revert to normal formatting
        figure.update_yaxes(exponentformat=None, tickformat=None)
        figure.update_xaxes(exponentformat=None, tickformat=None)

    timestamp = datetime.datetime.now().isoformat()
    if annotations:
        figure['layout']['annotations'] = annotations
    return figure, f"Update: {timestamp}", lines, lines2, values1, values2, ISA, [x_min_angle, x_max_angle], [y_min_angle, y_max_angle], change_annotation#, 'width': '100%', 'height' : '100%'}

@app.callback(
    [Output(component_id='graphs', component_property='figure', allow_duplicate=True),
     Output('change-annon', 'data', allow_duplicate=True)],
    [
    Input('graph-size-data', 'children'),
    
    Input('graph-lines', 'data'),
    Input('graph-lines2', 'data'),
    Input('graph-values', 'data'),
    Input('graph-values2', 'data'),
    Input('graph-isa', 'data'),
    Input('graph-xrange', 'data'),
    Input('graph-yrange', 'data'),
    Input('graphs', 'relayoutData'),
    Input('change-annon', 'data'),
    Input("disable-annotation-button", "children"),
    Input('annotation-size', 'value'),
    Input('filename', 'value'),
    ],
    State('graphs', 'figure'),
    prevent_initial_call=True,
)
def angle_updater(size, lines, lines2, values1, values2, ISA, xrange, yrange, relayout_data, change_anon, disable_anon, anon_size, filename, figure):
    #Callback to update the annotations angles when the graph scale/zoom changes
    if disable_anon != "Enable Annotations" and ISA:
        figure = go.Figure(figure)
        if figure:
            x_range = getattr(figure.layout.xaxis, 'range', None)
            if x_range is not None:
                xaxis_range = figure.layout.xaxis.range
            else:
                xaxis_range = [math.log10(xrange[0]), math.log10(xrange[1])]

            y_range = getattr(figure.layout.yaxis, 'range', None)
            if y_range is not None:
                yaxis_range = figure.layout.yaxis.range
            else:
                yaxis_range = [math.log10(yrange[0]), math.log10(yrange[1])]
        
        new_figure = go.Figure(figure)

        timestamp = datetime.datetime.now().isoformat()
        #Get trigger-id
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger_id == "filename":
            new_figure['layout']['annotations'] = []

        if size and len(size) >= 2:
            liner = size.split('\n')
            width = float(liner[0].replace('Plot area width:', '').replace('px','').strip())
            height = float(liner[1].replace('Plot area height:', '').replace('px','').strip())
        
            annotations = new_figure['layout']['annotations']
            cache_levels = ['L1', 'L2', 'L3', 'DRAM']
            angle_degrees = {}
            angle_degrees2 = {}
            new_angle = {}
            group_suffixes = ['_1', '_2']
            cache_level_suffix = ['L1_1', 'L2_1', 'L3_1', 'DRAM_1', 'FP_1', 'FP_FMA_1', 'L1_2', 'L2_2', 'L3_2', 'DRAM_2', 'FP_2', 'FP_FMA_2']
            for ann in annotations:
                    ann_name = ann["name"]

                    if ann_name in cache_level_suffix:
                        ann_vis = ann['opacity']
                        if ann_name[:-2] in cache_levels:
                            if ann_name[-1] == '1':
                                liner = lines
                            elif ann_name[-1] == '2' and lines2:
                                liner = lines2
                            else:
                                continue
                            log_x1, log_x2 = math.log10(liner[ann_name[:-2]]['start'][0]), math.log10(liner[ann_name[:-2]]['ridge'][0])
                            log_y1, log_y2 = math.log10(liner[ann_name[:-2]]['start'][1]), math.log10(liner[ann_name[:-2]]['ridge'][1])

                            log_xmin, log_xmax = xaxis_range[0], xaxis_range[1]
                            log_ymin, log_ymax = yaxis_range[0], yaxis_range[1]

                            #Compute pixel coordinates based on log scale
                            x1_pixel = ( (log_x1 - log_xmin) / (log_xmax - log_xmin) ) * width
                            x2_pixel = ( (log_x2 - log_xmin) / (log_xmax - log_xmin) ) * width
                            y1_pixel = height - ( (log_y1 - log_ymin) / (log_ymax - log_ymin) ) * height
                            y2_pixel = height - ( (log_y2 - log_ymin) / (log_ymax - log_ymin) ) * height

                            #Pixel slope
                            pixel_slope = (y2_pixel - y1_pixel) / (x2_pixel - x1_pixel)
                            ann['textangle'] = round(math.degrees(math.atan(pixel_slope)), 2)
            if not annotations or change_anon == 1:
                new_figure.layout.annotations = [
                    annotation for annotation in new_figure.layout.annotations
                    if annotation and not (hasattr(annotation, 'name') and annotation.name.endswith('_1'))
                ]
                for cache_level in ['L1', 'L2', 'L3', 'DRAM', 'FP', 'FMA']:
                    annotation = gut.draw_annotation(values1, lines, '1', ISA[0], cache_level, width, height, anon_size, x_range=[xaxis_range[0], xaxis_range[1]], y_range=[yaxis_range[0], yaxis_range[1]])
                    if annotation:
                        new_figure.add_annotation(annotation)
                
                new_figure.layout.annotations = [
                        annotation for annotation in new_figure.layout.annotations
                        if annotation != None and not (hasattr(annotation, 'name') and annotation.name.endswith('_2'))
                    ]
                if len(lines2) > 0:
                    
                    for cache_level in ['L1', 'L2', 'L3', 'DRAM', 'FP', 'FMA']:
                        annotation = gut.draw_annotation(values2, lines2, '2', ISA[1], cache_level, width, height, anon_size, x_range=[xaxis_range[0], xaxis_range[1]], y_range=[yaxis_range[0], yaxis_range[1]])
                        if annotation:
                            new_figure.add_annotation(annotation)
            if change_anon == 1:
                change_anon = 0
            return go.Figure(new_figure), change_anon
    else:
        return go.Figure(figure), change_anon


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

app.clientside_callback(
    """
    function(relayoutData) {
        // If no relayoutData, don't update
        if (!relayoutData) {
            return window.dash_clientside.no_update;
        }

        function getPlotSize(attempts) {
            const graphDiv = document.getElementById('graphs');
            if (!graphDiv) return null;

            const plotRect = graphDiv.querySelector('rect.nsewdrag[data-subplot="xy"]');
            if (plotRect) {
                const width = parseFloat(plotRect.getAttribute('width'));
                const height = parseFloat(plotRect.getAttribute('height'));
                return {width, height};
            } else {
                // Retry logic with delay
                if (attempts > 0) {
                    return new Promise(resolve => {
                        setTimeout(() => {
                            resolve(getPlotSize(attempts - 1));
                        }, 100);  // each retry is delayed by 200ms
                    });
                } else {
                    return null;
                }
            }
        }

        // Introduce an initial delay before making the first size query
        return new Promise(resolve => {
            setTimeout(() => {
                resolve(Promise.resolve(getPlotSize(5)).then(size => {
                    if (size) {
                        return `Plot area width: ${size.width}px\\nPlot area height: ${size.height}px`;
                    } else {
                        return 'Plot area not found after multiple attempts.';
                    }
                }));
            }, 100);  // initial delay of 300ms before starting the measurement process
        });
    }
    """,
    Output('graph-size-data', 'children'),
    Input('graph-size-update', 'children'),
)

if __name__ == '__main__':
    app.run_server(debug=False)