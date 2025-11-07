# Improved Dash GUI that can set parameters from av.py and run N_Moving_Objects
import importlib
import sys
import threading
import io
import contextlib
import os
import subprocess
import webbrowser
from threading import Timer
from dash import Dash, dcc, html, callback_context
from dash.dependencies import Input, Output, State

# Make sure av does not run heavy dataset I/O on import; GUI should open first.
os.environ.setdefault('AV_SKIP_LOAD', '1')
import av

# Globals to track background run and its output
run_thread = None
last_output = ""
last_status = "idle"
stop_requested = False


def run_moving_objects_in_background(params):
    """Background runner: set av parameters then import/reload N_Moving_Objects.
    Captures stdout/stderr into last_output and updates last_status.
    """
    global last_output, last_status, stop_requested
    stop_requested = False  # Reset stop flag
    last_status = "running"
    buf = io.StringIO()
    try:
        # Apply parameters to av module
        for k, v in params.items():
            setattr(av, k, v)

        # Ensure dataset_name is provided via environment so av will pick it up on reload
        if 'dataset_name' in params and params['dataset_name']:
            os.environ['AV_DATASET'] = params['dataset_name']

        # Ensure results_dir is provided via environment so modules can write into it
        if 'results_dir' in params and params['results_dir']:
            results_dir = params['results_dir']
            os.environ['AV_RESULTS_DIR'] = results_dir
            try:
                os.makedirs(results_dir, exist_ok=True)
            except Exception:
                pass

        # Allow av to run its loading logic on reload
        os.environ['AV_SKIP_LOAD'] = '0'
        # Capture stdout while importing/reloading heavy module
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            # If already imported, reload to rerun top-level logic
            try:
                # Reload av so it processes the dataset with the provided AV_DATASET
                importlib.reload(av)

                # Re-apply parameters AFTER reload (reload resets av to defaults!)
                for k, v in params.items():
                    setattr(av, k, v)

                # Debug prints to help diagnose flag propagation and results_dir
                print('\n--- DEBUG: run parameters passed from GUI ---')
                try:
                    print('params:', {k: params[k] for k in params})
                except Exception:
                    print('params (could not stringify)')
                print('AV_RESULTS_DIR env:', os.environ.get('AV_RESULTS_DIR'))
                print('av.N_VA_InequalityMatrices =', getattr(av, 'N_VA_InequalityMatrices', None))
                print('av.N_PDP =', getattr(av, 'N_PDP', None))
                print('av.PDPg_buffer_active =', getattr(av, 'PDPg_buffer_active', None))
                print('av.PDPg_rough_active =', getattr(av, 'PDPg_rough_active', None))
                print('av.PDPg_fundamental_active =', getattr(av, 'PDPg_fundamental_active', None))
                print('--- end DEBUG ---\n')

                import N_Moving_Objects
                importlib.reload(N_Moving_Objects)
            except ModuleNotFoundError:
                # First run - module not imported yet
                import N_Moving_Objects

        # Check if stop was requested during execution
        if stop_requested:
            last_output = buf.getvalue() + "\n⚠️ Analysis stopped by user.\n"
            last_status = "stopped"
        else:
            last_output = buf.getvalue()
            last_status = "finished"
    except Exception as e:
        if stop_requested:
            last_output = buf.getvalue() + f"\n⚠️ Analysis stopped by user.\n"
            last_status = "stopped"
        else:
            last_output = buf.getvalue() + f"\nERROR: {e}\n"
            last_status = "error"



# Initialize Dash app
app = Dash(__name__)

# Build a cleaner layout with grouped sections and fancy styling
CARD_STYLE = {
    'border': '2px solid #e0e0e0', 
    'borderRadius': '12px', 
    'padding': '20px',
    'boxShadow': '0 4px 12px rgba(0,0,0,0.1)', 
    'backgroundColor': '#ffffff',
    'background': 'linear-gradient(to bottom, #ffffff, #f9f9f9)'
}

HEADER_STYLE = {
    'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    'padding': '24px',
    'borderRadius': '12px',
    'color': 'white',
    'marginBottom': '20px',
    'boxShadow': '0 6px 20px rgba(102, 126, 234, 0.4)'
}

BUTTON_STYLE = {
    'backgroundColor': '#667eea',
    'color': 'white',
    'border': 'none',
    'borderRadius': '8px',
    'padding': '10px 20px',
    'fontSize': '14px',
    'fontWeight': 'bold',
    'cursor': 'pointer',
    'boxShadow': '0 2px 8px rgba(102, 126, 234, 0.3)',
    'transition': 'all 0.3s ease'
}

app.layout = html.Div([
    # Header with gradient
    html.Div([
        html.H1('🚀 Moving Objects Analysis Dashboard', style={'margin': '0 0 8px 0', 'fontSize': '32px'}),
        html.Div('⚙️ Configureer parameters, selecteer modules en start de analyse', 
                style={'fontSize': '16px', 'opacity': '0.95'})
    ], style=HEADER_STYLE),

    html.Div([
        # Left column: main parameters
        html.Div([
            html.H4('📊 PDP Types', style={'color': '#667eea', 'marginTop': '0'}),
            html.Div([
                dcc.Checklist(
                    id='pdp-fundamental',
                    options=[{'label': ' 🔹 Fundamental (required)', 'value': 'fundamental'}],
                    value=['fundamental'],
                    style={'fontSize': '14px', 'color': '#999'},
                    inputStyle={'cursor': 'not-allowed'},
                    labelStyle={'cursor': 'not-allowed'}
                ),
                dcc.Checklist(
                    id='pdp-types',
                    options=[
                        {'label': ' 🔸 Buffer', 'value': 'buffer'},
                        {'label': ' 🔶 Rough', 'value': 'rough'},
                        {'label': ' 🔷 Buffer + Rough', 'value': 'bufferrough'},
                    ],
                    value=[k for k, v in [('buffer', av.PDPg_buffer), ('rough', av.PDPg_rough), ('bufferrough', av.PDPg_bufferrough)] if v == 1],
                    style={'fontSize': '14px', 'marginTop': '8px'}
                ),
            ]),

            html.H4('🔢 Core Numeric Parameters', style={'marginTop': '20px', 'color': '#667eea'}),
            html.Div([
                html.Label('⏱️ window_length_tst', style={'fontWeight': '500', 'fontSize': '13px'}), 
                dcc.Input(id='window_length_tst', type='number', value=av.window_length_tst, 
                         style={'width': '100%', 'padding': '6px', 'borderRadius': '6px', 'border': '1px solid #ddd', 'fontSize': '13px'})
            ], style={'marginTop': '6px'}),

            html.H4('📏 Buffer / Rough Parameters', style={'marginTop': '20px', 'color': '#667eea'}),
            html.Div(style={'display': 'flex', 'gap': '8px'}, children=[
                html.Div([
                    html.Label('↔️ buffer_x', style={'fontWeight': '500', 'fontSize': '13px'}), 
                    dcc.Input(id='buffer_x', type='number', value=av.buffer_x, disabled=True,
                             style={'width': '100%', 'padding': '6px', 'borderRadius': '6px', 'border': '1px solid #ddd', 'fontSize': '13px'})
                ], style={'flex': '1'}),
                html.Div([
                    html.Label('↕️ buffer_y', style={'fontWeight': '500', 'fontSize': '13px'}), 
                    dcc.Input(id='buffer_y', type='number', value=av.buffer_y, disabled=True,
                             style={'width': '100%', 'padding': '6px', 'borderRadius': '6px', 'border': '1px solid #ddd', 'fontSize': '13px'})
                ], style={'flex': '1'})
            ]),
            html.Div(style={'display': 'flex', 'gap': '8px', 'marginTop': '6px'}, children=[
                html.Div([
                    html.Label('↔️ rough_x', style={'fontWeight': '500', 'fontSize': '13px'}), 
                    dcc.Input(id='rough_x', type='number', value=av.rough_x, disabled=True,
                             style={'width': '100%', 'padding': '6px', 'borderRadius': '6px', 'border': '1px solid #ddd', 'fontSize': '13px'})
                ], style={'flex': '1'}),
                html.Div([
                    html.Label('↕️ rough_y', style={'fontWeight': '500', 'fontSize': '13px'}), 
                    dcc.Input(id='rough_y', type='number', value=av.rough_y, disabled=True,
                             style={'width': '100%', 'padding': '6px', 'borderRadius': '6px', 'border': '1px solid #ddd', 'fontSize': '13px'})
                ], style={'flex': '1'})
            ]),

        ], style={**CARD_STYLE, 'flex': '1', 'marginRight': '12px'}),

        # Right column: toggles / advanced
        html.Div([
            html.H4('📈 Visualization & Report Modules', style={'marginTop': '0', 'color': '#667eea'}),
            dcc.Checklist(
                id='nva-options',
                options=[
                    {'label': 'PDP Core Code', 'value': 'N_PDP'},
                    {'label': 'Static Absolute', 'value': 'N_VA_StaticAbsolute'},
                    {'label': 'HeatMap', 'value': 'N_VA_HeatMap'},
                    {'label': 'Hierarchical Clustering', 'value': 'N_VA_HClust'},
                    {'label': 'Cluster Map', 'value': 'N_VA_ClusterMap'},
                    {'label': 'MDS', 'value': 'N_VA_Mds'},
                    {'label': 't-SNE', 'value': 'N_VA_TSNE'},
                    {'label': 'UMAP', 'value': 'N_VA_UMAP'},
                    {'label': 'Inequality Matrices', 'value': 'N_VA_InequalityMatrices'},
                    {'label': 'Top K', 'value': 'N_VA_TopK'},
                    {'label': 'Generate PDF Report', 'value': 'N_VA_Report'},
                    {'label': 'MDS Autoencoder', 'value': 'N_VA_Mds_autoencoder'},
                    {'label': 'Static Relative', 'value': 'N_VA_StaticRelative'},
                    {'label': 'Static Finetuned', 'value': 'N_VA_StaticFinetuned'},
                    {'label': 'Dynamic Absolute', 'value': 'N_VA_DynamicAbsolute'},
                    {'label': 'PDP Inverse', 'value': 'N_VA_Inverse'},
                ],
                value=[k for k in ['N_VA_Report','N_VA_StaticAbsolute','N_VA_StaticRelative','N_VA_StaticFinetuned','N_VA_DynamicAbsolute','N_PDP','N_VA_InequalityMatrices','N_VA_HeatMap','N_VA_HClust','N_VA_ClusterMap','N_VA_Mds','N_VA_TSNE','N_VA_UMAP','N_VA_Mds_autoencoder','N_VA_TopK','N_VA_Inverse'] if getattr(av, k, 0) == 1],
                style={'fontSize': '14px', 'lineHeight': '1.8'}
            ),

            html.H4('💾 Dataset & Run', style={'marginTop': '20px', 'color': '#667eea'}),
            html.Div([
                html.Label('📁 dataset_name (CSV path)', style={'fontWeight': '500', 'fontSize': '13px'}), 
                dcc.Input(id='dataset_name', type='text', value=getattr(av, 'dataset_name', 'N_C_Dataset.csv'), 
                         style={'width': '100%', 'padding': '6px', 'borderRadius': '6px', 'border': '1px solid #ddd', 'fontSize': '13px'})
            ], style={'marginTop': '6px'}),
            html.Div([
                html.Label('📂 results_dir (output folder)', style={'fontWeight': '500', 'fontSize': '13px'}), 
                dcc.Input(id='results_dir', type='text', value=os.environ.get('AV_RESULTS_DIR', ''), 
                         style={'width': '100%', 'padding': '6px', 'borderRadius': '6px', 'border': '1px solid #ddd', 'fontSize': '13px'})
            ], style={'marginTop': '8px'}),

            html.Div(style={'marginTop': '16px', 'display': 'flex', 'gap': '10px', 'flexDirection': 'column'}, children=[
                html.Button('▶️ Run Analysis', id='run-button', n_clicks=0, style=BUTTON_STYLE),
                html.Button('⏹️ Stop Analysis', id='stop-button', n_clicks=0, 
                           style={**BUTTON_STYLE, 'backgroundColor': '#e53e3e'}),
                html.Button('📊 Get Status & Output', id='status-button', n_clicks=0, 
                           style={**BUTTON_STYLE, 'backgroundColor': '#48bb78'}),
                html.Div([
                    html.Label('🔗 Results viewer script (path to .py or Streamlit script):', style={'fontSize': '12px', 'fontWeight': '500', 'marginBottom': '6px'}),
                    dcc.Input(
                        id='viewer-script-path',
                        type='text',
                        value=os.path.abspath(os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'visualisations', 'app_PDP_results.py'))),
                        style={'width': '100%', 'padding': '6px', 'borderRadius': '6px', 'border': '1px solid #ddd', 'fontSize': '13px', 'marginBottom': '8px'}
                    ),
                    html.Button('📈 View Results', id='view-results-button', n_clicks=0, disabled=True,
                               style={**BUTTON_STYLE, 'backgroundColor': '#805ad5'}),
                    html.Button('📂 Open Existing Results', id='open-existing-button', n_clicks=0,
                               title='Open viewer using existing results folder without rerunning the analysis',
                               style={**BUTTON_STYLE, 'backgroundColor': '#4299e1'})
                ], style={'display': 'flex', 'flexDirection': 'column'})
            ]),

        ], style={**CARD_STYLE, 'width': '400px'}),

    ], style={'display': 'flex', 'gap': '12px', 'alignItems': 'flex-start'}),


    # Collapsible advanced submenu
    html.Details([
        html.Summary('⚙️ Advanced Settings (click to expand)', 
                    style={'fontSize': '16px', 'fontWeight': 'bold', 'color': '#667eea', 
                           'cursor': 'pointer', 'padding': '12px', 'backgroundColor': '#f5f5f5', 
                           'borderRadius': '8px', 'marginTop': '16px'}),
        html.Div([
            html.Div([
                html.Label('🎞️ num_frames', style={'fontWeight': '500', 'fontSize': '13px'}), 
                dcc.Input(id='num_frames', type='number', value=getattr(av,'num_frames',20), 
                         style={'width': '100%', 'padding': '6px', 'borderRadius': '6px', 'border': '1px solid #ddd', 'fontSize': '13px'})
            ], style={'marginTop': '8px'}),

            html.H4('🗺️ Spatial Bounds', style={'marginTop': '16px', 'color': '#667eea'}),
            html.Div(style={'display': 'flex', 'gap': '8px'}, children=[
                html.Div([
                    html.Label('⬅️ min_x', style={'fontWeight': '500', 'fontSize': '13px'}), 
                    dcc.Input(id='min_boundary_x', type='number', value=av.min_boundary_x,
                             style={'width': '100%', 'padding': '6px', 'borderRadius': '6px', 'border': '1px solid #ddd', 'fontSize': '13px'})
                ], style={'flex': '1'}),
                html.Div([
                    html.Label('➡️ max_x', style={'fontWeight': '500', 'fontSize': '13px'}), 
                    dcc.Input(id='max_boundary_x', type='number', value=av.max_boundary_x,
                             style={'width': '100%', 'padding': '6px', 'borderRadius': '6px', 'border': '1px solid #ddd', 'fontSize': '13px'})
                ], style={'flex': '1'})
            ]),
            html.Div(style={'display': 'flex', 'gap': '8px', 'marginTop': '6px'}, children=[
                html.Div([
                    html.Label('⬇️ min_y', style={'fontWeight': '500', 'fontSize': '13px'}), 
                    dcc.Input(id='min_boundary_y', type='number', value=av.min_boundary_y,
                             style={'width': '100%', 'padding': '6px', 'borderRadius': '6px', 'border': '1px solid #ddd', 'fontSize': '13px'})
                ], style={'flex': '1'}),
                html.Div([
                    html.Label('⬆️ max_y', style={'fontWeight': '500', 'fontSize': '13px'}), 
                    dcc.Input(id='max_boundary_y', type='number', value=av.max_boundary_y,
                             style={'width': '100%', 'padding': '6px', 'borderRadius': '6px', 'border': '1px solid #ddd', 'fontSize': '13px'})
                ], style={'flex': '1'})
            ]),

            html.H4('🔧 Other Advanced Parameters', style={'marginTop': '16px', 'color': '#667eea'}),
            html.Div([
                html.Label('📐 DD', style={'fontWeight': '500', 'fontSize': '13px'}), 
                dcc.Input(id='DD', type='number', value=getattr(av,'DD',2),
                         style={'width': '100%', 'padding': '6px', 'borderRadius': '6px', 'border': '1px solid #ddd', 'fontSize': '13px'})
            ], style={'marginTop': '6px'}),
            html.Div([
                html.Label('📊 des', style={'fontWeight': '500', 'fontSize': '13px'}), 
                dcc.Input(id='des', type='number', value=getattr(av,'des',2),
                         style={'width': '100%', 'padding': '6px', 'borderRadius': '6px', 'border': '1px solid #ddd', 'fontSize': '13px'})
            ], style={'marginTop': '6px'}),
            html.Div([
                html.Label('📏 dim', style={'fontWeight': '500', 'fontSize': '13px'}), 
                dcc.Input(id='dim', type='number', value=getattr(av,'dim',2),
                         style={'width': '100%', 'padding': '6px', 'borderRadius': '6px', 'border': '1px solid #ddd', 'fontSize': '13px'})
            ], style={'marginTop': '6px'}),

            html.Div([
                html.Label('🔄 num_similar_configurations', style={'fontWeight': '500', 'fontSize': '13px'}), 
                dcc.Input(id='num_similar_configurations', type='number', value=getattr(av,'num_similar_configurations',5),
                         style={'width': '100%', 'padding': '6px', 'borderRadius': '6px', 'border': '1px solid #ddd', 'fontSize': '13px'})
            ], style={'marginTop': '6px'}),
            html.Div([
                html.Label('⚡ new_configuration_step', style={'fontWeight': '500', 'fontSize': '13px'}), 
                dcc.Input(id='new_configuration_step', type='number', value=getattr(av,'new_configuration_step',3),
                         style={'width': '100%', 'padding': '6px', 'borderRadius': '6px', 'border': '1px solid #ddd', 'fontSize': '13px'})
            ], style={'marginTop': '6px'}),
            html.Div([
                html.Label('➗ division_factor', style={'fontWeight': '500', 'fontSize': '13px'}), 
                dcc.Input(id='division_factor', type='number', value=getattr(av,'division_factor',5),
                         style={'width': '100%', 'padding': '6px', 'borderRadius': '6px', 'border': '1px solid #ddd', 'fontSize': '13px'})
            ], style={'marginTop': '6px'}),

        ], style={'padding': '12px', 'backgroundColor': '#fafafa', 'borderRadius': '8px', 'marginTop': '8px'})
    ], open=False, style={'marginTop': '16px', 'backgroundColor': '#fff', 'border': '2px solid #e0e0e0', 
                          'borderRadius': '10px', 'padding': '8px', 'boxShadow': '0 2px 8px rgba(0,0,0,0.08)'}),

    html.Div(id='output-div', style={
        'whiteSpace': 'pre-wrap', 
        'border': '2px solid #e0e0e0', 
        'padding': '16px', 
        'marginTop': '20px', 
        'minHeight': '200px', 
        'backgroundColor': '#ffffff',
        'borderRadius': '10px',
        'fontFamily': 'Consolas, monospace',
        'fontSize': '13px',
        'boxShadow': '0 2px 8px rgba(0,0,0,0.08)'
    }, children='💡 Ready to start. Configure parameters and click "Run Analysis".'),
    # Background interval to auto-check status and enable View Results when finished
    dcc.Interval(id='auto-status-interval', interval=2000, n_intervals=0),

], style={
    'fontFamily': "'Segoe UI', Arial, sans-serif", 
    'padding': '24px', 
    'maxWidth': '1200px', 
    'margin': '0 auto', 
    'backgroundColor': '#f5f7fa',
    'minHeight': '100vh'
})


# Callback to enable/disable buffer and rough parameters based on PDP type selection
@app.callback(
    [Output('buffer_x', 'disabled'), Output('buffer_y', 'disabled'),
     Output('rough_x', 'disabled'), Output('rough_y', 'disabled')],
    [Input('pdp-types', 'value')]
)
def toggle_parameters(pdp_list):
    """Enable buffer params if buffer/bufferrough selected; enable rough params if rough/bufferrough selected."""
    pdp_list = pdp_list or []
    
    # Buffer params enabled if 'buffer' or 'bufferrough' is selected
    buffer_disabled = not ('buffer' in pdp_list or 'bufferrough' in pdp_list)
    
    # Rough params enabled if 'rough' or 'bufferrough' is selected
    rough_disabled = not ('rough' in pdp_list or 'bufferrough' in pdp_list)
    
    return buffer_disabled, buffer_disabled, rough_disabled, rough_disabled


# Callback to enable "View Results" button after successful analysis
@app.callback(
    Output('view-results-button', 'disabled'),
    [Input('status-button', 'n_clicks'), Input('auto-status-interval', 'n_intervals')],
    [State('view-results-button', 'disabled')]
)
def enable_view_results_button(status_clicks, n_intervals, current_disabled):
    """Enable the View Results button when analysis is finished.

    This callback listens both to the manual "Get Status & Output" button and to the
    background Interval (`auto-status-interval`). The Interval allows automatic
    enabling of the View Results button when `last_status` becomes 'finished' without
    requiring the user to press "Get Status & Output".
    """
    # If the analysis finished, enable the button.
    try:
        if last_status == 'finished':
            return False
    except Exception:
        # If something weird happens reading last_status, keep current state.
        return current_disabled

    # Otherwise return the current state (keep it disabled or enabled as it is).
    return current_disabled


# Callback to handle View Results button click
@app.callback(
    Output('output-div', 'children', allow_duplicate=True),
    [Input('view-results-button', 'n_clicks')],
    [State('viewer-script-path', 'value'), State('dataset_name', 'value'), State('results_dir', 'value')],
    prevent_initial_call=True
)
def launch_results_viewer(n_clicks, script_path, dataset_name, results_dir):
    """Launch the provided visualization application (preferably Streamlit) using the given script path.
    The user can provide a path to a .py Streamlit script. We try to start it with `streamlit run <script>`.
    If that fails, we fall back to launching with plain `python <script>`.
    """
    if n_clicks and n_clicks > 0:
        try:
            if not script_path:
                return '❌ No script path provided. Please fill the "Results viewer script" field.'

            app_path = os.path.abspath(os.path.expanduser(script_path))
            if not os.path.exists(app_path):
                return f'❌ Error: script not found at {app_path}'

            # Use the script's directory as cwd so relative imports/resources resolve
            script_dir = os.path.dirname(app_path) or os.getcwd()

            # Prepare environment for subprocess: pass AV_RESULTS_DIR and AV_DATASET so the viewer auto-fills paths
            env = os.environ.copy()
            if results_dir:
                env['AV_RESULTS_DIR'] = results_dir
            # dataset_name state can override
            if dataset_name:
                env['AV_DATASET'] = dataset_name

            # Prefer running Streamlit with the same Python interpreter (works inside venvs):
            try:
                # Check if streamlit is importable in this Python environment
                try:
                    import streamlit  # type: ignore
                    has_streamlit = True
                except Exception:
                    has_streamlit = False

                if has_streamlit:
                    # Launch via `python -m streamlit run <script>` using the same interpreter
                    subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', app_path], cwd=script_dir, env=env)
                    return (f'✅ Streamlit started using {sys.executable} -m streamlit run {app_path} (AV_RESULTS_DIR={env.get("AV_RESULTS_DIR")}, AV_DATASET={env.get("AV_DATASET")}). '
                            "Open http://localhost:8501/ if it does not open automatically.")
                else:
                    # Fall back to launching the script as a plain Python program
                    subprocess.Popen([sys.executable, app_path], cwd=script_dir, env=env)
                    return (f'⚠️ `streamlit` is not installed in the Python environment ({sys.executable}). '
                            f'Launched with `python {app_path}` instead (AV_RESULTS_DIR={env.get("AV_RESULTS_DIR")}, AV_DATASET={env.get("AV_DATASET")} ).\n\nInstall Streamlit in this environment to use the interactive viewer:\n'
                            f'    {sys.executable} -m pip install streamlit')
            except Exception as e:
                # Final fallback: try plain python and report the error
                try:
                    subprocess.Popen([sys.executable, app_path], cwd=script_dir)
                    return f'⚠️ Failed to start via streamlit ({str(e)}). Launched with python instead.'
                except Exception as e2:
                    return f'❌ Error launching results viewer: {str(e2)}'

        except Exception as e:
            return f'❌ Error launching results viewer: {str(e)}'
    return ''


@app.callback(
    Output('output-div', 'children', allow_duplicate=True),
    [Input('open-existing-button', 'n_clicks')],
    [State('viewer-script-path', 'value'), State('results_dir', 'value'), State('dataset_name', 'value')],
    prevent_initial_call=True
)
def open_existing_results(n_clicks, script_path, results_dir, dataset_name):
    """Open the viewer using an existing results folder without rerunning the analysis."""
    if n_clicks and n_clicks > 0:
        try:
            if not script_path:
                return '❌ No script path provided. Please fill the "Results viewer script" field.'

            app_path = os.path.abspath(os.path.expanduser(script_path))
            if not os.path.exists(app_path):
                return f'❌ Error: script not found at {app_path}'

            script_dir = os.path.dirname(app_path) or os.getcwd()

            # Prepare environment for subprocess: pass AV_RESULTS_DIR so the viewer reads the right folder
            env = os.environ.copy()
            if results_dir:
                env['AV_RESULTS_DIR'] = results_dir
            if dataset_name:
                env['AV_DATASET'] = dataset_name

            # Try starting streamlit with the same interpreter if available
            try:
                try:
                    import streamlit  # type: ignore
                    has_streamlit = True
                except Exception:
                    has_streamlit = False

                if has_streamlit:
                    subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', app_path], cwd=script_dir, env=env)
                    return (f'✅ Streamlit started using {sys.executable} -m streamlit run {app_path} (AV_RESULTS_DIR={env.get("AV_RESULTS_DIR")} ). '
                            "Open http://localhost:8501/ if it does not open automatically.")
                else:
                    subprocess.Popen([sys.executable, app_path], cwd=script_dir, env=env)
                    return (f'⚠️ `streamlit` is not installed in the Python environment ({sys.executable}). '
                            f'Launched with `python {app_path}` instead (AV_RESULTS_DIR={env.get("AV_RESULTS_DIR")} ).\n\nInstall Streamlit in this environment to use the interactive viewer:\n'
                            f'    {sys.executable} -m pip install streamlit')
            except Exception as e:
                try:
                    subprocess.Popen([sys.executable, app_path], cwd=script_dir, env=env)
                    return f'⚠️ Failed to start via streamlit ({str(e)}). Launched with python instead.'
                except Exception as e2:
                    return f'❌ Error launching results viewer: {str(e2)}'

        except Exception as e:
            return f'❌ Error launching results viewer: {str(e)}'
    return ''


@app.callback(
    Output('output-div', 'children'),
    [Input('run-button', 'n_clicks'), Input('stop-button', 'n_clicks'), Input('status-button', 'n_clicks')],
    [
        State('pdp-types', 'value'),
        State('nva-options', 'value'),
        State('window_length_tst', 'value'),
        State('num_frames', 'value'),
        State('buffer_x', 'value'),
        State('buffer_y', 'value'),
        State('rough_x', 'value'),
        State('rough_y', 'value'),
        State('min_boundary_x', 'value'),
        State('max_boundary_x', 'value'),
        State('min_boundary_y', 'value'),
        State('max_boundary_y', 'value'),
        State('DD', 'value'),
        State('des', 'value'),
        State('dim', 'value'),
        State('num_similar_configurations', 'value'),
        State('new_configuration_step', 'value'),
        State('division_factor', 'value'),
    State('dataset_name', 'value'),
    State('results_dir', 'value')
    ]
)
def control_runner(run_clicks, stop_clicks, status_clicks, pdp_list, nva_list, window_length_tst, num_frames, buffer_x, buffer_y, rough_x, rough_y, min_boundary_x, max_boundary_x, min_boundary_y, max_boundary_y, DD, des, dim, num_similar_configurations, new_configuration_step, division_factor, dataset_name, results_dir):
    """Start background run when Run button clicked; stop on Stop button; return status/output on status button.
    """
    global run_thread, last_output, last_status, stop_requested

    ctx = callback_context
    if not ctx.triggered:
        return ''

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'run-button':
        # Map checklist selections to 0/1 flags for PDP types
        # Fundamental is always 1 (required)
        pdp_flags = {
            'PDPg_fundamental': 1,
            'PDPg_buffer': 1 if 'buffer' in (pdp_list or []) else 0,
            'PDPg_rough': 1 if 'rough' in (pdp_list or []) else 0,
            'PDPg_bufferrough': 1 if 'bufferrough' in (pdp_list or []) else 0,
        }

        # Map visualization/report checklist to flags
        nva_defaults = ['N_VA_Report','N_VA_StaticAbsolute','N_VA_StaticRelative','N_VA_StaticFinetuned','N_VA_DynamicAbsolute','N_PDP','N_VA_InequalityMatrices','N_VA_HeatMap','N_VA_HClust','N_VA_ClusterMap','N_VA_Mds','N_VA_TSNE','N_VA_UMAP','N_VA_Mds_autoencoder','N_VA_TopK','N_VA_Inverse']
        nva_flags = {k: (1 if k in (nva_list or []) else 0) for k in nva_defaults}
        
        # Auto-enable N_PDP if any module that needs it is selected
        pdp_dependent_modules = ['N_VA_HeatMap', 'N_VA_HClust', 'N_VA_ClusterMap', 'N_VA_Mds', 'N_VA_TSNE', 'N_VA_UMAP', 'N_VA_TopK', 'N_VA_InequalityMatrices']
        if any(nva_flags.get(mod, 0) == 1 for mod in pdp_dependent_modules):
            nva_flags['N_PDP'] = 1  # Force N_PDP to run to generate DistanceMatrix and/or InequalityMatrices
        
        # Auto-enable N_VA_StaticAbsolute if N_VA_Report is selected (Report needs StaticAbsolute images)
        if nva_flags.get('N_VA_Report', 0) == 1:
            nva_flags['N_VA_StaticAbsolute'] = 1  # Force StaticAbsolute to run for Report images

        # Prepare numeric params with fallbacks to av defaults
        params = {
            **pdp_flags,
            **nva_flags,
            'window_length_tst': int(window_length_tst) if window_length_tst is not None else av.window_length_tst,
            'num_frames': int(num_frames) if num_frames is not None else getattr(av,'num_frames',20),
            'buffer_x': int(buffer_x) if buffer_x is not None else av.buffer_x,
            'buffer_y': int(buffer_y) if buffer_y is not None else av.buffer_y,
            'rough_x': int(rough_x) if rough_x is not None else av.rough_x,
            'rough_y': int(rough_y) if rough_y is not None else av.rough_y,
            'min_boundary_x': int(min_boundary_x) if min_boundary_x is not None else av.min_boundary_x,
            'max_boundary_x': int(max_boundary_x) if max_boundary_x is not None else av.max_boundary_x,
            'min_boundary_y': int(min_boundary_y) if min_boundary_y is not None else av.min_boundary_y,
            'max_boundary_y': int(max_boundary_y) if max_boundary_y is not None else av.max_boundary_y,
            'DD': int(DD) if DD is not None else getattr(av,'DD',2),
            'des': int(des) if des is not None else getattr(av,'des',2),
            'dim': int(dim) if dim is not None else getattr(av,'dim',2),
            'num_similar_configurations': int(num_similar_configurations) if num_similar_configurations is not None else getattr(av,'num_similar_configurations',5),
            'new_configuration_step': int(new_configuration_step) if new_configuration_step is not None else getattr(av,'new_configuration_step',3),
            'division_factor': int(division_factor) if division_factor is not None else getattr(av,'division_factor',5),
            'dataset_name': dataset_name or getattr(av, 'dataset_name', 'N_C_Dataset.csv')
        }

        # Also set the *_active variants which some modules (e.g. N_PDP) check at runtime
        # The codebase uses both PDPg_buffer and PDPg_buffer_active in different places.
        params['PDPg_fundamental_active'] = int(params.get('PDPg_fundamental', 1))
        params['PDPg_buffer_active'] = int(params.get('PDPg_buffer', 0))
        params['PDPg_rough_active'] = int(params.get('PDPg_rough', 0))
        params['PDPg_bufferrough_active'] = int(params.get('PDPg_bufferrough', 0))

        # include results_dir if supplied
        if results_dir:
            params['results_dir'] = results_dir

        if run_thread is None or not run_thread.is_alive():
            run_thread = threading.Thread(target=run_moving_objects_in_background, args=(params,), daemon=True)
            run_thread.start()
            import datetime
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            return f'✅ [{timestamp}] Analysis STARTED in background.\n\n⏳ Processing... Click "Get Status & Output" to check progress.'
        else:
            return '⚠️ A run is already in progress. Use "Stop Analysis" to cancel or "Get Status & Output" to check progress.'

    elif trigger_id == 'stop-button':
        stop_requested = True
        if run_thread and run_thread.is_alive():
            import datetime
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            return f'⏹️ [{timestamp}] STOP requested.\n\n⏳ Waiting for analysis to terminate at next safe point...\n\nClick "Get Status & Output" to confirm termination.'
        else:
            return '⚠️ No analysis is currently running.'

    elif trigger_id == 'status-button':
        # Show last status and captured output with emoji
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        status_emoji = {
            'idle': '💤',
            'running': '⚙️',
            'finished': '✅',
            'stopped': '⏹️',
            'error': '❌'
        }
        emoji = status_emoji.get(last_status, '❓')
        header = f'[{timestamp}] {emoji} Status: {last_status.upper()}\n'
        separator = '='*60 + '\n'
        body = last_output or '(no output captured yet)'
        return header + separator + body

    return ''


def open_browser():
    """Open the browser after a short delay to ensure server is running."""
    webbrowser.open_new('http://127.0.0.1:8050/')


if __name__ == '__main__':
    # Open browser automatically after 1 second
    Timer(1, open_browser).start()
    app.run(debug=False)
