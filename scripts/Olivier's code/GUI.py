# Import required modules
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

# Initialize Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    dcc.Input(id='PDPg_fundamental', type='number', value=1),
    dcc.Input(id='PDPg_buffer', type='number', value=1),
    #... add more Inputs here for each parameter ...,
    html.Button('Run Script', id='button'),
    html.Div(id='output-div')
])

# Callback to update output-div when button is clicked
@app.callback(
    Output('output-div', 'children'),
    [Input('button', 'n_clicks')],
    [State('PDPg_fundamental', 'value'),
     State('PDPg_buffer', 'value')]
     #... add more States here for each parameter ...]
)
def update_output(n_clicks, PDPg_fundamental, PDPg_buffer):
    if n_clicks is not None:
        #... insert your code here ...,
        return f'Results: {PDPg_fundamental} and {PDPg_buffer}' # this is where you'd return your results,
    else:
        return ''

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
