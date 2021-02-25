# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import pandas as pd
import dash
from dash import no_update
import dash_table
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc

app = dash.Dash(__name__)

lipsum = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.'

style_row = {'display': 'inline-block', 'width': '500px', 'padding':0}
style_output = {'display': 'inline-block', 'width': '500px',
                'whiteSpace': 'pre-line', 'font-size':'110%'}
style_textarea = {'height': 200, 'width': 400}
# style_output = { 'font-size':'110%'}
app.layout = html.Div([
    html.Div([
        html.Div(dcc.Textarea(id='text1',value=lipsum,style=style_textarea), style=style_row),
        html.Div(id='text_output1', style={**style_output,**{'padding-right':100}}),
        html.Div(id='text_output2', style={**style_output,**{'padding':0}})
    ]),
    html.Button('Submit', id='submit_button', n_clicks=0),
    html.Br(), html.Br(),
    #html.Div(id='text_output', style=style_output),
])

@app.callback(
    [Output('text_output1', 'children'), Output('text_output2', 'children')],
    Input('submit_button', 'n_clicks'),
    State('text1', 'value')
)
def update_output(n_clicks, value):
    if n_clicks > 0:
        val1 = 'world: '+value
        val2 = 'hello: '+value
        return val1, val2
    else:
        return no_update, no_update

# app.layout = html.Div([
#     html.H2("Create your own enciphered poem"),
#     html.H3("Using only 12 letters: a, c, d, e, h, i, l, n, o, r, s, t"),
#     html.Br(),
#     dcc.Textarea(id='textarea_state',  value='Your poem goes here', style={'width': 400, 'height': 200}),
#     html.Br(),
#     html.Button('Submit', id='submit_button', n_clicks=0),
#     html.H4('Your enciphered poem is:'),
#     html.Div(id='textarea_output', style={'whiteSpace': 'pre-line', 'font-size':'110%'})
# ])
#
# @app.callback(
#     Output('textarea_output', 'children'),
#     Input('submit_button', 'n_clicks'),
#     State('textarea_state', 'value')
# )
# def update_output(n_clicks, value):
#     if n_clicks > 0:
#         return value


if __name__ == '__main__':
    app.run_server(host='127.0.0.1', port='8050', debug=True)

