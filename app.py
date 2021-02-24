# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import pandas as pd
import dash
import dash_table
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc

app = dash.Dash(__name__)

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/solar.csv').iloc[:,0:2]

app.layout = html.Div([
    html.H2("Create your own enciphered poem"),
    html.H3("Using only 12 letters: a, c, d, e, h, i, l, n, o, r, s, t"),
    html.Br(),
    html.Div([dcc.Textarea(id='textarea_state',  value='Your poem goes here',
                     style={'width': 400, 'height': 200}),
             html.Button('Submit', id='submit_button', n_clicks=0)],
             style={'width': '49%', 'display': 'inline-block'}),
    html.Div(dash_table.DataTable(id='table', 
                                  columns=[{"name": i, "id": i} for i in df.columns],
                                  data=df.to_dict('records'),
                                  style_table={'minWidth': '5px', 'width': '5px', 'maxWidth': '5px'})),
])


# app.layout = html.Div([

#     html.H4('Your enciphered poem is:'),
#     html.Div(id='textarea_output', style={'whiteSpace': 'pre-line', 'font-size':'110%'})
# ])

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

