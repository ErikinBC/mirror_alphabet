# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import os
import pandas as pd
import numpy as np
import string
from datetime import datetime

# from urllib.parse import quote as urlquote
# from flask import Flask, send_from_directory
import io

import dash
from dash import no_update
import dash_table
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash_extensions import Download
from dash_extensions.snippets import send_data_frame

from funs import alpha_trans, get_cipher, annot_vocab_cipher#, npair_max, 


app = dash.Dash(__name__)

# Set up directories
dir_base = os.getcwd()
dir_olu = os.path.join(dir_base, '..')
dir_output = os.path.join(dir_olu, 'output')

# Load in the data
df_idx = pd.read_csv(os.path.join(dir_output, 'dat_12.csv'),usecols=['idx','n'])
df_idx = df_idx.rename_axis('uix').reset_index()
df_words = pd.read_csv(os.path.join(dir_output,'words_etoaisnrlchd.csv'))
words12 = df_words.word
jalphabet12 = 'etoaisnrlchd'
alphabet12 = list(jalphabet12)
path_spacy = os.path.join(dir_output, 'pos_spacy.csv')
pos_spacy = pd.read_csv(path_spacy)


lipsum = 'rant tired sad ant'

style_row = {'display': 'inline-block', 'width': '500px', 'padding':0}
style_output = {'display': 'inline-block', 'width': '400px',
                'whiteSpace': 'pre-line', 'font-size':'110%'}
style_textarea = {'height': 200, 'width': 400}
# style_output = { 'font-size':'110%'}
app.layout = html.Div([
    html.H2("Create your own enciphered poem"),
    html.H3("Using only 12 letters: a, c, d, e, h, i, l, n, o, r, s, t"),
    html.Br(),
    html.Div([
        html.Div(dcc.Textarea(id='text1',value=lipsum,style=style_textarea), style=style_row),
        html.Div(id='text_output1', style={**style_output,**{'padding-right':100}}),
        html.Div(id='text_output2', style={**style_output,**{'padding':0}})
    ]),
    html.Button('Submit', id='submit_button', n_clicks=0),
    html.Br(), html.Br(),
    html.H3('Pick an index from 0-10394 (0 has most words, 10394 has the fewest)'),
    dcc.Input(id='user_idx',placeholder='Enter an interger...',type='number', value=0,min=0,max=10394,step=1),
    html.Br(), 
    html.Div(id='text_output3', style={**style_output,**{'padding-right':100}}),
    html.Div(id='text_output4', style={**style_output,**{'padding-right':100}}),
    html.Div(id='text_output5', style={**style_output,**{'padding-right':100}}),
    html.Br(), html.Br(),
    html.Div([html.Button("Download", id="btn"), Download(id="download")]),
    html.Br(), html.Br(), html.Br(), html.Br(), 
    html.Div(id='table-container',  className='tableDiv'),
    html.Br(), html.Br(), html.Br(), html.Br(), 
])

#df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [2, 1, 5, 6], 'c': ['x', 'x', 'y', 'y']})
@app.callback(
    Output("download", "data"),
    [Input("btn", "n_clicks")]
)
def func(n_clicks):
    if os.path.exists('tmp.csv'):
        print('exists')
        df = pd.read_csv('tmp.csv')
        fn = 'poem_' + datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '.csv'
        os.remove('tmp.csv')
        return send_data_frame(df.to_csv, fn, index=False)

@app.callback(
    [Output('text_output1', 'children'), Output('text_output2', 'children'), 
     Output('text_output3', 'children'), Output('text_output4', 'children'), Output('text_output5', 'children'),
     Output('table-container','children'), ],
    [Input('user_idx', 'value'), Input('submit_button', 'n_clicks')],
    State('text1', 'value')
)
def update_output(idx, n_clicks, txt):
    # Get the encipher index
    idx = df_idx.iloc[idx]['idx']
    xmap_idx = get_cipher(idx, alphabet12)
    cipher_map = ', '.join(pd.DataFrame(xmap_idx).apply(lambda x: x[0]+':'+x[1],1).to_list())
    punct = ''.join(['\\'+z for z in list(string.punctuation)])
    if n_clicks > 0:
        txt = pd.Series(txt).str.replace('[^\\s'+punct + jalphabet12 + ']','',regex=True)[0]
        val1 = 'original:\n' + txt
        ttxt = alpha_trans(txt, xmap_idx)
        val2 = 'enciphered:\n' + ttxt
        vv = [val1, val2]
        pd.DataFrame({'idx':idx, 'original':txt, 'enciphered':ttxt},index=[0]).to_csv('tmp.csv',index=False)
    else:
        vv = [no_update, no_update]
    vocab_idx = annot_vocab_cipher(corpus=words12, PoS=pos_spacy,  cipher=get_cipher(idx, alphabet12))
    vocab_idx = vocab_idx.rename_axis("word_num").reset_index()

    res_idx = vocab_idx.pos_x.value_counts().reset_index().rename(columns={'index':'pos','pos_x':'n'})
    n_idx = 'Parts of speech:\n' + ', '.join(list(res_idx.apply(lambda x: x.astype(str).str.cat(sep=' ('),1) + ')'))
    neq_idx = vocab_idx.groupby(['pos_x','pos_y']).size().reset_index().query('pos_x == pos_y')
    neq_idx = 'Matching PoS:\n' + ', '.join(list(neq_idx[['pos_x',0]].apply(lambda x: x.astype(str).str.cat(sep=' ('),1) + ')'))

    qq = dash_table.DataTable(id='table',
        columns=[{"name": i, "id": i} for i in vocab_idx.columns],
        data=vocab_idx.to_dict('records'),
        style_table={'height': '4100px', 'width':'1000px', 'overflowY': 'auto', 'overflowX': 'auto'})
    return vv + [cipher_map, n_idx, neq_idx, qq, ]


if __name__ == '__main__':
    app.run_server(host='127.0.0.1', port='8050', debug=True)

