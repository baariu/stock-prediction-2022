import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from datetime import datetime as dt
import pandas as pd
import yfinance as yf
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objs as go
# model
from Stock_model import prediction
from sklearn.svm import SVR


app = dash.Dash(__name__)
server = app.server

options = [
    {'label': 'Apple', 'value': 'AAPL'},
    {'label': 'Google', 'value': 'GOOGL'},
    {'label': 'IBM', 'value': 'IBM'},
    {'label': 'Microsoft', 'value': 'MSFT'},
    {'label': 'Netflix', 'value': 'NFLX'},
    {'label': 'Tesla', 'value': 'TSLA'}
          ]

colors = {
    'background': '#191414',
    'text': '#1DB954'
         }

app.layout = html.Div([ #Main division

html.Div([ #Division 1
        html.H2("Welcome to the stock Dash App!", className="start", style={'marginTop':'45px', 'color': '#7FDBFF'}),
        #stock code input
        html.Div([
                    dcc.Dropdown(
                        id='dropdown_symbol',
                        options= options,
                        value="",
                        placeholder="Select a stock...",
                        multi=False,
                        style={'width':'300px','height':'25px','fontSize': 18,'marginLeft':'-15px','backgroundColor':'##800000'}),
                #submit button
                    html.Button(id='submit_button',
                        n_clicks=0,
                        children= "Submit",
                        style={'width': '100px','height':'37px','fontSize': 18,'marginLeft':'-15px','backgroundColor':'#FFFF00'}), 
                ],style = {'display': 'flex','marginTop':'70px','color': '##800000' }),

        html.Div([#Date range picker input
                    html.H2('Enter start and end date:',style={'color':'#FFFFFF'}), 
                    dcc.DatePickerRange(
                        id='date_range',
                        min_date_allowed=dt(2015, 1, 1),
                        max_date_allowed=dt.now(),
                        start_date_placeholder_text="Start",
                        end_date=dt.now(),
                        number_of_months_shown=2,
                        style={'fontSize': 18,'marginTop':'2px'})
                 ],style={'marginTop':'40px'}),

        html.Div([ #Stock price button
                    html.Button(id='stock-price-button',
                        n_clicks=0,
                        children='Stockprice',
                        style={'fontSize': 18,'height':'40px','marginLeft':'100px','marginBottom':'35px','backgroundColor': '#00FF00'}),

                   #Indicators button
                    html.Button(id='Indicators_button',
                        n_clicks=0,
                        children='Indicators',
                        style={'fontSize': 18,'height':'40px','marginLeft':'80px','marginTop':'15px','backgroundColor': '#00FF00'}),
                        
                   #Number of days of forecast input, Forecast button
                    dcc.Input(id="number_of_days",
                        type='number',
                        placeholder="Number of days",
                        value= "",
                        min= 1,
                        debounce= False,
                        style={'fontSize': 18,'height':'40px','marginLeft':'50px','backgroundColor': '##800000'}),

                   #forecast button    
                    html.Button(id='forecast_button',
                        n_clicks=0,
                        children='Forecast',
                        style={'fontSize': 18,'height':'40px','marginLeft':'30px','backgroundColor': '#00FF00'})

                ],style={'display':'inline','marginTop':'50px','verticalAlign': 'middle'})
],style={'height': '45vw','width': '35vw','align-items':'center','display':'flex','flex-direction':'column','justify-content': 'flex-start','background-color':'rgb(5, 107, 107)'}, className="nav"),

html.Div([#Division 2

        html.Div([html.H2(id='ticker'), html.Img(id="logo")],className= "header", style={'margin-bottom':'10px'}),

        #Description
        html.Div( id= "description", className="description_ticker"),

        #Stock price plot
        dcc.Graph(id="graphs-content", figure={}), 

        #Indicator plot
        dcc.Graph(id="main-content",figure={}),

        #Forecast plot
        html.Div([],id="forecast-content")

],style={'marginTop':'-550px','marginLeft':'550px','width': '50vw','align-items':'centre','display':'flex','flex-direction':'column','justify-content': 'flex-start'},className='content'), 
],className="container")

@app.callback( #1st callback for outputing logo,name and description of the stock chosen.
    
    [Output("ticker","children"), Output("logo","src"),Output("description", "children")], 

    [Input('submit_button','n_clicks'), State('dropdown_symbol','value')]
    )
def update_data(n,value): # n represents the input component_property "n_clicks". 
    #input is what triggers the call back so if you have a button,it has to be your input because it's the one effecting the change.
    #value_chosen represents the component_property "value"
    if n==0:
        raise PreventUpdate
    else:
        ticker= yf.Ticker(value)
        inf= ticker.info
        df= pd.DataFrame().from_dict(inf,orient = "index").T
        dff=df[["logo_url","shortName","longBusinessSummary"]]
        return dff['shortName'].values[0], dff['logo_url'].values[0], dff['longBusinessSummary'].values[0]

@app.callback(#2nd callback. This updates the stock plot graph by using the stockprice button to get the graph 
     Output("graphs-content","figure"),
    [Input('stock-price-button','n_clicks')],
    [State('date_range','start_date'),State('date_range','end_date'),State('dropdown_symbol','value')]
    )
def update_graph(n,start,end,value):                                                                                                                                            
    if n==0:
        raise PreventUpdate  
    else:
        df = yf.download(value, start= start , end= end, period= max)
        df.reset_index(inplace=True)
        
        def get_stock_price_fig(df):
            fig = px.line(df,
                x='Date', # Date str,
                y=['Open','Close'],# list of 'Open' and 'Close',
                title="Closing and Opening Price vs Date")  
            return fig

    figure = get_stock_price_fig(df)
    return figure # plot the graph of fig using DCC function

@app.callback( #3rd Callback. Displays the indicator graph over time
  Output("main-content",'figure'), 
  [Input('Indicators_button','n_clicks')],
  [State('date_range','start_date'),State('date_range','end_date'),State('dropdown_symbol','value')]
  ) 
def update_indicator(n,start,end,value):
    if n==0:
        raise PreventUpdate
    else:
        df = yf.download(value, start= start,end= end, period= max)
        df.reset_index(inplace=True)

        def get_indicator(df):
            df['EWA_20'] = df['Close'].ewm(span=20, adjust=False).mean() #The estimated weighted mean of selected stock.
                                                                                        #Moving Average is the indicator we are using.                                                                           
            fig = px.scatter(df,
                    x= 'Date',# Date str,
                    y= 'EWA_20',# EWA_20 str,
                    title="Exponential Moving Average vs Date")

            fig.update_traces(mode='lines+markers') # appropriate mode
            return fig
        figure = get_indicator(df)
        return figure

@app.callback( #Last CallBack for FORECASTING Using the ML model

     Output("forecast-content","children"),
     [Input('forecast_button','n_clicks')],
     [State('dropdown_symbol','value'),State("number_of_days",'value')]
     
     )
def predict_stock(n,value,n_days):
     if n==0:
         raise PreventUpdate
     else:
        fig = prediction(value, int(n_days) + 1)
        return [dcc.Graph(figure=fig)] # plot the graph of fig using DCC function


if __name__ == '__main__':
    app.run_server(debug = True) 