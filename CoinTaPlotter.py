import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import sqlite3
import mplfinance as mpf
from tabulate import tabulate
import plotly.graph_objects as go
import json, requests, urllib, os, time, urllib.request, sqlite3, datetime, math
from plotly.offline import iplot
from plotly.subplots import make_subplots
from binance.client import Client
import plotly


class stock:
    def __init__(self, name):
        self.name = name
        self.data = {'TIME': 0, 'OPEN': 0, 'HIGH': 0, 'LOW': 0, 'CLOSE': 0, 'VOLUME': 0}
        self.df60 = pd.DataFrame(data=self.data, index=[0])
        self.df240 = pd.DataFrame(data=self.data, index=[0])
        self.df365 = pd.DataFrame(data=self.data, index=[0])
        self.State = 1
        self.C1, self.C2, self.C3 = 0, 0, 0
        print(f"{self.name} is created...")
        # self.filename_tf1 = "C:/Users/forga/OneDrive/Masaüstü/Codecamp/Python/TeleReport/DBs/H1_" + self.name

    @staticmethod
    def update_indicators(df):
        df['RSI'] = calc_rsi(df['CLOSE'])
        df['IFTRSI'] = calc_iftrsi(df['RSI'])
        df['BOLL_H'], df['BOLL_L'], df['BOLL_M'] = calc_bollinger(df['CLOSE'])
        df['MACD'], df['MACD_S'] = calc_macd(df['CLOSE'])
        df['STD'] = calc_std(df['CLOSE'])
        df['MA_200'] = sma(df['CLOSE'], 200)
        df['MA_50'] = sma(df['CLOSE'], 50)
        df['MA_20'] = sma(df['CLOSE'], 20)
        df['VOLUME(10)'] = sma(df['VOLUME'], 10)

    def initialize_dfs(self, time_before_end, end_time, resolution, exchange):
        res_dfs = {'60': self.df60, '240': self.df240, '365': self.df365}
        if exchange == 'BIST':
            start_time = end_time - time_before_end
            for res in resolution:
                t = time.time()
                url = 'https://web-cloud-new.foreks.com/tradingview-services/trading-view/history?symbol=' + self.name + '.E.BIST&resolution=' + res + '&from=' + str(
                    int(start_time)) + '&to=' + str(int(end_time)) + '&currency=TRL'
                data = urllib.request.urlopen(url).read()
                data = json.loads(data.decode('utf-8'))
                print(f'{self.name} data for {res} timeframe were collected in {time.time() - t:.2f} seconds...')
                res_dfs[res].drop(index=res_dfs[res].index[0], axis=0, inplace=True)
                for i in range(len(data['t'])):
                    res_dfs[res] = res_dfs[res].append({
                        'TIME': data['t'][i],
                        'DATE': str(datetime.datetime.fromtimestamp(data['t'][i] / 1000.0)),
                        'OPEN': data['o'][i],
                        'HIGH': data['h'][i],
                        'LOW': data['l'][i],
                        'CLOSE': data['c'][i],
                        'VOLUME': data['v'][i],
                        'RSI': 0, 'IFTRSI': 0, 'BOLL_H': 0, 'BOLL_M': 0, 'BOLL_L': 0, 'STD': 0, 'MACD': 0, 'MACD_S': 0,
                        'MA_200': 0, 'MA_50': 0, 'MA_20': 0, 'VOLUME(10)': 0, 'BUY': 0
                    }, ignore_index=True)
                self.update_indicators(res_dfs[res])
                print(f"{self.name} dataframe is initialized for {res} timeframe...")
        else:
            for res in resolution:
                if res == '60':
                    timeframe = Client.KLINE_INTERVAL_1HOUR
                elif res == '240':
                    timeframe = Client.KLINE_INTERVAL_4HOUR
                    time_before_end = int(time_before_end * 1)
                elif res == '365':
                    timeframe = Client.KLINE_INTERVAL_1DAY
                    time_before_end *= 2
                AnlApi_key = 'n3iweYuOAx3UXKyT95t2q9MiT4dy7cMiwghjs8TSAFLDYBpeEwElWgbLIVn1Nu7B'
                AnlApi_keySecret = 'kreH4pr4kX22FXrbni0ExyJuXasnMeXNPDfW6P1NzZbVKRq4WVgOg2mja4gb9rhT'
                client = Client(AnlApi_key, AnlApi_keySecret)
                start_time = end_time - time_before_end
                klines = client.get_historical_klines(self.name, timeframe, start_time, end_time)
                for i in range(len(klines) - 1):
                    res_dfs[res] = res_dfs[res].append({
                        'TIME': float(klines[i][0]),
                        'DATE': str(datetime.datetime.fromtimestamp(float(klines[i][0]) / 1000.0)),
                        'OPEN': float(klines[i][1]),
                        'HIGH': float(klines[i][2]),
                        'LOW': float(klines[i][3]),
                        'CLOSE': float(klines[i][4]),
                        'VOLUME': float(klines[i][5]),
                        'RSI': 0, 'IFTRSI': 0, 'BOLL_H': 0, 'BOLL_M': 0, 'BOLL_L': 0, 'STD': 0, 'MACD': 0, 'MACD_S': 0,
                        'MA_200': 0, 'MA_50': 0, 'MA_20': 0, 'VOLUME(10)': 0, 'BUY': 0
                    }, ignore_index=True)
                res_dfs[res] = res_dfs[res][1:]
                self.update_indicators(res_dfs[res])
                print(f"{self.name} dataframe is initialized for {res} timeframe...")
        self.df60, self.df240, self.df365 = res_dfs['60'], res_dfs['240'], res_dfs['365']

def calc_rsi(series):
    rsi_length = 14
    delta = series.diff().dropna()
    delta = delta[1:]
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up1 = up.ewm(alpha=1 / rsi_length).mean()
    roll_down1 = down.abs().ewm(alpha=1 / rsi_length).mean()
    rsi1 = 100.0 - (100.0 / (1.0 + roll_up1 / roll_down1))
    return rsi1


def calc_iftrsi(series):
    period = 9
    v1 = 0.1 * (series - 50)
    v2 = v1.ewm(alpha=1 / period).mean()
    iftrsi = (np.exp(2 * v2) - 1) / (np.exp(2 * v2) + 1)
    return iftrsi


def calc_bollinger(df):
    length = 20
    std = 2
    return (df.rolling(window=length).mean() + std * df.rolling(window=length).apply(
        np.std)), (df.rolling(window=length).mean() - std * df.rolling(window=length).apply(np.std)), (
               df.rolling(window=length).mean())


def sma(df, period):
    return df.rolling(window=period).mean()


def calc_std(df):
    length = 20
    return df.rolling(window=length).apply(np.std)


def calc_macd(series):
    exp1 = series.ewm(span=12, adjust=False).mean()
    exp2 = series.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal


def get_pivots(df, left, right):
    # df = df.set_index('TIME')
    # df = df.reset_index()
    loopback = int(len(df) * 0.9)
    pivot_point_high, pivot_point_high_time = [], []
    pivot_point_low, pivot_point_low_time = [], []
    piv_idx_high, piv_idx_low = np.array([0]), np.array([0])
    i = len(df) - loopback + left
    while i < len(df) - right - 1:
        if left == np.argmax(df['High'][i - left:i + right + 1].values):
            piv_idx_high = np.append(piv_idx_high, i - left + 1 + np.argmax(df['High'][i - left:i + right + 1].values))
        if left == np.argmin(df['Low'][i - left:i + right + 1].values):
            piv_idx_low = np.append(piv_idx_low, i - left + 1 + np.argmin(df['Low'][i - left:i + right + 1].values))
        i += 1

    piv_idx_high = np.delete(piv_idx_high, 0)
    piv_idx_low = np.delete(piv_idx_low, 0)

    pivot_point_high = df['High'][piv_idx_high].values
    pivot_point_high_time = df['TIME'][piv_idx_high].values
    pivot_point_low = df['Low'][piv_idx_low].values
    pivot_point_low_time = df['TIME'][piv_idx_low].values

    return pivot_point_high, pivot_point_low, pivot_point_high_time, pivot_point_low_time


def find_supports(pivot_point_low, pivot_point_low_time, threshold):
    supports = np.array([pivot_point_low[0]])
    support_width = np.array([0])
    support_time = np.array([pivot_point_low_time[0]])
    for idx, piv in enumerate(pivot_point_low):
        supports = np.append(supports, piv)
        support_width = np.append(support_width, 1)
        support_time = np.append(support_time, pivot_point_low_time[idx])
        # print(f"pivot: {piv}\nsupports: {supports}\nwidth: {support_width}\n")
        for s in range(len(supports) - 1):
            if piv * (1 + threshold / 100) > supports[s] and supports[s] > piv * (1 - threshold / 100):
                # print(True, s)
                supports = np.delete(supports, -1)
                support_width = np.delete(support_width, -1)
                support_time = np.delete(support_time, -1)
                support_width[s] += 1
                # print(f"pivot: {piv}\nsupports: {supports}\nwidth: {support_width}\n")
                break
        # print(f"pivot: {piv}\nsupports: {supports}\nwidth: {support_width}\n\n")

    return supports, support_width, support_time


def find_resistances(pivot_point_high, pivot_point_high_time, threshold):
    resistances = np.array([pivot_point_high[0]])
    resistance_width = np.array([0])
    resistance_time = np.array([pivot_point_high_time[0]])
    for idx, piv in enumerate(pivot_point_high):
        resistances = np.append(resistances, piv)
        resistance_width = np.append(resistance_width, 1)
        resistance_time = np.append(resistance_time, pivot_point_high_time[idx])
        for s in range(len(resistances) - 1):
            if piv * (1 + threshold / 100) > resistances[s] and resistances[s] > piv * (1 - threshold / 100):
                resistances = np.delete(resistances, -1)
                resistance_width = np.delete(resistance_width, -1)
                resistance_time = np.delete(resistance_time, -1)
                resistance_width[s] += 1
                break

    return resistances, resistance_width, resistance_time


def plot_pivot(resolution):
    if resolution == '60':
        df = stock_.df60.copy()
        left = 8
        right = 8
        SUPPORT_THRESHOLD = 0.25
        RESISTANCE_THRESHOLD = 0.25
    elif resolution == '240':
        df = stock_.df240.copy()
        left = 8
        right = 8
        SUPPORT_THRESHOLD = 0.8
        RESISTANCE_THRESHOLD = 0.8
    elif resolution == '365':
        df = stock_.df365.copy()
        left = 4
        right = 4
        SUPPORT_THRESHOLD = 1
        RESISTANCE_THRESHOLD = 1
    df = df.rename(columns={'OPEN': 'Open', 'HIGH': 'High', 'LOW': 'Low', 'CLOSE': 'Close', 'VOLUME': 'Volume'})
    df['TIME'] = pd.to_datetime(df['TIME'], unit='ms')
    pivot_point_high, pivot_point_low, pivot_point_high_time, pivot_point_low_time = get_pivots(df, left, right)

    trace1 = go.Candlestick(x=df['TIME'],
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close'], showlegend=False,
                            name='candlestick')

    trace2 = go.Scatter(x=df['TIME'],
                        y=df['MA_20'],
                        line_color='black',
                        name='MA_20')

    trace3 = go.Scatter(x=df['TIME'],
                        y=df['MA_200'],
                        line_color='red',
                        name='MA_200')

    trace4 = go.Scatter(x=df['TIME'],
                        y=df['BOLL_H'],
                        line_color='gray',
                        line={'dash': 'solid'},
                        name='upper band',
                        opacity=0.5)

    trace5 = go.Scatter(x=df['TIME'],
                        y=df['BOLL_L'],
                        line_color='gray',
                        line={'dash': 'solid'},
                        name='lower band',
                        opacity=0.5)

    data = [trace1, trace2, trace3, trace4, trace5]

    layout = {
        "xaxis": {"rangeselector": {
            "x": 0,
            "y": 0.95,
            "font": {"size": 13},
            "visible": True,
            "bgcolor": "rgba(150, 200, 250, 0.4)",
            "buttons": [
                {
                    "step": "all",
                    "count": 1,
                    "label": "reset"
                },
                {
                    "step": "month",
                    "count": 1,
                    "label": "1month",
                    "stepmode": "backward"
                },
                {
                    "step": "day",
                    "count": 14,
                    "label": "1 week",
                    "stepmode": "backward"
                },
                {
                    "step": "day",
                    "count": 28,
                    "label": "2 weeks",
                    "stepmode": "backward"
                },
                {"step": "all"}
            ]
        },
            "rangeslider": {"visible": False}
        },
        "yaxis": {
            "domain": [0, 0.2],
            "showticklabels": False
        },
        "legend": {
            "x": 0.3,
            "y": 0.95,
            "yanchor": "bottom",
            "orientation": "h"
        },
        "margin": {
            "b": 10,
            "l": 10,
            "r": 10,
            "t": 10
        }
    }

    fig = go.Figure(data=data, layout=layout)

    fig.update_layout(
        autosize=False,
        margin=dict(l=10, r=10, t=40, b=10),
        width=1400,
        height=800, )

    # Type casting and Drawing pivot points
    for i in range(len(pivot_point_high_time)):
        pivot_point_high_time[i] = str(pivot_point_high_time[i])

    for i in range(len(pivot_point_low_time)):
        pivot_point_low_time[i] = str(pivot_point_low_time[i])

    fig.add_trace(
        go.Scatter(
            x=pivot_point_high_time,  # ["2022-02-08 14:00:00"],
            y=np.multiply(pivot_point_high, 1.006),
            mode="markers",
            marker=dict(symbol='triangle-down', size=12, color='red'),
            name='pivot_high'
        )
    )

    fig.add_trace(
        go.Scatter(
            x=pivot_point_low_time,  # ["2022-02-08 14:00:00"],
            y=np.multiply(pivot_point_low, 0.994),
            mode="markers",
            marker=dict(symbol='triangle-up', size=12, color='blue'),
            name='pivot_low'
        )
    )

    # FIND SUPPORT LINES AND REGIONS
    supports, support_width, support_time = find_supports(pivot_point_low, pivot_point_low_time, SUPPORT_THRESHOLD)
    s = 0
    for s in range(len(supports)):
        if supports[s] < df['MA_20'].iloc[-1]:
            if support_width[s] > 3:
                fig.add_shape(type="rect",
                              x0=str(support_time[s]), y0=supports[s] * (1 + SUPPORT_THRESHOLD / 100),
                              x1=str(df['TIME'].iloc[-1]), y1=supports[s] * (1 - SUPPORT_THRESHOLD / 100),
                              line=dict(color="RoyalBlue", width=2),
                              fillcolor="LightSkyBlue",
                              opacity=0.5
                              )
            else:
                fig.add_shape(type="line",
                              x0=str(support_time[s]), y0=supports[s], x1=str(df['TIME'].iloc[-1]), y1=supports[s],
                              line=dict(color="Blue", width=support_width[s], dash="dashdot")
                              )

    # FIND RESISTANCE LINES AND REGIONS
    resistances, resistance_width, resistance_time = find_resistances(pivot_point_high, pivot_point_high_time,
                                                                      RESISTANCE_THRESHOLD)
    s = 0
    for s in range(len(resistances)):
        if resistances[s] > df['MA_20'].iloc[-1]:
            if resistance_width[s] > 3:
                fig.add_shape(type="rect",
                              x0=str(resistance_time[s]), y0=resistances[s] * (1 + RESISTANCE_THRESHOLD / 100),
                              x1=str(df['TIME'].iloc[-1]), y1=resistances[s] * (1 - RESISTANCE_THRESHOLD / 100),
                              line=dict(color="darkred", width=2),
                              fillcolor="mediumvioletred",
                              opacity=0.5
                              )
            else:
                fig.add_shape(type="line",
                              x0=str(resistance_time[s]), y0=resistances[s], x1=str(df['TIME'].iloc[-1]),
                              y1=resistances[s],
                              line=dict(color="Red", width=resistance_width[s], dash="dashdot")
                              )

    # fig.add_hline(x0 = pivot_point_low_time[4], x1 = pivot_point_low_time[7], type = 'line', xsizemode = 'scaled', y=supports[s], line_width=support_width[s], line_color="blue", line_dash="dash")

    fig2 = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.01, row_width=[0.15, 0.1, 0.1, 0.65],
                         figure=fig)  # , subplot_titles = (stock_.name, 'RSI', 'Volume')

    fig2.add_trace(go.Bar(x=df['TIME'], y=df['Volume'], showlegend=False), row=2, col=1)

    fig2.add_trace(go.Scatter(x=df['TIME'], y=df['RSI'], showlegend=False), row=3, col=1)
    fig2.add_hline(y=70, line_width=1, line_color="black", line_dash="dash", row=3, col=1)
    fig2.add_hline(y=30, line_width=1, line_color="black", line_dash="dash", row=3, col=1)

    fig2.add_trace(go.Scatter(x=df['TIME'], y=df['MACD'], showlegend=False), row=4, col=1)
    fig2.add_trace(go.Scatter(x=df['TIME'], y=df['MACD_S'], showlegend=False), row=4, col=1)

    return fig2


def create_Stocks():
    #StockList = ['ASELS',]
    StockList = ['BTCUSDT',]
    MyList = np.array([])
    for i in range(len(StockList)):
        MyList = np.append(MyList, stock(StockList[i]))
    return MyList


stock_list = create_Stocks()
stock_ = stock_list[0]

END_TIME = int(time.time()*1000)
TIME_BEFORE_END = 60*60*24*30*1000 #60

EXCHANGE = 'CRYPTO' #'BIST'
RESOLUTION = ['60', '240', ]#'365'

stock_.initialize_dfs(TIME_BEFORE_END, END_TIME, RESOLUTION, EXCHANGE)
#print(tabulate(stock_.df365.tail(2), headers = stock_.df365.columns))

'''fig2 = plot_pivot('240')
fig2.update_layout(
    dragmode='drawrect', # define dragmode
    newshape=dict(line_color='cyan'))
# Add modebar buttons
#fig2.show(config={'modeBarButtonsToAdd':['drawline',
                                        'drawopenpath',
                                        'drawclosedpath',
                                        'drawcircle',
                                        'drawrect',
                                        'eraseshape'
                                       ]})
#fig2.show()'''

def plot_pivot_1H_from(resolution):
    if resolution == '240':
        df = stock_.df240.copy()
        left = 8
        right = 8
        SUPPORT_THRESHOLD = 0.8
        RESISTANCE_THRESHOLD = 0.8
    elif resolution == '365':
        df = stock_.df365.copy()
        left = 4
        right = 4
        SUPPORT_THRESHOLD = 0.8
        RESISTANCE_THRESHOLD = 0.8
    df = df.rename(columns={'OPEN': 'Open', 'HIGH': 'High', 'LOW': 'Low', 'CLOSE': 'Close', 'VOLUME': 'Volume'})
    df['TIME'] = pd.to_datetime(df['TIME'], unit='ms')

    # Finding pivot points from specified timeframe chart
    pivot_point_high, pivot_point_low, pivot_point_high_time, pivot_point_low_time = get_pivots(df, left, right)

    # Declaring 1H chart data
    df = stock_.df60.copy()
    df = df.rename(columns={'OPEN': 'Open', 'HIGH': 'High', 'LOW': 'Low', 'CLOSE': 'Close', 'VOLUME': 'Volume'})
    df['TIME'] = pd.to_datetime(df['TIME'], unit='ms')

    # Trimming data that is not present in 1H chart(old datetime data)
    for idx, val in enumerate(pivot_point_high_time):
        if val > df['TIME'].iloc[0]:
            pivot_point_high_time = pivot_point_high_time[idx:]
            pivot_point_high = pivot_point_high[idx:]
            break
    for idx, val in enumerate(pivot_point_low_time):
        if val > df['TIME'].iloc[0]:
            pivot_point_low_time = pivot_point_low_time[idx:]
            pivot_point_low = pivot_point_low[idx:]
            break

    # Constructing the candlestick(base) figure with MA and Bollinger Bands
    trace1 = go.Candlestick(x=df['TIME'],
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close'], showlegend=False,
                            name='candlestick')

    trace2 = go.Scatter(x=df['TIME'],
                        y=df['MA_20'],
                        line_color='black',
                        name='MA_20')

    trace3 = go.Scatter(x=df['TIME'],
                        y=df['MA_200'],
                        line_color='red',
                        name='MA_200')

    trace4 = go.Scatter(x=df['TIME'],
                        y=df['BOLL_H'],
                        line_color='gray',
                        line={'dash': 'solid'},
                        name='upper band',
                        opacity=0.5)

    trace5 = go.Scatter(x=df['TIME'],
                        y=df['BOLL_L'],
                        line_color='gray',
                        line={'dash': 'solid'},
                        name='lower band',
                        opacity=0.5)

    # Appending traces and configuring the layout
    data = [trace1, trace2, trace3, trace4, trace5]
    layout = {
        "xaxis": {"rangeselector": {
            "x": 0,
            "y": 0.95,
            "font": {"size": 13},
            "visible": True,
            "bgcolor": "rgba(150, 200, 250, 0.4)",
            "buttons": [
                {
                    "step": "all",
                    "count": 1,
                    "label": "reset"
                },
                {
                    "step": "month",
                    "count": 1,
                    "label": "1month",
                    "stepmode": "backward"
                },
                {
                    "step": "day",
                    "count": 14,
                    "label": "1 week",
                    "stepmode": "backward"
                },
                {
                    "step": "day",
                    "count": 28,
                    "label": "2 weeks",
                    "stepmode": "backward"
                },
                {"step": "all"}
            ]
        },
            "rangeslider": {"visible": False}
        },
        "yaxis": {
            "domain": [0, 0.2],
            "showticklabels": False
        },
        "legend": {
            "x": 0.3,
            "y": 0.95,
            "yanchor": "bottom",
            "orientation": "h"
        },
        "margin": {
            "b": 10,
            "l": 10,
            "r": 10,
            "t": 10
        }
    }

    fig = go.Figure(data=data, layout=layout)

    # Adjusting the figure size
    fig.update_layout(
        autosize=False,
        margin=dict(l=10, r=10, t=40, b=10),
        width=1400,
        height=800, )

    # Type casting and Drawing pivot points
    for i in range(len(pivot_point_high_time)):
        pivot_point_high_time[i] = str(pivot_point_high_time[i])

    for i in range(len(pivot_point_low_time)):
        pivot_point_low_time[i] = str(pivot_point_low_time[i])

    fig.add_trace(
        go.Scatter(
            x=pivot_point_high_time,  # ["2022-02-08 14:00:00"],
            y=np.multiply(pivot_point_high, 1.006),
            mode="markers",
            marker=dict(symbol='triangle-down', size=12, color='red'),
            name='pivot_high'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pivot_point_low_time,  # ["2022-02-08 14:00:00"],
            y=np.multiply(pivot_point_low, 0.994),
            mode="markers",
            marker=dict(symbol='triangle-up', size=12, color='blue'),
            name='pivot_low'
        )
    )

    # FIND SUPPORT LINES AND REGIONS
    supports, support_width, support_time = find_supports(pivot_point_low, pivot_point_low_time, SUPPORT_THRESHOLD)
    s = 0
    for s in range(len(supports)):
        if supports[s] < df['MA_20'].iloc[-1]:
            if support_width[s] > 3:
                fig.add_shape(type="rect",
                              x0=str(support_time[s]), y0=supports[s] * (1 + SUPPORT_THRESHOLD / 100),
                              x1=str(df['TIME'].iloc[-1]), y1=supports[s] * (1 - SUPPORT_THRESHOLD / 100),
                              line=dict(color="RoyalBlue", width=2),
                              fillcolor="LightSkyBlue",
                              opacity=0.5
                              )
            else:
                fig.add_shape(type="line",
                              x0=str(support_time[s]), y0=supports[s], x1=str(df['TIME'].iloc[-1]), y1=supports[s],
                              line=dict(color="Blue", width=support_width[s], dash="dashdot")
                              )

    # FIND RESISTANCE LINES AND REGIONS
    resistances, resistance_width, resistance_time = find_resistances(pivot_point_high, pivot_point_high_time,
                                                                      RESISTANCE_THRESHOLD)
    s = 0
    for s in range(len(resistances)):
        if resistances[s] > df['MA_20'].iloc[-1]:
            if resistance_width[s] > 3:
                fig.add_shape(type="rect",
                              x0=str(resistance_time[s]), y0=resistances[s] * (1 + RESISTANCE_THRESHOLD / 100),
                              x1=str(df['TIME'].iloc[-1]), y1=resistances[s] * (1 - RESISTANCE_THRESHOLD / 100),
                              line=dict(color="darkred", width=2),
                              fillcolor="mediumvioletred",
                              opacity=0.5
                              )
            else:
                fig.add_shape(type="line",
                              x0=str(resistance_time[s]), y0=resistances[s], x1=str(df['TIME'].iloc[-1]),
                              y1=resistances[s],
                              line=dict(color="Red", width=resistance_width[s], dash="dashdot")
                              )

    # fig.add_hline(x0 = pivot_point_low_time[4], x1 = pivot_point_low_time[7], type = 'line', xsizemode = 'scaled', y=supports[s], line_width=support_width[s], line_color="blue", line_dash="dash")

    fig2 = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.01, row_width=[0.15, 0.1, 0.1, 0.65],
                         figure=fig)  # , subplot_titles = (stock_.name, 'RSI', 'Volume')

    fig2.add_trace(go.Bar(x=df['TIME'], y=df['Volume'], showlegend=False), row=2, col=1)

    fig2.add_trace(go.Scatter(x=df['TIME'], y=df['RSI'], showlegend=False), row=3, col=1)
    fig2.add_hline(y=70, line_width=1, line_color="black", line_dash="dash", row=3, col=1)
    fig2.add_hline(y=30, line_width=1, line_color="black", line_dash="dash", row=3, col=1)

    fig2.add_trace(go.Scatter(x=df['TIME'], y=df['MACD'], showlegend=False), row=4, col=1)
    fig2.add_trace(go.Scatter(x=df['TIME'], y=df['MACD_S'], showlegend=False), row=4, col=1)

    return fig2


fig2 = plot_pivot_1H_from('240')
fig2.update_layout(
    dragmode='drawrect', # define dragmode
    newshape=dict(line_color='cyan'))
# Add modebar buttons
fig2.show(config={'modeBarButtonsToAdd':['drawline',
                                        'drawopenpath',
                                        'drawclosedpath',
                                        'drawcircle',
                                        'drawrect',
                                        'eraseshape'
                                       ]})
#fig2.show()