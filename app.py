from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as po

app = Flask(__name__)

def smma(series, period):
    return series.ewm(alpha=1/period, adjust=False).mean()

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def mfi(data, period=14):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    money_flow = typical_price * data['Volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(), 0)
    positive_mf = positive_flow.rolling(window=period).sum()
    negative_mf = negative_flow.rolling(window=period).sum()
    mfi = 100 * (positive_mf / (positive_mf + negative_mf))
    return mfi

def fractals(high, low):
    up_fractal = (high.shift(2) < high.shift(1)) & (high.shift(0) > high.shift(1)) & (high.shift(0) > high.shift(-1)) & (high.shift(2) < high.shift(0))
    down_fractal = (low.shift(2) > low.shift(1)) & (low.shift(0) < low.shift(1)) & (low.shift(0) < low.shift(-1)) & (low.shift(2) > low.shift(0))
    return up_fractal, down_fractal

@app.route('/', methods=['GET', 'POST'])
def index():
    chart_html = ''
    if request.method == 'POST':
        ticker = request.form['ticker']
        data = yf.download(ticker, period='6mo')

        data['Jaw'] = smma(data['Close'], 13).shift(8)
        data['Teeth'] = smma(data['Close'], 8).shift(5)
        data['Lips'] = smma(data['Close'], 5).shift(3)
        data['EMA200'] = ema(data['Close'], 200)
        data['EMA60'] = ema(data['Close'], 60)
        data['RSI'] = rsi(data['Close'])
        data['MFI'] = mfi(data)
        up_fractal, down_fractal = fractals(data['High'], data['Low'])
        data['UpFractal'] = data['High'][up_fractal]
        data['DownFractal'] = data['Low'][down_fractal]

        fig = make_alligator_chart(data, ticker)
        chart_html = po.plot(fig, output_type='div')

    return render_template('index.html', chart_html=chart_html)

def make_alligator_chart(data, ticker):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Candlestick',
        increasing_line_color='red',
        decreasing_line_color='green'
    ))

    fig.add_trace(go.Scatter(x=data.index, y=data['Jaw'], line=dict(color='blue', width=1), name='Jaw (13,8)'))
    fig.add_trace(go.Scatter(x=data.index, y=data['Teeth'], line=dict(color='red', width=1), name='Teeth (8,5)'))
    fig.add_trace(go.Scatter(x=data.index, y=data['Lips'], line=dict(color='green', width=1), name='Lips (5,3)'))
    fig.add_trace(go.Scatter(x=data.index, y=data['EMA200'], line=dict(color='black', width=3, dash='dash'), name='EMA 200'))
    fig.add_trace(go.Scatter(x=data.index, y=data['EMA60'], line=dict(color='orange', width=3, dash='dash'), name='EMA 60'))
    fig.add_trace(go.Scatter(x=data.index, y=data['UpFractal'], mode='markers', marker=dict(symbol='triangle-up', color='red', size=10), name='Up Fractal'))
    fig.add_trace(go.Scatter(x=data.index, y=data['DownFractal'], mode='markers', marker=dict(symbol='triangle-down', color='blue', size=10), name='Down Fractal'))
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', yaxis='y2', marker_color='lightgray', opacity=0.3))

    fig.update_layout(
        title=f'{ticker} Alligator Chart + EMA + Fractals + RSI + MFI',
        yaxis=dict(title='Price'),
        yaxis2=dict(title='Volume', overlaying='y', side='right', showgrid=False),
        xaxis_rangeslider_visible=False,
        height=1000,
        template='plotly_white'
    )

    return fig

if __name__ == '__main__':
    app.run(debug=True)
