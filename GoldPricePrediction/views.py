from django.shortcuts import render
from django.http import HttpResponse


#========== IMPORT LIBRARIES ==========
from sklearn.linear_model import LinearRegression

import pandas as pd
import numpy as np
# Turn off pandas warning
pd.set_option('mode.chained_assignment', None)

# For Getting Dataset From YahooFinance
import yfinance as yf

# Get today's date
from datetime import datetime, timedelta
import pytz
today = datetime.now(tz=pytz.timezone('US/Eastern')).strftime('%Y-%m-%d')

# For Graphs
import plotly.offline as opy
from plotly.graph_objs import Scatter
import plotly.graph_objects as go

#========== READ DATA ==========
Df = yf.download('GLD', '2008-01-01', today, auto_adjust=True)
# Only keep close columns
Df = Df[['Close']]
# Drop rows with missing values
Df = Df.dropna()

# Plot the closing price of GLD
x_data = Df.index
y_data = Df['Close']
ClosingPricePlot_div = opy.plot({
    'data': [Scatter(x=x_data, y=y_data, mode='lines', name='test', opacity=0.8, marker_color='green')],
    'layout': {'title': 'Gold ETF Price Series', 'xaxis': {'title': 'Date'}, 'yaxis': {'title': 'Gold ETF price (in $)'}}
}, output_type='div')

#========== DEFINE EXPLANATORY VARIABLES ==========
Df['S_3'] = Df['Close'].rolling(window=3).mean()
Df['S_9'] = Df['Close'].rolling(window=9).mean()
Df['next_day_price'] = Df['Close'].shift(-1)

Df = Df.dropna()
X = Df[['S_3', 'S_9']]

# Define dependent variable
y = Df['next_day_price']

#========== TRAIN AND TEST DATASET ==========
t = .8
t = int(t*len(Df))

# Train dataset
X_train = X[:t]
y_train = y[:t]

# Test dataset
X_test = X[t:]
y_test = y[t:]

#========== LINEAR REGRESSION MODEL ==========
linear = LinearRegression().fit(X_train, y_train)
RegressionModelFormula = "Gold ETF Price (y) = %.2f * 3 Days Moving Average (x1) + %.2f * 9 Days Moving Average (x2) + %.2f (constant)" % (linear.coef_[0], linear.coef_[1], linear.intercept_)

#========== PREDICTING GOLD ETF PRICES ==========
predicted_price = linear.predict(X_test)
predicted_price = pd.DataFrame(predicted_price, index=y_test.index, columns=['price'])
# Attach y_tese series to dataframe
predicted_price['close'] = y_test
# Plot graph
x_data = predicted_price.index
y_data_predicted = predicted_price['price']
y_data_actual = predicted_price['close']
fig = go.Figure()
fig.add_trace(go.Scatter(x=x_data, y=y_data_predicted,
                    mode='lines',
                    name='Predicted Price'))
fig.add_trace(go.Scatter(x=x_data, y=y_data_actual,
                    mode='lines',
                    name='Actual Price'))
PredictionPlot_div = opy.plot({
    'data': [Scatter(x=x_data, y=y_data_predicted, mode='lines', name='Predicted Price', opacity=0.8),
    Scatter(x=x_data, y=y_data_actual, mode='lines', name='Actual Price', opacity=0.8)],
    'layout': {'title': 'Predicted VS Actual Price', 'xaxis': {'title': 'Date'}, 'yaxis': {'title': 'Gold ETF price (in $)'}}
}, auto_open=False, output_type='div')

#========== CUMULATIVE RETURNS ==========
gold = pd.DataFrame()

gold['price'] = Df[t:]['Close']
gold['predicted_price_next_day'] = predicted_price['price']
gold['actual_price_next_day'] = y_test
gold['gold_returns'] = gold['price'].pct_change().shift(-1)
    
gold['signal'] = np.where(gold.predicted_price_next_day.shift(1) < gold.predicted_price_next_day,1,0)
    
gold['strategy_returns'] = gold.signal * gold['gold_returns']
x_data = gold.index
y_data = ((gold['strategy_returns']+1).cumprod()).values
CumulativeReturns_div = opy.plot({
    'data': [Scatter(x=x_data, y=y_data, mode='lines', name='test', opacity=0.8, marker_color='green')],
    'layout': {'title': 'Cumulative Returns', 'xaxis': {'title': 'Date'}, 'yaxis': {'title': 'Cumulative Returns (X 100%)'}}
}, output_type='div')

#========== PREDICT DAILY MOVES ==========
data = yf.download('GLD', '2008-06-01', today, auto_adjust=True)
data['S_3'] = data['Close'].rolling(window=3).mean()
data['S_9'] = data['Close'].rolling(window=9).mean()
data = data.dropna()
data['predicted_gold_price'] = linear.predict(data[['S_3', 'S_9']])
data['signal'] = np.where(data.predicted_gold_price.shift(1) < data.predicted_gold_price,"Buy","No Position")


# Return to Context functions
def PlotClosingPrice():
    return ClosingPricePlot_div

def RegressionModel():
    return RegressionModelFormula

def PredictionPlot():
    return PredictionPlot_div

def r2_scoreCalculate():
    # R square
    r2_score = linear.score(X[t:], y[t:])*100
    r2_score = float("{0:.2f}".format(r2_score))
    return r2_score

def CumulativeReturns():
    return CumulativeReturns_div

def SharpeRatioCalculate():
    return '%.2f' % (gold['strategy_returns'].mean()/gold['strategy_returns'].std()*(252**0.5))

def MovingAverage_S3():
    return round(data['S_3'].iloc[-1], 2)

def MovingAverage_S9():
    return round(data['S_9'].iloc[-1], 2)

def GetSignal():
    return data['signal'].iloc[-1]

def GetPredictedPrice():
    return round(data['predicted_gold_price'].iloc[-1], 2)

def GetNextDay():
    NextDate = (data.index[-1].date() + timedelta(days=1)).strftime('%d/%m/%Y')
    return NextDate

def GetClosingPrice():
    return round(data["Close"].iloc[-1], 2)

def GetClosingPriceDate():
    return data.index[-1].strftime("%d/%m/%y")

#Ended

def home(request):
    #Added

    context = {
        'ClosingPricePlot_div' : PlotClosingPrice(),
        'PredictionPlot_div' : PredictionPlot(),
        'CumulativeReturns_div' : CumulativeReturns(),
        'SharpeRatio' : SharpeRatioCalculate(),
        'S_3' : MovingAverage_S3(),
        'S_9' : MovingAverage_S9(),
        'Signal' : GetSignal(),
        'PredictedPrice' : GetPredictedPrice(),
        'NextDate' : GetNextDay(),
        'ClosingPrice' : GetClosingPrice(),
        'ClosingDate' : GetClosingPriceDate(),
    }

    #Ended
    return render(request, 'GoldPricePrediction/home.html', context)

def base(request):
    return render(request, 'GoldPricePrediction/base.html')

def information(request):
    context = {
        'RegressionModelFormula' : RegressionModel(),
        'r2_score' : r2_scoreCalculate(),
    }
    return render(request, 'GoldPricePrediction/information.html', context)