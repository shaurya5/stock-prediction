from datetime import datetime
from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import math
import PySimpleGUI as sg
import matplotlib.dates as mdates
import datetime as dt
from plots import *
import matplotlib as mpl
mpl.use('tkagg')

def initWindow():
    sg.theme('DarkAmber')
    layout = [[sg.Text('Enter a stock name: '), sg.InputText()], 
                [sg.Button('Enter'), sg.Button('Cancel')]
            ]
    window = sg.Window('Stock Prediction', layout)

    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Cancel': # if user closes window or clicks cancel
            break
        else:
            print('You entered', values[0])
            break
    
    window.close()
    return values[0]

def graphOptionsWindow(data, adj_data, closing_prices):
    layout = [
                [sg.Button('Adjusted Closing Graph')],
                [sg.Button('Daily Log Returns')],
                [sg.Button('RSI Plot')],
                [sg.Button('MACD Plot')],
                [sg.Button('Modified Histogram')]
            ]
    window = sg.Window('Graph Selection', layout)

    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED: # if user closes window or clicks cancel
            break
        if event == 'Adjusted Closing Graph':
            showAdjCloseGraph(adj_data)
        if event == 'Daily Log Returns':
            plotDailyLogReturns(adj_data)
        if event == 'RSI Plot':
            RSIplot(data)
        if event == 'MACD Plot':
            MACDplot(data)
        if event == 'Modified Histogram':
            plotModifiedHist(adj_data, closing_prices)
    
    window.close()

def downloadStockData(value):
    start = datetime(2019, 1, 1)
    end = date.today()
    adj_data = yf.download(value, start, end)['Adj Close']
    data = yf.download(value, start, end)
    return adj_data, data

# Probablitiy calculations
def probabilityCalculations(adj_data):
    log_returns = np.log(1 + adj_data.pct_change())
    u = log_returns.mean()
    var = log_returns.var()
    drift = u - (0.5*var)
    stdev = log_returns.std()
    # CAGR and initializations
    number_of_trials = 3000
    time_elapsed = (adj_data.index[-1] - adj_data.index[0]).days
    total_growth = (adj_data[-1] / adj_data[1]) # current price / first price
    number_of_years = time_elapsed / 365.0
    cagr = total_growth ** (1/number_of_years) - 1
    closing_prices = []
    number_of_trading_days = 252
    price_series_cumulative = []

    for i in range(number_of_trials):
        daily_return_percentages = np.random.normal(cagr/number_of_trading_days, stdev/math.sqrt(number_of_trading_days),number_of_trading_days)+1
        price_series = [adj_data[-1]]

        for j in daily_return_percentages:
            price_series.append(price_series[-1] * j)

        # price_series_cumulative.append(price_series)
        closing_prices.append(price_series[-1])
        plt.plot(price_series)
    
    plt.show()
    return number_of_trials, price_series_cumulative, closing_prices

# Daily return percentages for the next 50 days and Monte Carlo Sim
def monteCarloSim(number_of_trials, price_series_cumulative):
    for i in range(number_of_trials):
        plt.plot(price_series_cumulative)
    plt.show()
    
# Expected price
def showExpectedPrice(closing_prices):
    mean_end_price = round(np.mean(closing_prices), 2)
    return mean_end_price

def main():
    stockName = initWindow()
    adj_data, data = downloadStockData(stockName)
    number_of_trials, price_series_cumulative, closing_prices = probabilityCalculations(adj_data)
    graphOptionsWindow(data, adj_data, closing_prices)
    predicted_price = showExpectedPrice(closing_prices)
    current_price = adj_data[len(data) - 1]
    if(current_price > predicted_price):
        print("Sell the Stock")
    else:
        print("Buy the Stock")

if __name__ == '__main__':
    main()
