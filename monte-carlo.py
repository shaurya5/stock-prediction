from datetime import datetime
from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from scipy.stats import norm
import yfinance as yf
import math
import PySimpleGUI as sg
import matplotlib.dates as mdates
import datetime as dt

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
                # [sg.Button('50 day return history')],
                # [sg.Button('Modified Histogram (based on top percentiles)')],
                [sg.Button('RSI Plot')],
                [sg.Button('MACD Plot')]
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
        # if event == '50 day return history':
        #     plotReturnHist(closing_prices)
        # if event == 'Modified Histogram (based on top percentiles)':
        #     plotModifiedHist(adj_data, closing_prices)
        if event == 'RSI Plot':
            RSIplot(data)
        if event == 'MACD Plot':
            MACDplot(data)
    
    window.close()

def downloadStockData(value):
    start = datetime(2019, 1, 1)
    end = date.today()
    adj_data = yf.download(value, start, end)['Adj Close']
    data = yf.download(value, start, end)
    return adj_data, data

# dict = ['GOOG', 'AAPL','MSFT','AMZN','NVDA', 'TSLA']
# valid_Input = False
# # User ticker selection
# while not valid_Input:
#     x = 0 
#     val = -1
#     tickerSelection = "|| "
#     for tickers in dict:
#         tickerSelection += tickers + " [" + str(x) + "] || " 
#         x += 1 
#     tickerSelection += "\n"
    # try:
    #     val = int(input("Please enter the number for the ticker you would like to track: \n" + tickerSelection))
    # except:
    #     print("Please input integer only...") 
    # print("Yo"val)
    # if val >= 0 and val <= len(dict) -1:
    #     valid_Input = True
    # else:
    #     print("Invalid input please try again \n\n")

# Ticker data pull
# ticker =  dict[val]
# def plot_stock_price(stock_price_data, name):
#     print(stock_price_data.columns)
#     plt.figure(figsize=(15,10))
#     plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
#     plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=200))

#     x_dates = []
#     for date in stock_price_data.index:
#         x_dates.append(date.strftime("%Y-%m-%d"))
#     # x_dates = [dt.datetime.strptime(d, '%Y-%m-%d').date() for d in stock_price_data.index.values]
#     # print(str(stock_price_data.index.values[0])[:10])
#     # x_dates = []
#     # for date in stock_price_data.index.values:
#         # ts = pd.to_datetime(str(date))
#         # d = ts.strftime('%Y.%m.%d')
#         # x_dates.append(str(stock_price_data.index.values[0])[:10])
    
#     plt.plot(x_dates, stock_price_data['High'], label='High')
#     plt.plot(x_dates, stock_price_data['Low'], label='Low')
#     plt.xlabel('Time')
#     plt.ylabel('Price')
#     plt.title(name)
#     plt.legend()
#     plt.gcf().autofmt_xdate()
#     plt.show()
    
def showAdjCloseGraph(adj_data):
    # ticker = values[0]
    # ticker = inputText
    adj_data.plot(figsize=(15,6))
    plt.show()

# Plot daily log returns
def plotDailyLogReturns(adj_data):
    log_returns = np.log(1 + adj_data.pct_change())
    sns.histplot(log_returns.iloc[1:])
    plt.xlabel("Daily Return")
    plt.ylabel("Frequency")
    plt.show()

# Probablitiy calculations
def probabilityCalculations(adj_data):
    log_returns = np.log(1 + adj_data.pct_change())
    u = log_returns.mean()
    var = log_returns.var()
    drift = u - (0.5*var)
    stdev = log_returns.std()
    # CAGR and initializations
    days = 50
    number_of_trials = 3000
    time_elapsed = (adj_data.index[-1] - adj_data.index[0]).days
    total_growth = (adj_data[-1] / adj_data[1])
    number_of_years = time_elapsed / 365.0
    cagr = total_growth ** (1/number_of_years) - 1
    closing_prices = []
    number_of_trading_days = 252
    price_series_cumulative = []

    for i in range(number_of_trials):
        daily_return_percentages = np.random.normal(cagr/number_of_trading_days, stdev/math.sqrt(number_of_trading_days),number_of_trading_days)+1
        price_series = [adj_data[-1]]
        daily_return_percentages = np.random.normal(cagr/number_of_trading_days, stdev/math.sqrt(number_of_trading_days),number_of_trading_days)+1
        price_series = [adj_data[-1]]

        for j in daily_return_percentages:
            price_series.append(price_series[-1] * j)
        for j in daily_return_percentages:
            price_series.append(price_series[-1] * j)

        price_series_cumulative.append(price_series)
        closing_prices.append(price_series[-1])
        price_series_cumulative.append(price_series)
        closing_prices.append(price_series[-1])

    return number_of_trials, price_series_cumulative, closing_prices

# Daily return percentages for the next 50 days and Monte Carlo Sim
def monteCarloSim(number_of_trials, price_series_cumulative):
    for i in range(number_of_trials):
        plt.plot(price_series_cumulative)
    plt.show()

# Histogram of 50 day returns
def plotReturnHist(closing_prices):
    plt.hist(closing_prices,bins=50)
    plt.show()

def RSIplot(data):
    delta = data["Adj Close"].diff(1)
    delta.dropna(inplace=True)

    positive=delta.copy()
    negative=delta.copy()
    positive[positive < 0]=0
    negative[negative > 0]=0

    days=14
    average_gain = positive.rolling(window=days).mean();
    average_loss = abs(negative.rolling(window=days).mean());

    relative_strength = average_gain / average_loss
    RSI = 100.0 - (100.0 / (1.0 + relative_strength))

    combined= pd.DataFrame()
    combined['Adj Close'] = data['Adj Close']
    combined['RSI'] = RSI

    plt.plot(combined.index,combined['RSI'],color='red')
    plt.axhline(y=30, color='blue', linestyle='dashed')
    plt.axhline(y=70, color='blue', linestyle='dashed')
    plt.grid(True,color='#555555')
    plt.show()

def MACDplot(data):
    shortEMA = data["Close"].ewm(span=12,adjust=False).mean()
    longEMA = data["Close"].ewm(span=26,adjust=False).mean()
    MACD = shortEMA - longEMA
    signal = MACD.ewm(span=9, adjust=False).mean()
    plt.figure(figsize=(12,8))
    plt.plot(data.index,MACD,color='red',label='MACD')
    plt.plot(data.index,signal,color='blue',label='Signal')
    plt.show()
    
# Expected price
def showExpectedPrice(closing_prices):
    mean_end_price = round(np.mean(closing_prices),2)
    print("Expected price: $", str(mean_end_price))

# Modified histogram (Top & bottom 10 percentiles)
def plotModifiedHist(adj_data, closing_prices):
    top_ten = np.percentile(closing_prices,100-10)
    bottom_ten = np.percentile(closing_prices,10)
    plt.hist(closing_prices,bins=40)
    plt.axvline(top_ten, color='r', linestyle='dashed', linewidth=2)
    plt.axvline(bottom_ten, color='r', linestyle='dashed', linewidth=2)
    plt.axvline(adj_data[-1], color='g', linestyle='dashed', linewidth=2)
    plt.show()

def main():
    stockName = initWindow()
    adj_data, data = downloadStockData(stockName)
    print(data.columns)
    number_of_trials, price_series_cumulative, closing_prices = probabilityCalculations(adj_data)
    graphOptionsWindow(data, adj_data, closing_prices)	
    # print(price_series_cumulative)
    # showAdjCloseGraph(data)
    # plotDailyLogReturns(data)
    # monteCarloSim(number_of_trials, price_series_cumulative)
    # plotReturnHist(closing_prices)
    # plotModifiedHist(data, closing_prices)
    # showExpectedPrice(closing_prices)


if __name__ == '__main__':
    main()
