import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

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

def plotDailyLogReturns(adj_data):
    log_returns = np.log(1 + adj_data.pct_change())
    sns.histplot(log_returns.iloc[1:])
    plt.xlabel("Daily Return")
    plt.ylabel("Frequency")
    plt.show()

def showAdjCloseGraph(adj_data):
    adj_data.plot(figsize=(15,6))
    plt.show()

# Modified histogram (Top & bottom 10 percentiles)
def plotModifiedHist(adj_data, closing_prices):
    top_ten = np.percentile(closing_prices,100-10)
    bottom_ten = np.percentile(closing_prices,10)
    plt.hist(closing_prices,bins=40)
    plt.axvline(top_ten, color='r', linestyle='dashed', linewidth=2)
    plt.axvline(bottom_ten, color='r', linestyle='dashed', linewidth=2)
    plt.axvline(adj_data[-1], color='g', linestyle='dashed', linewidth=2)
    plt.show()