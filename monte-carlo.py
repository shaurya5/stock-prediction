from datetime import datetime
from datetime import date
# import PyQt5
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib
import seaborn as sns
# from scipy.stats import norm
import yfinance as yf
import math
import PySimpleGUI as sg
# matplotlib.use('Qt5Agg')

def initWindow():
	sg.theme('DarkAmber')
	layout = [[sg.Text('Enter a company name: '), sg.InputText()], 
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

def graphOptionsWindow(data, closing_prices):
	layout = [
				[sg.Button('Adjusted Closing Graph')],
				[sg.Button('Daily Log Returns')],
				[sg.Button('50 day return history')],
				[sg.Button('Modified Histogram (based on top percentiles)')]
			]
	window = sg.Window('Graph Selection', layout)

	while True:
		event, values = window.read()
		if event == sg.WIN_CLOSED: # if user closes window or clicks cancel
			break
		if event == 'Adjusted Closing Graph':
			showAdjCloseGraph(data)
		if event == 'Daily Log Returns':
			plotDailyLogReturns(data)
		if event == '50 day return history':
			plotReturnHist(closing_prices)
		if event == 'Modified Histogram (based on top percentiles)':
			plotModifiedHist(data, closing_prices)
	
	window.close()

def downloadStockData(value):
	start = datetime(2019, 1, 1)
	end = date.today()
	data = yf.download(value, start , end)['Adj Close']
	return data

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
def showAdjCloseGraph(data):
	# ticker = values[0]
	# ticker = inputText
	data.plot(figsize=(15,6))
	plt.show(block=False)

# Plot daily log returns
def plotDailyLogReturns(data):
	log_returns = np.log(1 + data.pct_change())
	sns.displot(log_returns.iloc[1:])
	plt.xlabel("Daily Return")
	plt.ylabel("Frequency")
	plt.show(block=False)

# Probablitiy calculations
def probabilityCalculations(data):
	log_returns = np.log(1 + data.pct_change())
	u = log_returns.mean()
	var = log_returns.var()
	drift = u - (0.5*var)
	stdev = log_returns.std()
	# CAGR and initializations
	days = 50
	number_of_trials = 3000
	time_elapsed = (data.index[-1] - data.index[0]).days
	total_growth = (data[-1] / data[1])
	number_of_years = time_elapsed / 365.0
	cagr = total_growth ** (1/number_of_years) - 1
	closing_prices = []
	number_of_trading_days = 252
	price_series_cumulative = []

	for i in range(number_of_trials):
		daily_return_percentages = np.random.normal(cagr/number_of_trading_days, stdev/math.sqrt(number_of_trading_days),number_of_trading_days)+1
		price_series = [data[-1]]
		daily_return_percentages = np.random.normal(cagr/number_of_trading_days, stdev/math.sqrt(number_of_trading_days),number_of_trading_days)+1
		price_series = [data[-1]]

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
	plt.hist(closing_prices,bins=40)
	plt.show()

# Expected price
def showExpectedPrice(closing_prices):
	mean_end_price = round(np.mean(closing_prices),2)
	print("Expected price: $", str(mean_end_price))

# Modified histogram (Top & bottom 10 percentiles)
def plotModifiedHist(data, closing_prices):
	top_ten = np.percentile(closing_prices,100-10)
	bottom_ten = np.percentile(closing_prices,10)
	plt.hist(closing_prices,bins=40)
	plt.axvline(top_ten, color='r', linestyle='dashed', linewidth=2)
	plt.axvline(bottom_ten, color='r', linestyle='dashed', linewidth=2)
	plt.axvline(data[-1], color='g', linestyle='dashed', linewidth=2)
	plt.show()

def main():
	stockName = initWindow()
	data = downloadStockData(stockName)
	# print(data)
	number_of_trials, price_series_cumulative, closing_prices = probabilityCalculations(data)
	graphOptionsWindow(data, closing_prices)	
	# print(price_series_cumulative)
	# showAdjCloseGraph(data)
	# plotDailyLogReturns(data)
	# monteCarloSim(number_of_trials, price_series_cumulative)
	# plotReturnHist(closing_prices)
	# plotModifiedHist(data, closing_prices)
	# showExpectedPrice(closing_prices)


if __name__ == '__main__':
	main()