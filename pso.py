import pandas as pd
# import pandas_datareader as web
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
from scipy.stats import norm
import yfinance as yf
import math
import matplotlib.dates as mdates
import PySimpleGUI as sg
import numpy as np
from datetime import datetime
from datetime import date
from plots import *

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

def graphOptionsWindow(data, adj_data):
    layout = [
                [sg.Button('Adjusted Closing Graph')],
                [sg.Button('Daily Log Returns')],
                [sg.Button('RSI Plot')],
                [sg.Button('MACD Plot')],
                [sg.Button('Buy Or Sell Plot')]
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
        if event == 'Buy Or Sell Plot':
            buyOrSellPlot()
    
    window.close()

def downloadStockData(value):
    start = datetime(2019, 1, 1)
    end = date.today()
    # adj_data = yf.download(value, start, end)['Adj Close']
    data = yf.download(value, start, end)
    return data

timestamps_buy = []
timestamps_sell = []

def pso_buy_sell(df):
    # Define the objective function
    def objective(x):
        # Compute the moving averages
        short_ma = df['Close'].rolling(window=int(x[0])).mean()
        long_ma = df['Close'].rolling(window=int(x[1])).mean()

        # Compute the buy/sell signal
        buy_signal = (short_ma > long_ma).astype(int)

        # Compute the profit
        profit = np.sum((buy_signal.shift(1) * (df['Close'] - df['Close'].shift(1))))

        return -profit

    # Define the PSO parameters
    num_particles = 10
    num_dimensions = 2
    max_iterations = 50
    c1 = 2.0
    c2 = 2.0
    w = 0.7

    # Initialize the particles
    particles = np.random.rand(num_particles, num_dimensions) * 100.0

    # Initialize the velocities
    velocities = np.zeros((num_particles, num_dimensions))

    # Initialize the personal best positions and values
    personal_best_positions = particles.copy()
    personal_best_values = np.zeros(num_particles)

    for i in range(num_particles):
        personal_best_values[i] = objective(particles[i])

    # Initialize the global best position and value
    global_best_position = particles[personal_best_values.argmin()].copy()
    global_best_value = personal_best_values.min()

    # Run the PSO algorithm
    for iteration in range(max_iterations):
        # Update the velocities and positions
        for i in range(num_particles):
            r1 = np.random.rand(num_dimensions)
            r2 = np.random.rand(num_dimensions)
            velocities[i] = w * velocities[i] + c1 * r1 * (personal_best_positions[i] - particles[i]) + c2 * r2 * (global_best_position - particles[i])
            particles[i] = particles[i] + velocities[i]

            # Apply bounds to the positions
            particles[i] = np.maximum(particles[i], np.array([1, 1]))
            particles[i] = np.minimum(particles[i], np.array([len(df), len(df)]))

        # Evaluate the objective function
        for i in range(num_particles):
            value = objective(particles[i])

            # Update the personal best position and value
            if value < personal_best_values[i]:
                personal_best_positions[i] = particles[i].copy()
                personal_best_values[i] = value

                # Update the global best position and value
                if value < global_best_value:
                    global_best_position = particles[i].copy()
                    global_best_value = value

    # Compute the moving averages with the best parameters
    short_ma = df['Close'].rolling(window=int(global_best_position[0])).mean()
    long_ma = df['Close'].rolling(window=int(global_best_position[1])).mean()

    
    buy_signal = (short_ma > long_ma).astype(int)

    
    for i in range(len(buy_signal)):
        if buy_signal[i] == 1:
            print('Buy on', df['Date'][i])
            timestamps_buy.append(df['Date'][i])
        else:
            print('Sell on', df['Date'][i])
            timestamps_sell.append(df['Date'][i])

def buyOrSellPlot():
    plt.plot(timestamps_buy, color='r', label="Buy")
    plt.plot(timestamps_sell, color='g', label="Sell")
    plt.legend()
    plt.show()

def main():
    stock_name = initWindow()
    data = downloadStockData(stock_name)
    data = data.reset_index()
    pso_buy_sell(data)
    graphOptionsWindow(data, data['Adj Close'])

if __name__ == '__main__':
    main()
