import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import yfinance as yf
import PySimpleGUI as sg
from datetime import datetime
from datetime import date
import numpy as np
import random
import operator
import pandas as pd
import seaborn as sns
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

NO_OF_GENERATIONS = 4
MUTATION_RATE = 5
POPULATION_SIZE = 750
NumReturn = 5

class Chromosome():
    def __init__(self, min=None, max=None, prev_min=None, prev_max=None, buy=None, score =None):
        self.min = min
        self.max = max
        self.prev_min = prev_min
        self.prev_max = prev_max
        self.buy = buy
        self.score = score

    def mutate(self):
        mu, sigma = 0, 0.15 # mean and standard deviation
        s = np.random.normal(mu, sigma, 1) 
        x = iter(s)
        toChange = random.randint(0,5)
        if toChange == 0:
           self.buy = random.randint(0,999) % 2
        if toChange == 1:
            self.min = next(x)
        if toChange == 2:
            self.max = next(x)
        if toChange == 3:
            self.prev_min = next(x)
        if toChange == 4:
            self.prev_max = next(x)
        if self.min > self.max:
            self.min, self.max = self.max, self.min
        if self.prev_min > self.prev_max:
            self.prev_min, self.prev_max = self.prev_max, self.prev_min

class TrainingData():
    population = []
    nextGen = []
    dayChange = []
    nextDayChange = []
    profit = []

    def __init__(self, popSize = None, mRate = None, mChange = None):
        self.popSize = popSize
        self.mRate = mRate
        self.mChange = mChange
    
    def generateData(self, data):
        closes = data['Adj Close'].tolist()
        opens = data['Open'].tolist()
        # file = open('stock_data', 'w')

        for i in range(len(data)-2):
            dayChangeVal = (float(closes[i])-float(opens[i+1])) / 100
            nextDayChangeVal = (float(closes[i+1]) - float(opens[i+2])) / 100
            profitVal = float(opens[i]) - float(opens[i+1])

            # file.write(str(dayChangeVal) + ' ' + str(nextDayChangeVal) + ' ' + str(profitVal) + '\n')
            self.dayChange.append(dayChangeVal)
            self.nextDayChange.append(nextDayChangeVal)
            self.profit.append(profitVal)
        
        global DataSize
        DataSize = len(self.dayChange)
        # file.close()

    def populationInit(self):
        mean = 0
        sd = 0.15  #standard deviation
        s = np.random.normal(mean, sd, 4*POPULATION_SIZE)
        it = iter(s)

        for i in range(POPULATION_SIZE):
            temp = Chromosome(next(it), next(it), next(it), next(it), random.randint(0, 999) % 2, 0)

            if temp.min > temp.max:
                temp.min, temp.max = temp.max, temp.min  #swap the values
            if temp.prev_min > temp.prev_max:
                temp.prev_min, temp.prev_max = temp.prev_max, temp.prev_min
            
            self.population.append(temp)
    
    def fitnessFunction(self):
        for i in range(len(self.population)):
            match = False
            for j in range(DataSize):
                if(self.population[i].prev_min < self.dayChange[j] and self.population[i].prev_max > self.dayChange[j]):
                    if(self.population[i].min < self.nextDayChange[j] and self.population[i].max > self.nextDayChange[j]):
                        if(self.population[i].buy == 1):
                            match = True
                            self.population[i].score += self.profit[j]

                #Match is found and we short
                if(self.population[i].prev_min < self.dayChange[j] and self.population[i].prev_max > self.dayChange[j]):
                    if(self.population[i].min < self.nextDayChange[j] and self.population[i].max > self.nextDayChange[j]):
                        if(self.population[i].buy == 0):
                            match = True
                            self.population[i].score -= self.profit[j]

                #We have not found any matches = -5000
                if match == False:
                    self.population[i].score = -5000
    
    def randomChoiceWeighted(self):
        self.exists()
        self.fitnessFunction()
        max = self.population[0].score
        for i in self.population[1:]:
            max += i.score

        pick = random.uniform(0, max)
        current = 0
        for i in range(len(self.population)):
            current += self.population[i].score
            if current > pick:
                self.nextGen.append(self.population[i])


    def exists(self):
        i = 0
        while i < len(self.population):
            if self.population[i].score is None:
                del self.population[i]
            else:
                i += 1

    def crossover(self,z):
        children = []

        for i in range(POPULATION_SIZE-len(self.nextGen)):
            child = Chromosome(0,0,0,0,0)
            chromosome1 = self.nextGen[random.randint(0,999999) % len(self.nextGen)]
            chromosome2 = self.nextGen[random.randint(0,999999) % len(self.nextGen)]
            if(random.randint(0,999) %2):
                child.min = chromosome1.min
                child.max = chromosome1.max
                child.prev_min = chromosome1.prev_min
                child.prev_max = chromosome1.prev_max 
                child.buy = chromosome1.buy  
            else:
                child.min = chromosome2.min
                child.min = chromosome2.max
                child.prev_min = chromosome2.prev_min
                child.prev_max = chromosome2.prev_max
                child.buy = chromosome2.buy

            if child.max < child.min:
                child.max, child.min = child.min, child.max

            if child.prev_max < child.prev_min:
                child.prev_max, child.prev_min = child.prev_min, child.prev_max

            children.append(child)

            for i in range(len(children)):
                if random.randint(0,999) % 100 <= z:
                    children[i].mutate()
                try:
                    self.population[i] = children[i]
                except IndexError:
                    break

            print(f'Children {len(children)}')

            for i in range(len(children), len(self.population), 1):
                if i > len(children):
                    self.population[i] = self.nextGen[i-len(children)]
                else:
                    break
               
            self.exists()
            self.fitnessFunction()
            self.population.sort(key=operator.attrgetter('score'))

    def determineStockAction(self):
        buyData = []
        sellData = []
        for i in range(len(self.population)):
            if self.population[i].buy == 1:
                buyData.append(self.population[i])
            elif self.population[i].buy == 0:
                sellData.append(self.population[i])

        # plot(sellData)
        buyOutput = []
        sellOutput = []
        fieldnames=["Scores"]
        i=1
        size=len(buyData)
        while i < NumReturn+1:
            index = size-i
            try:
                print("Minimum %f | Maximum %f | Previous Min %f | Previous Max %f | Score %f" % (buyData[index].min, buyData[index].max, buyData[index].prev_min, buyData[index].prev_max, buyData[index].score))
                buyOutput.append(buyData[index].score)
            except IndexError:
                pass
            i += 1
            # print("The best Data when shorting" % (NumReturn))
        i = 1
        size = len(sellData)
        while i < NumReturn+1:
            try:
                index = size-i
                # print("Minimum %f | Maximum %f | Previous Min %f | Previous Max %f | Score %f" % (sellData[index].min, sellData[index].max, sellData[index].prev_min, sellData[index].prev_max, sellData[index].score))
                sellOutput.append(sellData[index].score)
            except:
                pass
            i += 1
        
        print('output scores when we buy today',buyData)
        print('output scores when we short today',sellData)

        my_list = []
        print(len(buyOutput), len(sellOutput))
        for i in range(len(buyOutput)):
            if buyOutput[i]>sellOutput[i]:
                my_list.append(1)
            else: my_list.append(0)
        avg=sum(my_list)/len(my_list)
        print(my_list)
        if avg>=0.5:
            print('Buy the Stock')
        else:
            print('Short/Sell the Stock')

def main():
    stock_name = initWindow()
    adj_data, data = downloadStockData(stock_name)
    graphOptionsWindow(data, adj_data)
    x = TrainingData()
    x.generateData(data)
    x.populationInit()
    x.randomChoiceWeighted()
    x.crossover(MUTATION_RATE)
    x.determineStockAction()

if __name__ == '__main__':
    main()

