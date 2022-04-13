#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 09:40:28 2021

@author: ianwallgren
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import math
import seaborn as sns
import datetime
import statistics
from datetime import date
from  KDEpy import FFTKDE

import hashlib

import numpy as np
import scipy
import warnings
from KDEpy.binning import linear_binning
from KDEpy.utils import autogrid
from scipy import fftpack
from scipy.optimize import brentq

from KDEpy import FFTKDE
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np

#%%
# Days of the interest rate announcement

events = ['2012-01-25',
          '2012-03-13',
          '2012-04-25',
          '2012-06-20',
          '2012-07-31',
          '2012-09-13',
          '2012-10-24',
          '2012-12-12',
          
          '2013-01-30',
          '2013-03-20',
          '2013-05-01',
          '2013-06-19',
          '2013-07-31',
          '2013-09-18',
          '2013-10-30',
          '2013-12-18',
          
          '2014-01-29',
          '2014-03-19',
          '2014-04-30',
          '2014-06-18',
          '2014-07-30',
          '2014-09-17',
          '2014-10-29',
          '2014-12-17',
          
          '2015-01-28',
          '2015-03-18',
          '2015-04-29',
          '2015-06-17',
          '2015-07-29',
          '2015-09-17',
          '2015-10-28',
          '2015-12-16',
          
          '2016-01-27',
          '2016-03-16',
          '2016-04-27',
          '2016-06-15',
          '2016-07-27',
          '2016-09-21',
          '2016-11-02',
          '2016-12-14',
                    
          '2017-02-01', 
          '2017-03-05',
          '2017-05-03',
          '2017-06-14',
          '2017-07-26',
          '2017-09-20',
          '2017-11-01',
          '2017-12-13',

          '2018-02-01',
          '2018-03-01',
          '2018-05-02',
          '2018-06-03',
          '2018-08-01',
          '2018-09-06',
          '2018-11-08',
          '2018-12-19',

          '2019-01-30',
          '2019-03-20',
          '2019-05-01',
          '2019-06-19',
          '2019-07-31',
          '2019-09-18',
          '2019-10-30',
          '2019-12-11',
        
          '2020-01-29',
          '2020-03-31',
          '2020-04-29',
          '2020-06-10',
          '2020-07-29',
          '2020-09-16',
          '2020-11-05',
          '2020-12-16']

#%% 
# Reading data
#prices = pd.read_csv('USA500.csv')

# Excluding all data with zero volume
#prices = prices[prices['Volume'] > 0]


#%%

# Normalising, removing GMT, and setting index

#prices['date_norm'] = pd.to_datetime(prices['Local time'],utc=True,dayfirst=True).dt.normalize()
#prices.set_index(prices['date_norm'],inplace=True,drop=True)

#prices['date_norm'] = pd.to_datetime(prices['Local time'],dayfirst=True).dt.normalize()
#prices.set_index(prices['date_norm'],inplace=True,drop=True)

#%%

# Normalising 
# Adding days prior and after events to "dates" dataframe

from pandas.tseries.offsets import BDay

dates = pd.DataFrame(events,columns=["event_date"])
dates['event_date'] = pd.to_datetime(dates["event_date"],utc=True)
#utc=True
#dates['start_date'] = dates["event_date"] - BDay(1)
dates['start_date'] = dates["event_date"] - BDay(1)
dates['end_date'] = dates["event_date"] + BDay(1)

#had to swap start_date and event_date. Getting first of jan and first of feb instead of two consecutive days
columns_titles = ["start_date","event_date","end_date"]
dates=dates.reindex(columns=columns_titles)

dates.dropna(axis=0, inplace=True)

#%%
# Creating dictionary where keys = event dates,
# values = intraday data for day of announcement and two days surrounding it
# In case announcement is on weekend, only two days of data are shown in values
#event_price_dict = dict()
#event_price_list = []
#for i in dates.index:
#    event_price_dict[dates.iloc[i,0]] = prices.loc[dates.iloc[i,1]:dates.iloc[i,2]]
#    event_price_list.append(prices.loc[dates.iloc[i,1]:dates.iloc[i,2]])
   # print(“Didn’t work: “,dates.iloc[i,0], ” “, dates.iloc[i,0].day_name())
   
#%%

#parameters to optimize: targeted_profit, MACD, RSI (as much as possible).

#plots to include: accumulated returns, histogram of returns

#to do
#look over statistics and code in general since we flipped the dataframe

#to do
#Eventually make sure we buy/sell the day following the signal
 

#%%
#Indicator settings
#to do: look into RSI settings, numbers are almost always between 30 and 70. This is normal

#RSI settings
#1
#(RSI_days,Overbought_level,Oversold_level) = (10,60,40)
#2
(RSI_days,Overbought_level,Oversold_level) = (8,70,30) #this dataset works better without RSI filter
#3
#(RSI_days,Overbought_level,Oversold_level) = (10,90,10)
#4
#(RSI_days,Overbought_level,Oversold_level) = (13,60,40)
#
#(RSI_days,Overbought_level,Oversold_level) = (5,100,0)


#MACD settings
#1
#(Short_EMA,Long_EMA,Signal_line_EMA) = (5,35,5)
#2
#(Short_EMA,Long_EMA,Signal_line_EMA) = (12,26,9)
#3
#(Short_EMA,Long_EMA,Signal_line_EMA) = (5,40,5)
#4
(Short_EMA,Long_EMA,Signal_line_EMA) = (5,40,5)
#5
#(Short_EMA,Long_EMA,Signal_line_EMA) = (5,35,5)

 
Short_signal = []
Long_signal = []
Neutralize_long = []
Neutralize_short = []

taregeted_profit = 0


#%%
def definitions(file):
    df = pd.read_csv('2012_2014.csv')
    df = df[df['Volume'] > 0]

   # df2 = pd.read_csv('2015_2017.csv')
   # df2 = df2[df2['PX_VOLUME'] > 0]

   # df3 = pd.read_csv('2018_2020.csv')
   # df3 = df3[df3['PX_VOLUME'] > 0]

#    df = pd.merge(df1, df2, how = 'outer')
#    df = pd.merge(df, df3, how = 'outer')

   # df = pd.read_csv('USA500.csv')

    #df = prices
    df = df.rename(columns={'Gmt time': 'Exchange Date'})
    df['Exchange Date'] = pd.to_datetime(df['Exchange Date'].values, utc=True, dayfirst=True)
    df = df.set_index(df['Exchange Date'])
    df.sort_index(ascending = True, inplace = True)
  #  df['Exchange Date'].apply(lambda x: x.replace(tzinfo=None))
    
    #ATR (22)
  
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    ATR = true_range.rolling(22).sum()/14
    df['ATR'] = ATR

    # Chandelier exit

    ## for long position
    df["Max_high"] = df.rolling(22, min_periods=22)['High'].max()  
    df["Ch_exit_long"] = df["Max_high"] - df["ATR"] * 3
    
    ## for short position
    df["Lowest_low"] = df.rolling(22, min_periods=22)['Low'].min()  
    df["Ch_exit_short"] = df["Lowest_low"] + df["ATR"] * 3
    
   
    #df['Exchnage Date'] = pd.to_datetime(df["Exchange Date"],utc=True).dt.normalize()

  #  df = df.set_index(pd.DatetimeIndex(df['Exchange Date'].values))
 #   df['Exchange Date'] = df['Exchange Date'].dt.tzoffset(None,-3600)
 #   df['Exchange Date'].dt.tz_localize(None)
 #   df.sort_index(ascending = True, inplace = True)
    
    #Calculating RSI df column
    
   # df.sort_index(ascending = True, inplace = True)    #the new data seems to be in ascending order already
    df['difference'] = df['Close'].diff(1)
    df['pct_change'] = df["Close"].pct_change() 
    df['pos'] = df.difference.copy()
    df['neg'] = df.difference.copy()
    
    df['pos'][df['pos'] < 0] = 0 
    df['neg'][df['neg'] > 0] = 0 
    
   # df.loc[df['pos'] < 0, 'pos'] = 0
   # df.loc[df['neg'] < 0, 'neg'] = 0
    
    df['avg_gain'] = df['pos'].rolling(window=RSI_days).mean()
    df['avg_loss'] = abs(df['neg'].rolling(window=RSI_days).mean())
    df['RS'] = df['avg_gain'] / df['avg_loss']
    df['RSI'] = 100.0 - (100.0 / (1.0 + df['RS']))
    
  
   # df['Log_returns'] = abs(np.log(df['Close'].iloc[i]/df['Close'].iloc[i-4])
  
    #Calculating MACD df  and signal line df column
    df['short_period_EMA'] = df.Close.ewm(span=Short_EMA, adjust=False).mean()
    df['long_period_EMA'] = df.Close.ewm(span=Long_EMA, adjust=False).mean()
    df['MACD'] =  df['short_period_EMA'] - df['long_period_EMA']
    df['Signal_line'] = df['MACD'].ewm(span=Signal_line_EMA, adjust=False).mean()
    

#Signal strength

#to do: confirm that signal strengths are calculated correctly
#redo this, sell signals will always be 0,5 atm



    min_cp = [] 
    max_cp = []
    Signal_strength = []
    ATR = []
    rolling_mean = []
    for i in range(0,len(df)):
        #TR parameter
        if i > len(df):
            min_cp.append(min(df['Close'].iloc[i],df['Close'].iloc[i-1],df['Close'].iloc[i-2]))
            max_cp.append(max(df['Close'].iloc[i],df['Close'].iloc[i-1],df['Close'].iloc[i-2]))
            Signal_strength.append(((df['Close'].iloc[i] - min_cp[i])/(max_cp[i]-min_cp[i]))*0.5+0.5)
            rolling_mean.append((df['Close'].iloc[i]+df['Close'].iloc[i-1]+df['Close'].iloc[i-2])/3)
        else:
            min_cp.append(min(df['Close'].iloc[i],df['Close'].iloc[i-1],df['Close'].iloc[i-2]))
            max_cp.append(max(df['Close'].iloc[i],df['Close'].iloc[i-1],df['Close'].iloc[i-2]))
            Signal_strength.append(((df['Close'].iloc[i] - min_cp[i])/(max_cp[i]-min_cp[i]))*0.5+0.5)
            rolling_mean.append((df['Close'].iloc[i]+df['Close'].iloc[i-1]+df['Close'].iloc[i-2])/3)
            
    for i in range(0,len(df)):
        ATR.append(sum(Signal_strength[i-4:i])/4)
        
    
    df['min_cp'] = min_cp
    df['max_cp'] = max_cp
    df['Signal_strength'] = Signal_strength
#    df['ATR'] = ATR
    df['rolling_mean'] = rolling_mean
       
    logr = [] 
    for i in range(0,len(df)) :
        logr.append(np.log(df['Close'].iloc[i]/df['Close'].iloc[i-1]))
    
    df['logr'] = logr
    df['logr_ema'] = df.logr.ewm(span=8, adjust=False).mean()    
        
   
       
    
    #dropping all NaN values caused by the lag in calculation- If we drop na then Signal_strength is messed up
    df.dropna(inplace=True)
    
#    return df

#df = definitions(r'Downloads/1Apple_30minintradaily_6months.xlsx')
#df = definitions(r'Desktop/SP500_daily_oneyear_copy.xlsx')
#df = definitions(r'Desktop/SP500_daily_oneyear_copy.xlsx')
#df = definitions('//Users//ianwallgren//Desktop//SP500_intraday_copy.xlsx')
  


#Conditions
    
#def conditions(df,RSI_below_upper_level,RSI_above_lower_level):
    Bull_cross = []
    Bear_cross = []
    flag = 1
    
    df.loc[(df['RSI'] < Overbought_level), 'RSI_below_upper_level'] = 1
    df.loc[(df['RSI'] > Oversold_level), 'RSI_above_lower_level'] = 1
    
    
       
    for i in range(0,len(df)):
          if df['MACD'][i] < df['Signal_line'][i]:
              Bull_cross.append(0)
              if flag != 2:
                  Bear_cross.append(1)
                  flag = 2
              else:
                  Bear_cross.append(0)
         
                
          elif df['MACD'][i] > df['Signal_line'][i]:
              Bear_cross.append(0)
              if flag != 3:
                  Bull_cross.append(1)
                  flag = 3
              else:
                  Bull_cross.append(0)              
        
          else:
              Bull_cross.append(0)
              Bear_cross.append(0)
              
    df['Bull_cross'] = Bull_cross
    df['Bear_cross'] = Bear_cross
    
#    return conditions



#Generate signals   
#to do: Evaluate targeted profit level
 
#def signals(df,conditions):
 #   Short_signal = []
 #   Long_signal = []
 #   Neutralize_long = []
 #   Neutralize_short = []
    in_long = -1
    in_short = -2
    long_time = 0 
    short_time = 0    
    #targeted_profit = 0
    targeted_profit_long = 0
    targeted_profit_short = 0
   # taregeted_profit = 0
    #loop through data in reversed order 
    #(df['ATR'].iloc[i]) * 
    for i in range(0,len(df)):
      #  print(abs(np.log(df['Close'].iloc[i]/df['Close'].iloc[i-4])))
        #print(np.log(df['Close'].iloc[i]))
        #alternatively make targeted profit df column
        #targeted_profit parameter, change this to optimize results 
    
        if df['RSI_below_upper_level'][i] == 1 and df['Bull_cross'][i] == 1 and in_long != 1 and i !=0 :
             
            Long_signal.append(df['Exchange Date'].iloc[i])
            #targeted_profit = df['Signal_strength'].iloc[i] * 1/(np.log(df['Close'].iloc[i])*50)
            #targeted_profit = (df['Signal_strength'].iloc[i]+0.5) * abs(np.log(df['Close'].iloc[i]/df['Close'].iloc[i-4])) #want to make this level relative to the price level
            targeted_profit_long = df["Ch_exit_long"].iloc[i]
            ###targeted_profit_long = (df['ATR'].iloc[i]) *  abs(np.log(df['Close'].iloc[i])/df['Close'].iloc[i-8]) #calculated over last two hours
            #the ATR is already accounting for general price level
            #print(targeted_profit)
            
            in_long= 1
          
            continue
        
        if in_long == 1:
            long_time += 1
            
        if df['Close'].iloc[i] < (3/3) * targeted_profit_long and in_long == 1 and i != 0: 
            #or df['Close'].iloc[i] / df['Close'][i-long_time] <(-1) * (3/3) * targeted_profit_long \ 
            
                
      #  if (df['Close'].iloc[i] - df['Close'].iloc[i-long_time]) / df['Close'].iloc[i-long_time] > targeted_profit \
      #      or (df['Close'].iloc[i] - df['Close'].iloc[i - long_time]) / df['Close'][i-long_time] < -targeted_profit \
      #      and in_long == 1 and i != 0: 
                
            Neutralize_long.append(df['Exchange Date'].iloc[i])
            in_long = 0
            long_time = 0
            
            
        if in_long == 1 and i == (len(df)-1): 
            Neutralize_long.append(df['Exchange Date'].iloc[i])
            long_time = 0 
            in_long = 0
            
            #* ((3/2 - (df['ATR'].iloc[i])) * 
     
              #it neutralizes immediately after last short signal, this will cause problems when including more than one datasets    
    for i in range(0,len(df)):
        
    
        if df['RSI_above_lower_level'][i] == 1 and df['Bear_cross'][i] == 1 and in_short != 1 and i !=0:
             
            Short_signal.append(df['Exchange Date'].iloc[i])
           # targeted_profit = (-1) * df['logr_ema'].iloc[i]
            targeted_profit_short = df["Ch_exit_short"].iloc[i]
            ##targeted_profit_short = (-1) * ((3/2 - (df['ATR'].iloc[i]))  * abs(np.log(df['Close'].iloc[i])/df['Close'].iloc[i-8])) #reversed order since trend is likely sloaping downwards
            in_short= 1 
            continue
            
            #bryt så att den rullar över första när vi har appendat för signal
  
            #targeted_profit = (-1) * (3/2 - df['Signal_strength'].iloc[i]) * 20 / df['rolling_mean'].iloc[i] #adjusting for signal strength during downard trend. Also want to make this level relative to the price level
            #print(targeted_profit)
           
            
        
        if in_short == 1:
            short_time += 1
            
            
        if df['Close'].iloc[i] > (3/3) * targeted_profit_short and in_short == 1 and i != 0:      
            #or df['Close'].iloc[i] / df['Close'][i-short_time] > (-1) * (3/3) * targeted_profit_short \
           
            
   #     if ((df['Close'].iloc[i] - df['Close'].iloc[i-short_time]) / df['Close'].iloc[i-short_time] < targeted_profit \
    #        or (df['Close'].iloc[i] - df['Close'].iloc[i - short_time]) / df['Close'][i-short_time] > -targeted_profit) \
   #         and in_short == 1 and i != 0: 
                
    
    #is this correct...
            Neutralize_short.append(df['Exchange Date'].iloc[i])
            in_short = 0
            short_time = 0
            
            
        if in_short == 1 and i == (len(df)-1): 
            Neutralize_short.append(df['Exchange Date'].iloc[i])
            short_time = 0 
            in_short = 0   
            
         
            
        #%%


#        short_positions = []
#        long_positions = []
#        
#        for i in range(0,len(Neutralize_short)):
#            short_positions.append(Short_signal[i])
#            short_positions.append(Neutralize_short[i])
            
#        for i in range(0,len(Long_signal)):   
#            long_positions.append(Long_signal[i])
#            long_positions.append(Neutralize_long[i])
            
#        print(long_positions)
#        print(short_positions)    
    #print(Long_signal,Neutralize_long)  
  #  print(df['Close'].loc[Short_signal],df['Close'].loc[Neutralize_short])  
 #   print(df['Close'].loc[Long_signal],df['Close'].loc[Neutralize_long])  
      
   
    #return signals
    return df
   
#x = '//Users//ianwallgren//Desktop//SP500_intraday_copy.xlsx'  
#x = event_price_list[0]    
#df = definitions(x)    
#df = definitions('//Users//ianwallgren//Desktop//SP500_intraday_copy.xlsx' ) 

#df = definitions('//Users//ianwallgren//Documents//GitHub//momentum-events//1_week.csv')

df = definitions(1)  

#df = 0



#%%


#print((len(Short_signal)))
#print((len(Neutralize_short)))

#print(Short_signal)
#print(Neutralize_short)

#print((len(Long_signal)))
#print((len(Neutralize_long)))

#print(Long_signal)
#print(Neutralize_long)
        
#%%
#Stats
#To do: revise all stats methods
#%%
#%%
#Earnings

def total_returns(df):
    long_short_returns = []
    trades_long = min(len(Long_signal),len(Neutralize_long)) #list should be of same length either way
    long_returns = []
   # cumu_long = 0
 #   cumulative_long = []
    
    trades_short = min(len(Short_signal),len(Neutralize_short))
    short_returns = [] 
    
    long_returns_noweight = []
    short_returns_noweight = []
    
    long_returns_noweight_event = []
    short_returns_noweight_event = []
    
    long_returns_outside_event = []
    short_returns_outside_event = []
    
    
    lo = pd.DataFrame()
    lo['long'] = Long_signal
    
    sh = pd.DataFrame()
    sh['short'] = Short_signal  #here is the problem, but what, eller e det i sho['short]
  #  print(len(sh['short']))
    
 #   Lon_signal = tuple(Long_signal)
    long_returns_event = []
    short_returns_event = [] 
    
    #redo it so that we calculate it same way as long returns byut with a negative sign 
    #minus
    
    for i in range (0,trades_long):
        long_returns_noweight.append((df['Close'].loc[Neutralize_long][i]*df['ATR'].loc[Long_signal][i] - df['Close'].loc[Long_signal][i]*df['ATR'].loc[Long_signal][i]) / df['Close'].loc[Long_signal][i]*df['ATR'].loc[Long_signal][i])
       
    for i in range (0,trades_short):
        short_returns_noweight.append((-1)*(df['Close'].loc[Neutralize_short][i]*((3/2)-df['ATR'].loc[Short_signal][i]) - df['Close'].loc[Short_signal][i]*((3/2)-df['ATR'].loc[Short_signal][i])) / df['Close'].loc[Short_signal][i]*((3/2)-df['ATR'].loc[Short_signal][i]))
    
       #should be trades_short
    for i in range (0,len(Short_signal)):
        for s in range(0,len(dates)):
            if sh['short'].iloc[i].year == dates['start_date'].iloc[s].year and sh['short'].iloc[i].month  == dates['start_date'].iloc[s].month and sh['short'].iloc[i].day  == dates['start_date'].iloc[s].day:
                short_returns.append(3*(-1)*(df['Close'].loc[Neutralize_short][i]*((3/2)-df['ATR'].loc[Short_signal][i]) - df['Close'].loc[Short_signal][i]*((3/2)-df['ATR'].loc[Short_signal][i])) / df['Close'].loc[Short_signal][i]*((3/2)-df['ATR'].loc[Short_signal][i]))
                short_returns_event.append((-1)*(df['Close'].loc[Neutralize_short][i]*((3/2)-df['ATR'].loc[Short_signal][i]) - df['Close'].loc[Short_signal][i]*((3/2)-df['ATR'].loc[Short_signal][i])) / df['Close'].loc[Neutralize_short][i]*((3/2)-df['ATR'].loc[Short_signal][i]))
                short_returns_noweight_event.append('Event')
                break
               
            elif sh['short'].iloc[i].year == dates['event_date'].iloc[s].year and sh['short'].iloc[i].month == dates['event_date'].iloc[s].month and sh['short'].iloc[i].day == dates['event_date'].iloc[s].day:
                short_returns.append(3*(-1)*(df['Close'].loc[Neutralize_short][i]*((3/2)-df['ATR'].loc[Short_signal][i]) - df['Close'].loc[Short_signal][i]*((3/2)-df['ATR'].loc[Short_signal][i])) / df['Close'].loc[Short_signal][i]*((3/2)-df['ATR'].loc[Short_signal][i]))
                short_returns_event.append((-1)*(df['Close'].loc[Neutralize_short][i]*((3/2)-df['ATR'].loc[Short_signal][i]) - df['Close'].loc[Short_signal][i]*((3/2)-df['ATR'].loc[Short_signal][i])) / df['Close'].loc[Neutralize_short][i]*((3/2)-df['ATR'].loc[Short_signal][i]))
                short_returns_noweight_event.append('Event')
                break
        
            elif sh['short'].iloc[i].year == dates['end_date'].iloc[s].year and sh['short'].iloc[i].month == dates['end_date'].iloc[s].month and sh['short'].iloc[i].day == dates['end_date'].iloc[s].day:
                short_returns.append(3*(-1)*(df['Close'].loc[Neutralize_short][i]*((3/2)-df['ATR'].loc[Short_signal][i]) - df['Close'].loc[Short_signal][i]*((3/2)-df['ATR'].loc[Short_signal][i])) / df['Close'].loc[Short_signal][i]*((3/2)-df['ATR'].loc[Short_signal][i]))
                short_returns_event.append((-1)*(df['Close'].loc[Neutralize_short][i]*((3/2)-df['ATR'].loc[Short_signal][i]) - df['Close'].loc[Short_signal][i]*((3/2)-df['ATR'].loc[Short_signal][i])) / df['Close'].loc[Neutralize_short][i]*((3/2)-df['ATR'].loc[Short_signal][i]))
                short_returns_noweight_event.append('Event')
                break
            elif s==len(dates)-1:
                short_returns.append((-1)*(df['Close'].loc[Neutralize_short][i]*((3/2)-df['ATR'].loc[Short_signal][i]) - df['Close'].loc[Short_signal][i]*((3/2)-df['ATR'].loc[Short_signal][i])) / df['Close'].loc[Short_signal][i]*((3/2)-df['ATR'].loc[Short_signal][i]))
                short_returns_noweight_event.append('Outside event')
                #short_returns_outside_event.append((-1)*(df['Close'].loc[Neutralize_short][i]*((3/2)-df['ATR'].loc[Short_signal][i]) - df['Close'].loc[Short_signal][i]*((3/2)-df['ATR'].loc[Short_signal][i])) / df['Close'].loc[Short_signal][i]*((3/2)-df['ATR'].loc[Short_signal][i]))
            else:
                continue     
    
    for i in range (0,trades_long):
        for k in range(0,len(dates)):
            if lo['long'].iloc[i].year == dates['start_date'].iloc[k].year and lo['long'].iloc[i].month  == dates['start_date'].iloc[k].month and lo['long'].iloc[i].day  == dates['start_date'].iloc[k].day:
                long_returns.append(3*((df['Close'].loc[Neutralize_long][i]*df['ATR'].loc[Long_signal][i] - df['Close'].loc[Long_signal][i]*df['ATR'].loc[Long_signal][i])) / df['Close'].loc[Long_signal][i]*df['ATR'].loc[Long_signal][i])
                long_returns_event.append((df['Close'].loc[Neutralize_long][i]*df['ATR'].loc[Long_signal][i] - df['Close'].loc[Long_signal][i]*df['ATR'].loc[Long_signal][i]) / df['Close'].loc[Long_signal][i]*df['ATR'].loc[Long_signal][i])
                long_returns_noweight_event.append('Event')
                break
               
            elif lo['long'].iloc[i].year == dates['event_date'].iloc[k].year and lo['long'].iloc[i].month == dates['event_date'].iloc[k].month and lo['long'].iloc[i].day == dates['event_date'].iloc[k].day:
                long_returns.append(3*((df['Close'].loc[Neutralize_long][i]*df['ATR'].loc[Long_signal][i] - df['Close'].loc[Long_signal][i]*df['ATR'].loc[Long_signal][i])) / df['Close'].loc[Long_signal][i]*df['ATR'].loc[Long_signal][i])
                long_returns_event.append((df['Close'].loc[Neutralize_long][i]*df['ATR'].loc[Long_signal][i] - df['Close'].loc[Long_signal][i]*df['ATR'].loc[Long_signal][i]) / df['Close'].loc[Long_signal][i]*df['ATR'].loc[Long_signal][i])
                long_returns_noweight_event.append('Event')
                break
        
            elif lo['long'].iloc[i].year == dates['end_date'].iloc[k].year and lo['long'].iloc[i].month == dates['end_date'].iloc[k].month and lo['long'].iloc[i].day == dates['end_date'].iloc[k].day:
                long_returns.append(3*((df['Close'].loc[Neutralize_long][i]*df['ATR'].loc[Long_signal][i] - df['Close'].loc[Long_signal][i]*df['ATR'].loc[Long_signal][i])) / df['Close'].loc[Long_signal][i]*df['ATR'].loc[Long_signal][i])
                long_returns_event.append((df['Close'].loc[Neutralize_long][i]*df['ATR'].loc[Long_signal][i] - df['Close'].loc[Long_signal][i]*df['ATR'].loc[Long_signal][i]) / df['Close'].loc[Long_signal][i]*df['ATR'].loc[Long_signal][i])
                long_returns_noweight_event.append('Event')
                break
            
            
           # if k==len(dates)-1:
           # elif k==len(dates):
                
                
            elif k==len(dates)-1:
                long_returns.append((df['Close'].loc[Neutralize_long][i]*df['ATR'].loc[Long_signal][i] - df['Close'].loc[Long_signal][i]*df['ATR'].loc[Long_signal][i])  / df['Close'].loc[Long_signal][i]*df['ATR'].loc[Long_signal][i])
                long_returns_noweight_event.append('Outside event')
                #long_returns_outside_event.append((df['Close'].loc[Neutralize_long][i]*df['ATR'].loc[Long_signal][i] - df['Close'].loc[Long_signal][i]*df['ATR'].loc[Long_signal][i])  / df['Close'].loc[Long_signal][i]*df['ATR'].loc[Long_signal][i])
                
       
            else:
                continue
    
        
    
    # if [Long_signal][i].year, Long_signal][i].year, Long_signal][i].year == dates['start_date'].iloc[i].date() or [Long_signal][i].date() == dates['event_date'].iloc[i].date()  or [Long_signal][i].date() == dates['end_date'].iloc[i].date():
            
    
            
           # cumu_long = cumu_long + long_returns[i]
          #  cumulative_long.append(cumu_long)
            
       
   
    long_short_returns.append(long_returns)
    
    
    
  #  for i in range (0,trades_short):
   #      if [Short_signal][i] in dates['start_date'] or dates['event_date'] or dates['end_date']:
 #            short_returns.append(3*(df['Close'].loc[Short_signal][i]*df['ATR'].loc[Short_signal][i] - df['Close'].loc[Neutralize_short][i]*df['ATR'].loc[Short_signal][i]))
  #       else:
  #           short_returns.append(df['Close'].loc[Short_signal][i]*df['ATR'].loc[Short_signal][i] - df['Close'].loc[Neutralize_short][i]*df['ATR'].loc[Short_signal][i])
    
  
 #   long_short_returns.append(short_returns)
    #total_returns = sum(long_short_returns)
    total_noweight = sum(short_returns_noweight) + sum(long_returns_noweight)
    total = sum(short_returns) + sum(long_returns)
    total_event = sum(short_returns_event) + sum(long_returns_event)
    
   # print(short_returns.describe())
    #print(long_returns.describe())
   # print(total)
   
    short_noweights = pd.DataFrame()
    short_noweights['Short returns'] = short_returns_noweight
    short_noweights['Classification'] = short_returns_noweight_event
    
    
    long_noweights = pd.DataFrame()
    long_noweights['Long returns'] = long_returns_noweight
    long_noweights['Classification'] = long_returns_noweight_event
    
 #   long_returns_outside_event = pd.DataFrame()
 #   long_returns_outside_event['long_returns_outside_event'] = long_returns_outside_event
    
   # short_returns_outside_event = pd.DataFrame()
   # short_returns_outside_event['short_returns_outside_event'] = short_returns_outside_event

    sho = pd.DataFrame()
    sho['sho'] = short_returns
    
    lon = pd.DataFrame()
    lon['lon'] = long_returns
    
    long_event = pd.DataFrame()
    long_event['Long returns'] = long_returns_event
    
    short_event = pd.DataFrame()
    short_event['Short returns'] = short_returns_event
    
   # print(lon['lon'].describe())
   # print(sho['sho'].describe())
    
    return total, lon['lon'], sho['sho'], long_event['Long returns'], short_event['Short returns'], long_noweights, short_noweights, total_noweight, total_event
#%%
#t = total_returns(df)

t = total_returns(df)
totall = t[0]
longg = t[1]
shortt = t[2]
longg_event = t[3]
shortt_event = t[4]
longg_noweights = t[5]
shortt_noweights = t[6]
totall_noweight = t[7]
totall_event = t[8]
#longg_outside_events = t[9]
#shortt_outside_events = t[10]


        
#%%
#shortt_noweight_positive = []
#shortt_noweight_negative = []
##shortt_noweight_zero = []
#for i in shortt_noweights['Short returns']:
#    if i > 0:
#    elif i < 0:
#        shortt_noweight_negative.append(i)
#    else:
#        shortt_noweight_zero.append(i)
        
#longg_noweight_positive = []
##longg_noweight_negative = []
longg_noweight_zero = []
#for i in longg_noweights['Long returns']:
#    if i > 0:
#        longg_noweight_positive.append(i)
#    elif i < 0:
#        longg_noweight_negative.append(i)
#    else:
#        longg_noweight_zero.append(i)

#shortt_event_positive = []
#shortt_event_negative = []
#shortt_event_zero = []
##for i in shortt_event['Short returns']:
#    if i > 0:
#        shortt_event_positive.append(i)
#    elif i < 0:
#        shortt_event_negative.append(i)
#    else:
#        shortt_event_zero.append(i)

#longg_event_positive = []
#longg_event_negative = []
#longg_event_zero = []
#for i in longg_event['Long returns']:
#    if i > 0:
##        longg_event_positive.append(i)
#    elif i < 0:
#        longg_event_negative.append(i)
#    else:
#        longg_event_zero.append(i)    
#%%
#win rate (excluding zeroes)

#short_noweight
#wr_short_noweight = len(shortt_noweight_positive)/(len(shortt_noweight_positive)+len(shortt_noweight_negative))
#short_weight
#short_event
#wr_short_event = len(shortt_event_positive)/(len(shortt_event_positive)+len(shortt_event_negative))

#long_noweight
#wr_long_noweight = len(longg_noweight_positive)/(len(longg_noweight_positive)+len(longg_noweight_negative))
#long_weight
#long_event
#wr_long_event = len(longg_event_positive)/(len(longg_event_positive)+len(longg_event_negative))
long_event_pos = []
long_event_neg = []
long_event_ze = []
for i in longg_event:
    if i>0:
        long_event_pos.append(i)
    if i<0:
        long_event_neg.append(i)
    if i==0:
        long_event_ze.append(i)
        
short_event_pos = []
short_event_neg = []
short_event_ze = []
for i in shortt_event:
    if i>0:
        short_event_pos.append(i)
    if i<0:
        short_event_neg.append(i)
    if i==0:
        short_event_ze.append(i)
        
long_nowe_pos = []
long_nowe_neg = []
long_nowe_ze = []
for i in longg_noweights['Long returns']:
    if i>0:
        long_nowe_pos.append(i)
    if i<0:
        long_nowe_neg.append(i)
    if i==0:
        long_nowe_ze.append(i)
        
short_nowe_pos = []
short_nowe_neg = []
short_nowe_ze = []
for i in shortt_noweights['Short returns']:
    if i>0:
        short_nowe_pos.append(i)
    if i<0:
        short_nowe_neg.append(i)
    if i==0:
        short_nowe_ze.append(i)
        

#%%
#avg return

#short_noweight
#avg_return_shortt_noweight = sum(shortt_noweights) / len(shortt_noweights)
#short_weight
#avg_return_shortt = sum(shortt) / len(shortt)
#short_event
#avg_return_shortt_event = sum(shortt_event) / len(shortt_event)

#long_noweight
#avg_return_longg_noweight = sum(longg_noweights) / len(longg_noweights)
#long_weight
#avg_return_longg = sum(longg) / len(longg)
#long_event
#avg_return_longg_event = sum(longg_event) / len(longg_event)

#%%
#l = pd.DataFrame()
#l['Date_op'] = Long_signal
#l['Date_cl'] = Neutralize_long
#l['Signal'] = 'Long'
#date_list = df['Exchange Date'].reset_index(drop=True)
#date_list = date_list.to_frame()
#date_list2 = df.reset_index(drop=True)
#date_list2 = date_list2[date_list2['Exchange Date'].isin(Neutralize_long)]
#date_list3 = df.reset_index(drop=True)
#date_list3 = date_list3[date_list3['Exchange Date'].isin(Long_signal)]
#time_diff = (date_list2.index - date_list3.index)
#l['Hold_per'] = time_diff
#l['Returns'] = longg_noweights['Classification'] #l['Returns'] = longg_noweights['Classification']
#sns.set(rc = {'figure.figsize':(15,8)})
#sns.barplot(x = "Date_op", y = "Hold_per", data = l, hue = longg_noweights['Classification'])
#plt.savefig('long_position_holding_per.png', dpi=300)





#longg_event.to_excel("longg_event.xlsx",index=False)


#%%
#total return

#short_noweight
#total_return_shortt_noweight = sum(shortt_noweights)
#short_weight
#total_return_shortt = sum(shortt)
#short_event
#total_return_shortt_event = sum(shortt_event)

#long_noweight
#total_return_longg_noweight = sum(longg_noweights)
#long_weight
#total_return_longg = sum(longg)
#long_event
#total_return_longg_event = sum(longg_event)

#total
#total_return_long_short_noweight = totall_noweight
#total_return_long_short = totall
#total_return_long_short_event = totall_event
#%%
#number of trades

#short_noweight
#trades_shortt_noweight = len(shortt_noweights)
#short_weight
#trades_shortt = len(shortt)
#short_event
#trades_shortt_event = len(shortt_event)

#long_noweight
trades_longg_noweight = len(longg_noweights)
#long_weight
trades_longg = len(longg)
#long_event
trades_longg_event = len(longg_event)

#%%


#acc_longg_returns = []
#sum_longg = 0

#for i in longg_noweights:    
#    sum_longg = sum_longg + i
#    acc_longg_returns.append(sum_longg)
    
#plt.plot(Neutralize_long, acc_long_returns)        
#plt.plot(acc_longg_returns, label = 'cumulative long returns')      

#acc_shortt_returns = []
#sum_shortt = 0

#for i in shortt_noweights:    
#    sum_shortt = sum_shortt + i
#    acc_shortt_returns.append(sum_shortt)
    

    
#plt.plot(Neutralize_long, acc_long_returns)        
#plt.plot(acc_shortt_returns, label = 'cumulative short returns')    
#xlabel = ('Cumulative returns')
#plt.savefig('cumulative_long_short_returns')

#we should use sheatherjones bandwidth, don't know the keyword argument


#sns.kdeplot(longg_noweights, shade = True)
#sns.kdeplot(shortt_event, shade = True, clip = (-30,30), bw = 0.5, color = "orange")
#plt.savefig('short_event_KDE_bw05')


#ax = sns.boxplot(x=shortt)
#sns.color_palette("dark:salmon_r")

#sns.lineplot(data=df,x='Close')

#fig, axes = plt.subplots(1,2)
#ax = sns.histplot(data = shortt_noweights, x = 'Short returns', bins = 100, binrange = [-0.0075,0.0075], stat = 'probability', hue = 'Classification', multiple='stack', palette='dark')
#ax = sns.histplot(data = longg_noweights, x = 'Long returns', bins = 100, binrange = [-0.0075,0.0075], stat = 'probability', hue = 'Classification', common_norm = False, element='step', fill = False)
#ax2 = sns.histplot(data = shortt_noweights, x = 'Short returns', bins = 100, binrange = [-0.0075,0.0075], stat = 'probability', hue = 'Classification', element='step', multiple='stack', ax=axes[1])
#ax1.legend(fontsize='medium')
#plt.legend(loc='center right')
 
#ax = sns.histplot(data = longg_noweights, x = 'Long returns', bins = 100, binrange = [-0.0075,0.0075], stat = 'probability', hue = 'Classification', common_norm = False, element='step', fill = False, ax=axes[1])
#multiple = 'stack',
#,  palette = 'dark:salmon_r'
#ax2 = ax.twinx()
#ax2.plot(shortt_event)
#shortt_event.plot(ax=ax2)
#ax.set_title('Long returns 2012-2021, independent normalization')
#ax.set_xlabel('Long returns in percentages')

#fig = plt.hist(longg_noweights['long returns'],bins = 200, range=[-0.0075,0.0075], color="green")

#plt.savefig('histogram_long_returns_noweights_hue_event_nofill',dpi=1200)


#%%

# Compute density estimates using 'silverman'



# Generate a distribution and some data
#dist = norm(loc=0, scale=1)
#data = shortt
#dist.rvs(2**8) # Generate 2**8 points


# Generate a distribution and some multimodal data
#dist1 = norm(loc=0, scale=1)
#dist2 = norm(loc=10, scale=1)
#data = np.hstack([dist1.rvs(2**8), dist2.rvs(2**8)])

# Compute density estimates using 'silverman'
#x, y = FFTKDE(kernel='gaussian', bw='silverman').fit(data).evaluate()
#plt.plot(x, y, label='KDE /w silverman')

# Compute density estimates using 'ISJ' - Improved Sheather Jones
#y = FFTKDE(kernel='gaussian', bw='isj').fit(data).evaluate(x)
#plt.plot(x, y, label='KDE /w ISJ')

#plt.plot(x, (dist1.pdf(x) + dist2.pdf(x))/2, label='True pdf')
#plt.grid(True, ls='--', zorder=-15); plt.legend()
#plt.show()

#sns.kdeplot(shortt_noweights, shade = True, clip = (-30,30), bw = 'ISJ', color = "orange") #, bw_adjust = .2
#plt.ylabel = ("Returns")
#plt.show()
#plt.savefig('long_events_KDE')
#sns.kdeplot(shortt_event, shade = True)
#sns.kdeplot(longg_event, shade = True, cut = -2) #, showfliers = False



#%%
#total, long_returns, short_returns, 
#total_returns = total_returns(df)

#sns.set()
#plt.figure(figsize=(12,4)) 

#.xlabel('Signal #', fontsize=15)
#plt.ylabel('Returns', fontsize=15)

#plt.plot(longg, label = 'long returns')
#plt.plot(shortt, label = 'short returns')

#plt.legend(loc=0,fontsize='medium')

#plt.plot(df['Close'], label = 'Close price', alpha = 0.7)

#plt.legend(loc=0,fontsize='medium')
#plt.savefig('SP500_daily_(14,70,30)_(3,40,3)_updated_code', bbox_inches = 'tight')

#f = total_returns(df)
#c = long_returns
#d = short_returns

#sns.set()
#plt.figure(figsize=(12,4)) 

#plt.xlabel('Date', fontsize=15)
#plt.ylabel('Return', fontsize=15)

#
#plt.plot(d, label = 'short returns')

#
#def total_returns(df):
#    long_short_returns = []
#    trades_long = min(len(Long_signal),len(Neutralize_long)) #list should be of same length either way
#    long_returns = []
 #   cumu_long = 0
#    cumulative_long = []
#    for i in range (0,trades_long):
#        long_returns.append(df['Close'].loc[Neutralize_long][i]*df['ATR'].loc[Long_signal][i] - df['Close'].loc[Long_signal][i]*df['ATR'].loc[Long_signal][i])
 #       cumu_long = cumu_long + long_returns[i]
 #       cumulative_long.append(cumu_long)
 #   long_short_returns.append(long_returns)
    
 #   trades_short = min(len(Short_signal),len(Neutralize_short))
 #   short_returns = []
#    for i in range (0,trades_short):
 #       short_returns.append(df['Close'].loc[Short_signal][i]*df['ATR'].loc[Short_signal][i] - df['Close'].loc[Neutralize_short][i]*df['ATR'].loc[Short_signal][i])
    #print(short_returns)
  #
#    long_short_returns.append(short_returns)
    #total_returns = sum(long_short_returns)
 #   total = sum(short_returns) + sum(long_returns)
    
   
    
#    return total






#%%
#Accumulated returns
#accumulated_returns = [] 

#for i in range (0,trades_long):
#    accumulated_returns.append(df['Close'].loc[Neutralize_long][i]*df['Signal_strength'].loc[Long_signal][i] - df['Close'].loc[Long_signal][i]*df['Signal_strength'].loc[Long_signal][i])
    
    
    
#    loc.accumulated_returns[i] = loc.Neutralize_long[i]
#print(long_returns)

#for i in range (0,trades_short):
 #   accumulated_returns.append(df['Close'].loc[Short_signal][i]*df['Signal_strength'].loc[Short_signal][i] - df['Close'].loc[Neutralize_short][i]*df['Signal_strength'].loc[Short_signal][i])




#%%
#win rate
#trades with 0 return are accounted for as positive returns 

def win_rate(df):
    trades_long = min(len(Long_signal),len(Neutralize_long)) #list should be of same length either way
    long_returns = []
    for i in range (0,trades_long):
        long_returns.append(df['Close'].loc[Neutralize_long][i]*df['Signal_strength'].loc[Long_signal][i] - df['Close'].loc[Long_signal][i]*df['Signal_strength'].loc[Long_signal][i])
    #print(long_returns)
    
    trades_short = min(len(Short_signal),len(Neutralize_short))
    short_returns = []
    for i in range (0,trades_short):
        short_returns.append(df['Close'].loc[Short_signal][i]*df['Signal_strength'].loc[Short_signal][i] - df['Close'].loc[Neutralize_short][i]*df['Signal_strength'].loc[Short_signal][i])
    
    long_short_returns = long_returns + short_returns  
    pos_returns = []
    neg_returns = []
    
    for i in long_short_returns:
         if i >= 0:
             pos_returns.append(i)
         else:
             neg_returns.append(i)
     
    
    win_rate = (len(pos_returns))/(len(pos_returns)+len(neg_returns))
    return win_rate

win_rate = win_rate(df)

#%%
#Number of trades
def total_trades(df):
    long_short_returns = []
    trades_long = min(len(Long_signal),len(Neutralize_long)) #list should be of same length either way
    long_returns = []
    for i in range (0,trades_long):
        long_returns.append(df['Close'].loc[Neutralize_long][i]*df['Signal_strength'].loc[Long_signal][i] - df['Close'].loc[Long_signal][i]*df['Signal_strength'].loc[Long_signal][i])
    
    
    trades_short = min(len(Short_signal),len(Neutralize_short))
    short_returns = []
    for i in range (0,trades_short):
        short_returns.append(df['Close'].loc[Short_signal][i]*df['Signal_strength'].loc[Short_signal][i] - df['Close'].loc[Neutralize_short][i]*df['Signal_strength'].loc[Short_signal][i])
 
    total_trades = trades_long + trades_short
    return total_trades

total_trades = total_trades(df)

#%%
#Having a problem with the syntax atm

#Average return

long_short_returns = []
trades_long = min(len(Long_signal),len(Neutralize_long)) #list should be of same length either way
long_returns = []
for i in range (0,trades_long):
    long_returns.append(df['Close'].loc[Neutralize_long][i]*df['ATR'].loc[Long_signal][i] - df['Close'].loc[Long_signal][i]*df['ATR'].loc[Long_signal][i])
    
    
trades_short = min(len(Short_signal),len(Neutralize_short))
short_returns = []
for i in range (0,trades_short):
    short_returns.append(df['Close'].loc[Short_signal][i]*df['ATR'].loc[Short_signal][i] - df['Close'].loc[Neutralize_short][i]*df['ATR'].loc[Short_signal][i])
 
long_short_returns.append(sum(long_returns))
long_short_returns.append(sum(short_returns))
long_short_returns = sum(long_short_returns)
total_trades = trades_long + trades_short
Avg_return = long_short_returns / total_trades

    
#%%

#l_d = {'Long_signal': Long_signal, 'Neutralize_long': Neutralize_long}
#s_d = {'Short_signal': Short_signal, 'Neutralize_short': Neutralize_short}

#long_results_file = pd.DataFrame(l_d)
#short_results_file = pd.DataFrame(s_d)


#pd.to_excel(long_results_file,"Longs")
#pd.to_excel(short_results_file,"Shorts")


#%%
#Average position length. Code needs to be updated according to the new structure.

def holding_period(df):
    long_locations = df.loc[Long_signal].index
    short_locations = df.loc[Short_signal].index
    neutralize_long_locations = df.loc[Neutralize_long].index
    neutralize_short_locations = df.loc[Neutralize_short].index
    
    holding_period_long = []
    holding_period_long_seconds = []
    
    holding_period_short = []
    holding_period_short_seconds = []
    
    for i in range(0,len(long_locations)):
        if long_locations[i].date() == neutralize_long_locations[i].date():
           holding_period_long.append(neutralize_long_locations[i]-long_locations[i])
           holding_period_long_seconds.append(datetime.timedelta.total_seconds(holding_period_long[i]))
            
        else:
            holding_period_long.append(neutralize_long_locations[i]-long_locations[i])
            holding_period_long_seconds.append(datetime.timedelta.total_seconds(holding_period_long[i]
            -datetime.timedelta(seconds=63000))) #63000 seconds between stock market closes and opens (?)
            
    for i in range(0,len(short_locations)):
        if short_locations[i].date() == neutralize_short_locations[i].date():
           holding_period_short.append(neutralize_short_locations[i]-short_locations[i])
           holding_period_short_seconds.append(datetime.timedelta.total_seconds(holding_period_short[i]))
            
        else:
            holding_period_short.append(neutralize_short_locations[i]-short_locations[i])
            holding_period_short_seconds.append(datetime.timedelta.total_seconds(holding_period_short[i]
            -datetime.timedelta(seconds=63000)))
    
    total_holding_period_seconds = holding_period_long_seconds + holding_period_short_seconds
    avg_position_length = (sum(total_holding_period_seconds) / len(total_holding_period_seconds)) / 60 
    return avg_position_length

avg_position_length = holding_period(df)

#%%
#Print statements

print('Average return:' +' ' +str(Avg_return))
print('Average position length (min):' +' ' +str(avg_position_length))
print('Win rate:' +' ' +str(win_rate))
print('Number of trades:' +' ' +str(total_trades))
print('Total returns:'+' '+ str(total_returns))



#%%    
#Plotting       
    

sns.set()
plt.figure(figsize=(12,4)) 
plt.scatter(df.loc[Long_signal].index,df.loc[Long_signal]['Close'], marker = '^',label = 'Long', color = 'green')
plt.scatter(df.loc[Short_signal].index,df.loc[Short_signal]['Close'], marker = '^',label = 'Short', color = 'red')
plt.scatter(df.loc[Neutralize_long].index,df.loc[Neutralize_long]['Close'], marker = 'x',label = 'Neutralize long', color = 'green')
plt.scatter(df.loc[Neutralize_short].index,df.loc[Neutralize_short]['Close'], marker = 'o', label = 'Neutralize short', color = 'red')
plt.plot(df['Close'], label = 'Close price', alpha = 0.7)
plt.xticks(rotation=25)
plt.title('SP500 Close price and buy/sell signals,' +''+ 'win rate:'+''+str(win_rate), fontsize= 15)
plt.xlabel('Date', fontsize=15)
plt.ylabel('Close price', fontsize=15)
plt.legend(loc=0,fontsize='medium')
#plt.savefig('SP500_daily_(14,70,30)_(3,40,3)_updated_code', bbox_inches = 'tight')



#%%
#Results
#Positive accumulated profit on short position trades
#Negative accumulated profit in long position trades

acc_long_returns = []
sum_long = 0

for i in longg:    
    sum_long = sum_long
    acc_long_returns.append(sum_long)
    
#plt.plot(Neutralize_long, acc_long_returns)

acc_short_returns = []
sum_short = 0

for i in shortt:    
    sum_short = sum_short + i
    acc_short_returns.append(sum_short)
    
#plt.plot(Neutralize_short, acc_short_returns)

l = pd.DataFrame()
l['Date_op'] = Long_signal
l['Date_cl'] = Neutralize_long
l['Signal'] = 'Long'

perc_ch = [] 
for i in range (0,trades_long):
   perc_ch.append((df['Close'].loc[Neutralize_long][i] - df['Close'].loc[Long_signal][i])/df['Close'].loc[Long_signal][i])

l['Perc_ch'] = perc_ch

#%%

#sns.set(rc = {'figure.figsize':(15,8)})
#sns.lineplot("Date_op", "Perc_ch", data = l)
#plt.savefig('long_position_holding_per.png', dpi=300)


#%%

l = pd.DataFrame()
l['Date_op'] = Long_signal
l['Date_cl'] = Neutralize_long
l['Signal'] = 'Long'
date_list = df['Exchange Date'].reset_index(drop=True)
date_list = date_list.to_frame()
date_list2 = df.reset_index(drop=True)
date_list2 = date_list2[date_list2['Exchange Date'].isin(Neutralize_long)]
date_list3 = df.reset_index(drop=True)
date_list3 = date_list3[date_list3['Exchange Date'].isin(Long_signal)]
time_diff = (date_list2.index - date_list3.index)
l['Hold_per'] = time_diff
l['Returns'] = longg_noweights['Classification']
sns.set(rc = {'figure.figsize':(15,8)})
ax = sns.barplot(x = "Returns", y = "Hold_per", data = l, lw=0., ci=99)
ax.set_title('Long returns holding period, measured in 15 min bars with 99% CI', fontsize=25)
ax.set_xlabel('Classification',fontsize=15)
ax.set_ylabel('Holding period mean',fontsize=15)
plt.savefig('long_position_holding_per_mean_ci99.png', dpi=1200)
#%%
s = pd.DataFrame()
s['Date_op'] = Short_signal
s['Date_cl'] = Neutralize_short
s['Signal'] = 'Long'
date_list = df['Exchange Date'].reset_index(drop=True)
date_list = date_list.to_frame()
date_list2 = df.reset_index(drop=True)
date_list2 = date_list2[date_list2['Exchange Date'].isin(Neutralize_short)]
date_list3 = df.reset_index(drop=True)
date_list3 = date_list3[date_list3['Exchange Date'].isin(Short_signal)]
time_diff = (date_list2.index - date_list3.index)
s['Hold_per'] = time_diff
s['Returns'] = shortt_noweights['Classification']
sns.set(rc = {'figure.figsize':(15,8)})
ax = sns.barplot(x = "Returns", y = "Hold_per", data = s, lw=0., ci=99)
ax.set_title('Short returns holding period, measured in 15 min bars with 99% CI', fontsize=25)
ax.set_xlabel('Classification',fontsize=15)
ax.set_ylabel('Holding period mean',fontsize=15)
plt.savefig('short_position_holding_per_mean_ci99.png', dpi=1200)


#sns.barplot(x = "Date_op", y = "Hold_per", data = l, hue = "Returns", lw=0.)
#palette = sns.color_palette("RdBu", n_colors=7),
#
#plt.savefig('long_position_holding_per.png', dpi=300)
#date_list = df['Exchange Date'].reset_index(drop=True)
#date_list = date_list.to_frame()

#date_list2 = df.reset_index(drop=True)
#date_list2 = date_list2[date_list2['Exchange Date'].isin(Neutralize_long)]

#date_list3 = df.reset_index(drop=True)
#date_list3 = date_list3[date_list3['Exchange Date'].isin(Long_signal)]

#time_diff = (date_list2.index - date_list3.index)
#value_1 = date_list['Exchange Date'].loc[Neutralize_long]
#value_1 = date_list2['Close'].loc[Neutralize_long]

#l['Hold_per'] = time_diff
#l['Returns'] = longg_noweights

#%%
#sns.set(rc = {'figure.figsize':(15,8)})
#sns.lineplot("Date_op", "Hold_per", data = l, hue = longg_noweights['Classification'])
#plt.savefig('long_position_holding_per.png', dpi=300)

#l['Hold_per'] = l['Date_cl'] - l['Date_op']
#l['Hold_per'] = l['Hold_per']/np.timedelta64(1,'m')
#l = l.drop([193])

#sns.barplot(x='Date_op', y='Hold_per', data=l)

#sns.set(rc = {'figure.figsize':(15,8)})
#sns.lineplot("Date", "Perc_ch", data = l[0:1000])
#plt.savefig('l.png', dpi=300)

longg_noweights.to_excel('longg_noweights.xlsx',index=False)

#%%
#s = pd.DataFrame()
#s['Date_op'] = Short_signal
#s['Date_cl'] = Neutralize_short
#s['Signal'] = 'Short'
#s['Perc_ch'] = s['Return'].pct_change()   # Cumulative return

#date_list4 = df.reset_index(drop=True)
#date_list4 = date_list4[date_list4['Exchange Date'].isin(Neutralize_short)]

#date_list5 = df.reset_index(drop=True)
#date_list5 = date_list5[date_list5['Exchange Date'].isin(Short_signal)]

#time_diff = (date_list4.index - date_list5.index)
#value_1 = date_list['Exchange Date'].loc[Neutralize_long]
#value_1 = date_list2['Close'].loc[Neutralize_long]

#s['Hold_per'] = time_diff

#sns.set(rc = {'figure.figsize':(15,8)})
#sns.lineplot("Date_op", "Hold_per", data = s)
#plt.savefig('short_position_holding_per.png', dpi=300)

#%%
#sns.set(rc = {'figure.figsize':(15,8)})
#sns.lineplot("Date", "Perc_ch", data = l[0:1000])
#plt.savefig('l.png', dpi=300)

#ls = pd.merge(s, l, how = 'outer')
#ls['Date'] = pd.to_datetime(ls['Date'].values, utc=True)
#ls.sort_values(by='Date', inplace=True)
#ls = ls.reset_index(drop=True)
#ls['Perc_ch'] = ls['Return'].pct_change()  

#sns.set(rc = {'figure.figsize':(15,8)})
#sns.lineplot("Date", "Perc_ch", data = ls)
#plt.savefig('ls_perc_ch.png', dpi=300)


#plt.plot(ls['Date'], ls['Return'])

#x = []
#summa = 0
#for i in range(0,len(ls)):
 #   summa = summa + ls['Return'].iloc[i]longlo g



#%%

#df = df.loc[: 1328] 





#for i in skew_list:
#    for j in timeperiod_list:
#        data = trading(df, i, j)
#        returns_list = returns(data)
#        stats_dict = {"Skewness":i,
#                      "Timeperiod":j,
#                      "Cumulative Profit":profit(data),
#                      "Mean Return":returns_list.mean(),
#                      "Std Return":returns_list.std(),
#                      "Avg Pos. Len":trade_len(data).mean(),
#                      "Max Return":returns_list.max(),
#                      "Min Return":returns_list.min(),
#                      "No Trades":len(returns_list),
#                      "Win Ratio":win_ratio(returns_list),
#                      "Profit Factor":profit_factor(data),
#                      "Sharpe Ratio":sharpe_ratio(returns_list.mean(), returns_list.std())
#                      }
#        results = results.append(stats_dict,ignore_index=True)
#Collapse






