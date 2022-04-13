#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 13:39:12 2022

@author: emildamirov
"""

import linchackathon as lh
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from matplotlib.dates import date2num
from pandas.tseries.offsets import BDay

#LAST UPDATED: 24 MARCH

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



dates = pd.DataFrame(events,columns=["event_date"])
dates['event_date'] = pd.to_datetime(dates["event_date"],utc=True)
dates['start_date'] = dates["event_date"] - BDay(1)
dates['end_date'] = dates["event_date"] + BDay(1)
columns_titles = ["start_date","event_date","end_date"]
dates=dates.reindex(columns=columns_titles)
dates.dropna(axis=0, inplace=True)
#%%
class TradingIndicator:
    
    # Obs time_series_data must be a df!
    def __init__(self, df):
        self.df = df

    # Simple MA with some default params
    def ma(self, window_size = 30, win_type='triang', col="Adj Close"):
        return self.df[col].rolling(window=window_size, win_type=win_type).mean()

    # Exponentially weighted MA
    def ewma(self, span = 10, adjust=False, col="Adj Close"):
        return self.df[col].ewm(span=span, adjust=adjust).mean()
    
     # Stochastic implementation.
    def stochastic_oscillator(self, span=14, ma_span=3, col="Adj Close"):
        data = self.df
        data["span_high"] = data.High.rolling(span).max()
        data["span_low"] = data.Low.rolling(span).min()
        data['%K'] = (data[col]- data['span_low'])*100/(data['span_high'] - data['span_low'])
        data['%D'] = data['%K'].rolling(ma_span).mean()
        return data
    
    # Moving average convergence divergence
    # Buy MACD over signal line.
    def macd(self, span1=12, span2=26, span_signal=9, col="Adj Close"):
        df = self.df
        
        df["macd"]=self.ewma(span=span1, col=col)-self.ewma(span=span2, col=col)
        df["signal"]=df["macd"].ewm(span=span_signal, adjust=False).mean()
        return df
    
    # Bollinger band for LT trading signals.
    # Trading signal sell when prices hit upper band, buy when prices hit lower band
    def bollinger_bands(self, rate=20, col="Adj Close"):
        sma = self.ma(window_size=rate)
        std = self.df[col].rolling(rate).std()
        bollinger_up = sma + std * 2 # Calculate top band
        bollinger_down = sma - std * 2 # Calculate bottom band
        return bollinger_up, bollinger_down
    
    # RSI indicator
    def rsi(self, periods=14, col="Adj Close", ema=True):
        deltas=self.df[col].diff(1)
     # Make two series: one for lower closes and one for higher closes
        up = deltas.clip(lower=0)
        down = -1 * deltas.clip(upper=0)
    
        if ema == True:
	    # Use exponential moving average
            ma_up = up.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
            ma_down = down.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
        else:
        # Use simple moving average
            ma_up = up.rolling(window = periods, adjust=False).mean()
            ma_down = down.rolling(window = periods, adjust=False).mean()
        
        rsi = ma_up / ma_down
        rsi = 100 - (100/(1 + rsi))
        return rsi
    
df = pd.read_csv('/Users/ianwallgren/Documents/GitHub/momentum-events/'+'1_week.csv')

    

#%%
#class Signal:
def __init__(self,df):
    self.df = df
  
def atr(self): #believe this is correct but not entirely sure...
    ind = range(0,len(df))
    indexlist = list(ind)
    df.index = indexlist
    z = np.zeros(len(df))
    df['True Range'] = z

    for index, row in df.iterrows():
        if index != 0:
            tr1 = row["High"] - row["Low"]
            tr2 = abs(row["High"] - df.iloc[index-1]["Close"])
            tr3 = abs(row["Low"] - df.iloc[index-1]["Close"])
            true_range = max(tr1, tr2, tr3)
            df['True Range'].iloc[index] = true_range
            #df.set_value(index,"True Range", true_range)
    df["Avg TR"] = df["True Range"].rolling(min_periods=14, window=14, center=False).mean()
    return df["Avg TR"]

        
def prepare_input_data(df): #WILL FIX THIS SO THAT WE UTILISE THE CONSTRUCTED FUNCTIONS OUTSIDE PREPARE_INPUT_DATA
    df['Gmt time'] = pd.to_datetime(df['Gmt time'].values,utc=True,dayfirst=True)
   # df = df[df['Volume'] > 0]   
    
    df['Gmt time'] = pd.to_datetime(df['Gmt time'].values, utc=True, dayfirst=True)
    ind = range(0,len(df))
    indexlist = list(ind)
    df=df.dropna()    
    high_close = np.abs(df['High']-df['Close'].shift())
    low_close = np.abs(df['Low']-df['Close'].shift())
    high_low = df['High']-df['Low']
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(10).sum()/10
    df["ATR"]=atr   
    df["Max_high"] = df["High"].rolling(10, min_periods=10).max() 
    df["Min_min"] = df["Low"].rolling(10, min_periods=10).min()  
    df["Ch_exit_long"] = df["Max_high"] - df["ATR"] * 3
    df["Ch_exit_short"] = df["Min_min"] + df["ATR"] * 3
    df['difference'] = df['Close'].diff(1)
    df['pct_change'] = df["Close"].pct_change() 
    df['rolling_returns'] = df['pct_change'].rolling(96).mean() 
    
    #24hr market return: open between 07-21 according to data...?
    df['24hr market return'] = df['Close'].diff(periods=56)    
    df['pos'] = df['rolling_returns'].diff()
    #df['pos'] = df.difference.copy()
    df['neg'] = df['rolling_returns'].diff()
    #df['neg'] = df.difference.copy()   
    df['pos'][df['pos'] < 0] = 0 
    df['neg'][df['neg'] > 0] = 0 
    df['avg_gain'] = df['pos'].rolling(window=14).mean()
    df['avg_loss'] = abs(df['neg'].rolling(window=14).mean())
    df['RS'] = df['avg_gain'] / df['avg_loss']
    df['rsi'] = 100.0 - (100.0 / (1.0 + df['RS']))
    ind = TradingIndicator(df)
    df = ind.macd(5, 35, 5, col="24hr market return")
    #df = ind.macd(5, 35, 5, col="Close")
    signal = df["signal"]
    macd= df["macd"]    
    #double check this 
    df["diff"]=(signal-macd)
    df["sign"] = np.sign(df["diff"])
    df["buy"] = (df["sign"] > df["sign"].shift(1))*1
    df["sell"] = (df["sign"] < df["sign"].shift(1))*1
   # df=df.dropna()
    return df
#%%
#df = pd.read_csv('/Users/ianwallgren/Documents/GitHub/momentum-events/'+'1_week.csv')
df = pd.read_csv('/Users/ianwallgren/Documents/GitHub/momentum-events/'+'2012_2014.csv')
df = df.iloc[-4000:] #last (?) three weeks of 2012 included
df_ready = prepare_input_data(df)

#%%
class Signal:
    def __init__(self,df_ready):
        self.df_ready = df_ready
    
    def long_signal(self):
        signal_to_buy = []
        signal_to_sell = []
        counter = []
        in_long = False
        a = np.zeros(len(df_ready))
        df_ready['buy it'] = a
        df_ready['close buy'] = a
        df_ready['in long'] = a
        df_ready['first crossover long'] = a
        df_ready['macd_signal_diff'] = df_ready['macd'] - df_ready['signal']
        s_above_m = False
        s_below_m = False
        buy_it = []
        close_long = []
        
        
        signal_to_sell = []
        signal_to_close_sell = []
        counter_sell = []
        in_short = False
        a_short = np.zeros(len(df_ready))
        df_ready['short it'] = a_short
        df_ready['close short'] = a_short
        df_ready['in short'] = a_short
        df_ready['first crossover short'] = a_short
        df_ready['macd_signal_diff'] = df_ready['macd'] - df_ready['signal']
        df_ready['not in a position'] = a_short
        df_ready['updated returns_short'] = a_short
        s_above_m = False
        s_below_m = False
        short_it = []
        close_short = []
        
        for i in range(len(df_ready)):
            if df_ready['macd_signal_diff'].iloc[i] < 0 and s_above_m == False:
                df_ready['first crossover long'].iloc[i] = 1
                s_above_m = True
            elif df_ready['macd_signal_diff'].iloc[i] < 0 and s_above_m == True:
                 df_ready['first crossover long'].iloc[i] = 0
               #  if 
                 
            elif df_ready['macd_signal_diff'].iloc[i] >= 0 and s_above_m == True:
                 s_above_m = False    
        
        for i in range(len(df_ready)):
            if df_ready['macd_signal_diff'].iloc[i] > 0 and s_below_m == False:
                df_ready['first crossover short'].iloc[i] = 1
                s_below_m = True
            elif df_ready['macd_signal_diff'].iloc[i] > 0 and s_below_m == True:
                 df_ready['first crossover short'].iloc[i] = 0
            elif df_ready['macd_signal_diff'].iloc[i] <= 0 and s_below_m == True:
                 s_below_m = False     
        
       

        for i in range(len(df_ready)):
            if in_long == False and in_short == False and df_ready['macd_signal_diff'].iloc[i] <0 and df_ready['first crossover long'].iloc[i] == 1 and df_ready['Volume'].iloc[i] != 0:
                df_ready['buy it'].iloc[i+1] = 1
                buy_it.append(df_ready.index[i+1])
                in_long = True
                s_above_m = True
                continue
            
            elif in_short == False and in_long == False and df_ready['in long'].iloc[i] == 0 and \
                df_ready['macd_signal_diff'].iloc[i] >0 and df_ready['first crossover short'].iloc[i] == 1 \
                    and df_ready['Volume'].iloc[i] != 0 and df_ready['in long'].iloc[i] ==0:
                df_ready['short it'].iloc[i+1] = 1
                short_it.append(df_ready.index[i+1])
                in_short = True
                s_below_m = True
                continue
            
            elif in_short == True:
                df_ready['in short'].iloc[i] = df_ready['Close'].iloc[i]
                if (df_ready['Ch_exit_short'].iloc[i] < df_ready['Close'].iloc[i]) and df_ready['Volume'].iloc[i] != 0:
                    df_ready['close short'].iloc[i+1] = 1
                    close_short.append(df_ready.index[i])
                    in_short = False
                    continue
                elif i == len(df_ready)-1 and in_short == True:
                    df_ready['close short'].iloc[i] = 1
                    close_short.append(df_ready.index[i])
                    in_short = False
                    break
        
            elif in_long == True:
                 df_ready['in long'].iloc[i] = df_ready['Close'].iloc[i]
            
                 if (df_ready['Ch_exit_long'].iloc[i] > df_ready['Close'].iloc[i]) and df_ready['Volume'].iloc[i] != 0:
                     df_ready['close buy'].iloc[i+1] = 1
                     close_long.append(df_ready.index[i])
                     in_long = False
                     continue
                 elif i == len(df_ready)-1 and in_long == True:
                    df_ready['close buy'].iloc[i] = 1
                    close_long.append(df_ready.index[i+1])
                    in_long = False
                    break
                 
        a_not = np.zeros(len(df_ready))
        df_ready['not in a position'] = a_not
        for i in range(len(df_ready)):
           # print(df_ready['in long'].iloc[i], df_ready['in short'].iloc[i])
            if df_ready['in long'].iloc[i] == 0 and df_ready['in short'].iloc[i] == 0:
                df_ready['not in a position'].iloc[i] = df_ready['Close'].iloc[i]
            else:
                df_ready['not in a position'].iloc[i] = 0
             
        return df_ready,buy_it,close_long
    
#    def short_signal(self):
 #       signal_to_sell = []
  #      signal_to_close_sell = []
   #     counter_sell = []
    #    in_short = False
     #   a_short = np.zeros(len(df_ready))
      #  df_ready['short it'] = a_short
       # df_ready['close short'] = a_short
       # df_ready['in short'] = a_short
       # df_ready['first crossover short'] = a_short
       # df_ready['macd_signal_diff'] = df_ready['macd'] - df_ready['signal']
        #df_ready['not in a position'] = a_short
       # df_ready['updated returns_short'] = a_short
      #  s_above_m = False
        #s_below_m = False
        #short_it = []
        #close_short = []
        
        #for i in range(len(df_ready)):
        #    if df_ready['macd_signal_diff'].iloc[i] > 0 and s_below_m == False:
         #       df_ready['first crossover short'].iloc[i] = 1
          #      s_below_m = True
           # elif df_ready['macd_signal_diff'].iloc[i] > 0 and s_below_m == True:
            #     df_ready['first crossover short'].iloc[i] = 0
           # elif df_ready['macd_signal_diff'].iloc[i] <= 0 and s_below_m == True:
            #     s_below_m = False        

     #   for i in range(len(df_ready)):
      #      if in_short == False and df_ready['in long'].iloc[i] == 0 and \
       #         df_ready['macd_signal_diff'].iloc[i] >0 and df_ready['first crossover short'].iloc[i] == 1 \
        #            and df_ready['Volume'].iloc[i] != 0 and df_ready['in long'].iloc[i] ==0:
         #       df_ready['short it'].iloc[i] = 1
          #      short_it.append(df_ready.index[i])
           #     in_short = True
            #    s_below_m = True
             #   continue      
        
       ##     elif in_short == True:
         #       df_ready['in short'].iloc[i] = df_ready['Close'].iloc[i]
          #      if (df_ready['Ch_exit_short'].iloc[i] < df_ready['Close'].iloc[i]) and df_ready['Volume'].iloc[i] != 0:
           #         df_ready['close short'].iloc[i] = 1
            #        close_short.append(df_ready.index[i])
             #       in_short = False
              #      continue
               # elif i == len(df_ready)-1 and in_short == True:
               #     df_ready['close short'].iloc[i] = 1
                #    close_short.append(df_ready.index[i])
                 #   in_short = False
                  #  break
                
        #a_not = np.zeros(len(df_ready))
       # df_ready['not in a position'] = a_not
        #for i in range(len(df_ready)):
        #    print(df_ready['in long'].iloc[i], df_ready['in short'].iloc[i])
          #  if df_ready['in long'].iloc[i] == 0 and df_ready['in short'].iloc[i] == 0:
         #       df_ready['not in a position'].iloc[i] = df_ready['Close'].iloc[i]
        #    else:
         #       df_ready['not in a position'].iloc[i] = 0
        #return df_ready, short_it, close_short
    
    def in_event(self): #cant seem to find a way of doing this maneuver without a nested foor loop
        a_not = np.zeros(len(df_ready))
        df_ready['Classification'] = a_not
        for i in range (0,len(df_ready)):
            for s in range(0,len(dates)):
                if (df_ready['Gmt time'].iloc[i].year == dates['start_date'].iloc[s].year \
                    and df_ready['Gmt time'].iloc[i].month  == dates['start_date'].iloc[s].month \
                    and df_ready['Gmt time'].iloc[i].day  == dates['start_date'].iloc[s].day) or \
                    (df_ready['Gmt time'].iloc[i].year == dates['event_date'].iloc[s].year and \
                     df_ready['Gmt time'].iloc[i].month  == dates['event_date'].iloc[s].month and \
                         df_ready['Gmt time'].iloc[i].day  == dates['event_date'].iloc[s].day) or \
                        (df_ready['Gmt time'].iloc[i].year == dates['end_date'].iloc[s].year and \
                         df_ready['Gmt time'].iloc[i].month  == dates['end_date'].iloc[s].month \
                         and df_ready['Gmt time'].iloc[i].day  == dates['end_date'].iloc[s].day):
                    df_ready['Classification'].iloc[i] = 'in event'
                    break
                else:
                    df_ready['Classification'].iloc[i] = 'outside event'   
            
        return df_ready


            
datan = Signal.long_signal(df_ready)[0]
#datan_short = Signal.short_signal(df_ready)[0]           
datan = Signal.long_signal(df_ready)[0]
#datan_short = Signal.short_signal(df_ready)[0]
datan_event = Signal.in_event(df_ready)


#%%
#BACKTESTING
df_backtest = pd.DataFrame()
df_backtest_short = pd.DataFrame()
class Backtest:
    def __init__(self,datan,df_backtest,df_backtest_short):
        self.datan = datan
        self.df_backtest = df_backtest
        self.df_backtest_short = df_backtest_short
        
    def returns_long(self):
        returns = []
        buys = []
        closes_buys = []
        counter1 = []
        pos_long = []
        neg_long = []
        pos_long_inevent = []
        neg_long_inevent = []
        pos_long_outevent = []
        neg_long_outevent = []
        classification = []
                        
        for i in range(len(datan)):
            if datan['buy it'].iloc[i] != 0:
                buys.append(df_ready['Close'].iloc[i])
              #  print(datan['Close'].iloc[i], 'buyprice', i)
            elif datan['close buy'].iloc[i] != 0:
                closes_buys.append(datan['Close'].iloc[i])
           #     print(datan['Close'].iloc[i], 'sellprice', i)
         
        for k in range(len(buys)):
            returns.append((closes_buys[k]-buys[k])/buys[k])
            
        for i in range(len(datan)):
            if datan['Classification'].iloc[i] == 'in event' and datan['buy it'].iloc[i] != 0: 
                classification.append('in event')
            if datan['Classification'].iloc[i] == 'outside event' and datan['buy it'].iloc[i] != 0: 
                classification.append('outside event')
            
        df_backtest['returns'] = returns
        df_backtest['class'] = classification
        df_backtest['cumsum returns'] = np.cumsum(returns)
        
        returns_long_inevent = sum(df_backtest['returns'].where(df_backtest['class']=='in event',0))
        returns_long_outevent = sum(df_backtest['returns'].where(df_backtest['class']=='outside event',0))
        
        for j in range(len(returns)):
            if returns[j] > 0:
                pos_long.append(returns[j])
            elif returns[j] < 0:
                neg_long.append(returns[j])
                
        win_rate_long = len(pos_long) / (len(pos_long)+len(neg_long))
        returns_inevent = (df_backtest['returns'].where(df_backtest['class']=='in event',0))
        returns_outevent = (df_backtest['returns'].where(df_backtest['class']=='outside event',0))
        all_returns_long_inevent = df_backtest['returns'].where(df_backtest['class']=='in event',0)
        all_returns_long_outevent = df_backtest['returns'].where(df_backtest['class']=='outside event',0)
        
        pos_returns_outevent = returns_outevent[returns_outevent>0]
        neg_returns_outevent = returns_outevent[returns_outevent<0] 
        
        pos_returns_inevent = returns_inevent[returns_inevent>0]
        neg_returns_inevent = returns_inevent[returns_inevent<0]   
        
        
        
        win_rate_long_outevent = len(pos_returns_outevent)/(len(pos_returns_outevent)+len(neg_returns_outevent))
        win_rate_long_inevent = len(pos_returns_inevent)/(len(pos_returns_inevent)+len(neg_returns_inevent))
        
        length1 = len(df_backtest[df_backtest['class'] == 'in event'])
        length2 = len(df_backtest[df_backtest['class'] == 'outside event'])
        avg_returns_long_inevent = returns_long_inevent/length1
        avg_returns_long_outevent = returns_long_outevent/length2
        return returns, df_backtest, returns_long_inevent, returns_long_outevent, avg_returns_long_inevent, \
            avg_returns_long_outevent, win_rate_long, win_rate_long_inevent, win_rate_long_outevent,\
               all_returns_long_inevent, all_returns_long_outevent 
    
    def returns_short(self):
        returns_short = []
        shorts = []
        closes_shorts = []
        counter_short = []
        pos_short = []
        neg_short = []
        classification = []
           
        for i in range(len(datan)):
             if datan['short it'].iloc[i] != 0:
                 shorts.append(df_ready['Close'].iloc[i])
             elif datan['close short'].iloc[i] != 0:
                 closes_shorts.append(datan['Close'].iloc[i])

        for k in range(len(shorts)):
            returns_short.append((closes_shorts[k]-shorts[k])/shorts[k])
                
        for i in range(len(datan)):
            if datan['Classification'].iloc[i] == 'in event' and datan['short it'].iloc[i] != 0: 
                classification.append('in event')
            if datan['Classification'].iloc[i] == 'outside event' and datan['short it'].iloc[i] != 0: 
                classification.append('outside event')
        
        df_backtest_short['returns short'] = returns_short
        df_backtest_short['class'] = classification
        df_backtest_short['cumsum returns short'] = np.cumsum(returns_short)
        
        for j in range(len(returns_short)):
            if returns_short[j] > 0:
                pos_short.append(returns_short[j])
            elif returns_short[j] < 0:
                neg_short.append(returns_short[j])
                
        win_rate_short = len(pos_short) / (len(pos_short)+len(neg_short)) 
        
        
        returns_inevent = (df_backtest_short['returns short'].where(df_backtest_short['class']=='in event',0))
        returns_outevent = (df_backtest_short['returns short'].where(df_backtest_short['class']=='outside event',0))
        pos_returns_outevent = returns_outevent[returns_outevent>0]
        neg_returns_outevent = returns_outevent[returns_outevent<0] 
        
        pos_returns_inevent = returns_inevent[returns_inevent>0]
        neg_returns_inevent = returns_inevent[returns_inevent<0]   
        
        
        
        
        win_rate_short_outevent = len(pos_returns_outevent)/(len(pos_returns_outevent)+len(neg_returns_outevent))
        win_rate_short_inevent = len(pos_returns_inevent)/(len(pos_returns_inevent)+len(neg_returns_inevent))
        
        returns_short_inevent = sum(df_backtest_short['returns short'].where(df_backtest_short['class']=='in event',0))
        returns_short_outevent = sum(df_backtest_short['returns short'].where(df_backtest_short['class']=='outside event',0))
        
        all_returns_short_inevent = df_backtest_short['returns short'].where(df_backtest_short['class']=='in event',0)
        all_returns_short_outevent = df_backtest_short['returns short'].where(df_backtest_short['class']=='outside event',0)
    
        length1 = len(df_backtest_short[df_backtest_short['class'] == 'in event'])
        length2 = len(df_backtest_short[df_backtest_short['class'] == 'outside event'])
        avg_returns_short_inevent = returns_short_inevent/length1
       # print(length1)
     #   print(df_backtest_short['returns short'].where(df_backtest_short['class']=='in event',0))
        avg_returns_short_outevent = returns_short_outevent/length2
        
        return returns_short, df_backtest_short, win_rate_short, returns_short_inevent, returns_short_outevent,\
            avg_returns_short_inevent, avg_returns_short_outevent, win_rate_short_outevent, win_rate_short_inevent\
                , all_returns_short_inevent, all_returns_short_outevent
        
    def holding_period_long(self):
        hold_buy = []
        hold_close_buy = []
        hold_long = []
        in_hold_buy = False
        for i in range(len(datan)):
            if datan['buy it'].iloc[i] != 0 and in_hold_buy == False:
                hold_buy.append(i)
                in_hold_buy = True
                for k in range(i,len(datan)):
                    if datan['close buy'].iloc[k] !=0:
                        hold_close_buy.append(k)
                        in_hold_buy = False
        
        for c in range(len(hold_buy)):
            hold_long.append(hold_close_buy[c]-hold_buy[c])
            average_hold_long = sum(hold_long)/len(hold_long)
            
        df_backtest['holding period'] = hold_long
        
        length1 = len(df_backtest[df_backtest['class'] == 'in event'])
        hold_per_long_inevent = (sum(df_backtest['holding period'].where(df_backtest['class']=='in event',0))) / length1
     
        hold_per_long_outevent = sum(df_backtest['holding period'].where(df_backtest['class']=='outside event',0)) / (len(df_backtest)-length1)
        return hold_long, average_hold_long, df_backtest, hold_per_long_inevent, hold_per_long_outevent
    
    def holding_period_short(self):
        hold_short = []
        hold_close_short = []
        hold_short_total = []
        in_hold_short = False
        for i in range(len(datan)):
            if datan['short it'].iloc[i] != 0 and in_hold_short == False:
                hold_short.append(i)
                in_hold_short = True
                for k in range(i,len(datan)):
                    if datan['close short'].iloc[k] !=0:
                        hold_close_short.append(k)
                        in_hold_short = False
        
        for c in range(len(hold_short)):
            hold_short_total.append(hold_close_short[c]-hold_short[c])
        average_hold_short = sum(hold_short_total)/len(hold_short_total)
            
        df_backtest_short['holding period'] = hold_short_total
        
        length1 = len(df_backtest_short[df_backtest_short['class'] == 'in event'])
        length2 = len(df_backtest_short[df_backtest_short['class'] == 'outside event'])
        
        hold_per_short_inevent = (sum(df_backtest_short['holding period'].where(df_backtest_short['class']=='in event',0))) / length1
        hold_per_short_outevent = sum(df_backtest_short['holding period'].where(df_backtest_short['class']=='outside event',0)) / length2
        return hold_short_total, average_hold_short, df_backtest_short, hold_per_short_inevent, hold_per_short_outevent
       

    def plot_hold_per_hist_long(self):
        hold_long = Backtest.holding_period_long(datan)[2]
        ax = sns.histplot(data = hold_long, x = 'holding period', stat = 'probability', hue = 'class', multiple='stack', palette='dark')
        plt.title('Histogram of long position holding periods measured in 15 min bars')
        plt.xlabel('Size of holding period')
        
        
        
    def plot_hold_per_hist_short(self):
        hold_short = Backtest.holding_period_short(datan)[2]
        sns.histplot(data = hold_short, x = 'holding period', hue = 'class',  stat = 'probability', multiple='stack', palette='dark',label='Holding period in 15 min bars short position')
        plt.title('Histogram of short position holding periods measured in 15 min bars')
        plt.xlabel('Size of holding period')
        
    def plot_returns_hist_long(self):
        returns_long = Backtest.returns_long(datan)[1]
       # sns.histplot(data = returns_long, x ='returns', hue ='class', stat = 'probability', multiple='stack', palette='dark',label='Returns in percent long')
        sns.histplot(data = returns_long, x = 'returns', stat = 'probability', hue = 'class', common_norm = False, element='step', fill = False)
        # plt.legend(loc='best')
        plt.title('Histogram of long position returns in percent, categories independently normalized')
        plt.xlabel('Size of return')
        
    def plot_returns_hist_short(self):
        returns_short = Backtest.returns_short(datan)[1]
       # sns.histplot(data = returns_short, x='returns short',hue='class', stat = 'probability', multiple='stack', palette='dark',label='Returns in percent short')
        sns.histplot(data = returns_short, x = 'returns short', stat = 'probability', hue = 'class', common_norm = False, element='step', fill = False)
        #plt.legend(loc='best')
        plt.title('Histogram of short position returns in percent, categories independently normalized')
        plt.xlabel('Size of return')
        
    def plot_cumulative_returns_long(self):
        returns_long = Backtest.returns_long(datan)[1]
        longs_event = returns_long['cumsum returns'].where(returns_long['class'] == 'in event', np.nan)
        longs_outevent = returns_long['cumsum returns'].where(returns_long['class'] == 'outside event', np.nan)
        longs_overall = returns_long['cumsum returns']
        
        
        
        plt.plot(longs_event,linewidth=1, color='green', label = 'in event')
        plt.plot(longs_outevent,linewidth=1, color = 'red', label = 'outside event')
        plt.plot(longs_overall,linewidth=1, alpha=0.3, color = 'grey')
        plt.legend(loc='best')
        plt.title('Cumulative returns long positions (grey transitioning to red should be interpreted as red, vice versa for green)')
        plt.ylabel('Cumulative return')
        plt.xlabel('Number of long trades')
    
    def plot_cumulative_returns_short(self):
        returns_short = Backtest.returns_short(datan)[1]
        shorts_event = returns_short['cumsum returns short'].where(returns_short['class'] == 'in event', np.nan)
        shorts_outevent = returns_short['cumsum returns short'].where(returns_short['class'] == 'outside event', np.nan)
        shorts_overall =returns_short['cumsum returns short']
       # cum_close = np.cumsum(datan['Close'].iloc[:])
        plt.plot(shorts_event,lw=1, c='red', label='in event')
        plt.plot(shorts_outevent,linewidth=1, color = 'green', label = 'outside event')
        plt.plot(shorts_overall,linewidth=1, alpha=0.3, color = 'grey')
        plt.legend(loc='best')
        plt.title('Cumulative returns short positions (grey transitioning to red should be interpreted as red, vice versa for green)')
        plt.ylabel('Cumulative return')
        plt.xlabel('Number of long trades')
        
        
       # sns.set()
        #longs = datan['Close'].where(datan['in long'] > 0, np.nan)
        #shorts = datan['Close'].where(datan['in short'] > 0, np.nan)
        #not_in_position = datan['Close'].where(datan['not in a position'] > 0, np.nan) 
        #plt.plot(cum_close)
     
        
    def plot_positions(self): #does not work atm
        buy_it = Signal.long_signal(df_ready)[1]
        short_it = Signal.short_signal(df_ready)[1]
        close_buy = Signal.long_signal(df_ready)[2]
        close_short = Signal.short_signal(df_ready)[2]
        sns.set()
        plt.figure(figsize=(12,4)) 
        plt.scatter(df_ready.loc[buy_it].index,df_ready.loc[buy_it]['Close'], marker = '^',label = 'Long', color = 'green')
        plt.scatter(datan_short[short_it].index,datan_short.loc[short_it]['Close'], marker = '^',label = 'Short', color = 'red')
        plt.scatter(df_ready.loc[close_buy].index,df_ready.loc[close_buy]['Close'], marker = 'x',label = 'Close long', color = 'green')
        plt.scatter(datan_short.loc[close_short].index,datan_short.loc[close_short]['Close'], marker = 'o', label = 'Close short', color = 'red')
        plt.plot(df_ready['Close'], label = 'Close price', alpha = 0.7)
        plt.xticks(rotation=25)
        plt.title('SP500 Close price, long and short signals,' +''+ 'return in percent:'+''+str(round(sum(Backtest.returns_long(datan)+Backtest.returns_short),3)))
        plt.xlabel('Date', fontsize=15)
        plt.ylabel('Close price', fontsize=15)
        plt.legend(loc=0,fontsize='medium')
        
    def plot_equity_curve(self):
        #sns.set()
        #longs = datan['Close'].where(datan['in long'] > 0, np.nan)
        #shorts = datan['Close'].where(datan['in short'] > 0, np.nan)
       # not_in_position = datan['Close'].where(datan['not in a position'] > 0, np.nan) 
       # plt.plot(longs,linewidth=3, color='green', label = 'in long position')
       # plt.plot(shorts,linewidth=3, color = 'red', label = 'in short position')
       # plt.plot(not_in_position,linewidth=3, color = 'orange', label = 'in neutral position')
       # plt.legend(loc='best')
       # plt.title('When we are long,short and neutral')
       # plt.xlabel('Data point')
       # plt.ylabel('Close price')
        
        #sns.set()
        longs = datan['Close'].where(datan['in long'] > 0, np.nan)
        shorts = datan['Close'].where(datan['in short'] > 0, np.nan)
        not_in_position = datan['Close'].where(datan['not in a position'] > 0, np.nan) 
        
        fig, ax = plt.subplots(figsize=(10,6))
        
        ax.plot(datan['Gmt time'], longs, linewidth=1, color='green', label = 'in long position')
        ax.plot(datan['Gmt time'],shorts,linewidth=1, color = 'red', label = 'in short position')
        ax.plot(datan['Gmt time'],not_in_position,linewidth=1, color = 'orange', label = 'in neutral position')
        
        for i in range(23,24):
            ax.axvspan(date2num(dates['start_date'].iloc[i]), date2num(dates['end_date'].iloc[i]), 
           label="Event day",color="green", alpha=0.3)
        
        #ax.axvspan(date2num(datetime(2014,12,27)), date2num(datetime(2014,12,29)), 
         #  color="green", alpha=0.3)
        
        ax.legend(loc='best')
        ax.set_title('When we are long, short and neutral')
        ax.set_xlabel('Data point')
        ax.set_ylabel('Close price')       

#%%
#LONG STATS
returns_long = Backtest.returns_long(datan)[1]
returns_long_inevent = Backtest.returns_long(datan)[2]
returns_long_outevent = Backtest.returns_long(datan)[3]
avg_returns_long_inevent = Backtest.returns_long(datan)[4]
avg_returns_long_outevent = Backtest.returns_long(datan)[5]
win_rate_long = Backtest.returns_long(datan)[6]
win_rate_long_inevent = Backtest.returns_long(datan)[7]
win_rate_long_outevent = Backtest.returns_long(datan)[8]
total_long_trades = len(returns_long)

avg_hold_per_long = Backtest.holding_period_long(datan)[1]
avg_hold_per_long_inevent = Backtest.holding_period_long(datan)[3]
avg_hold_per_long_outevent = Backtest.holding_period_long(datan)[4]


#LONG PLOTS
#hist_plot_holding_period_long = Backtest.plot_hold_per_hist_long(datan)
#plt.savefig('sensitive_setting_hist_holdper_long_lastsixweeks_2012.png',dpi=1200)

#hist_plot_returns_long = Backtest.plot_returns_hist_long(datan)   
#plt.savefig('sensitive_setting_hist_returns_long_lastsixweeks_2012.png',dpi=1200)

cum_ret_long = Backtest.plot_cumulative_returns_long(datan)
#plt.savefig('sensitive_settings_cum_ret_long_lastsixweeks_2012.png',dpi=1200)


#%%
#SHORT STATS
returns_short = Backtest.returns_short(datan)[0]
win_rate_short = Backtest.returns_short(datan)[2]
returns_short_inevent = Backtest.returns_short(datan)[3]
returns_short_outevent = Backtest.returns_short(datan)[4]
avg_returns_short_inevent = Backtest.returns_short(datan)[5]
avg_returns_short_outevent = Backtest.returns_short(datan)[6]
win_rate_short_inevent = Backtest.returns_short(datan)[7]
win_rate_short_outevent = Backtest.returns_short(datan)[8]
total_short_trades = len(returns_short)


avg_holding_per_short =  Backtest.holding_period_short(datan)[1]
hold_per_short =  Backtest.holding_period_short(datan)[2]
avg_hold_per_short_inevent =  Backtest.holding_period_short(datan)[3]
avg_hold_per_short_outevent =  Backtest.holding_period_short(datan)[4]

#SHORT PLOTS
#hist_plot_holding_period_short = Backtest.plot_hold_per_hist_short(datan)
#plt.savefig('sensitive_setting_hist_holdper_short_lastsixweeks_2012.png',dpi=1200)

#hist_plot_returns_short = Backtest.plot_returns_hist_short(datan)   
#plt.savefig('sensitive_setting_hist_returns_short_lastsixweeks_2012.png',dpi=1200)

cum_ret_short = Backtest.plot_cumulative_returns_short(datan)
#plt.savefig('sensitive_settings_cum_ret_short_lastsixweeks_2012.png',dpi=1200)


#%%
#TOTAL
#STATS

long_trades_inevent = Backtest.returns_long(datan)[9]
long_trades_inevent = long_trades_inevent[long_trades_inevent!=0]
long_trades_outevent = Backtest.returns_long(datan)[10]
long_trades_outevent = long_trades_outevent[long_trades_outevent!=0]
short_trades_inevent = Backtest.returns_short(datan)[9]
short_trades_inevent = short_trades_inevent[short_trades_inevent!=0]
short_trades_outevent = Backtest.returns_short(datan)[10]
short_trades_outevent = short_trades_outevent[short_trades_outevent!=0]
total_returns_intevent = returns_short_inevent+returns_long_inevent
total_returns_outevent = returns_short_outevent+returns_long_outevent
total_returns = total_returns_intevent+total_returns_outevent


long_short_ratio_inevent = len(long_trades_inevent)/(len(long_trades_inevent)+len(short_trades_inevent))
long_short_ratio_outevent = len(long_trades_outevent)/(len(long_trades_outevent)+len(short_trades_outevent))

#plot_positions = Backtest.plot_positions(datan)   #does not work atm, but equity_curve below is preferred anyway
#plt.savefig('sensitive_setting_positions_lastthreeweeks_2012.png',dpi=1200)      
plot_equity_curve = Backtest.plot_equity_curve(datan)
#plt.savefig('sensitive_setting_ALLpositions_lastsixweeks_2012.png',dpi=1200)

#%%
returns_long = Backtest.returns_long(datan)[1]
longs_event = returns_long['cumsum returns'].where(returns_long['class'] == 'in event', np.nan)
longs_outevent = returns_long['cumsum returns'].where(returns_long['class'] == 'outside event', np.nan)

#%%

datan_event = datan.iloc[1210:1240]
fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10,5))
fig.suptitle('Illustrative example of how a long trade is entered and exited')
axes[1].set_title('How a long trade is exited')
axes[0].set_title('How a long trade is entered')
sns.lineplot(ax=axes[1], data=datan_event,x='Gmt time', y='Close', label='close price')
sns.lineplot(ax=axes[1], data=datan_event,x='Gmt time', y='Ch_exit_long', label='chandelier exit')
sns.lineplot(ax=axes[0], data=datan_event,x='Gmt time', y='macd',label='macd line')
sns.lineplot(ax=axes[0], data=datan_event,x='Gmt time', y='signal',label='signal line')
#axes[1].set_xticks(range(len(datan_event))) # <--- set the ticks first
axes[1].set_xticklabels(['2014-12-02','2014-12-03','2014-12-04','2014-12-05','2014-12-06'\
                         ,'2014-12-07','2014-12-08'],rotation=45)
axes[0].set_xticklabels(['2014-12-02','2014-12-03','2014-12-04','2014-12-05','2014-12-06'\
                         ,'2014-12-07','2014-12-08'],rotation=45)
axes[0].set(xlabel='Date', ylabel='macd')
axes[1].set(xlabel='Date', ylabel='Close price')

#axes.savefig('Illustration_signals',dpi=1200)
#plt.savefig('Illustration_signals',dpi=1200,bbox_inches = 'tight')


#ax = sns.lineplot(data =datan_event, x ='Gmt time', y = 'Ch_exit_long',label='chandelier exit').set(title='Illustrative example of how a long trade is exited')
#ax = sns.lineplot(data =datan_event, x ='Gmt time', y = 'Close', label ='close price')
#ax.set(xlabel='Gmt time', ylabel='Close price')






#sns.lineplot(data =datan_event, x ='Gmt time', y = 'macd')
#sns.lineplot(data =datan_event, x ='Gmt time', y = 'signal')
