#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 09:06:59 2022

@author: ianwallgren
"""

import linchackathon as lh
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from matplotlib.dates import date2num

#LAST UPDATED: 24 MARCH
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
        data = self.df
        
        data["macd"]=self.ewma(span=span1, col=col)-self.ewma(span=span2, col=col)
        data["signal"]=data["macd"].ewm(span=span_signal, adjust=False).mean()
        return data
    
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
df = df.iloc[-2000:] #last (?) three weeks of 2012 included
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
            if in_long == False and df_ready['macd_signal_diff'].iloc[i] <0 and df_ready['first crossover long'].iloc[i] == 1 and df_ready['Volume'].iloc[i] != 0:
                df_ready['buy it'].iloc[i] = 1
                buy_it.append(df_ready.index[i])
                in_long = True
                s_above_m = True
                continue  
        
            elif in_long == True:
                 df_ready['in long'].iloc[i] = df_ready['Close'].iloc[i]
            
                 if (df_ready['Ch_exit_long'].iloc[i] > df_ready['Close'].iloc[i]) and df_ready['Volume'].iloc[i] != 0:
                     df_ready['close buy'].iloc[i] = 1
                     close_long.append(df_ready.index[i])
                
                     in_long = False
                     continue
             
        return df_ready,buy_it,close_long
    
    def short_signal(self):
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
        s_above_m = False
        s_below_m = False
        short_it = []
        close_short = []
        
        for i in range(len(df_ready)):
            if df_ready['macd_signal_diff'].iloc[i] > 0 and s_below_m == False:
                df_ready['first crossover short'].iloc[i] = 1
                s_below_m = True
            elif df_ready['macd_signal_diff'].iloc[i] > 0 and s_below_m == True:
                 df_ready['first crossover short'].iloc[i] = 0
            elif df_ready['macd_signal_diff'].iloc[i] <= 0 and s_below_m == True:
                 s_below_m = False        

        for i in range(len(df_ready)):
            if in_short == False and df_ready['in long'].iloc[i] == 0 and df_ready['macd_signal_diff'].iloc[i] >0 and df_ready['first crossover short'].iloc[i] == 1 and df_ready['Volume'].iloc[i] != 0:
                df_ready['short it'].iloc[i] = 1
                short_it.append(df_ready.index[i])
                in_short = True
                s_below_m = True
                continue      
        
            elif in_short == True:
                df_ready['in short'].iloc[i] = df_ready['Close'].iloc[i]
                if (df_ready['Ch_exit_short'].iloc[i] < df_ready['Close'].iloc[i]) and df_ready['Volume'].iloc[i] != 0:
                    df_ready['close short'].iloc[i] = 1
                    close_short.append(df_ready.index[i])
                    in_short = False
                    continue
        a_not = np.zeros(len(df_ready))
        df_ready['not in a position'] = a_not
        for i in range(len(df_ready)):
        #    print(df_ready['in long'].iloc[i], df_ready['in short'].iloc[i])
            if df_ready['in long'].iloc[i] == 0 and df_ready['in short'].iloc[i] == 0:
                df_ready['not in a position'].iloc[i] = df_ready['Close'].iloc[i]
            else:
                df_ready['not in a position'].iloc[i] = 0
        return df_ready, short_it, close_short

            
datan = Signal.long_signal(df_ready)[0]
datan_short = Signal.short_signal(df_ready)[0]


#%%
#BACKTESTING
class Backtest:
    def __init__(self,datan):
        self.datan = datan
        
    def returns_long(self):
        returns = []
        buys = []
        closes_buys = []
        counter1 = []
        pos_long = []
        neg_long = []
                        
        for i in range(len(datan)):
            if datan['buy it'].iloc[i] != 0:
                buys.append(df_ready['Close'].iloc[i])
                print(datan['Close'].iloc[i], 'buyprice', i)
            elif datan['close buy'].iloc[i] != 0:
                closes_buys.append(datan['Close'].iloc[i])
                print(datan['Close'].iloc[i], 'sellprice', i)
        
        for k in range(len(buys)-1):
            returns.append((closes_buys[k]-buys[k])/buys[k])
            
        for j in range(len(returns_long)-1):
            if returns_long[j] > 0:
                pos_long.append(returns_long[j])
            elif returns_long[j] < 0:
                neg_long.append(returns_long[j])
                
        win_rate_long = len(pos_long) / (len(pos_long)+len(neg_long))
        return returns, win_rate_long
    
    def returns_short(self):
        returns_short = []
        shorts = []
        closes_shorts = []
        counter_short = []
        pos_short = []
        neg_short = []
                        
        for i in range(len(datan)):
            if datan['short it'].iloc[i] != 0:
                shorts.append(df_ready['Close'].iloc[i])
                #print(datan['Close'].iloc[i], 'buyprice', i)
            elif datan['close short'].iloc[i] != 0:
                closes_shorts.append(datan['Close'].iloc[i])
                #print(datan['Close'].iloc[i], 'sellprice', i)
        
        for k in range(len(shorts)-1):
            returns_short.append((shorts[k]-closes_shorts[k])/shorts[k])
        for j in range(len(returns_short)-1):
            if returns_short[j] > 0:
                pos_short.append(returns_short[j])
            elif returns_short[j] < 0:
                neg_short.append(returns_short[j])
                
        win_rate_short = len(pos_short) / (len(pos_short)+len(neg_short))
            
        return returns_short, win_rate_short
        
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
        
        for c in range(len(hold_buy)-1):
            hold_long.append(hold_close_buy[c]-hold_buy[c])
            average_hold_long = sum(hold_long)/len(hold_long)
        return hold_long, average_hold_long
    
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
        
        for c in range(len(hold_short)-1):
            hold_short_total.append(hold_close_short[c]-hold_short[c])
           # print(hold_close_short[c],hold_short[c])
           # print(hold_close_short[c]-hold_short[c])
            average_hold_short = sum(hold_short_total)/len(hold_short_total)
        return hold_short_total, average_hold_short
       

    def plot_hold_per_hist_long(self):
        hold_long = Backtest.holding_period_long(datan)[0]
        sns.histplot(data = hold_long, bins = len(hold_long),  stat = 'probability', multiple='stack', palette='dark',label='Holding period in 15 min bars')
        plt.legend(loc='best')
        plt.title('Histogram of long position holding periods measured in 15 min bars')
        plt.xlabel('Size of holding period')
        
    def plot_hold_per_hist_short(self):
        hold_short = Backtest.holding_period_short(datan)[0]
        sns.histplot(data = hold_short, bins = len(hold_short),  stat = 'probability', multiple='stack', palette='dark',label='Holding period in 15 min bars short position')
        plt.legend(loc='best')
        plt.title('Histogram of short position holding periods measured in 15 min bars')
        plt.xlabel('Size of holding period')
        
    def plot_returns_hist_long(self):
        returns_long = Backtest.returns_long(datan)[0]
        sns.histplot(data = returns_long, bins = len(returns_long), stat = 'probability', multiple='stack', palette='dark',label='Returns in percent long')
        plt.legend(loc='best')
        plt.title('Histogram of long position returns in percent')
        plt.xlabel('Size of return')
        
    def plot_returns_hist_short(self):
        returns_short = Backtest.returns_short(datan)[0]
        sns.histplot(data = returns_short, bins = len(returns_short), stat = 'probability', multiple='stack', palette='dark',label='Returns in percent short')
        plt.legend(loc='best')
        plt.title('Histogram of short position returns in percent')
        plt.xlabel('Size of return')
        
    def plot_cumulative_returns_long(self):
        returns_long = Backtest.returns_long(datan)[0]
        cum_ret_long = np.cumsum(returns_long)
       # cum_close = np.cumsum(datan['Close'].iloc[:])
        plt.plot(cum_ret_long,lw=1, c='green', label='Long')
        #plt.plot(cum_close)
        plt.legend(loc='best')
        plt.title('Cumulative returns long positions')
        plt.ylabel('Cumulative return')
        plt.xlabel('Number of long trades')
    
    def plot_cumulative_returns_short(self):
        returns_short = Backtest.returns_short(datan)[0]
        cum_ret_short = np.cumsum(returns_short)
       # cum_close = np.cumsum(datan['Close'].iloc[:])
        plt.plot(cum_ret_short,lw=1, c='red', label='short')
        #plt.plot(cum_close)
        plt.legend(loc='best')
        plt.title('Cumulative returns short positions')
        plt.ylabel('Cumulative return')
        plt.xlabel('Number of short trades')
        
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
        
        sns.set()
        longs = datan['Close'].where(datan['in long'] > 0, np.nan)
        shorts = datan['Close'].where(datan['in short'] > 0, np.nan)
        not_in_position = datan['Close'].where(datan['not in a position'] > 0, np.nan) 
        
        fig, ax = plt.subplots(figsize=(10,6))
        
        ax.plot(datan['Gmt time'], longs, linewidth=1, color='green', label = 'in long position')
        ax.plot(datan['Gmt time'],shorts,linewidth=1, color = 'red', label = 'in short position')
        ax.plot(datan['Gmt time'],not_in_position,linewidth=1, color = 'orange', label = 'in neutral position')
        ax.axvspan(date2num(datetime(2014,12,29)), date2num(datetime(2014,12,30)), 
           label="Event day",color="green", alpha=0.3)
        
        ax.axvspan(date2num(datetime(2014,12,27)), date2num(datetime(2014,12,28)), 
           color="green", alpha=0.3)
        
        ax.legend(loc='best')
        ax.set_title('When we are long,short and neutral')
        ax.set_xlabel('Data point')
        ax.set_ylabel('Close price')       

#%%
#LONG STATS
returns_long = Backtest.returns_long(datan)[0]
win_rate_long = Backtest.returns_long(datan)[1]
avg_hold_per_long = Backtest.holding_period_long(datan)[1]


#LONG PLOTS
#hist_plot_holding_period_long = Backtest.plot_hold_per_hist_long(datan)
#plt.savefig('sensitive_setting_hist_holdper_long_lastthreeweeks_2012.png',dpi=1200)

#hist_plot_returns_long = Backtest.plot_returns_hist_long(datan)   
#plt.savefig('sensitive_setting_hist_returns_long_lastthreeweeks_2012.png',dpi=1200)

cum_ret_long = Backtest.plot_cumulative_returns_long(datan)
#plt.savefig('sensitive_settings_cum_ret_long_lastthreeweeks_2012.png',dpi=1200)


#%%
#SHORT STATS
returns_short = Backtest.returns_short(datan)[0]
win_rate_short = Backtest.returns_short(datan)[1]
avg_holding_per_short =  Backtest.holding_period_short(datan)[1]

#SHORT PLOTS
#hist_plot_holding_period_short = Backtest.plot_hold_per_hist_short(datan)
#plt.savefig('sensitive_setting_hist_holdper_short_lastthreeweeks_2012.png',dpi=1200)

#hist_plot_returns_short = Backtest.plot_returns_hist_short(datan)   
#plt.savefig('sensitive_setting_hist_returns_short_lastthreeweeks_2012.png',dpi=1200)

#cum_ret_short = Backtest.plot_cumulative_returns_short(datan)
#plt.savefig('sensitive_settings_cum_ret_short_lastthreeweeks_2012.png',dpi=1200)


#%%
#TOTAL
#plot_positions = Backtest.plot_positions(datan)   #does not work atm, but equity_curve below is preferred anyway
#plt.savefig('sensitive_setting_positions_lastthreeweeks_2012.png',dpi=1200)      
plot_equity_curve = Backtest.plot_equity_curve(datan)
#plt.savefig('sensitive_setting_ALLpositions_lastthreeweeks_2012.png',dpi=1200)



