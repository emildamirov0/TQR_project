#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 13:48:42 2022

@author: ianwallgren
"""

import linchackathon as lh
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#%%
class TradingIndicator:
    
    # Obs time_series_data must be a df!
    def __init__(self, time_series_data):
        self.df = time_series_data

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
    
#%%

    
#%%
def avg_true_range(self, df): 
  ind = range(0,len(df))
  indexlist = list(ind)
  df.index = indexlist

  for index, row in df.iterrows():
    if index != 0:
      tr1 = row["High"] - row["Low"]
      tr2 = abs(row["High"] - df.iloc[index-1]["Close"])
      tr3 = abs(row["Low"] - df.iloc[index-1]["Close"])

      true_range = max(tr1, tr2, tr3)
      df.set_value(index,"True Range", true_range)

  df["Avg TR"] = df["True Range"].rolling(min_periods=14, window=14, center=False).mean()
  return df


def chandelier_exit(self, df): # default period is 22

  df_tr = self.avg_true_range(df)

  rolling_high = df_tr["High"][-22:].max()
  rolling_low = df_tr["Low"][-22:].max()

  chandelier_long = rolling_high - df_tr.iloc[-1]["Avg TR"] * 3
  chandelier_short = rolling_low - df_tr.iloc[-1]["Avg TR"] * 3
#%%
def prepare_input_data(df):
  

    df['Gmt time'] = pd.to_datetime(df['Gmt time'].values,utc=True,dayfirst=True)
 
    df = df[df['Volume'] > 0]
  
    
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
    df["Ch_exit_long"] = df["Max_high"] - df["ATR"] * 3


    df['difference'] = df['Close'].diff(1)
    df['pct_change'] = df["Close"].pct_change() 
    df['pos'] = df.difference.copy()
    df['neg'] = df.difference.copy()
    
    df['pos'][df['pos'] < 0] = 0 
    df['neg'][df['neg'] > 0] = 0 
    
    df['avg_gain'] = df['pos'].rolling(window=14).mean()
    df['avg_loss'] = abs(df['neg'].rolling(window=14).mean())
    df['RS'] = df['avg_gain'] / df['avg_loss']
    df['rsi'] = 100.0 - (100.0 / (1.0 + df['RS']))
    ind = TradingIndicator(df)
    df = ind.macd(12, 26, 9, col="Close")
    signal = df["signal"]
    macd= df["macd"]
    
    #double check this
    
    df["diff"]=(signal-macd)
    df["sign"] = np.sign(df["diff"])
    df["buy"] = (df["sign"] > df["sign"].shift(1))*1
    df["sell"] = (df["sign"] < df["sign"].shift(1))*1

    df=df.dropna()

    return df

df = pd.read_csv('/Users/ianwallgren/Documents/GitHub/momentum-events/'+'1_week.csv')
df_ready = prepare_input_data(df)

#%%


    #MACD
signal_to_buy = []
signal_to_sell = []
counter = []
in_long = False
a = np.zeros(len(df_ready))
df_ready['buy it'] = a
df_ready['sell it'] = a
df_ready['macd_signal_diff'] = df_ready['macd'] - df_ready['signal']   
for i in range(len(df_ready)):
    if in_long == False and df_ready['macd_signal_diff'].iloc[i] <0:
        df_ready['buy it'].iloc[i] = 1
        in_long = True
        continue
       # print(i,'buy')
        
        
    elif in_long == True and (df_ready['Ch_exit_long'].iloc[i] > df_ready['Close'].iloc[i]):
        df_ready['sell it'].iloc[i] = 1
        in_long = False
        continue
        #print(i,'sell',df_ready['Ch_exit_long'].iloc[i],df_ready['Close'].iloc[i], 'sums;',df_ready['buy it'].iloc[:i].sum(),df_ready['sell it'].iloc[:i].sum())


returns = []
buys = []
sells = []
counter1 = []
for i in range(len(df_ready)):
    if df_ready['buy it'].iloc[i] != 0:
        buys.append(df_ready['Close'].iloc[i])
        print(df_ready['Close'].iloc[i], 'buyprice', i)
    elif df_ready['sell it'].iloc[i] != 0:
        sells.append(df_ready['Close'].iloc[i])
        print(df_ready['Close'].iloc[i], 'sellprice', i)
        
for k in range(len(buys)-1):
    returns.append(sells[k]-buys[k])
        

#%%

Long_signal = []
Neutralize_long = []
def backtest_macd(df):
    long=False
    enter_price = 0
    close_price = 0
    enter_date = 0
    exit_date = 0
    returns = []
    
    
    
    for i in range(len(df)):
        if (df["buy"].iloc[i]==1  & long==False & (df["rsi"].iloc[i]<70)) & (df["Ch_exit_long"].iloc[i] < df['Close'].iloc[i]):
            
            enter_price = df['Close'].iloc[i]
            enter_date= df.index[i]
            long = True
            
        if((df["Ch_exit_long"].iloc[i] > df['Close'].iloc[i]) & long==True):
               
            
            stop_price = df['Close'].iloc[i]
            exit_date=df.index[i]
            returns.append([(stop_price-enter_price)/enter_price, enter_date, exit_date])
            long=False  
        
    ret=np.mean([x[0] for x in returns])     
    print(ret)
    return returns, df

returns = backtest_macd(df_ready)[0]
df1 = backtest_macd(df_ready)[1]

#%%
enter_date = []
exit_date = []
for i in range(0,len(returns)-1):
    enter_date.append(returns[i][1])
    exit_date.append(returns[i][2])
    
#%%
#Plotting

df = df.set_index(df['Gmt time'],inplace=False, drop=True)


sns.set()
plt.figure(figsize=(12,4)) 
plt.scatter(df1.loc[enter_date].index,df1.loc[enter_date]['Close'], marker = '^',label = 'Long', color = 'green')
#plt.scatter(df.loc[Short_signal].index,df.loc[Short_signal]['Close'], marker = '^',label = 'Short', color = 'red')
plt.scatter(df1.loc[exit_date].index,df1.loc[exit_date]['Close'], marker = 'x',label = 'Neutralize long', color = 'green')
#plt.scatter(df.loc[Neutralize_short].index,df.loc[Neutralize_short]['Close'], marker = 'o', label = 'Neutralize short', color = 'red')
plt.plot(df1['Close'], label = 'Close price', alpha = 0.7)
plt.xticks(rotation=25)
#plt.title('SP500 Close price and buy/sell signals,' +''+ 'return:'+''+str(returns))
plt.xlabel('Date', fontsize=15)
plt.ylabel('Close price', fontsize=15)
plt.legend(loc=0,fontsize='medium')
#plt.savefig('Plot_long_p',dpi=1200)


#%%
#Holding period


l = pd.DataFrame()
l['Date_op'] = enter_date
l['Date_cl'] = exit_date
l['Signal'] = 'Long'
date_list = df['Gmt time'].reset_index(drop=True)
date_list = date_list.to_frame()
date_list2 = df.reset_index(drop=True)
date_list2 = date_list2[date_list2['Gmt time'].isin(Neutralize_long)]
date_list3 = df.reset_index(drop=True)
date_list3 = date_list3[date_list3['Gmt time'].isin(Long_signal)]
time_diff = (date_list2.index - date_list3.index)
l['Hold_per'] = time_diff
#l['Returns'] = longg_noweights['Classification']
sns.set(rc = {'figure.figsize':(15,8)})

#ax = sns.barplot(x = "Returns", y = "Hold_per", data = l, lw=0., ci=99)
ax = sns.barplot(data = l, lw=0., ci=95)

ax.set_title('Long returns holding period, measured in 15 min bars with 95% CI', fontsize=25)
ax.set_xlabel('All long positions considered',fontsize=15)
ax.set_ylabel('Holding period mean',fontsize=15)
#plt.savefig('long_position_holding_per_mean_ci95.png', dpi=1200)










