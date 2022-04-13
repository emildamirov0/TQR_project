#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 13:39:52 2022

@author: emildamirov
"""

import pandas as pd

df = pd.read_csv('2012_2014.csv')
df = df[df['Volume'] > 0]
df = df.rename(columns={'Gmt time': 'Exchange Date'})
df['Exchange Date'] = pd.to_datetime(df['Exchange Date'].values, utc=True, dayfirst=True)
ind = range(0,len(df))
indexlist = list(ind)

#df = df.set_index(df['Exchange Date'])
#df.sort_index(ascending = True, inplace = True)



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
