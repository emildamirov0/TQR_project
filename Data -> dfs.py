import pandas as pd

# Days of the interest rate announcement

events =['2012-01-25',
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
          '2019-12-11'
        
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
prices = pd.read_csv('USA500.csv')

# Excluding all data with zero volume
prices = prices[prices['Volume'] > 0]


#%%

# Normalising, removing GMT, and setting index

prices['date_norm'] = pd.to_datetime(prices['Local time'],utc=True,dayfirst=True).dt.normalize()
prices.set_index(prices['date_norm'],inplace=True,drop=True)


#%%

# Normalising 
# Adding days prior and after events to "dates" dataframe

from pandas.tseries.offsets import BDay

dates = pd.DataFrame(events,columns=["event_date"])
dates['event_date'] = pd.to_datetime(dates["event_date"],utc=True).dt.normalize()
dates['start_date'] = dates["event_date"] - BDay(1)
dates['end_date'] = dates["event_date"] + BDay(1)
dates.dropna(axis=0, inplace=True)

