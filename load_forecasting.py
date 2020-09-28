# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 20:42:08 2019

@author: Bidong Liu
"""
#read files from TXT file
import pandas as pd
#import numpy as np 
#import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

import os
os.chdir('C:/Users/Bidong Liu/Desktop/Electric_load_forecasting')

pd.set_option('display.max_columns', None)  
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

# =============================================================================
# #for load data file
# =============================================================================
#import load data as a dataframe
dataLoad = pd.read_csv("Load_history.csv") 

#reshape dataframe from wide to long

dataLoadL=pd.melt(dataLoad, id_vars =['zone_id', 'year', 'month', 'day'], value_name='load', \
                  value_vars =['h{}'.format(h) for h in range(1, 25)]) 
#change varaible values 
hourList=['h{}'.format(h) for h in range(1, 25)]
numList=[num for num in range(1,25)]
for (hour, num) in zip(hourList,numList):
    dataLoadL['variable']=dataLoadL['variable'].mask(dataLoadL['variable']==hour, num)

#change column data type 
dataLoadL['hour']=dataLoadL['variable'].astype(int)



#subset dataframe by conditions to Zone 1 with year between 2004 and 2006
zone1 = dataLoadL[(dataLoadL.zone_id == 1) &  (dataLoadL.year <=2006) ]

#create trend variable
zone1['trend'] = pd.Series(range(1,(len(zone1.index)+1))).values
zone1['trend'].describe()

#drop columns
zone1=zone1.drop(columns=['variable', 'zone_id'])

zone1.head()
zone1.describe(include='all')


# =============================================================================
# #for temperature data file
# =============================================================================
#import temperature data file
dataTemperature = pd.read_csv("temperature_history.csv") 
dataTemperature.head()
#reshape dataframe from wide to long
dataTempL=pd.melt(dataTemperature, id_vars =['station_id', 'year', 'month', 'day'], value_name='temper', \
                   value_vars =['h{}'.format(i) for i in range(1, 25)]) 
#change varaible values 
hourList=['h{}'.format(h) for h in range(1, 25)]
numList=[num for num in range(1,25)]

for (hour, num) in zip(hourList,numList):
    dataTempL['variable']=dataTempL['variable'].mask(dataTempL['variable']==hour, num)
    
dataTempL['hour']=dataTempL['variable'].astype(int)


#subset dataframe by conditions
station1 = dataTempL[(dataTempL.station_id == 1) &  (dataTempL.year <=2006) ]
#drop columns
station1=station1.drop(columns=['variable', 'station_id'])
station1.head()
station1.describe(include='all')

# =============================================================================
# combine load data file and temperature data file
# =============================================================================
zone1.head()
station1.head()
#merge two data files
zone1LT = pd.merge(zone1, station1, how='left', left_on=['year', 'month', \
                  'day', 'hour'], right_on=['year', 'month', 'day', 'hour'])

#create column of weekday
zone1LT['weekday'] = pd.to_datetime(zone1LT[['year', 'month', 'day']]).dt.dayofweek

zone1LT.describe()
zone1LT.head()
zone1LT.dtypes

# =============================================================================
# Explore the relationships 
#         between load and temperature
#         between load and month, weekday and hour
#         using the package plotnine
# =============================================================================

#Scatter plot for overall
allArray = zone1LT.values
  
load=allArray[: ,3]
temperature=allArray[: ,5]

zone1LT.plot.scatter(x='temper', y='load', c='DarkBlue')
plt.title('Scatter plot Electric load VS Temperature')
# Set x-axis label
plt.xlabel('Temperature (F)')
# Set y-axis label
plt.ylabel('Electric load (MW)')
 
 
#Scatter plot by month
for month in range(1,13):
     zone1LTM=zone1LT[(zone1LT['month'] == month)]
     #zone1LTM['month'].describe()
     
     zone1LTM.plot.scatter(x='temper', y='load', c='DarkBlue')
     plt.title('Scatter plot Electric load VS Temperature for Month ' + str(month))
     # Set x-axis label
     plt.xlabel('Temperature (F)')
     # Set y-axis label
     plt.ylabel('Electric load (MW)')
 
#Scatter plot by hour
for hour in range(1,25):
     zone1LTH=zone1LT[(zone1LT['hour'] == hour)]
     #zone1LTM['month'].describe()
     
     zone1LTH.plot.scatter(x='temper', y='load', c='DarkBlue')
     plt.title('Scatter plot Electric load VS Temperature for Hour ' + str(hour))
     # Set x-axis label
     plt.xlabel('Temperature (F)')
     # Set y-axis label
     plt.ylabel('Electric load (MW)')



# =============================================================================
# Modeling process
# =============================================================================

#subset for training datasets
training=zone1LT[(zone1LT.year < 2006)]
test=zone1LT[(zone1LT.year == 2006)]

#check whether has missing value for load
test['load'].isnull().values.any()
test['load'].isnull().sum()
#drop rows with missing value
test=test.dropna()

test.describe()

#zone1LT['load']=zone1LT['load'].mask(zone1LT['year']==2006, np.nan)

#check missing values
#zone1LT.isnull() 

training['trend'].describe()
test['trend'].describe()


res = smf.ols(formula='load ~ trend + C(month) + C(weekday) + C(hour) + \
              C(weekday)*C(hour) +\
              temper + temper*temper + temper*temper*temper +\
              month*temper + month*temper*temper + month*temper*temper*temper +\
              hour*temper + hour*temper*temper + hour*temper*temper*temper', data=training).fit()

test['prediction']=res.predict(test)

APE=abs(test['prediction']-test['load'])/test['load']
print(APE.mean())

#add a test code to for github




 

