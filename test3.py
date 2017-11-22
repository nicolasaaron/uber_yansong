# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 18:12:22 2017

@author: iris
"""

import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% load data
filename = 'data_set_170705.xlsx'
xls_file = pd.read_excel(filename, sheetname =['driver_trips',
                                               'rider_trips',
                                               'driver_data',
                                               'rider_data',
                                               'city_metrics'])

#%%
driver_trips =  xls_file['driver_trips']
rider_trips = xls_file['rider_trips']
driver_data = xls_file['driver_data']
rider_data = xls_file['rider_data']
city_metrics = xls_file['city_metrics']

rider_trips['hour'] = rider_trips.request_time.dt.hour
rider_trips['date'] = rider_trips.request_time.dt.date
rider_trips['time_slog'] = pd.to_datetime(rider_trips.date) + rider_trips['hour'].apply(lambda x: pd.Timedelta(x, 'h'))



#%%
rider_trips_filtered = rider_trips
print(rider_trips_filtered.shape)

# filter canceled trips
rider_trips_filtered = rider_trips_filtered[rider_trips_filtered.trip_status=='completed']
print(rider_trips_filtered.shape)

#filter nan
rider_trips_filtered = rider_trips_filtered.dropna(subset=['estimated_time_to_arrival','surge_multiplier'], how='any')
print(rider_trips_filtered.shape)

#common noise : surge price = 1
trips_sp_non_active = rider_trips_filtered[ rider_trips_filtered.surge_multiplier ==1]
trips_sp_active = rider_trips_filtered[rider_trips_filtered.surge_multiplier > 1]


#filer <0 ETA
#rider_trips_filtered = rider_trips_filtered[ rider_trips_filtered.estimated_time_to_arrival >=0]
#print(rider_trips_filtered.shape)

#rush hour
FLAG_rush_hour = False

if FLAG_rush_hour:
    rush_hour_morning = ( (rider_trips_filtered.hour >=7) & (rider_trips_filtered.hour <=9) )
    rush_hour_evening =  ((rider_trips_filtered.hour >=14) & (rider_trips_filtered.hour <=18))
    rider_trips_filtered = rider_trips_filtered[ rush_hour_evening]
    print(rider_trips_filtered.shape)



#%% ditribution of ETA without common noise

ETA_sp_non_active = trips_sp_non_active.estimated_time_to_arrival.mean()
ETA_sp_active = trips_sp_active.estimated_time_to_arrival.mean()

#%% plot evolution of ETA along time slog

dict_info_ETA = dict()
city_average_ETA= trips_sp_active.groupby('time_slog').estimated_time_to_arrival.agg(['count',np.mean]).reset_index()

for name, region in trips_sp_active.groupby('start_geo'):
    region_grouped = region.groupby('time_slog').estimated_time_to_arrival.agg(['count',np.mean]).reset_index()
    dict_info_ETA[name] = region_grouped

#figure
plt.figure()
CC = dict_info_ETA['Chelsea Court']
plt.scatter(x = CC.time_slog.astype(np.int64), y = CC['mean'], color = 'c', s = np.array(CC['count'])/5, alpha = 0.5)
plt.scatter(x = city_average_ETA.time_slog.astype(np.int64) , y = city_average_ETA['mean'], color = 'orange', s = np.array(city_average_ETA['count'])/5, alpha = 0.5)

line1, = plt.plot(CC.time_slog.astype(np.int64), CC['mean'],color='c',alpha = 1,label='CC')
line2, = plt.plot(city_average_ETA.time_slog.astype(np.int64), city_average_ETA['mean'], color='orange', linestyle = '-', alpha = 0.5, label='city avg')

line3, = plt.axhline(y = ETA_sp_non_active, color= 'r',label = 'avg ETA sp non active')
line4, = plt.axhline(y = ETA_sp_active, color = 'g', label = 'avg ETA sp active')

plt.legend(handles=[line1,line2], loc = 1)
plt.ylabel('ETA')

tick_hour = np.array(rider_trips_filtered.sort_values('time_slog').time_slog.head(1).dt.hour)[0]
new_xtick_label = city_average_ETA.time_slog.apply(lambda tm : tm.strftime('%m/%d %H:00') if tm.hour == tick_hour else "");
plt.xticks(city_average_ETA.time_slog.astype(np.int64), np.array(new_xtick_label), rotation = 'vertical');

      
