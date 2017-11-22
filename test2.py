# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 19:11:24 2017

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

#%% ####################################
# driver_trips
#######################################

driver_trips.head()
driver_trips['request_time'] = pd.to_datetime(driver_trips.request_time)
driver_trips.dtypes
#%%
CC = driver_trips[driver_trips.start_geo == 'Chelsea Court'].sort_values('request_time')
#CC.set_index('request_time', inplace=True)
#CC.index.name = 'request_time'
#CC.reset_index(inplace=True)
CC.request_time.astype(dt.datetime, inplace=True).head()

CC['time_int']= CC.request_time.astype(np.int64)
CC.dtypes

#%% plot figure of surge pricing and request time
#plt.figure()
#plt.scatter(CC.request_time.dt, CC.surge_multiplier, s=1,c='b')
#plt.xticks(rotation='vertical')

CC.plot(kind='scatter',x='time_int',y='surge_multiplier', s = 3,color='b', alpha = 0.5)
ax = plt.gca()
xticks = ax.get_xticks()
ax.set_xticklabels([pd.to_datetime(tm, unit='ns').strftime('%Y-%m-%d\n %H:%M:%S') for tm in xticks],
                    rotation=50)
ax.set_xlabel('request_time')

#%%



#%%#############################################
# rider_trips
################################################

rider_trips['hour'] = rider_trips.request_time.dt.hour
rider_trips['date'] = rider_trips.request_time.dt.date
rider_trips['time_slog'] = pd.to_datetime(rider_trips.date) + rider_trips['hour'].apply(lambda x: pd.Timedelta(x, 'h'))

rider_trips_filtered = rider_trips

#filter rush hour
rush_hour_morning = ((rider_trips.hour >=7) & (rider_trips.hour <=9) )
rush_hour_evening =  ((rider_trips.hour >=16) & (rider_trips.hour <=18))
#rider_trips_filtered = rider_trips_filtered[ rush_hour_morning | rush_hour_evening]
print(rider_trips_filtered.shape)

# filter canceled trips
rider_trips_filtered = rider_trips_filtered[rider_trips_filtered.trip_status=='completed']
print(rider_trips_filtered.shape)

#filter nan
rider_trips_filtered = rider_trips_filtered.dropna(subset=['estimated_time_to_arrival','surge_multiplier'], how='any')
print(rider_trips_filtered.shape)

#filter too small ETA
#rider_trips_filtered = rider_trips_filtered[rider_trips_filtered.estimated_time_to_arrival >=1]
#print(rider_trips_filtered.shape)

#filter end_geo
#rider_trips_filtered = rider_trips_filtered[rider_trips_filtered.end_geo == 'Chelsea Court']
#print(rider_trips_filtered.shape)

#filtration surge price
#rider_trips_filtered = rider_trips_filtered[rider_trips.surge_multiplier > 1]
#print(rider_trips_filtered.shape)


# grouped by region
grouped = rider_trips_filtered.groupby(['start_geo'])
dict_region = dict()
for name, region in grouped:
    dict_region[name] = region

c_array={'Allen Abby':'r',
          'Blair Bend':'g',
          'Chelsea Court':'b',
          'Daisy Drive':'orange'}
               
#%% plot sugre_price vs ETA by region

dict_info = dict()
plt.figure()
for name, region in grouped:
    print(name)
    region_grouped = region.groupby('surge_multiplier').estimated_time_to_arrival.agg(['count','mean'])
    region_grouped.reset_index(inplace=True)
    dict_info[name] = region_grouped
    plt.plot(region_grouped.surge_multiplier, region_grouped['mean'],color=c_array[name], linestyle = ':',alpha = 0.7, label = name)
    
#city_average = rider_trips_filtered.groupby('surge_multiplier').estimated_time_to_arrival.agg(['count','mean'])
city_average = rider_trips_filtered[rider_trips_filtered.start_geo != 'Chelsea Court']\
                                   .groupby('surge_multiplier').estimated_time_to_arrival.agg(['count','mean'])

city_average.reset_index(inplace=True)
plt.plot(city_average.surge_multiplier, city_average['mean'],linestyle ='-', color='purple',alpha=1, label = 'city avg')
plt.legend()
plt.xlabel('surge multiplier')
plt.ylabel('mean ETA during rush hour')

#%% plot number of riders along time in different regions
plt.figure()
dict_info_ridernb = dict()
for name, region in rider_trips_filtered.groupby(['start_geo']):
    df= region.groupby(['time_slog']).size().reset_index().rename(columns= {0:'rider_nb'})
    plt.plot(df.time_slog.astype(np.int64), df['rider_nb'],alpha = 0.5, label = name)
    dict_info_ridernb[name] = df

plt.ylabel('number of riders')
x = pd.DataFrame(rider_trips_filtered.time_slog.unique()).rename(columns={0:'time_slog'}).sort_values('time_slog')
tick_hour = np.array(rider_trips_filtered.sort_values('time_slog').time_slog.head(1).dt.hour)[0]
new_xtick_label = x.time_slog.apply(lambda tm : tm.strftime('%m/%d %H:00') if tm.hour == tick_hour else "");
plt.xticks(x.time_slog.astype(np.int64), new_xtick_label, rotation= 'vertical');

lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')
          
#%% plot number of rider by hour
plt.figure()
for name, region in rider_trips_filtered.groupby('start_geo'):
    df = region.groupby('hour').size().reset_index().rename(columns= {0:'rider_nb'})
    plt.plot(df.hour, df['rider_nb'],alpha = 0.5, label = name)
plt.ylabel('number of riders') 
plt.xlabel('hour') 
ax = plt.gca()
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')

#%%plt.figure()

for name, region in rider_trips_filtered.groupby('start_geo'):
    df = region.groupby('hour').estimated_time_to_arrival.agg(np.mean).reset_index()
    plt.plot(df.hour, df['estimated_time_to_arrival'],alpha = 0.5, label = name)
    
plt.axhline(y=rider_trips_filtered.estimated_time_to_arrival.mean(), color='r', linestyle='-')
plt.ylabel('ETA mean') 
plt.xlabel('hour') 
ax = plt.gca()
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')
           
#%% plot evolution of avg surge price along time

dict_info_sp = dict()
for name, region in rider_trips_filtered.groupby(['start_geo']):
    #region_grouped = region.groupby(['date']).surge_multiplier.agg(['count',np.mean])
    region_grouped = region.groupby(['time_slog']).surge_multiplier.agg(['count',np.mean])
    region_grouped.reset_index(inplace = True)
    dict_info_sp[name] = region_grouped

city_average_sp= rider_trips_filtered[rider_trips_filtered.start_geo != 'Chelsea Court'].groupby(['time_slog']).surge_multiplier.agg(['count',np.mean])
#city_average_sp= rider_trips_filtered.groupby(['time_slog']).surge_multiplier.agg(['count',np.mean])
city_average_sp.reset_index(inplace = True)

#figure
CC = dict_info_sp['Chelsea Court']

plt.figure()
#x = pd.to_datetime(CC.date).astype(np.int64)
#x_label= pd.to_datetime(CC.date).dt.strftime('%Y/%m/%d')
#x = CC.time_slog.astype(np.int64)
#x_label = CC.time_slog.dt.strftime('%m/%d %H:00')
                 
plt.scatter(x = CC.time_slog.astype(np.int64), y = CC['mean'], color = 'c', s = np.array(CC['count'])/5, alpha = 0.5)
plt.scatter(x =city_average_sp.time_slog.astype(np.int64), y = city_average_sp['mean'], color = 'r', s = np.array(city_average_sp['count'])/5, alpha = 0.5)

line1,=plt.plot(CC.time_slog.astype(np.int64), CC['mean'],color='c',alpha = 1, label='CC')
line2,=plt.plot(city_average_sp.time_slog.astype(np.int64), city_average_sp['mean'], color='r', linestyle = '-', alpha = 0.5,label='city avg')
plt.legend(handles=[line1,line2], loc = 2)
plt.ylabel('surge multiplier')

tick_hour = np.array(rider_trips_filtered.sort_values('time_slog').time_slog.head(1).dt.hour)[0]
new_xtick_label = city_average_sp.time_slog.apply(lambda tm : tm.strftime('%m/%d %H:00') if tm.hour == tick_hour else "");
plt.xticks(city_average_sp.time_slog.astype(np.int64), new_xtick_label, rotation= 'vertical');


#%% plot evolution of avg ETA

dict_info_ETA = dict()

#city_average_ETA= rider_trips_filtered[rider_trips_filtered.start_geo != 'Chelsea Court'].groupby('time_slog').estimated_time_to_arrival.agg(['count',np.mean])
city_average_ETA= rider_trips_filtered.groupby('time_slog').estimated_time_to_arrival.agg(['count',np.mean])

city_average_ETA.reset_index(inplace = True)

for name, region in grouped:
    region_grouped = region.groupby('time_slog').estimated_time_to_arrival.agg(['count',np.mean])
    region_grouped.reset_index(inplace = True)
    dict_info_ETA[name] = region_grouped

#figure
plt.figure()
CC = dict_info_ETA['Chelsea Court']
#x = pd.to_datetime(CC.date).astype(np.int64)
#x_label= pd.to_datetime(CC.date).dt.strftime('%Y/%m/%d')

#x = CC.time_slog.astype(np.int64)
#x_label = CC.time_slog.dt.strftime('%m/%d %H:00')
                 
plt.scatter(x = CC.time_slog.astype(np.int64), y = CC['mean'], color = 'c', s = np.array(CC['count'])/5, alpha = 0.5)
plt.scatter(x = city_average_ETA.time_slog.astype(np.int64) , y = city_average_ETA['mean'], color = 'r', s = np.array(city_average_ETA['count'])/5, alpha = 0.5)

line1, = plt.plot(CC.time_slog.astype(np.int64), CC['mean'],color='c',alpha = 1,label='CC')
line2, = plt.plot(city_average_ETA.time_slog.astype(np.int64), city_average_ETA['mean'], color='r', linestyle = '-', alpha = 0.5, label='city avg')

plt.legend(handles=[line1,line2], loc = 1)
plt.ylabel('ETA')

tick_hour = np.array(rider_trips_filtered.sort_values('time_slog').time_slog.head(1).dt.hour)[0]
new_xtick_label = city_average_ETA.time_slog.apply(lambda tm : tm.strftime('%m/%d %H:00') if tm.hour == tick_hour else "");
plt.xticks(city_average_ETA.time_slog.astype(np.int64), np.array(new_xtick_label), rotation = 'vertical');

          
#%% check the difference of ETA city avarage(other regions) and CC region

diff_ETA = pd.concat( [city_average_ETA.set_index('time_slog').rename(columns={'count':'count_other', 'mean':'mean_other'})
                    , dict_info_ETA['Chelsea Court'].set_index('time_slog').rename(columns={'count':'count_CC', 'mean':'mean_CC'})]
                    , axis = 1, join='outer').reset_index()

diff_ETA.fillna(0, inplace=True)
               
diff_ETA['difference_in_time'] = diff_ETA['mean_CC'] - diff_ETA['mean_other']

# figure scatter
plt.figure()
plt.scatter(x= diff_ETA.time_slog.astype(np.int64), y=diff_ETA.difference_in_time, color = 'c', edgecolor='w', alpha = 0.5)
plt.axhline(y=0, color='r', linestyle='-')
plt.ylabel('difference in ETA (CC - city_avg)')
plt.title('difference in ETA')
new_xtick_label = diff_ETA.time_slog.apply(lambda tm : tm.strftime('%m/%d %H:00') if tm.hour == 7 else "");
plt.xticks(diff_ETA.time_slog.astype(np.int64), np.array(new_xtick_label), rotation = 'vertical');

#figure bar         
fig= plt.figure()
ax = plt.subplot(111)
ax.bar( np.array(diff_ETA[diff_ETA.difference_in_time >=0].index)
        , np.array(diff_ETA[diff_ETA.difference_in_time >=0].difference_in_time) , color='r',alpha=0.5)
ax.bar( np.array(diff_ETA[diff_ETA.difference_in_time <0].index)
        , np.array(diff_ETA[diff_ETA.difference_in_time <0].difference_in_time) , color='b',alpha=0.5)
plt.ylabel('difference in ETA (CC - city_avg)')
ax.set_title('difference in ETA')
new_xtick_label = diff_ETA.time_slog.apply(lambda tm : tm.strftime('%m/%d %H:00') if tm.hour == 7 else "");
plt.xticks(diff_ETA.index, np.array(new_xtick_label), rotation = 'vertical');



#%%##################################################
# rider number
#####################################################

rider_trips_month = rider_trips
rider_trips_month['month'] = rider_trips.request_time.dt.month
rider_trips_month = rider_trips_month[rider_trips_month.trip_status == 'completed'] 
rider_trips_mohtn = rider_trips_month.dropna(subset=['estimated_time_to_arrival','surge_multiplier'], how='any')


city_info_month = rider_trips_month.groupby(['start_geo','month']).trip_status.agg(['count'])
city_info_month.reset_index(inplace=True)
city_info_month.sort_values('start_geo')


# number of trips per day
grouped_nb = rider_trips_month.groupby(['start_geo'])
dict_info_nb =dict()
city_avg_nb = rider_trips_month.groupby('date').trip_status.agg(['count'])
city_avg_nb.reset_index(inplace = True)

x = pd.to_datetime(city_avg_nb.date).astype(np.int64)
x_label= pd.to_datetime(city_avg_nb.date).dt.strftime('%Y/%m/%d')
   
plt.figure()
for name, region in grouped_nb:
    region_grouped = region.groupby(['date']).trip_status.agg(['count'])
    region_grouped.reset_index(inplace=True)
    dict_info_nb[name] = region_grouped    
    plt.scatter(x, y = region_grouped['count'], color = c_array[name], s = np.array(region_grouped['count'])/5, alpha = 0.5, label=name)
    plt.plot(x, region_grouped['count'],color=c_array[name],alpha = 0.5,label=name)

    
plt.scatter(x,y=city_avg_nb['count'],color = 'b', s = np.array(city_avg_nb['count']) / 5, alpha =1, label = name)
plt.plot(x, city_avg_nb['count'], color ='b', alpha=1, label = 'city avg')
xticks = plt.xticks(x, np.array(x_label), rotation = 'vertical')
plt.legend()
plt.ylabel('number of trips')




