# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 14:29:25 2017

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


#%%
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

#filer <0 ETA
#rider_trips_filtered = rider_trips_filtered[ rider_trips_filtered.estimated_time_to_arrival >=0]
#print(rider_trips_filtered.shape)

#rush hour
rush_hour_morning = ( (rider_trips_filtered.hour >=7) & (rider_trips_filtered.hour <=9) )
rush_hour_evening =  ((rider_trips_filtered.hour >=16) & (rider_trips_filtered.hour <=18))
rider_trips_filtered = rider_trips_filtered[ rush_hour_morning | rush_hour_evening]
print(rider_trips_filtered.shape)







#%%
grouped = rider_trips_filtered.groupby(['start_geo'])
df_region = dict()
region_names = []
for name, region in grouped:
    df_region[name] = region
    region_names.append(name)
             
color_array={'Allen Abby':'r',
              'Blair Bend':'g',
              'Chelsea Court':'b',
              'Daisy Drive':'orange',
              'city_average':'c'}

AA = df_region['Allen Abby']
BB = df_region['Blair Bend']
CC = df_region['Chelsea Court']
DD = df_region['Daisy Drive']

#%% plot distribution of CC and the whole city

ax = plt.figure()

binvalues = np.linspace(rider_trips_filtered.estimated_time_to_arrival.min()
                        , rider_trips_filtered.estimated_time_to_arrival.max()
                        , 300)

labels = ['CC', 'city avg']
ax = CC.estimated_time_to_arrival.plot(kind='density',color='c')
ax = CC.estimated_time_to_arrival.plot(kind='hist', normed = 1, bins = binvalues, alpha = 0.3, edgecolor='w')
         
ax = rider_trips_filtered.estimated_time_to_arrival.plot(kind='density',color='orange')                
ax = rider_trips_filtered.estimated_time_to_arrival.plot(kind='hist', normed=1, bins = binvalues, alpha = 0.3, edgecolor='w')

#ax.set_xlim(rider_trips_filtered.estimated_time_to_arrival.min(), rider_trips_filtered.estimated_time_to_arrival.max())
ax.set_xlim(0,10)
lines, _ = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')
ax.set_xlabel('ETA')
ax.set_ylabel('density')
plt.show()


#%% plot only the density
ax = plt.figure()
labels = []

ax = CC.estimated_time_to_arrival.plot(kind='kde')
labels.append('CC')                                
ax = rider_trips_filtered.estimated_time_to_arrival.plot(kind='kde')
labels.append('city avg')
ax.set_xlim(rider_trips_filtered.estimated_time_to_arrival.min(), rider_trips_filtered.estimated_time_to_arrival.max())
lines, _ = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')
ax.set_xlabel('ETA')
plt.show()


#%% plot distribution (density) for all regions
ax = plt.figure()
labels = []
for name, region in  rider_trips_filtered.groupby(['start_geo']):
    #ax = region.estimated_time_to_arrival.plot(kind='hist', color = color_array[name], alpha = 0.5, bins = 50)
    ax = region.estimated_time_to_arrival.plot(kind='bar')
    labels.append(name)
    
ax = rider_trips_filtered.estimated_time_to_arrival.plot(kind = 'bar')
labels.append('city avg')

lines, _ = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')
plt.show()


#%% plot according to surge price


df_sp_CC = {}
df_sp_city = {}
sp_levels = []

for key, region in CC.groupby(['surge_multiplier']):
    df_sp_CC[key] = region
    sp_levels.append(key)
for key, region in rider_trips_filtered.groupby(['surge_multiplier']):
    df_sp_city[key] = region

'''
# figure for CC 

ax = plt.figure()
labels = []
for key in sp_levels:
    if df_sp_CC[key].shape[0] > 1:
        ax = df_sp_CC[key].estimated_time_to_arrival.plot(kind='density')
        labels.append(name)
    
ax = CC.estimated_time_to_arrival.plot(kind = 'density')
labels.append('CC whole')

lines, _ = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')
ax.set_xlim(0,15)
plt.title('CC')
plt.show()

# figure for the whole city
ax = plt.figure()
labels = []
for key in sp_levels:
    if df_sp_CC[key].shape[0] > 1:
        ax = df_sp_CC[key].estimated_time_to_arrival.plot(kind='density')
        labels.append(name)
    
ax = CC.estimated_time_to_arrival.plot(kind = 'density')
labels.append('CC whole')

lines, _ = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')
ax.set_xlim(0,15)
plt.title('city')
plt.show()
'''

# subplots comparison between CC and the city at different sp levels
n_figures = len(sp_levels)

fig, axes = plt.subplots(nrows=4, ncols=10)

for i in range(4):
    for j in range(10):
        idx = i*10 + j
        if idx < n_figures:
            if df_sp_CC[sp_levels[idx]].shape[0] > 1:
                axes[i,j] = df_sp_CC[sp_levels[idx]].estimated_time_to_arrival.plot(ax = axes[i,j], kind='density', color = 'blue')
                axes[i,j] = df_sp_city[sp_levels[idx]].estimated_time_to_arrival.plot(ax = axes[i,j], kind='density', color = 'orange')
                axes[i,j].set_ylim(0,0.5)
                lines, _ = axes[i,j].get_legend_handles_labels()
                axes[i,j].legend(lines, ['CC','city'], loc='best') 
                axes[i,j].set_title(sp_levels[idx])
        if j !=0:
            axes[i,j].yaxis.set_visible(False)
            #axes[i,j].set_yticks([])
            
            
#%% figure 1 for distribution of surge price 

ax = plt.figure()
labels = []

CC_rush = CC[CC.surge_multiplier >1]
city_rush = rider_trips_filtered[rider_trips_filtered.surge_multiplier > 1]
bins_values = np.append(rider_trips_filtered.surge_multiplier.sort_values().unique(), [rider_trips_filtered.surge_multiplier.max()+0.1])
           
ax = city_rush.surge_multiplier.plot(kind='density',color='orange')                
ax = city_rush.surge_multiplier.plot(kind='hist', normed=1, bins = bins_values, alpha = 0.3,edgecolor='w', color='orange')
labels.append('city avg')
ax = CC_rush.surge_multiplier.plot(kind='density',color='c')
ax = CC_rush.surge_multiplier.plot(kind='hist', normed = 1, bins = bins_values, alpha = 0.3, edgecolor='w', color='c')
labels.append('CC')   

ax.set_xlim(city_rush.surge_multiplier.min(), city_rush.surge_multiplier.max())
lines, _ = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')
ax.set_xlabel('surge_multiplier')
ax.set_ylabel('density')
plt.show()


#%% proportion of surge prices

proportion_regions = rider_trips_filtered.groupby(['surge_multiplier','start_geo']).size().reset_index().rename(columns={0:'trips_nb'})

grouped = proportion_regions.groupby('start_geo')
proportion = pd.DataFrame()
for key,value in rider_trips_filtered.groupby('start_geo'):
    proportion = pd.concat([proportion 
                            ,value.groupby('surge_multiplier').size().reset_index().set_index('surge_multiplier').rename(columns={0:key})]
                            ,axis=1, join = 'outer')

proportion = pd.concat([proportion, rider_trips_filtered.groupby('surge_multiplier').size().reset_index().set_index('surge_multiplier').rename(columns={0:'city'})]
                        , axis = 1, join='outer')
    
proportion.fillna(0, inplace=True)
proportion['other_regions_CC'] = proportion['Allen Abby']+proportion['Blair Bend']+proportion['Daisy Drive']

percentage = proportion / proportion.sum()

#alternative way to get the proportion

#bins_values = np.append(rider_trips_filtered.surge_multiplier.sort_values().unique(), [rider_trips_filtered.surge_multiplier.max()+0.1])
#hist_CC,_ = np.histogram(CC.surge_multiplier, bins = bins_values)  
#hist_city,_ =np.histogram(rider_trips_filtered.surge_multiplier, bins = bins_values)

ax = percentage.loc[1.1:,['Allen Abby','Blair Bend','Chelsea Court','Daisy Drive','city']].plot(kind='bar',alpha = 0.7)
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')
ax.set_title('surge multiplier proportion')
vals = ax.get_yticks()
ax.set_yticklabels(['{:.0f}%'.format(x*100) for x in vals])
plt.show()

ax = percentage.loc[1.1:,['city','Chelsea Court','other_regions_CC']].plot(kind='bar', alpha =0.7, edgecolor ='w')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')
ax.set_title('surge multiplier proportion')
vals = ax.get_yticks()
ax.set_yticklabels(['{:.0f}%'.format(x*100) for x in vals])
plt.show()




#%% #############################################################
# compare CC and the reset regions of the cities
#################################################################
#%% surge price
proportion_CC = proportion['Chelsea Court'] 
proportion_others = proportion['Allen Abby']+ proportion['Blair Bend']+proportion['Daisy Drive']
df = pd.concat([proportion_CC,proportion_others], axis=1, join='outer')
df.rename(columns= {0:'other regions'},inplace=True)
df = df / df.sum()

ax = df.loc[1.1:,:].plot(kind='bar',stacked=False)
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')
ax.set_title('surge price proportion')
plt.show()

#%% ETA
other_regions = rider_trips_filtered[rider_trips_filtered.start_geo != 'Blair Bend' ]

binvalues = np.linspace(rider_trips_filtered.estimated_time_to_arrival.min()
                        , rider_trips_filtered.estimated_time_to_arrival.max()
                        , 300)

ax = plt.figure()
labels = []
ax = BB.estimated_time_to_arrival.plot(kind='density',color='c')
ax = BB.estimated_time_to_arrival.plot(kind='hist', normed = 1, bins = binvalues, alpha = 0.3, edgecolor='w')
labels.append('BB')              
ax = other_regions.estimated_time_to_arrival.plot(kind='density',color='orange')                
ax = other_regions.estimated_time_to_arrival.plot(kind='hist', normed=1, bins = binvalues, alpha = 0.3, edgecolor='w')
labels.append('other regions')

lines, _ = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')
ax.set_xlabel('ETA')
#ax.set_xlim(rider_trips_filtered.estimated_time_to_arrival.min(), rider_trips_filtered.estimated_time_to_arrival.max())
ax.set_xlim(0,10)

plt.show()


