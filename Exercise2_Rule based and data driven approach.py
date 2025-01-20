# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 10:43:50 2023

@author: HP
"""
"Task 1"
import pandas as pd
annotatedfile1 = pd.read_csv("C:/Users/HP/Documents/Master's/Decision Support/Exercise 2/annotation1.AN1", skiprows = 2, names = ['Time', 'VType', 'Variable', 'Value', 'Status'], index_col=False)
category_counts = annotatedfile1['Variable'].value_counts()

#filtered the required data
filteredCI = annotatedfile1.loc[annotatedfile1['Variable'] == 30001000]
filteredTp = annotatedfile1.loc[annotatedfile1['Variable'] == 400]
filteredPCWP = annotatedfile1.loc[annotatedfile1['Variable'] == 800]

filteredCIcount = filteredCI.shape[0]
filteredTpcount = filteredTp.shape[0]
filteredPCWPcount = filteredPCWP.shape[0]

"Task 2"
#merge the filtered data
mergedfiltereddata = pd.concat([filteredCI, filteredTp, filteredPCWP])

#defining the rules 
import numpy as np

mergedfiltereddata['CI value'] = np.where((mergedfiltereddata['Variable'] == 30001000) & (mergedfiltereddata['Value'] < 2.0), 1, 0)
mergedfiltereddata['Tp value'] = np.where((mergedfiltereddata['Variable'] == 400) & (mergedfiltereddata['Value'] < 32.5), 1, 0)
mergedfiltereddata['PCWP'] = np.where((mergedfiltereddata['Variable'] == 800) & (mergedfiltereddata['Value'] > 10), 1, 0)

mergedfiltereddata['Cardiac failure'] = np.where((mergedfiltereddata['CI value'] == 1) | (mergedfiltereddata['Tp value'] == 1) | (mergedfiltereddata['PCWP'] == 1), 1, 0)

#changing the time format
#To modify the time in HH:MM we can use this code:\n",
timefile = pd.read_table("C:/Users/HP/Documents/Master's/Decision Support/Exercise 2/info.txt", skiprows = 0, names = ["Time"], index_col=False)

date_time_str = str(timefile['Time'][0][3:5]+ timefile['Time'][0][6:8])

mergedfiltereddata['Time_formatted']=date_time_str+mergedfiltereddata['Time'].astype(str)
mergedfiltereddata['Time_formatted']=pd.to_datetime((mergedfiltereddata['Time_formatted']),format='%m%y%d%H%M%S')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
date_form = mdates.DateFormatter("%H:%M")
 
#Plotting scatter plor
ax = mergedfiltereddata.plot.scatter(x='Time_formatted', y='Cardiac failure', marker='o')
plt.ylim(-1, 2)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
plt.xticks(rotation=45)
plt.show()

"Task3"
annotatedfile2 = pd.read_csv("C:/Users/HP/Documents/Master's/Decision Support/Exercise 2/annotation2.AN2", skiprows = 2, names = ['Time', 'VType', 'Variable', 'Value', 'Status'], index_col=False)

annotatedfile2['Manual Cardiac Failure'] =  np.where((annotatedfile2['Variable'] == 15001116) & (annotatedfile2['Value'] == 2), 1, 0)
mergedfiltereddata['Manual Cardiac Failure'] = annotatedfile2['Manual Cardiac Failure']

fig, ax = plt.subplots()
ax.scatter(mergedfiltereddata['Time_formatted'], mergedfiltereddata['Cardiac failure'], marker = 'o', label = 'Annotated Cardiac Failure')
ax.scatter(mergedfiltereddata['Time_formatted'], mergedfiltereddata['Manual Cardiac Failure'], marker = 'x', label = 'Manual Cardiac Failure')
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
plt.xticks(rotation=45)
plt.legend()
plt.xlabel("Time")
plt.ylabel("Cardiac Failure")
plt.show()

"Task 4"

date_time_str = str(timefile['Time'][0][3:5]+ timefile['Time'][0][6:8])
annotatedfile2['Time_formatted2']=date_time_str+annotatedfile2['Time'].astype(str)
annotatedfile2['Time_formatted2']=pd.to_datetime((annotatedfile2['Time_formatted2']),format='%m%y%d%H%M%S')

careactivity1 = annotatedfile2.loc[annotatedfile2['Variable'] == 92]
print(careactivity1)


date_time_str = str(timefile['Time'][0][3:5]+ timefile['Time'][0][6:8])
annotatedfile1['Time_formatted3']=date_time_str+annotatedfile1['Time'].astype(str)
annotatedfile1['Time_formatted3']=pd.to_datetime((annotatedfile1['Time_formatted3']),format='%m%y%d%H%M%S')