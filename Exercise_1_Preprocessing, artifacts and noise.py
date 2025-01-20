# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Task1 

import numpy as np #for defining time
import pandas as pd #for manipulating data
import matplotlib.pyplot as plt #for plotting

#import csv file and assign names to the columns
d = pd.read_csv("C:/Users/HP/Documents/Master's/Decision Support/Exercise 1/samples.csv", header = None, skiprows=2, names=['Elapsed time', 'ECGI', 'ECGI filtered'])

#create a time variable
fs = 500
start = 0
end = len(d)/fs
step = 1/fs
t = np.arange(start,end,step, dtype = None)

#plotting two signals on one graph
plt.plot(t, d['ECGI'], label = "raw ECG signal")
plt.plot(t, d['ECGI filtered'], label = "filtered ECG signal")
plt.legend()
plt.xlabel("Time")
plt.ylabel("ECG")
plt.show()

#Task2
plt.psd(d['ECGI'], NFFT=1024, Fs=500, label = "raw ECG signal")
plt.psd(d['ECGI filtered'], NFFT=1024, Fs=500, label = "filtered ECG signal")
plt.legend()


#Task3
import neurokit2 as nk 
from scipy import signal as signal


#Split data to 02 signals, prepare time axis for the plot 
ecg_raw = d['ECGI'] 
ecg_filtered = d['ECGI filtered']

_, rpeaks = nk.ecg_peaks(ecg_raw, sampling_rate=fs)
_, waves_peak = nk.ecg_delineate(ecg_raw, rpeaks, sampling_rate=fs, method="peak")   
_, waves_peak = nk.ecg_delineate(ecg_raw,  
                                 rpeaks,  
                                 sampling_rate=fs,  
                                 method="peak",  
                                 show=True,  
                                 show_type='peaks')

_, rpeaks = nk.ecg_peaks(ecg_filtered, sampling_rate=fs)
_, waves_peak = nk.ecg_delineate(ecg_filtered, rpeaks, sampling_rate=fs, method="peak")   
_, waves_peak = nk.ecg_delineate(ecg_filtered,  
                                 rpeaks,  
                                 sampling_rate=fs,  
                                 method="peak",  
                                 show=True,  
                                 show_type='peaks')

# 1. Design a bandpass filter for 0.01-150 Hz 
order = 4 
low = 0.01  
high = 150  
low_normalized = low/(0.5*fs) 
high_normalized = high/(0.5*fs) 
f1_b, f1_a = signal.butter(order, [low_normalized,high_normalized], btype='band') 
# Apply this filter to raw signal 
output_signal_first_filter = signal.lfilter(f1_b, f1_a, ecg_raw) 
plt.figure() 
plt.plot(t,ecg_raw, label='Raw', ) 
plt.plot(t,output_signal_first_filter, label='Filtered signal by Bandpass', ) 
plt.xlabel('Time') 
plt.ylabel('Signal Value') 
plt.title('Signal Data Plot') 
plt.legend() 
plt.grid() 
 
plt.figure() 
frequencies, psd_raw = plt.psd(ecg_raw, NFFT=1024, Fs=fs, label='raw ECG PSD') 
frequencies, psd_filtered = plt.psd(output_signal_first_filter, NFFT=1024, Fs=fs, label='Filtered w. 1st filter') 
plt.xlabel('Frequency (Hz)') 
plt.ylabel('Power/Frequency (dB/Hz)') 
plt.title('Power Spectral Density of Raw and 1st Filter ECG Signal') 
plt.legend() 
plt.show()

# 2. Notch filter for 50Hz 
f0 = 50.0  # Frequency to be removed from signal (Hz) 
Q = 7  # Quality factor, adjust affect height of signal being altered 
# Design notch filter 
f2_b, f2_a = signal.iirnotch(f0, Q, fs) 
# Apply notch filter 
output_signal_second_filter = signal.lfilter(f2_b, f2_a, ecg_raw) 
# Then plot second filtered signal 
plt.figure() 
plt.plot(t,ecg_raw, label='Raw', ) 
plt.plot(t,output_signal_second_filter, label='Filtered signal by Notch 50Hz', ) 
plt.xlabel('Time') 
plt.ylabel('Signal Value') 
plt.title('Signal Data Plot') 
plt.legend() 
plt.grid() 
 
plt.figure() 
frequencies, psd_raw = plt.psd(ecg_raw, NFFT=1024, Fs=fs, label='raw ECG PSD') 
frequencies, psd_filtered = plt.psd(output_signal_second_filter, NFFT=1024, Fs=fs, label='Filtered ECG PSD') 
plt.xlabel('Frequency (Hz)') 
plt.ylabel('Power/Frequency (dB/Hz)') 
plt.title('Power Spectral Density of Raw and 2nd Filter ECG Signal') 
plt.legend()

#if you want to print a few lines of the data to view it
print(d.head())