# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 10:03:03 2022

@author: hao
"""

# Complex signals in time
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft
# x: signal; sRate: sampling rate; is_plot: plot it?; f_start: = 1: remove DC 
def fft_(x=[], sRate=None, is_plot=True, case = 1, f_start=1):
    if x==[]:
        N = 2000  # no. of data points
        sRate = 1000.0  # sampling rate
        t = np.linspace(0.0, N/sRate, N) #
        w0 = 10    # noise weighting
        bias = -80
        f1 = 50.0 # in Hz
        w1 = 5    # weight     t
        f2 = 300.0
        w2 = 2
        if case == 1: 
            x = bias + w1*np.sin(f1*2.0*np.pi*t) + w2*np.sin(f2*2.0*np.pi*t)+ w0*np.random.random(N)
        else: 
            gen = map(lambda k: w1*np.cos(2.0*np.pi*f1*k) if 0.4 <= k < 0.6 else w2*np.sin(2*np.pi*f2*k), t)
            x0 = np.array(list(gen))
            f = np.linspace(0.0, 1.0/(2.0/sRate), int(N/2))
            x = x0 + w0 * np.random.rand(len(t)) + bias
     
    N = len(x)   # 資料長度    
    y=fft(x)
    y_mag = 2.0/N*abs(y.real[:int(N/2)])
    if is_plot== True :
        plt.figure(1)        
        plt.subplot(311)        
        t = np.linspace(0.0, N/sRate, N) # time 
        plt.plot(t, x, lw=2)
        plt.ylabel('Amplitude')
        plt.xlabel('Time')
        plt.grid(True)
        
        plt.subplot(312)
        f = np.linspace(0.0, 1.0/(2.0/sRate), int(N/2))
        plt.plot(f[f_start:], y_mag[f_start:]) 
        plt.ylabel('Amplitude')
        plt.xlabel('Frequency')
        plt.grid(True)
        
        plt.subplot(313)
        powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(x, Fs=sRate)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')        
        plt.show()  

    return y_mag
if __name__ == '__main__':
    fft_(case=1) # for testing