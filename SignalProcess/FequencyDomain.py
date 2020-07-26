# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 21:00:32 2018

@author: hao
"""
import numpy as np
import matplotlib.pyplot as plt

# In[]
N = 1000      # no. of data points
sRate = 1000  # sampling rate
T = 1/sRate   # sampling period

w0 = 0    # noise weighting
bias = 0
f1 = 200.0 # in Hz
w1 = 1    # weight     t
f2 = 15.0
w2 = 0.5
t = np.linspace(0.0, N*T, N)

gen = map(lambda k: w1*np.cos(2.0*np.pi*f1*k) 
            if 0.4 <= k < 0.6 else w2*np.sin(2*np.pi*f2*k), t)
x0 = np.array(list(gen))
f = np.linspace(0.0, 1.0/(2.0/sRate), N/2)
x = x0 + w0 * np.random.rand(len(t)) + bias

f = np.linspace(0.0, 1.0/(2.0*T), len(x)) 
band = 10
total_band = 10

# In[]
def getBand(p, bandWidth, totalBand):
    ps = np.abs(np.power(p[1:int(len(s)/2)],2)) # power spectrum 
    maxFreq =  min(bandWidth * totalBand, len(ps))
        
    # power bands
    pb = [sum(ps[s:s+bandWidth]) for s in range(0, maxFreq, bandWidth)]
    pb = pb + np.zeros(totalBand) 
    return (pb[0:totalBand])

def getFreqFeature(t, s, f):
    fp = f[1:int(len(s)/2)]
    p = np.fft.fft(s)  #calculate fft
    py = np.abs(p[1:int(len(s)/2)]) #  amplitude spectrum, DCcomponent p[0] is removed  
    showFreq(t, s, fp,py)    
    
    mean = py.mean()
    asum = np.sum(py)
    
    order2 = np.mean(np.power((py-mean),2))
    change = np.sum(fp*py)/asum # frequency change
    
    rms = np.sqrt(np.sum((fp**2)*py)/asum) # rms
    bands = getBand(p, band, total_band)
      
    P = [mean, order2, change, rms, bands]
    return (P)

def showFreq(t, x, f, y):
    plt.figure(1)
    plt.subplot(211)  # plot subplot
    plt.plot(t, x, lw=2)
    plt.ylabel('Amplitude')
    plt.xlabel('Time')
    plt.grid(True)
    plt.subplot(212)
    plt.plot(f, y) 
    plt.ylabel('Amplitude')
    plt.xlabel('Frequency')
    plt.grid(True)
    plt.show()
  
# In[]
  
y=getFreqFeature(t, x, f)

# In[3]: Questions
'''
1. What purposes of the bands used for?
2. What differences of fetures between ppt and codes? 
'''