# -*- coding: utf-8 -*-
"""
Created on Thu May 11 11:20:21 2023

@author: hao
"""

import matplotlib.pyplot as plt #For plotting
from math import sin, pi #For generating input signals
import sys #For reading command line arguments

### Filter - 6KHz->8Khz Bandpass Filter
### @param [in] input - input unfiltered signal
### @param [out] output - output filtered signal
fs = 20000

def filter(x):
    y = [0]*fs
    for n in range(4, len(x)):
        y[n] = 0.0101*x[n] - 0.0202*x[n-2] + 0.0101*x[n-4] 
        + 2.4354*y[n-1] - 3.1869*y[n-2] + 2.0889*y[n-3] - 0.7368*y[n-4] 
    return y

###Read in desired frequency from command line
frequency = 10000 # int(sys.argv[1])

# In[]
### Create empty arrays
input = [0]*fs
output = [0]*fs

### Fill array with xxxHz signal
for i in range(fs):
    input[i] = sin(2 * pi * frequency * i / fs) #+ sin(2 * pi * 70 * i / 48000)

### Run the signal through the filter
output = filter(input)

### Grab samples from input and output #1/100th of a second
output_section = output[0:int(fs/100)]  
input_section = input[0:int(fs/100)] 

### Plot the signals for comparison
plt.figure(1)                
plt.subplot(211)   
plt.ylabel('Magnitude')
plt.xlabel('Samples') 
plt.title('Unfiltered Signal')      
plt.plot(input_section)
plt.subplot(212)             
plt.ylabel('Magnitude')
plt.xlabel('Samples') 
plt.title('Filtered Signal')
plt.plot(output_section)
plt.show()