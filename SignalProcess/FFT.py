'''
#===============================================================================
 for Rasbian: 
 sudo apt-get install python3-pip
 sudo pip3 install numpy
 sudo apt-get install python3-matplotlib
----------------------------------------------
 sudo apt-get upgrade
===============================================================================
===============================================================================
 for Windows:
 cd ./python/python36/scripts/
 pip install numpy
#===============================================================================
''' 
import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt

# In[1]: generate signals and calculate fft values 

N = 2000      # no. of data points
sRate = 200  # sampling rate
T = 1/sRate   # sampling period

t = np.linspace(0.0, N*T, N) #

w0 = 2    # noise weighting
bias = 0
f1 = 20.0 # in Hz
w1 = 5    # weight     t
f2 = 10.0
w2 = 0.5
x = bias + w1*np.sin(f1*2.0*np.pi*t) + w2*np.sin(f2*2.0*np.pi*t) + w0*np.random.random(N)
y = fft(x)
f = np.linspace(0.0, 1.0/(2.0*T), N/2)

# In[2]: plot
plt.figure(1)
plt.subplot(211)  # plot subplot
plt.plot(t, x, lw=2)
plt.ylabel('Amplitude')
plt.xlabel('Time')
plt.grid(True)
plt.subplot(212)
plt.plot(f, 2.0/N*abs(y.real[0:int(N/2)])) 
plt.ylabel('Amplitude')
plt.xlabel('Frequency')
plt.grid(True)
plt.show()

# In[3]: Questions
'''
1. What is the SNR?
2. Can we extract signals from high noises by fft? 
'''