"""
Created on Thu Mar 28 17:03:24 2019

Usgae:
    1. signal simulation
        gmode = 0
        weightings: w0, w1, and w2
        bias: bias
        frequency: f1, f2
        time interval: gt
        
    2. apply real data
        gmode = 1
        sampling rate: sRate
Reference: 
    https://morvanzhou.github.io/tutorials/data-manipulation/plt/3-3-contours/
    https://in.mathworks.com/help/wavelet/examples/wavelet-packets-decomposing-the-details.html
    http://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/
"""

import pywt
import matplotlib.pyplot as plt
from numpy import square
import numpy as np
import pandas as pd
import Denoise as de
import FT as ft 

# In[1]: variable defintion

output_path = './Output\\'
source_path = './Source\\testSignal.csv'

# In[2]: to understand what is wavlet packets?
def getSignal(source_path='', sRate=1000, gmode=0):
    if gmode == 0:       
        N = 2000  # no. of data points
        t = np.linspace(0.0, N/sRate, N) #
        w0 = 0    # noise weighting
        bias = 0      
        w1 = 5   
        f1 = 50.0 # in Hz     t

        w2 = 2
        f2 = 50.0
        gen = map(lambda k: w1*np.cos(2.0*np.pi*f1*k) if 0.4 <= k < 0.6 else w2*np.sin(2*np.pi*f2*k), t)
        s0 = np.array(list(gen))

        s = s0 + w0 * np.random.rand(len(t)) + bias
    else:
        s = pd.read_csv(source_path,index_col=None,header = None).values[:,0]
        N = len(s)    
        t = np.linspace(0.0, N/sRate, N) # time 
        
    return t,s

# In[3]: apply wavelet package to signal with a given level     
def wp_(t, s, waveletFun='db3', level=3):
    
#    for family in pywt.families():  # show wt family
#        print('%s family: '%(family) + ','.join(pywt.wavelist(family)))     
    
    wp = pywt.WaveletPacket(data=s, wavelet=waveletFun, maxlevel=level)
    # access list of node names
    node_space = [node.path for node in wp.get_level(level, 'freq')]
    wp_E = []  
    
    # calculate energes of wavelet packages
    for node in node_space:
        E = sum((square(wp['%s'%node].data))) # accumulate the energies by nodes
        wp_E.append(E)
    
    nwp = range(1,len(node_space)+1,1) # wp axis
        
    rowNo = 1+ len(node_space)/2
    plt.subplot(rowNo, 2, 1);  plt.plot(t,s)
    plt.title("Time"); plt.xlabel("sec"); plt.ylabel('amp')

    plt.subplot(rowNo,2,2);  
    plt.bar(nwp[:],np.array(wp_E)[:], edgecolor="black", lw=2)
    plt.title("Wavelet energy level%s"%level); plt.xlabel("nodes"); plt.ylabel('amp') 
    for i in nwp:
        plt.subplot(rowNo,2,2+i);  
        plt.plot(wp[node_space[i-1]].data); plt.title('node #%d (%s)'%(i,node_space[i-1]))
        
    #plt.figure(figsize=(3,3))
    plt.show()
        
    return wp_E
               
# In[4]        
if __name__ == '__main__':
    # gmode: using simulation; gmode=1: using csv file
    t,s = getSignal(sRate=1000, gmode=0)  
    wp_(t,s,level=4)  # calculate wavelet packages with level 3 
    
# In[5]: using actual data
    sRate = 2000  # sampling rate
    t,s = getSignal(source_path, sRate, gmode=1) # retreieve data from file  
    r = [0,len(t)]  # specify the analysis range
   # r = [0,100000]
   
    ft.fft_(s[r[0]:r[1]], sRate) # call fft function   
    wp_(t[r[0]:r[1]],s[r[0]:r[1]], waveletFun='db3', level=4)  # calculate wavelet packages with level n
               
# In[5]: apply low pass filter by wden with threshold
    ds = de.lowpassfilter(s, thresh = 0.3, wavelet="db4") # 
    wp_(t,ds, waveletFun='db3', level=4) 
               
# In[6]: apply wden filter with soft wavelets 
    # wden(X,TPTR,SORH,SCAL,N,'wname')
    #    TPTR: threshold selection rule, {rigrsure','heursure','sqtwolog', 'minimaxi'}
    #    SORH: ('soft' or 'hard') is for soft or hard thresholding
    #    SCAL: defines multiplicative threshold rescaling
    #        'one' for no rescaling
    #        'sln' for rescaling using a single estimation of level noise based on first-level coefficients
    #        'mln' for rescaling done using level-dependent estimation of level noise
    #    N: decomposition level N
    #    'wname': desired orthogonal wavelet, {'db1'-'db45, 'sym2'-'sym45',...}
    ds = de.wden(s, 'sqtwolog', 'soft', 'mln', 5, 'coif5') # 
    wp_(t,ds, waveletFun='db3', level=4) 
               
# In[6]    
    ds = de.wden(s, 'sqtwolog', 'soft', 'mln', 2, 'sym7') # 
    wp_(t,ds, waveletFun='db3', level=4)
