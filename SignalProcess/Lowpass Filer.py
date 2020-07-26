# -*- coding: utf-8 -*-
'''
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
'''

import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt

# In[1]: call low pass filter with cut-off frequency
'''
scipy.signal.butter(N, Wn, btype='low', analog=False, output='ba')
N: The order of the filter.
Wn: A scalar or length-2 sequence giving the critical frequencies. 
btype: {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}, optional
analog: bool, optional, When True, return an analog filter, otherwise a digital filter is returned.
output : {‘ba’, ‘zpk’, ‘sos’}, optional
Type of output: numerator/denominator (‘ba’), pole-zero (‘zpk’), or second-order sections (‘sos’). Default is ‘ba’.
'''
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# In[2]: Filter model
order = 6
fs = 60.0       # sample rate, Hz
cutoff = 3.67  # desired cutoff frequency of the filter, Hz

# Get the filter coefficients so we can check its frequency response.
b, a = butter_lowpass(cutoff, fs, order)

# Plot the frequency response.
'''
scipy.signal.freqz(b, a=1, worN=None, whole=0, plot=None)
b: numerator of a linear filter (b/a)
a: denominator of a linear filter
worN : {None, int, array_like}, optional
    If None (default), then compute at 512 frequencies equally spaced around the unit circle. 
    If a single integer, then compute at that many frequencies. 
    If an array_like, compute the response at the frequencies given (in radians/sample).
whole : bool, optional
    Normally, frequencies are computed from 0 to the Nyquist frequency, pi radians/sample (upper-half of unit-circle). 
    If whole is True, compute frequencies from 0 to 2*pi radians/sample.
plot : callable
    A callable that takes two arguments. If given, the return parameters w and h are passed to plot. 
    Useful for plotting the frequency response inside freqz.
'''
w, h = freqz(b, a, worN=8000)

plt.subplot(2, 1, 1)
plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
plt.axvline(cutoff, color='k')

plt.xlim(0, 0.5*fs)
plt.title("Lowpass Filter Frequency Response")
plt.xlabel('Frequency [Hz]')
plt.grid()

# In[3]: apply the data to the filter 
# Demonstrate the use of the filter.
# First make some data to be filtered.
T = 5.0         # in seconds
n = int(T * fs) # total number of samples
t = np.linspace(0, T, n, endpoint=False)

# "Noisy" data.  We want to recover the 1.2 Hz signal from this.
data = np.sin(1.2*2*np.pi*t) + 10*np.cos(9*2*np.pi*t) + 10*np.sin(12.0*2*np.pi*t)

# Filter the data, and plot both the original and filtered signals.
y = butter_lowpass_filter(data, cutoff, fs, order)

plt.subplot(2, 1, 2)
plt.plot(t, data, 'b-', label='data')
plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
plt.xlabel('Time [sec]')
plt.grid()
plt.legend()

plt.subplots_adjust(hspace=0.35)
plt.show()

# In[4]: Question
'''
1. Order and cut-off frequency of the filter?
2. Existing phase lag?
'''