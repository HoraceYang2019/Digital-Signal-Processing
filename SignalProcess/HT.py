# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, chirp

# In[1]: Generate signal
'''
scipy.signal.chirp(t, f0, t1, f1, method='linear', phi=0, vertex_zero=True): Frequency-swept cosine generator
t:  Times at which to evaluate the waveform.
f0: Frequency (e.g. Hz) at time t=0.
t1: Time at which f1 is specified.
f1: Frequency (e.g. Hz) of the waveform at time t1.
method : {‘linear’, ‘quadratic’, ‘logarithmic’, ‘hyperbolic’}, optional
    Kind of frequency sweep. If not given, linear is assumed.
phi: Phase offset, in degrees. Default is 0.
vertex_zero: This parameter is only used when method is ‘quadratic’. 
    It determines whether the vertex of the parabola that is the graph of the frequency is at t=0 or t=t1.
'''
duration = 1.0 # in second
fs = 400.0     # sampling rate smaples/sec
samples = int(fs*duration)  # sample size
t = np.arange(samples) / fs # time-stamp

signal = chirp(t, 20.0, t[-1], 100.0)
signal *= (1.0 + 0.5 * np.sin(2.0*np.pi*3.0*t) )

# In[2]: Apply HT to filter noise
'''
scipy.signal.hilbert(x, N=None, axis=-1)
x : array_like Signal data. Must be real.
N : int, optional. Number of Fourier components. Default: x.shape[axis]
axis : int, optional. Axis along which to do the transformation. Default: -1.
'''
analytic_signal = hilbert(signal)
amplitude_envelope = np.abs(analytic_signal)
instantaneous_phase = np.unwrap(np.angle(analytic_signal))
instantaneous_frequency = np.diff(instantaneous_phase) / (2.0*np.pi) * fs

fig = plt.figure()
ax0 = fig.add_subplot(211)
ax0.plot(t, signal, label='signal')
ax0.plot(t, amplitude_envelope, label='envelope')
ax0.set_xlabel("time in seconds")
ax0.legend()

ax1 = fig.add_subplot(212)
ax1.plot(t[1:], instantaneous_frequency)
ax1.set_xlabel("time in seconds")
ax1.set_ylim(0.0, 120.0)

plt.show()

# In[3]: Question
'''
What features can be extracted by HT?
'''


