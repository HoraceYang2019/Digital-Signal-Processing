import numpy as np
import numpy.random as random
import scipy.signal as signal
import matplotlib.pyplot as plt

fs = 1000
t = np.linspace( 0, 1, fs, endpoint = False )
noise = random.uniform( -1, 1, fs )	

f1, pxx1 = signal.periodogram( noise, fs )
f2, pxx2 = signal.welch( noise, fs )

plt.figure( 1 )
plt.plot( f1, pxx1 )
plt.xlabel( 'frequency (Hz)' )
plt.ylabel( 'PSD' )

plt.figure( 2 )
plt.plot( f2, pxx2 )
plt.xlabel( 'frequency (Hz)' )
plt.ylabel( 'PSD' )

plt.show( )