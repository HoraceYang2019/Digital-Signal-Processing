import numpy as np
import numpy.random as random
import scipy.signal as signal
import matplotlib.pyplot as plt

fs = 1000
t = np.linspace( 0, 1, fs, endpoint = False )
x = 10 * np.cos( 2 * np.pi * 100 * t ) + 5 * np.cos( 2 * np.pi * 200 * t ) 
noise = random.uniform( -1, 1, fs )	
y = x + noise

f1, pxx1 = signal.periodogram( y, fs )
f2, pxx2 = signal.welch( y, fs )

lp1 = 10 * np.log10( pxx1 )
lp2 = 10 * np.log10( pxx2 )

plt.figure( 1 )
plt.plot( f1, lp1 )
plt.xlabel( 'frequency (Hz)' )
plt.ylabel( 'PSD' )
plt.axis( [ 0, 500, -50, 10 ] )

plt.figure( 2 )
plt.plot( f2, lp2 )
plt.xlabel( 'frequency (Hz)' )
plt.ylabel( 'PSD' )
plt.axis( [ 0, 500, -40, 10 ] )

plt.show( )