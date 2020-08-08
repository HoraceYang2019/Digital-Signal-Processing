import numpy as np
import scipy.signal as signal
from numpy.fft import fft, fftshift, fftfreq
import matplotlib.pyplot as plt

t = np.linspace( 0, 1, 1000, endpoint = False )
x = signal.chirp( t, 0, 1, 100, 'linear' )

f = fftshift( fftfreq( 1000, 0.001 ) )
X = fftshift( fft( x ) ) 
Xm = abs( X )  

plt.plot( f, Xm )
plt.xlabel( 'f' )
plt.ylabel( 'Magnitude' )

plt.show( )