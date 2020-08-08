import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

fs = 1000
t = np.linspace( 0, 1, fs )

x = np.array( [ ] )
for i in range( 10 ):
	segment = np.cos( 2 * np.pi * ( ( i + 1 ) * 20 ) * t )
	x = np.append( x, segment )

f, t, Zxx = signal.spectrogram( x, fs )
	
plt.pcolormesh( t, f, abs(Zxx) )
plt.xlabel( 'Time(Second)' )
plt.ylabel( 'Frequency(Hz)' )

plt.show( )