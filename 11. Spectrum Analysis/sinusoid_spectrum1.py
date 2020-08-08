import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt

t = np.linspace( 0, 1, 1000, endpoint = False )
x = np.cos( 2 * np.pi * 100 * t ) 

X = fft( ( x ) ) 
Xm = abs( X )  

plt.plot( Xm )
plt.xlabel( 'k' )
plt.ylabel( 'Magnitude' )

plt.show( )