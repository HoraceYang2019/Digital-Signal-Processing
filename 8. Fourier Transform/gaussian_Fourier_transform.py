import numpy as np
import matplotlib.pyplot as plt

sigma = eval( input( "Enter sigma: " ) )

t = np.linspace( -7, 7, 100 )
x = np.exp( - ( t * t ) / ( 2 * sigma * sigma ) )

w = np.linspace( -7, 7, 1000 )
X = np.exp( - ( sigma * sigma * w * w ) / 2 )
X = np.sqrt( 2 * np.pi * sigma * sigma ) * X

plt.figure( 1 )
plt.plot( t, x )
plt.xlabel( 't' )
plt.ylabel( 'x(t)' )

plt.figure( 2 )
plt.plot( w, X )
plt.xlabel( r'$\omega$' )
plt.ylabel( r'X($\omega$)' )

plt.show( ) 