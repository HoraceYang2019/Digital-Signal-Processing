import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt

t = np.linspace( 0, 1, 200, endpoint = False )	# 定義時間陣列	
x = 10 * np.cos( 2 * np.pi * 5 * t )			# 原始訊號

n1 = random.uniform( -1, 1, 200 )				# 布朗尼雜訊
ns = np.cumsum( n1 )
mean = np.mean( ns )	
max = np.max( np.absolute( ns - mean ) )
noise = ( ns - mean ) / max

y = x + noise

plt.figure( 1 )
plt.plot( t, x )
plt.xlabel( 't (second)' )
plt.ylabel( 'Amplitude' )
plt.axis( [ 0, 1, -12, 12 ] )

plt.figure( 2 )
plt.plot( t, noise )
plt.xlabel( 't (second)' )
plt.ylabel( 'Amplitude' )
plt.axis( [ 0, 1, -1, 1 ] )

plt.figure( 3 )
plt.plot( t, y )
plt.xlabel( 't (second)' )
plt.ylabel( 'Amplitude' )
plt.axis( [ 0, 1, -12, 12 ] )

plt.show( )