import numpy as np
import numpy.random as random
import scipy.signal as signal
import matplotlib.pyplot as plt

t = np.linspace( 0, 1, 200, endpoint = False )		
x = 10 * np.cos( 2 * np.pi * 5 * t ) + random.uniform( -5, 5, 200 )

sigma = 3                             			# 標準差
filter_size = 6 * sigma + 1                		# 濾波器大小
gauss = signal.gaussian( filter_size, sigma )	# 濾波器係數
sum = np.sum( gauss )                   		# 正規化
gauss = gauss / sum

y = np.convolve( x, gauss, 'same' )           

plt.figure( 1 )
plt.plot( t, x )
plt.xlabel( 't (second)' )
plt.ylabel( 'Amplitude' )

plt.figure( 2 )
plt.plot( t, y )
plt.xlabel( 't (second)' )
plt.ylabel( 'Amplitude' )

plt.show( )
