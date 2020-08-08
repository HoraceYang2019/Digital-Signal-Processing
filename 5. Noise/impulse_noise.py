import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt

amplitude = eval( input( "Enter amplitude of impulse noise: " ) )
probability = eval( input( "Enter probability of impulse noise(%): " ) )

t = np.linspace( 0, 1, 200, endpoint = False )		# 定義時間陣列
x = 10 * np.cos( 2 * np.pi * 5 * t )				# 原始訊號

noise = np.zeros( x.size )							# 脈衝雜訊
for i in range( x.size ):
	p1 = random.uniform( 0, 1 )
	if p1 < probability / 100:
		p2 = random.uniform( 0, 1 )
		if p2 < 0.5:
			noise[i] = amplitude
		else:
			noise[i] = -amplitude
		
y = x + noise
			
plt.figure( 1 )										
plt.plot( t, x )
plt.xlabel( 't (second)' )
plt.ylabel( 'Amplitude' )
plt.axis( [ 0, 1, -12, 12 ] )

plt.figure( 2 )
plt.stem( t, noise )
plt.xlabel( 't (second)' )
plt.ylabel( 'Amplitude' )

plt.figure( 3 )
plt.plot( t, y )
plt.xlabel( 't (second)' )
plt.ylabel( 'Amplitude' )
plt.axis( [ 0, 1, -15, 15 ] )

plt.show( )