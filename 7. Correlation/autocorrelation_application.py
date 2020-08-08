import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt

def autocorr( x ):
	R = np.correlate( x, x, 'full' )
	return R[ int( R.size / 2 ) : ]

def main( ):
	t = np.linspace( 0, 1, 	100, endpoint = False )		# 定義時間陣列
	x = 10 * np.cos( 2 * np.pi * 5 * t )				# 原始訊號
	noise = random.uniform( -2, 2, 100 )				# 雜訊(均勻分佈)	
	y = x + noise

	auto_corr1 = autocorr( x )
	auto_corr2 = autocorr( noise )
	auto_corr3 = autocorr( y )

	plt.figure( 1 )										
	plt.subplot( 121 )
	plt.plot( t, x )
	plt.xlabel( 't (second)' )
	plt.ylabel( 'Amplitude' )

	plt.subplot( 122 )									
	plt.plot( auto_corr1 )
	plt.xlabel( 'Lag' )
	plt.ylabel( 'Auto Correlation' )

	plt.figure( 2 )
	plt.subplot( 121 )
	plt.plot( t, noise )
	plt.xlabel( 't (second)' )
	plt.ylabel( 'Amplitude' )

	plt.subplot( 122 )
	plt.plot( auto_corr2 )
	plt.xlabel( 'Lag' )
	plt.ylabel( 'Auto Correlation' )

	plt.figure( 3 )	
	plt.subplot( 121 )
	plt.plot( t, y )
	plt.xlabel( 't (second)' )
	plt.ylabel( 'Amplitude' )

	plt.subplot( 122 )
	plt.plot( auto_corr3 )
	plt.xlabel( 'Lag' )
	plt.ylabel( 'Auto Correlation' )

	plt.show( )
	
main( )