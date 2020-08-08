import numpy as np

def autocorr( x ):
	R = np.correlate( x, x, 'full' )
	return R[ int( R.size / 2 ) : ]

def main( ):
	x = np.array( [ 1, 2, 1, 2, 1 ] )
	R = autocorr( x )
	print( "x =", x )
	print( "Autocorrelation =", R )
	
main( )