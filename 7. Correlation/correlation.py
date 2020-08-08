import numpy as np

x = np.array( [ 1, 2, 4, 3, 2, 1, 1 ] )
h = np.array( [ 1, 2, 3, 1, 1 ] )
y = np.correlate( x, h, 'full' )
y1 = np.correlate( x, h, 'same' )

print( "x =", x )
print( "h =", h )
print( "Full Correlation y =", y )
print( "Correlation y =", y1 )