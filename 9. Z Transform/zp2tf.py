import numpy as np
import scipy.signal as signal

z = np.array( [ -0.8, 1 ] )
p = np.array( [ 0.6 + 0.8j, 0.6 - 0.8j, -1 ] )
k = 0.8

b, a = signal.zpk2tf( z, p, k )

print( "Numerator Polynomial Coefficients =", b )
print( "Denominator Polynomial Coefficients =", a )