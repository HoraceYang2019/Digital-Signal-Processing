from scipy.io.wavfile import read
import matplotlib.pyplot as plt

filename = input( "Please enter file name: " )
sampling_rate, x = read( filename )

plt.plot( x )
plt.xlabel( 'n' )
plt.ylabel( 'Amplitude' )

plt.show( )