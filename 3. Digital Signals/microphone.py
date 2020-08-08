import numpy as np
import pyaudio
import matplotlib.pyplot as plt

fs = 11000
CHUNK = 1024
pa = pyaudio.PyAudio( )
stream = pa.open( format = pyaudio.paInt16, channels = 1, rate = fs, 
				  input = True, output = False, frames_per_buffer = CHUNK )
			
try:			
	while True:						
		data = stream.read( CHUNK )
		x = np.fromstring( data, np.int16 )

		plt.clf( )
		plt.plot( x )
		plt.axis( [ 0, CHUNK, -30000, 30000 ] )

		plt.pause( 0.1 )

except KeyboardInterrupt:
	print( "Quit" )
	pa.close( stream )
	quit( )