import numpy as np
import wave
from scipy.io.wavfile import read, write
import struct
import scipy.signal as signal
import matplotlib.pyplot as plt

infile  = input( "Input File: " )	
fs, x = read( infile )	
f, t, Zxx = signal.spectrogram( x, fs )

plt.pcolormesh( t, f, abs( Zxx ) )
plt.xlabel( 'Time(Second)' )
plt.ylabel( 'Frequency(Hz)' )

plt.show( )