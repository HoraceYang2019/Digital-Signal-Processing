# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 17:24:27 2020

@author: hao
"""
# record and save a new audio file 
import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import wave

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

pa = pyaudio.PyAudio( )
stream = pa.open( format = FORMAT, channels = CHANNELS, rate = RATE, \
                 input = True, output = False, frames_per_buffer = CHUNK )
frames = []
stream.start_stream()

try:
    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        x = np.fromstring( data, np.int16 )
        plt.clf( )
        plt.plot( x )
        plt.axis( [ 0, CHUNK, -30000, 30000 ] )
        plt.pause( 0.1 )

except KeyboardInterrupt:
    print( "Quit" ) 
    pa.close( stream )
    quit( )

# Save the frames
filename = str(input( "Please enter file name [default: output.wav] : ") or 'output.wav')
wav = wave.open(filename, 'wb')
wav.setnchannels(CHANNELS)
wav.setsampwidth(pa.get_sample_size(FORMAT))
wav.setframerate(RATE)
wav.writeframes(b''.join(frames))
wav.close()