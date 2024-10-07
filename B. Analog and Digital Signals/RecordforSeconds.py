# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 17:27:31 2020

@author: hao
"""
# Receive audio data from a microphone, play and plot it
import pyaudio
import time
import matplotlib.pyplot as plt
import numpy as np

WIDTH = 2              #number of audio channels (1 for mono, 2 for stereo).
CHANNELS = 2
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5

pa = pyaudio.PyAudio()

# open a microphone as an input stream 
stream = pa.open(format=pa.get_format_from_width(WIDTH), channels=CHANNELS,rate=RATE, 
                input=True, output=True, frames_per_buffer=CHUNK)

stream.start_stream()
frames = []
print('now start!')
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    
    data = stream.read(CHUNK)  # read audio data from the stream (microphone)
   # stream.write(data, CHUNK)  # write the audio data to the stream (speaker)
    frames.append(data)
    #x = np.frombuffer(data, np.int16 )outout2.wav
    #plt.clf( )
    #plt.plot(x)
    #plt.axis([ 0, CHUNK, -30000, 30000 ])
    #plt.pause(0.1)
print('now stop!')
stream.stop_stream()
stream.close()

pa.terminate()

# In[]
# Save the frames
import wave

filename = str(input( "Please enter file name [default: output.wav] : ") or 'output.wav')
wav = wave.open(filename, 'wb')
wav.setnchannels(CHANNELS)
wav.setsampwidth(WIDTH)
wav.setframerate(RATE)
wav.writeframes(b''.join(frames))
wav.close()