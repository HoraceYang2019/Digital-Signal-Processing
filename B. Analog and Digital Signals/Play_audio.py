# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 17:14:40 2020

@author: hao
"""

import pyaudio
import wave
import sys
# In[]
import wave

filename = input( "Please enter file name: " ) # for example: r2d2.wav
wav = wave.open( filename, 'rb' )   # read the specified file  

num_channels = wav.getnchannels()   # number of audio channels (1 for mono, 2 for stereo).
sampwidth	= wav.getsampwidth()	# sample width in bytes.
frame_rate	= wav.getframerate()	# sampling frequency
num_frames	= wav.getnframes()		# number of audio frames 
comptype	= wav.getcomptype()	    # compression type ('NONE' is the only supported type). 
compname	= wav.getcompname()	    # Human-readable version of getcomptype(). Usually 'not compressed' parallels 'NONE'.

print( "Number of Channels =", num_channels )
print( "Sample Width =", sampwidth )
print( "Sampling Rate =", frame_rate )
print( "Number of Frames =", num_frames )
print( "Comptype =", comptype )
print( "Compname =", compname )

wav.close( )
# In[]
CHUNK = 1024

# if len(sys.argv) < 2:
#     print("Plays a wave file.\n\nUsage: %s filename.wav" % sys.argv[0])
#     sys.exit(-1)

wf = wave.open('output.wav', 'rb')

p = pyaudio.PyAudio()

stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)

data = wf.readframes(CHUNK) # read stream by frames

while data != b'':
    stream.write(data)
    data = wf.readframes(CHUNK)

# In[]
stream.stop_stream()
stream.close()

p.terminate()