# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 11:24:31 2021
pip install pyaudio
Note: 
    Fix.1 the remainder of the recording frame 
    Add.1 Add Fade_In and Fade_Out functions
Question: 
   
@author: hao
"""
import wave
import pyaudio
import numpy as np
import matplotlib.pyplot as plt

# In[0]
record_seconds = 5
# set the chunk size of 1024 samples
chunk_size = 1024
# sample format
audio_format = pyaudio.paInt32
# mono, change to 2 if you want stereo
channel_no = 2
# 44100 samples per second
sample_rate = 44100

# In[1]: 
def init_Audio(channels = channel_no):    
    # initialize PyAudio object
    p = pyaudio.PyAudio()
    # open stream object as input & output
    stream = p.open(format = audio_format,
                    channels = channel_no, 
                    rate = sample_rate,
                    input = True,
                    output = True,
                    frames_per_buffer = chunk_size)
    return p, stream

# In[2]: generate audio by key and duration 
# stream: handle of audio; t: time duration; key: piano key
def play_key(stream, duration=0.5, key = [49, 61, 73], mode='Normal'):  
    
    data = []
    active = 0.5 

    for i in range(len(key)):
        t = np.linspace(0, int(key[i]*duration)*channel_no, 
                        int(sample_rate*duration)*channel_no)
        point = int(len(t)*active)
        part1 = t[:point]
        part2 = t[point:]
        # standard piano frquency by key
        frequency = pow(pow(2, 1/12), key[i] - 49)*440
        
        signal = np.sin(2 * np.pi * frequency * part1) 
        
        # Add.1
        if mode == 'Normal':     
            Amp = np.ones(len(signal))
        else:
            if mode == 'Fade_In':
                Amp = np.linspace( 0, 1, len(signal), endpoint = False )
            else:
                if mode == 'Fade_Out':
                    Amp = np.linspace( 1, 0, len(signal), endpoint = False ) 
                
        signal = Amp*signal
        # Add.1      
        
        data.append(signal)  #single
        data.append(np.sin(2 * np.pi * 0 * part2))

    output = np.concatenate(data[:]).astype(np.float32)
    
    stream.write(output.tobytes())
    return output

# In[3]: Record the audio 

def reording(p, stream, filename = 'recorded.wav', record_seconds = 5):    
    frames = []
    print(f'Recording for {record_seconds} secs')
    for i in range(int(sample_rate / chunk_size * record_seconds)):
        data = stream.read(chunk_size)
        # if you want to hear your voice while recording
        frames.append(data)
    print(f'Finished {i+1} frames recording.')

    # Fix.1: compensate the tail of the frames
    tail = bytes(np.zeros(record_seconds*sample_rate 
                            - len(frames)*chunk_size))
    frames.append(tail)
    # Fix.1: for compensation
    
    # open the file in 'write bytes' mode
    wf = wave.open(filename, "wb")
    # set the channels
    wf.setnchannels(channel_no)
    # set the sample format
    wf.setsampwidth(p.get_sample_size(audio_format))
    # set the sample rate
    wf.setframerate(sample_rate)
    # write the frames as bytes
    wf.writeframes(b"".join(frames))
    # close the file
    wf.close()

# In[4]: end the audio opreation    
def close_Audio(p, stream):
    # stop and close stream
    stream.stop_stream()
    stream.close()
    # terminate pyaudio object
    p.terminate()
    # save audio file

# In[5]: read the audio file
def get_audio(filename = 'recorded.wav', play=True):
    # read the target file
    f = wave.open(filename,"rb")
    # get the audio parameters
    params = f.getparams()
    # get the specific parameters
    nchannels, sampwidth, framerate, nframes = params[:4] 
    str_data = f.readframes(nframes)
    if play == True:
        play_audio(stream, str_data)

    # transfer the audio object into an array
    wave_data = np.frombuffer(str_data, dtype = np.int32)
    # reshape the shape from (*)x1 to (*/2)x2
    #wave_data.shape = -1, nchannels
    # trasport data from (*/2)x2 to 2x(*/2)
    wave_data = wave_data.T
    # assign time interval
    time = np.arange(0, nframes*nchannels) * (1.0/(framerate*nchannels))
    f.close()
    return time, wave_data

# In[6]
def play_audio(stream, stream_data, chunk=chunk_size):
    for i in range(int(len(stream_data)/chunk)+1): 
        data = stream_data[i*chunk:(i+1)*chunk]
        stream.write(data)    # play the audio by chunk  
        
# In[7]: show the audio data
def plot_audio(time, data):
    # plot one channel of audio
    plt.plot(time, data)
    plt.xlabel('time (sec)')
    plt.ylabel('Amplitude')
    plt.show()

# In[8]
if __name__ == '__main__':
    p, stream = init_Audio()
    duration=1; key = [49, 61, 73, 85, 106]
    
    gen_data = play_key(stream, duration, key, mode='Fade_Out') # generate background voice
    plt.plot(gen_data) # length of gen_data: 441000
    
    reording(p, stream)
    time, rec_data = get_audio(play=True) # record front voice
    # length of rec_data: 441000
    plot_audio(time, rec_data)
    
    print(f'gen_data: {gen_data.shape}, rec_data: {rec_data.shape}')
    # the lengths of two data are the same 
    com_data = gen_data[:len(rec_data)]*100000000 + rec_data*1 
    
    plt.plot(com_data)
    play_audio(stream, bytes(np.array(com_data, np.int32))) # 
    close_Audio(p, stream)