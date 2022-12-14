import pyaudio
import numpy as np
import wave
from tuning import Tuning
import usb.core
import usb.util
import time
import serial
import os
from subprocess import Popen
import keras
import librosa

RESPEAKER_RATE = 16000
RESPEAKER_CHANNELS = 6 # change base on firmwares, 1_channel_firmware.bin as 1 or 6_channels_firmware.bin as 6
RESPEAKER_WIDTH = 2 # run getDeviceInfo.py to get index
RESPEAKER_INDEX = 2  # refer to input device id
CHUNK = 1024
RECORD_SECONDS = 5 # longer time recording increases accuracy of classification
WAVE_OUTPUT_FILENAME = "output.wav"
TEXT = True

p = pyaudio.PyAudio()
dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
reconstructed_model = keras.models.load_model("trained_model")
classes = ['dog', 'chirping_birds', 'vacuum_cleaner', 'thunderstorm', 'door_wood_knock', 'can_opening', 'crow', 'clapping', 'fireworks', 'chainsaw', 'airplane', 'mouse_click', 'pouring_water', 'train', 'sheep', 'water_drops', 'church_bells', 'clock_alarm', 'keyboard_typing', 'wind', 'footsteps', 'frog', 'cow',
'brushing_teeth', 'car_horn', 'crackling_fire', 'helicopter', 'drinking_sipping', 'rain', 'insects', 'hen', 'engine', 'breathing', 'crying_baby', 'hand-saw', 'glass_breaking', 'toilet_flush', 'pig', 'washing_machine', 'sea_waves', 'siren', 'cat', 'door_wood_creaks', 'crickets', 'coughing', 'sneezing',
'snoring', 'laughing', 'sigh', 'sniff', 'throatclearing', 'phone', 'speech']
directions = ['m_left', 'left', 'forward','right', 'm_right']

def predict():
    y,sr=librosa.load(WAVE_OUTPUT_FILENAME)
    S = np.mean(np.abs(librosa.stft(y)))
    if(S > 0.005):
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=36).T,axis=0)
        x_train=np.array(mfccs)
        x_train=np.reshape(x_train,(1, 9, 4, 1))
        prediction = reconstructed_model.predict(x_train)
        prediction_label=np.argmax(prediction)  
        
        Mic_tuning = Tuning(dev)

        if (Mic_tuning.direction > 180 and Mic_tuning.direction < 365) :
            if (Mic_tuning.direction < 270):
                turn_direction = 4
            elif (Mic_tuning.direction > 350):
                turn_direction = 2
            else:
                turn_direction = 3
        else:
            if (Mic_tuning.direction > 90):
                turn_direction = 0 
            elif (Mic_tuning.direction < 15):
                turn_direction = 2
            else:
                turn_direction = 1

        if (TEXT == True):
            prediction_label=classes[prediction_label]
            turn_direction = directions[turn_direction]
        
        print(turn_direction, prediction_label)
    else:
        print('X','X')

stream = p.open(
        rate=RESPEAKER_RATE,
        format=p.get_format_from_width(RESPEAKER_WIDTH),
        channels=RESPEAKER_CHANNELS,
        input=True,
        input_device_index=RESPEAKER_INDEX)

while(True):   

    frames = []
    stream.start_stream()

    for i in range(0, int(RESPEAKER_RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
   
    # Create record file
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(RESPEAKER_CHANNELS)
    wf.setsampwidth(p.get_sample_size(p.get_format_from_width(RESPEAKER_WIDTH)))
    wf.setframerate(RESPEAKER_RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    predict()
    # if delay is required: time.sleep(1)

stream.close()
p.terminate()