import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from librosa import display
import librosa
import keras
from numpy import genfromtxt
from keras.utils.np_utils import to_categorical
from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout

classes = ['dog', 'chirping_birds', 'vacuum_cleaner', 'thunderstorm', 'door_wood_knock', 'can_opening', 'crow', 'clapping', 'fireworks', 'chainsaw', 'airplane', 'mouse_click', 'pouring_water', 'train', 'sheep', 'water_drops', 'church_bells', 'clock_alarm', 'keyboard_typing', 'wind', 'footsteps', 'frog', 'cow',
'brushing_teeth', 'car_horn', 'crackling_fire', 'helicopter', 'drinking_sipping', 'rain', 'insects', 'hen', 'engine', 'breathing', 'crying_baby', 'hand-saw', 'glass_breaking', 'toilet_flush', 'pig', 'washing_machine', 'sea_waves', 'siren', 'cat', 'door_wood_creaks', 'crickets', 'coughing', 'sneezing',
'snoring', 'laughing', 'sigh', 'sniff', 'throatclearing', 'phone', 'speech']

x_train = genfromtxt('train_data.csv', delimiter=',')
y_train = genfromtxt('train_labels.csv', delimiter=',')
x_test = genfromtxt('test_data.csv', delimiter=',')
y_test = genfromtxt('test_labels.csv', delimiter=',')

reconstructed_model = keras.models.load_model("trained_model")

y,sr=librosa.load('/Users/kathrynbeggs/Desktop/frog_test.wav')
mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=36).T,axis=0)

x_train=np.array(mfccs)

x_train=np.reshape(x_train,(1, 9, 4, 1))

prediction = reconstructed_model.predict(x_train)

maxPosition=np.argmax(prediction)  
prediction_label=classes[maxPosition]

print(prediction_label)