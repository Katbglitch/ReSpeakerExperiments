import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from librosa import display
import librosa
from sklearn.model_selection import train_test_split

data = pd.read_csv('DataForReSpeaker/data30Nov.csv') #40 of each 
#data = pd.read_csv('DataForReSpeaker/AllAudio.csv')

x_train=[]
x_test=[]
y_train=[]
y_test=[]
mfccs =[0] * len(data)
label = data["category"]

path="DataForReSpeaker/allAudio"

for i in tqdm(range(len(data))):
    file=data.iloc[i]["filename"]
    filename=path+"/"+file
    y,sr=librosa.load(filename)
    mfccs[i] = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=36).T,axis=0)

x_train,x_test,y_train,y_test=train_test_split(mfccs,label,test_size=0.02, random_state=0)

# x_train=np.array(x_train)
# x_test=np.array(x_test)
# y_train=np.array(y_train)
# y_test=np.array(y_test)

# print(x_train.shape, y_train.shape)

np.savetxt("train_data.csv", x_train, delimiter=",")
np.savetxt("test_data.csv",x_test,delimiter=",")
np.savetxt("train_labels.csv",y_train,delimiter=",")
np.savetxt("test_labels.csv",y_test,delimiter=",")