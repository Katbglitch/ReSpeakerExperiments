import numpy as np
from numpy import genfromtxt
from keras.utils.np_utils import to_categorical
from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout,Activation

x_train = genfromtxt('train_data.csv', delimiter=',')
y_train = genfromtxt('train_labels.csv', delimiter=',')
x_test = genfromtxt('test_data.csv', delimiter=',')
y_test = genfromtxt('test_labels.csv', delimiter=',')

y_train = to_categorical(y_train, num_classes=53)
y_test = to_categorical(y_test, num_classes=53)


x_train=np.reshape(x_train,(x_train.shape[0],9, 4,1))
x_test=np.reshape(x_test,(x_test.shape[0],9, 4,1))

model=Sequential()
model.add(Conv2D(64,kernel_size=4,strides=1,padding="Same",activation="relu",input_shape=(9,4,1)))
model.add(MaxPooling2D(padding="same"))

model.add(Conv2D(128,kernel_size=4,strides=1,padding="same",activation="relu"))
model.add(MaxPooling2D(padding="same"))

model.add(Dense(128))
model.add(Activation("relu"))
model.add(Dropout(0.1))

model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.1))

model.add(Dense(128))
model.add(Activation("relu"))
model.add(Dropout(0.1))

model.add(Flatten())

model.add(Dense(53))
model.add(Activation("softmax"))

model.compile(optimizer="Adam",loss="categorical_crossentropy",metrics=["accuracy"])

model.fit(x_train,y_train,batch_size=27,epochs=200,validation_data=(x_test,y_test))

train_loss_score=model.evaluate(x_train,y_train)
test_loss_score=model.evaluate(x_test,y_test, verbose =0)

model.save('trained_model')

# print(train_loss_score)
print(test_loss_score)