from tinynn.layers.dense import Dense
from tinynn.models.sequential import Sequential
import numpy as np

X = np.array([[1,2,3],
     [5,4,3],
     [2,3,4]])
y = np.array([0, 0, 1])


model = Sequential()

model.add(Dense(3,64))
model.add(Dense(64,2,activation='softmax'))

model.compile_model(learning_rate=0.01, optimizer='adam') #Also available params: decay_rate,momentum(only for sgd) and optimizer = sgd,adagrad
model.fit(X,y,epochs=1000)
