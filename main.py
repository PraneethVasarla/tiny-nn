from tinynn.layers.dense import Dense
from tinynn.models.sequential import Sequential
import numpy as np

X = np.array([[1,2,3],
     [5,4,3],
     [2,3,4]])
y = np.array([0, 0, 1])

model = Sequential()

model.add(Dense(3,3,layer_num=1))
model.add(Dense(3,4,layer_num=2))
model.add(Dense(4,4,layer_num=3))
model.add(Dense(4,4,layer_num=4))
model.add(Dense(4,4,layer_num=5))
model.add(Dense(4,3,layer_num=6))
model.add(Dense(3,2,layer_num=7,activation='softmax'))

model.compile_model(learning_rate=1)
model.fit(X,y,epochs=5)
