import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation

dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

model = Sequential()

# 2 Hidden layers with 12 and 8 neurons respectively. One output node with sigmoid activation. 

model.add(Dense(12, input_dim=8, init='uniform'))
#model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(12, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

#Compile the model.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, nb_epoch=2000, batch_size=30,  verbose=2)

predictions = model.predict(X)

rounded = [round(x) for x in predictions]

print rounded
