import csv

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def loadLine(csvLine):
    line = []
    i=0
    for name in names:
        if name == csvLine[0]:
            break
        else:
            i += 1
    
    line.append(i)
   # line.append[1]
    line.append(csvLine[2])
    line.append(csvLine[3])
    line.append(csvLine[4])
    line.append(csvLine[5])
    line.append(csvLine[6])
    line.append(csvLine[7])
    line.append(csvLine[8])
    print line
    data.append(line)

def BaselineModel():
    model = Sequential()
    model.add(Dense(13, input_dim=7, init='uniform', activation='relu'))
    model.add(Dense(13, init='uniform', activation='relu'))
    model.add(Dense(6, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform'))
    #Compile the model.
    model.compile(loss='mse', optimizer='adam', metrics=['mape'])
    return model

data = []

names = ['adviser', 'amdahl','apollo', 'basf', 'bti', 'burroughs', 'c.r.d', 'cambex', 'cdc', 'dec', 
       'dg', 'formation', 'four-phase', 'gould', 'honeywell', 'hp', 'ibm', 'ipl', 'magnuson', 
       'microdata', 'nas', 'ncr', 'nixdorf', 'perkin-elmer', 'prime', 'siemens', 'sperry', 
       'sratus', 'wang']

with open('processors_data.csv') as localfile:
    reader = csv.reader(localfile,delimiter=',',quotechar='"')
    reader.next()
    for row in reader:
        loadLine(row)

dataset = numpy.array(data)
numpy.random.shuffle(dataset)
#dataset = numpy.loadtxt("processors_data.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:7]
Y = dataset[:,7]

#Init random. 
seed = 7
numpy.random.seed(seed)

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=BaselineModel, nb_epoch=1000, batch_size=5,  verbose=2)))

pipeline = Pipeline(estimators)
#pipeline.fit(X,Y)
kfold = KFold(n_splits=3, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)

print results.mean()
print results.std()
print results
