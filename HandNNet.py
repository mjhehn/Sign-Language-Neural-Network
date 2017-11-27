#Managing import of local python libs
#https://stackoverflow.com/questions/279237/import-a-module-from-a-relative-path
import os, sys, inspect

cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"lib")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"lib/Leap")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"lib/Anderson")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
#*******************************
#imports
import pandas as pd
import numpy as np
import neuralnetworks as nn
import mlutils as ml

#******************************

#implementations from Matt's CS440 A6
def trainNNs(X, T, trainFraction, hiddenLayerStructures, numberRepetitions, numberIterations, classify=False):
    import neuralnetworks as nn
    import mlutils as ml
    import numpy as np
    import time
    result = []
    for structure in hiddenLayerStructures:
        trainedResult = []
        testResult = []
        t0 = time.time()
        for n in range(0, numberRepetitions):
            Xtrain,Ttrain,Xtest,Ttest = ml.partition(X,T,(trainFraction, 1-trainFraction),classify)
            if classify:
                nnet = nn.NeuralNetworkClassifier(X.shape[1], structure, len(np.unique(T)))
                nnet.train(Xtrain, Ttrain, numberIterations, errorPrecision=1.e-8)
                trainedResult.append(np.sum(nnet.use(Xtrain)==Ttrain)/len(Ttrain))
                testResult.append(np.sum(nnet.use(Xtest)==Ttest)/len(Ttest))
            else:
                nnet = nn.NeuralNetwork(X.shape[1], structure, T.shape[1])
                nnet.train(Xtrain, Ttrain, numberIterations)
                trainedResult.append(np.sqrt(np.mean(((nnet.use(Xtrain)-Ttrain)**2))))
                testResult.append(np.sqrt(np.mean(((nnet.use(Xtest)-Ttest)**2))))

            
        result.append([structure, trainedResult, testResult, time.time() - t0])
    return result

def summarize(results):
    import numpy as np
    summaryResults = []
    for result in results:
        summaryResults.append([result[0], np.mean(result[1]), np.mean(result[2]), result[3]])
    return summaryResults


def bestNetwork(summary):
    best = min(summary, key=lambda l: l[2])
    return best
#****************************************


#test read of file/run of the neural net. Also/or see Test NNet.ipynb
filename = "templates/test.csv"
data = pd.read_csv(filename)        #get data from csv as pandas dataframe
names = list(data)                  #get csv column names
data['signcode'] = data["sign"].astype('category').cat.codes    #classify hand signs into categoricals in Pandas, and add a column for the numerical version of signs
data = data.values  #convert dataframe to numpy array

Xhands = data[:, 0:63]
Xhands = Xhands.astype(np.float64)
Tsign = data[:, 64:65]  #get the numerical version of sign/category/class
Tsign = Tsign.astype(np.int32)

handParts = names[0:63]
signHeader = names[63]
signs = data[:, 63:64]

result = trainNNs(Xhands, Tsign, 0.90, [[5,5], 5], 10, 100, classify=True)
bestNetwork(summarize(result))