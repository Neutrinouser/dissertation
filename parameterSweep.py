import Solver
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn

class Features:
    def __init__(self,timeSet=[],**kwargs):
        self.results = Solver.featureExtraction(**kwargs)
        if not timeSet:
            timeSet = self.results.keys()
        self.outputDict = {}; 
        k = np.array(list(self.results.keys()))
        self.timeSet = [k[np.argmin(np.abs( k - time ))] for time in timeSet]; 

    def outputDictionary(self):
        values = [self.results[j] for j in self.timeSet]; 
        features = values[0].keys(); 
        self.outputDict['times'] = self.timeSet
        for j in features:
            self.outputDict[j] = [v[j] for v in values] 
        return self.outputDict

    def plotting(self,featureSet = None ):
        if not featureSet:
            featureSet = self.outputDict.keys(); 
        _= self.outputDictionary(); 
        for j in featureSet:
            plt.plot(self.timeSet, self.outputDict[j])
        plt.show()

# s = Features(timeSet = np.linspace(0,1,11))
# d = s.outputDictionary()
# print(d)
# s = Features(); 
# s.plotting(); 
# print(Features().outputDictionary())

sList = np.linspace(5,12,10); 
mu_mList = np.linspace(5e-3,5e-2, 10); 
inputDictionary = dict(s=sList,mu_m=mu_mList); 

#d = [[Features(timeSet=[1.0],s=i,mu_m=j).outputDictionary() for i in sList] for j in mu_mList]

#print(d)

def parameterSweep(inputs):
    keys = list(inputs.keys())
    L = len(keys) - 1
    def iteration(input,l):
        key = keys[l] 
        out = {}
        for i in inputs[key]:
            inputNew = input; 
            inputNew.update({key: i}) 
            if l==L:
                inputNew.update(timeSet=[1.0])
                out[i] = Features(**inputNew).outputDictionary()
            else:
                out[i] = iteration(inputNew,l+1)
        return out
    return iteration({},0)

d = parameterSweep(inputDictionary); 
with open('data.p','wb') as fp:
    pickle.dump(d,fp)

#extra line