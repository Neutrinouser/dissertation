import Solver
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn

sList = np.linspace(7,13,3); 
mu_mList = np.linspace(5e-3,5e-2, 3); 
inputDictionary = dict(s=sList,mu_m=mu_mList); 


def parameterSweep(inputs,timeSet=[1.0]):
    keys = list(inputs.keys())
    L = len(keys) - 1
    def iteration(input,l):
        key = keys[l] 
        out = {}
        for i in inputs[key]:
            inputNew = input; 
            inputNew.update({key: i}) 
            if l==L:
                inputNew.update(timeSet=timeSet)
                out[i] = Solver.featureExtraction(**inputNew)
            else:
                out[i] = iteration(inputNew,l+1)
        return out
    return iteration({},0)

d = parameterSweep(inputDictionary); 
with open('parameterSweepData.p','wb') as fp:
    pickle.dump(d,fp)