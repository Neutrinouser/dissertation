import Solver
import numpy as np

sList = np.linspace(7.5,10.0,20)
mu_mList = np.linspace(5e-3,5e-2,12) 

data = [[Solver.featureExtraction(s=i,mu_m=j,tMax=5.0, timeSet = np.linspace(0,5.0,501) )  for j in mu_mList] for i in sList]

with open('final.p','wb') as f:
    np.save(f, data)