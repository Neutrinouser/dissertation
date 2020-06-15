import pdeSolver 
import pickle
import matplotlib.pyplot as plt

kwargs= {
        #Specify whether m represents 'microspheres' or 'cells' 
        'typeOfLabeledParticles':'microspheres',
        #Proliferation, apoptosis and dispersion rates
        'mu_m': 8e-3, 'mu_n':1e-2, 's': 10.0, 'd': 6.0,
        #Consumption of nutrients function and parameter
        'gamma': pdeSolver.gamma, 'Gamma':0.5,
        #Background nutrient levels 
        'cInfty': 1.0,
        #Initial conditions
        'RInit': 1.4,'mInit': pdeSolver.mInit,  
        #mInit parameters
        'labelProp': 0.05 , 'mInitSpaceInterval': [0.5,0.8],
        #time grid specifications 
        'tMax':1,'deltaT': 2e-3, 
        #space grid spefications 
        'deltaRho': 5e-3,
        } 

# Get dictionary output
d = pdeSolver.main(**kwargs)

#Merge input and output dictionaries and store them as a pickle
data = {'input': kwargs, 'output': d}

with open('data.p', 'wb') as fp:
    pickle.dump(data, fp)

# Plotting results
plt.rc('text', usetex=True)
#First we plot R(t)
plt.plot(d['t'], list(d['R'].values()) ), plt.show(block=False)
plt.xlabel(r"$\displaystyle t $"), plt.ylabel(r"$\displaystyle R(t) $")
plt.savefig('R')
#Then we plot c(rho),m(rho),v(rho) for given time instances (defined in timeSet)
timeSet = [0.0,0.5,1.0]; varNames = ['v','m','c']
i = len(timeSet); j = len(varNames)
for index1, time in enumerate(timeSet):
    for index2, varName in enumerate(varNames):
        plt.subplot(i,j,3*index1+index2+1)
        plt.plot(d['x'], d[varName][time])
        plt.xlabel(r"$\displaystyle \rho $")
        plt.ylabel(r"$\displaystyle "+varName+"$")
        plt.title("t="+str(time))
plt.savefig('vmc')
plt.show(block=False)    