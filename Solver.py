import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import matplotlib.pyplot as plt

try:
    from progress.bar import Bar
except:
    raise Exception('Progress package not installed. \
    Install it with pip install progress in cmd')

def plotting(**kwargs):
    d = solver(**kwargs); 
    plt.rc('text', usetex=True)
    #First we plot R(t)
    plt.plot(d['t'], list(d['R'].values()) )
    plt.xlabel(r"$\displaystyle t $"), plt.ylabel(r"$\displaystyle R(t) $")
    plt.savefig('R')
    plt.show(block=True)
    #Then we plot c(rho),m(rho),v(rho) for given time instances (defined in timeSet)
    timeSet = np.multiply(d['t'][len(d['t'])-1],[0.0,0.5,1.0]); varNames = ['v','m','c']
    j = len(varNames)
    for index2, varName in enumerate(varNames):
        #plot display
        plt.subplot(1,j,index2+1)
        plt.ylabel(r"$\displaystyle "+varName+"$")
        #plt.title("t="+str(round(time,2)))
        #Find nearest time in time grid
        k = np.array(list(d[varName].keys()))
        for time in timeSet:
            time = k[np.argmin(np.abs( k - time ))]
            #plot
            plt.plot(d['x'], d[varName][time])
        #plt.xlabel(r"$\displaystyle \rho $")
    plt.savefig('vmc')
    plt.show(block=True) 
    

def featureExtraction(timeSet = [1.0],**kwargs):
    # Gives features of microbead distribution at different time instances (in timeSet).

    # d is the output dictionary
    d = solver(**kwargs)

    # Change timeSet so that it aligns with the time data available from the numerical solver.
    k = d['t']; timeSet = [k[np.argmin(np.abs( k - time ))] for time in timeSet]; 
    
    # Output features initiated here
    radius = [] ; mode = [] ; stdev = [] ; 

    dRho = d['x']; 
    for time in timeSet: 
        dmt = d['m'][time]; dRt = d['R'][time]; 
        radius.append(dRt)
        mode.append(dRho[max([(x,i) for i,x in enumerate(dmt)])[1]] * dRt)
        mean  = dRt * np.sum(dRho * dmt) / np.sum(dmt)
        stdev.append(np.sqrt(np.sum( (dRho * dRt - mean)**2 * dmt )/np.sum(dmt)))
    
    return dict(t=timeSet,features=dict(tumourRadius=radius,frontPosition=mode,dispersion=stdev))
    
def mInit(spaceGrid, deltaRho=5e-3, labelProp = 0.011 , mInitSpaceInterval = [0.7,0.9], **kwargs):
    a = mInitSpaceInterval[0] ; b = mInitSpaceInterval[1]
    rho = spaceGrid
    out = (rho-a)**2 * (b- rho)**2 *  ((rho>a) & (rho < b))
    return out/np.sum(out) * labelProp/deltaRho

def rInit(cInfty=1.0,Gamma=0.5,s=10.0,d=6.0,**kwargs):
    return np.sqrt(3/Gamma*(cInfty-d/s))

def gamma(Gamma,m,**kwargs):
    return Gamma * m 

def solver(tMax=1,deltaT=2e-3, #time grid specifications
         **kwargs):
    # Initial radius of the tumour is the steady state radius in the abscence of labelled cells
    RInit = rInit(**kwargs); 

    # Set up spaceGrid and timeGrid
    spaceGrid, kwargs['deltaRho'] = getSpaceGrid(**kwargs)
    kwargs['spaceGrid'] = spaceGrid
    timeGrid, deltaT = getTimeGrid(tMax,deltaT) 

    # Set up progress bar
    bar = Bar('Processing', max= tMax/deltaT )

    # Initiate (R,m,v,c)
    currentR = RInit;   R = {0:currentR}
    currentM = mInit(**kwargs); m = {0:currentM}
    currentC = odeC(currentR,currentM,**kwargs);  c = {0:currentC}
    currentV = odeV(currentC,currentR,currentM,**kwargs);  v = {0:currentV}

    for t in timeGrid[1:]:
        bar.next()

        #Euler for R, Midpoint rule for m, ode sols for c and v
        currentM = pdeM(currentC,currentV,currentR,currentM,deltaT,**kwargs)
        currentR = currentR +  deltaT * integrandR(currentV,**kwargs)
        currentC = odeC(currentR,currentM,**kwargs)
        currentV = odeV(currentC,currentR,currentM,**kwargs)

        # Update dictionaries
        m.update({t: currentM })
        c.update({t: currentC })
        v.update({t: currentV })
        R.update({t: currentR })  

        # Identified Warnings; abs(R) becomes very big in case of instabilities. 
        if (currentR < 0) | (currentR > 1e+2):
            bar.finish(); 
            print('Unsuccessful; decreasing space and time increment.')
            factor = 0.5 
            deltaRho = kwargs['deltaRho'] * factor; 
            kwargs.pop('deltaRho')
            return solver(tMax,deltaT * factor**2,deltaRho = deltaRho,**kwargs)

    bar.finish()
    return dict(c=c,v=v,R=R,m=m,x=spaceGrid,t=timeGrid)

def odeC(R,m,deltaRho=5e-3,typeOfLabeledParticles='microspheres',gamma=gamma,Gamma=0.5,cInfty=1.0,**kwargs):
    # Solves C'' = R^2 (Gamma(1-m)+Gamma_m) st c'(0)=0, c(1)=c_infty as a BVP.
    rhs = R**2 * (gamma(Gamma,1-m,**kwargs) + (typeOfLabeledParticles == 'cells') * gamma(Gamma,m,**kwargs))
    rhs[len(rhs)-1] = cInfty ; rhs[0] = 0 ; N = len(rhs)
    der2 = sp.spdiags([  np.concatenate((np.ones(N-2),[0,0]))  ,  \
        np.concatenate(([-deltaRho],-2*np.ones(N-2),[deltaRho**2]))  , \
            np.concatenate(([0,deltaRho],np.ones(N-2) )) ], \
                [-1, 0, 1], N, N, format='csc')/deltaRho**2
    return spl.spsolve(der2,rhs)

def odeV(c,R,m,deltaRho=5e-3,typeOfLabeledParticles='microspheres',mu_m = 8e-3,mu_n=1e-2,s=10.0,d=6.0,**kwargs):
    # Solves v' = 1/R (mu_m - mu_n)  m'' + R ((s*c-d) (1-m) + S_m), v(0)=0 as a 1st order ODE.
    N = len(m)
    der2Op = sp.spdiags([  np.ones(N)  , np.concatenate(([-1],-2*np.ones(N-2) ,[-1])), np.ones(N) ], [-1,0,1],N,N,format='csc')
    m2ndDer = der2Op @ m/deltaRho**2
    rhs = 1/R * (mu_m - mu_n) * m2ndDer + R * (s*c-d) * (1 - m * (typeOfLabeledParticles=='microspheres') )
    v = np.array([0]) # BC: v(0)=0
    for I in range(N-1):
        v = np.append(v,v[len(v)-1] + deltaRho * rhs[I])
    return v

def integrandR(v,**kwargs):
    # dR/dt = v(1,t)
    return v[len(v)-1]

def pdeM(c,v,R,m,deltaT,spaceGrid,deltaRho=5e-3, typeOfLabeledParticles='microspheres',mu_m = 8e-3,mu_n=1e-2,s=10.0,d=6.0,**kwargs):
    N = len(m)
    #Calculate first and second derivative of m
    der1Op = sp.spdiags([ -np.ones(N)  , np.ones(N) ], [-1,0 ],N,N,format='csc')
    der2Op = sp.spdiags([  np.ones(N)  , -2*np.ones(N) , np.ones(N) ], [-1,0,1],N,N,format='csc')
    # m_t = f(m,m_x,m_xx) = netProliferation(m) + velocityTerm(m_x) + dispersion(m_xx)
    netProliferation = sp.diags(-(s*c - d )*(1-m) * (typeOfLabeledParticles=='microspheres'),0) 
    dispersion = 1/R**2 * sp.diags((mu_m- (mu_m-mu_n)*m),0) * der2Op/deltaRho**2
    velocityTerm = - 1/R * sp.diags(v- spaceGrid* v[len(v)-1],0) * der1Op/ deltaRho
    withoutBoundaries = sp.diags(np.concatenate(([0], np.ones(N-2) , [0])),0)
    f =  netProliferation + velocityTerm + dispersion

    boundaryConditions = 1/deltaRho * sp.diags([ np.concatenate((np.zeros(N-2),[-1] )), \
    np.concatenate(([-1],np.zeros(N-2),[1])), np.concatenate(([1],np.zeros(N-2)))  ], [-1,0,1] ) \
    # - v[len(v)-1]/mu_m * sp.diags(np.concatenate((np.zeros(N-1) , [1])),0)

    # Reduce to linear system A*m = b
    theta = 1.0 # theta-method scheme
    A = withoutBoundaries * (sp.identity(N) - deltaT * theta * f ) + boundaryConditions 
    b = (withoutBoundaries * (sp.identity(N) + deltaT *(1.0-theta) * f )) @ m
    return spl.spsolve(A,b)

def getGrid(high,increment,low=0.0):
    Grid = np.linspace(low,high,int(np.floor((high-low)/increment)+1)) 
    increment   = Grid[1] - Grid[0]
    return Grid, increment 

def getSpaceGrid(deltaRho=5e-3,**kwargs):
    return getGrid(1.0,deltaRho)

def getTimeGrid(tMax=1, deltaT=2e-3):
    return getGrid(tMax,deltaT)

if __name__ == '__main__':
    plotting()