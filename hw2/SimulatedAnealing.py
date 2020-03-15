import random
import math
import numpy as np

#################################################################
###############       Class SimulatedAnealing     ###############
#################################################################
'''
T           :  initial temperature
c           :  temperature attenuation coefficient
n           :  inner iteration times
cycle       :  cooling times
kernel      :  methed for perturbation
ksize       :  perturbation range for the kernel
showIter    :  [boolean] show each steps of iteration
x           :  initail design vector

'''

class SimulatedAnealing:
    
    def __init__(self, T, c, n, cycle=200, kernel='uniform', ksize=5, showIter=False):
        self.T = T
        self.c = c
        self.n = n
        self.cycle = cycle
        self.kernel = kernel
        self.ksize = ksize
        self.showIter = showIter
        self.x = []
    
    # Uniform ditribution kernel
    def uniform(self):
        base = self.x - self.ksize
        delta = [random.random()*self.ksize*2 for _ in range(len(base))]
        xNew = [b+d for b,d in zip(base, delta)]
        return(np.array(xNew))

    # Perturbation step
    # choose which kernel to use
    def perturb(self):
        if self.kernel == 'Unknonw':
            pass
        else:
            return self.uniform()
    
    # User define objective function
    def objFunc(self):
        raise Exception("Objective function not assigned")
    
    # Metropolitan criterion
    def evaluation(self, xNew):
        deltaE = self.objFunc(xNew) - self.objFunc(self.x)
        if deltaE<0:
            self.x = xNew
        else:
            thresh = random.random()
            boltz = math.exp(-deltaE/self.T)
            if boltz>thresh:
                self.x = xNew
        return
    
    # Annealing step
    # 1. generate new design vector (inner loop)
    # 2. go through metropolitan criterion (inner loop)
    # 3. reduce temperature for each iteration (outer loop)

    def fit(self, x):
        n=0; p=0 
        self.x = np.array(x)
        
        while p<self.cycle:
            while n<self.n:
                xNew = self.perturb()
                self.evaluation(xNew)
                if self.showIter:
                    self.show(p,n)
                n+=1
            self.T*=self.c
            n=0
            p+=1
            if self.showIter:
                print('Fitness Value: {}'.format(self.objFunc(self.x)))
        
    def show(self,p,n):
        log = 'Cycle: {} Iter: {} x:{}'.format(p,n,self.x)
        print(log)

#################################################################
###############        Simulation Problems        ###############
#################################################################

'''      
# Simulation for hw2-question2  
def main():
    def f1(x):
        assert x.size==2
        retVal = 6*x[0]**2 - 6*x[0]*x[1] + 2*x[1]**2 - x[0] - 2*x[1]
        return(retVal)
    
    x = [0,0]
    sa = SimulatedAnealing(T=10000, c=0.7, n=10, cycle=100, ksize=20, showIter=True)
    sa.objFunc = f1
    sa.fit(x)
'''

# Simulation for hw2-question3      
def main():
    def f2(x):
        assert x.size==1
        x = x[0]
        return(-x*(1.5-x))
    
    x = [2]
    sa = SimulatedAnealing(T=400, c=0.5, n=5, cycle=100, showIter=True)
    sa.objFunc = f2
    sa.fit(x)


if __name__=='__main__':
    main()