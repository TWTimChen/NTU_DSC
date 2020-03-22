import random
import numpy as np

#################################################################
###############           Class Pratical          ###############
#################################################################
'''
v           :   current velocity
x           :   current position
y           :   current objective function value
localMax    :   overall best objective function value
localMaxX   :   position of the overall best objective 
                function value
'''

class Particle():
    def __init__(self, x):
        self.v = np.array([0])
        self.x = np.array(x)
        self.y = np.array([0])
        self.localMax = float(0)
        self.localMaxX = self.x
    
    # Calculate velocity based on input
    def setV(self, globalMaxX, w, c1, c2, r1, r2):
        self.v = w*self.v \
             + c1*r1*(self.localMaxX - self.x) \
             + c2*r2*(globalMaxX - self.x)

    # Detemine the local maximun
    def setLocalMax(self):
        if self.y>=self.localMax:
            self.localMax = self.y
            self.localMaxX = self.x
    
    # Move the position by the velocity
    # Clip the position by boundaries
    def move(self, xLow, xUp):
        self.x = self.x + self.v
        outOfLow = self.x < xLow
        outOfUp = self.x > xUp
        self.x[outOfLow] = xLow[outOfLow]
        self.x[outOfUp] = xUp[outOfUp]

#################################################################
###############        Class ParticalSworm        ###############
#################################################################
'''
-------------------------- Data Member --------------------------

globalMax   :   overall best objective function value among particals
globalMaxX  :   position of the overall best objective function 
                value among particals
dim         :   dimesion of the design vector
par         :   list of particles
xLow        :   lower bound of the design vector
xUP         :   upper bound of the design vector

-------------------------- Parameters ---------------------------

showIter    :   show particles information for each iteration
n           :   number of particles
iterTime    :   iteration times
w           :   velocity damper coefficient
c1          :   weight of local max
c2          :   weight of global max
r1          :   random number for local max
r2          :   random number for global max
'''

class ParticleSworm():
    def __init__(self, xLow, xUp, n=20, showIter=False, iterTime=100, damperWeight=1, localWeight=1, globalWeight=1):
        assert(len(xLow) == len(xUp))

        ###################### Data Member ######################
        self.globalMax = 0
        self.globalMaxX = 0
        self.dim = len(xLow)
        self.par = []
        self.xLow = np.array(xLow)
        self.xUp = np.array(xUp)
        
        ###################### Parameters #######################
        self.showIter = showIter
        self.n = n
        self.iterTime = iterTime
        self.w = damperWeight
        self.c1 = localWeight
        self.c2 = globalWeight
        self.r1 = 0
        self.r2 = 0
    
    # Generate [n] particles within boundaries
    def genPar(self):
        xSeed = np.random.random((self.n, self.dim))
        xDiff =  self.xUp - self.xLow
        xBase = xSeed + self.xLow
        xRandom = [b*xDiff for b in xBase]
        self.par = [Particle(x) for x in xRandom]
    
    # User defined objective function
    def evaluation():
        raise Exception("Objective function not assigned")
    
    # Execute swarming process
    def fit(self, x=""):
        # Check input part 
        # 1. check if there is a user defined input
        #    - if true: check for its validation
        #    - if not true: call genPar to generate random points

        if x=="":
            self.genPar()
        else:
            assert(len(x[0])==self.dim)
            assert(sum(x<self.xLow)==0)
            assert(sum(x>self.xUp)==0)
            x = np.array(x)
            self.n = len(x)
            self.par = [Particle(xx) for xx in x]

        #Iteration part
        # 1. calculate each particle's objective function value
        # 2. detemine the global maximun
        # 3. calculate velocity for each particle
        # 4. move each particle

        for i in range(self.iterTime):
            for p in self.par:
                p.y = self.evaluation(p.x)
                p.setLocalMax()

            if self.showIter:
                self.show(i)
            
            maxIndex = np.argmax([p.y for p in self.par])
            if self.par[maxIndex].y >= self.globalMax:
                self.globalMax = self.par[maxIndex].y
                self.globalMaxX = self.par[maxIndex].x
            self.r1 = random.random()
            self.r2 = random.random()

            for p in self.par:
                p.setV(self.globalMaxX, self.w, self.c1, self.c2, self.r1, self.r2)
                p.move(self.xLow, self.xUp)

    def show(self, iter):
        print('Iteration: {}'.format(iter))
        for p in self.par:
            print('x: {:.2f} y: {:.1f}'.format(p.x[0], p.y[0]))

#################################################################
###############                main               ###############
#################################################################

def main():
    def f1(x):
        assert len(x)==1
        return(11 + 2*x - x**2)

    x = [[-1.5], [0.5], [0] ,[1.25]]
    
    ps = ParticleSworm(xLow=[-2], xUp=[2], showIter=True, damperWeight=0.3)
    ps.evaluation = f1
    ps.fit(x)
    ps.show('Result')

if __name__ == "__main__":
    main()