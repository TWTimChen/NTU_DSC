import random
import numpy as np

class Particle():
    def __init__(self, x):
        self.v = 0
        self.x = np.array(x)
        self.y = 0
        self.localMax = 0
    
    def setV(self, globalMax, w, c1, c2, r1, r2):
        self.v = w*self.v \
             + c1*r1*(self.x - self.localMax) \
             + c2*r2*(self.x - globalMax)

    def setLocalMax(self):
        if self.y>=self.localMax:
            self.localMax = self.y
    
    def move(self, xLow, xUp):
        self.x += self.y
        outOfLow = self.x < xLow
        outOfUp = self.x > xUp
        self.x[outOfLow] = xLow[outOfLow]
        self.x[outOfUp] = xLow[outOfUp]

class ParticleSworm():
    def __init__(self, xLow, xUp, n=20, iterTime=100, damperWeight=1, localWeight=1, globalWeight=1):
        assert(len(xLow) == len(xUp))
        self.globalMax = 0
        self.dim = len(xLow)
        self.par = []
        self.xLow = np.array(xLow)
        self.xUp = np.array(xUp)
        self.n = n
        self.iterTime = iterTime
        self.w = damperWeight
        self.c1 = localWeight
        self.c2 = globalWeight
        self.r1 = 0
        self.r2 = 0

    def genPar(self):
        xSeed = np.random.random((self.n, self.dim))
        xDiff =  self.xUp - self.xLow
        xBase = xSeed + self.xLow
        xRandom = [b*xDiff for b in xBase]
        self.par = [Particle(x) for x in xRandom]

    def evaluation():
        raise Exception("Objective function not assigned")

    def fit(self, x=""):
        if x=="":
            self.genPar()
        else:
            assert(len(x[0])==self.dim)
            assert(sum(x<self.xLow)==0)
            assert(sum(x>self.xUp)==0)
            x = np.array(x)
            self.n = len(x)
            self.par = [Particle(xx) for xx in x]
        
        for _ in range(self.iterTime):
            for p in self.par:
                p.y = self.evaluation(p.x)
                p.setLocalMax()

            self.globalMax = max([p.y for p in self.par])
            self.r1 = random.random()
            self.r2 = random.random()

            for p in self.par:
                p.setV(self.globalMax, self.w, self.c1, self.c2, self.r1, self.r2)
                p.move(self.xLow, self.xUp)

    def show(self):
        # res = [p.x for p in self.par]
        print([p.x for p in self.par])

def main():
    def f1(x):
        assert len(x)==1
        return(11 + 2*x - x**2)

    x = [[-1.5], [0.5], [0] ,[1.25]]
    
    ps = ParticleSworm(xLow=[-2], xUp=[2])
    ps.evaluation = f1
    ps.fit(x)
    ps.show()

if __name__ == "__main__":
    main()