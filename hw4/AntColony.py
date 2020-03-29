import numpy as np

#################################################################
###############             Utilities             ###############
#################################################################

def normSampling(x, n):
    div = sum(x)
    if div != 0:
        x = [i/div for i in x]
        index = np.arange(len(x))
        return(np.random.choice(index, size=n, p=x))
    return([-1]*len(x))


#################################################################
###############           Class Pratical          ###############
#################################################################
'''
xId         :   corresonding index of the design graph 
i           :   number of the agent
y           :   current objective function value
'''

class AntPar:
    def __init__(self, xId, i):
        self.xId = xId
        self.i = i
        self.y = None

#################################################################
###############        Class ParticalSworm        ###############
#################################################################
'''
-------------------------- Data Member --------------------------

nPar        :   number of the ants
graph       :   design graph
graphWeight :   pheromone for each node of the design graph
nLayer      :   number of the layer of the design graph
par         :   list fro the ants

-------------------------- Parameters ---------------------------

iterTime    :   iteration times
show        :   show agents' information for each iteration
eta         :   scaling parameter for update pheromone
rho         :   evaporization parameter for pheromone
'''

class AntColony:

    def __init__(self, nPar, graph, iterTime=10, show=False, eta=2, rho=.5):
        self.nPar = nPar
        self.graph = np.array([np.array(layer) for layer in graph])
        self.graphWeight = [np.ones_like(layer) for layer in graph]
        self.nLayer = len(self.graph)
        self.par = [AntPar(np.zeros(self.nLayer), i) for i in range(self.nPar)]

        assert(eta>=0)
        assert((rho>=0) & (rho<=1))
        self.iterTime = iterTime
        self.show = show
        self.eta = eta
        self.rho = rho

    def evalutation():
        raise Exception("Objective function not assigned")

    def setPar(self):
        layerIndex = [normSampling(layer, self.nPar) for layer in self.graphWeight]
        layerIndex = np.array(layerIndex).T
        for i in range(self.nPar):
            self.par[i].xId = layerIndex[i]
    
    def evalPar(self):
        for p in self.par:
            x = [layer[i] for i, layer in zip(p.xId, self.graph)]
            p.y = self.evalutation(x)

    def setGraphWeight(self):
        bestS = max([p.y for p in self.par])
        worstS = min([p.y for p in self.par])

        bestParId = [p.i for p in (self.par) if p.y == bestS]

        deltaW = self.eta * abs(bestS/worstS)
        for bi in bestParId:
            for i, xi in enumerate(self.par[bi].xId):
                self.graphWeight[i][xi] += deltaW

    def evapGraphWeight(self):
        for layer in self.graphWeight:
            for node in layer:
                node *= (1 - self.rho)

    def isNotConverge(self):
        return True

    def fit(self):
        iterTime = 0
        while self.isNotConverge() & (iterTime<self.iterTime):
            self.setPar()
            if self.show: self.showIter(iterTime)
            self.evalPar()
            self.setGraphWeight()
            self.evapGraphWeight()
            iterTime += 1

    def showIter(self, iterTime):
        print('Iteration: {}'.format(iterTime))
        for p in self.par:
            retVal = [layer[i] for i, layer in zip(p.xId, self.graph)]
            print(retVal)

    def showResult(self):
        resVal = max([p.y for p in self.par])
        resPar = [p.xId.tolist() for p in self.par if p.y == resVal]
        print('Optimal Solution:')
        for rp in resPar:
            resOpt = [layer[i] for i, layer in zip(rp, self.graph)]
            print(resOpt)

#################################################################
###############                Main               ###############
#################################################################
def main():
    def f(x):
        assert(len(x) == 1)
        xx = x[0]
        retVal = xx**2 - 2*xx -11
        return(-retVal)

    graph = [[0., 0.5, 1., 1.5, 2., 2.5, 3.]]    

    ac = AntColony(nPar=4, graph=graph, iterTime=20, show=True)
    ac.evalutation = f
    ac.fit()
    ac.showResult()

if __name__ == "__main__":
    main()