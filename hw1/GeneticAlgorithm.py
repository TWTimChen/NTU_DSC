import random 
import numpy as np

#################################################################
###############             Utilities             ###############
#################################################################

# transform decimal numbers to 
# string of 6-digit binary numbers
def dec2bin2str(d):
    bStr = "{0:b}".format(d)
    bStrFill = "{:0>6}".format(bStr)
    return(bStrFill)

#################################################################
###############        Class GeneticAlgorithm     ###############
#################################################################
'''
maxIter     :  iteration times for GA
popSize     :  number of initial gene combinition
poolSize    :  capacity of the mating pool
pop         :  list of initial gene combinition
pool        :  list of competing gene combinition
strLen      :  lenth of the gene combinition
pMute       :  probability of mutation for each gene

'''
class GeneticAlgorithm:
    maxIter = int()
    popSize = int()
    poolSize = int()
    pop = []
    pool = []
    strLen = int()
    pMute = float()
    
    def __init__(self, strLen, popSize=10, poolSize=100, maxIter=50, pMute = 0.01):
        self.strLen = strLen
        self.popSize = popSize
        self.poolSize = poolSize
        self.maxIter = maxIter
        self.pMute = pMute
    
    # User defined objective function
    def evaluation(self):
        raise Exception("Objective function not assigned")
    
    # User defined constrains
    def constrain(self):
        raise Exception("constrain function not assigned")
    
    # Generate mating pool
    # 1. randomly create {popSize} distinct individuals which observes conatrains
    # 2. sample {poolSize} individuals from {pop} 
    def generatePool(self):
        self.pool = []
        self.pop = []
        while len(self.pop) <= self.popSize:
            candid = random.randint(0, 2**self.strLen-1)
            candid = dec2bin2str(candid)
            if self.constrain(candid):
                if candid not in self.pop:
                    self.pop.append(int(candid, 2))
        self.pool = np.random.choice(self.pop, size=self.poolSize)
    
    # Evaluation and selection
    # 1. eliminate gene combinitions outside constrains
    # 2. evaluate each gene combinition's fitness value
    # 3. sample {poolSize} individuals based on thier fitness value
    def selection(self):
        poolSet = list(set(self.pool))
        poolStr = [dec2bin2str(p) for p in poolSet]
        poolStr = [ps for ps in poolStr if self.constrain(ps)]
        poolSet = [int(ps,2) for ps in poolStr]
        fitVal = [self.evaluation(ps) for ps in poolStr]
        fitValTot = sum(fitVal)
        poolRatio = [v/fitValTot for v in fitVal]
        self.pool = np.random.choice(poolSet, size=self.poolSize, p=poolRatio)
    
    # Crossover
    # 1. generate crossover position
    # 2. interchange genes
    def crossover(self, idx1, idx2):
        crossPos = random.randint(1,self.strLen-1)
        head1 = self.pool[idx1]%2**crossPos
        tail1 = self.pool[idx1]-head1
        head2 = self.pool[idx2]%2**crossPos
        tail2 = self.pool[idx2]-head2
        self.pool[idx1] = tail1 + head2
        self.pool[idx2] = tail2 + head1
    
    # Ｍutation
    # 1. generate mutation position by pMute
    # 2. indentify whether the muting gene is "1" or "0"
    # 3. mute the gene
    def mutation(self, idx):
        mutePos = [i for i in range(self.strLen) if(random.random()<self.pMute)]
        for mp in mutePos:
            muten = self.pool[idx]
            if muten%2**(mp+1) >= 2**(mp):
                muten -= 2**(mp)
            else:
                muten += 2**(mp)
    
    # Generate mating pool and execute interation
    def fit(self):
        self.generatePool()
        for _ in range(self.maxIter):
            self.selection()
            for i in range(0,self.poolSize,2):
                self.crossover(i, i+1)
            for i in range(self.poolSize):
                self.mutation(i)
        self.selection()
    # Print out the resulting gene combinition with the highest fitness value
    def show(self):
        poolSet = list(set(self.pool))
        poolStr = [dec2bin2str(p) for p in poolSet]
        fitVal = [self.evaluation(ps) for ps in poolStr]
        print("Iteration time: {}".format(self.maxIter))
        print("Best Combinition: {}\nFitness Value: {}".format(poolStr[np.argmax(fitVal)], max(fitVal)))

#################################################################
###############         Knapsack Problems         ###############
#################################################################
def main():
    KnapsackOptions = {}
    KnapsackOptions["AK-47 Rifle"] = {"Weight":15, "Survival Points":15}
    KnapsackOptions["Rope"] = {"Weight":3, "Survival Points":7}
    KnapsackOptions["Pocket Knife"] = {"Weight":2, "Survival Points":10}
    KnapsackOptions["Night-Vision Goggle"] = {"Weight":5, "Survival Points":5}
    KnapsackOptions["Handgun"] = {"Weight":9, "Survival Points":8}
    KnapsackOptions["Sniper Rifle"] = {"Weight":20, "Survival Points":17}
    
    # objective function for Knapsack Problem
    def SurviveObjFunc(optionsStr):
        assert len(optionsStr)==len(KnapsackOptions)

        SurvPtList = [i["Survival Points"] for i in KnapsackOptions.values()]
        SurvivePt = [sp*int(opt) for sp, opt in zip(SurvPtList, optionsStr)]
        return(sum(SurvivePt))

    # constrains for Knapsack Problem
    def WeightLimit(optionsStr, WtLimit = 30):
        assert len(optionsStr)==len(KnapsackOptions)

        WtLimit = WtLimit
        WtList = [i["Weight"] for i in KnapsackOptions.values()]
        Wt = [wt*int(opt) for wt, opt in zip(WtList, optionsStr)]
        return(sum(Wt) <= WtLimit)
    
    ga = GeneticAlgorithm(strLen=len(KnapsackOptions), popSize=10, poolSize=100, maxIter=2000, pMute=0.01)
    ga.evaluation = SurviveObjFunc
    ga.constrain = WeightLimit
    ga.fit()
    ga.show()

if __name__ == "__main__":
    main()