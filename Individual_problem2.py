#
# Individual.py
#
#

import math

#A simple 1-D Individual class
class Individual:
    """
    Individual
    """
    minSigma=1e-100
    maxSigma=1
    learningRate=1
    minLimit=None
    maxLimit=None
    uniprng=None
    normprng=None
    fitFunc=None
    num_mutivariables=None

    def __init__(self):
        x = []
        for i in range(self.num_mutivariables):
            x.append(self.uniprng.uniform(self.minLimit,self.maxLimit))
        self.x= x
        self.fit=self.__class__.fitFunc(self.x)
        self.sigma=self.uniprng.uniform(0.9,0.1) #use "normalized" sigma
        
    def crossover(self, other):
        #perform crossover "in-place"
        alpha=self.uniprng.random()
        for i in range(len(self.x)):
            tmp=self.x[i]*alpha+other.x[i]*(1-alpha)
            other.x[i]=self.x[i]*(1-alpha)+other.x[i]*alpha
            self.x[i]=tmp
        self.fit=None
        other.fit=None
    
    def mutate(self):
        self.sigma=self.sigma*math.exp(self.learningRate*self.normprng.normalvariate(0,1))
        if self.sigma < self.minSigma: self.sigma=self.minSigma
        if self.sigma > self.maxSigma: self.sigma=self.maxSigma
        for i in range(len(self.x)):
            self.x[i]=self.x[i]+(self.maxLimit-self.minLimit)*self.sigma*self.normprng.normalvariate(0,1)
        self.fit=None
    
    def evaluateFitness(self):
        if self.fit == None: self.fit=self.__class__.fitFunc(self.x)
        
    def __str__(self):
        return '%0.8e'%self.x+'\t'+'%0.8e'%self.fit+'\t'+'%0.8e'%self.sigma
