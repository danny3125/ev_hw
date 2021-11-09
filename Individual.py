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
    num_balls=None
    numParticleTypes = None
    def __init__(self):
        x = []
        for i in range(self.num_balls): x.append(int(self.uniprng.uniform(0,3)))
        self.x= x
        self.fit=self.__class__.fitFunc(self.x)
        self.sigma=self.uniprng.uniform(0.9,0.1) #use "normalized" sigma
        
    def crossover(self, other):
        #perform crossover "in-place"
        alpha = Randint(0,len(self.x))
        tmp = other.x[alpha + 1:len(self.x) - 1] + self.x[0:alpha]
        other.x=self.x[alpha + 1:len(self.x) - 1] + other.x[0:alpha]
        self.x=tmp
        self.fit=None
        other.fit=None
    
    def mutate(self):
        self.sigma=self.sigma*math.exp(self.learningRate*self.normprng.normalvariate(0,1))
        if self.sigma < self.minSigma: self.sigma=self.minSigma
        if self.sigma > self.maxSigma: self.sigma=self.maxSigma
        for i in range(len(self.num_balls)):
            if self.sigma > self.uniprng.random():
                self.x[i] = self.uniprng.randint(0,len(self.numParticleTypes) - 1)
        self.fit=None
    
    def evaluateFitness(self):
        if self.fit == None: self.fit=self.__class__.fitFunc(self.x)
        
    def __str__(self):
        return '%0.8e'%self.x+'\t'+'%0.8e'%self.fit+'\t'+'%0.8e'%self.sigma
