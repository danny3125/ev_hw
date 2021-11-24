#################################################################################
#
# Multi-objective binary-tournament selection
#  based on non-dominated front-ranking approach
#

#Notes: 1. In this code we will simplify things a bit by assuming that
#          minimization is desired for all objectives (in the more general
#          case you could have a mix of minimization and maximization on various objectives.)
#
#       2. As an example test case, we will use Deb's Min-Ex function here.
#          Min-Ex State is 2-D real, with 2 minimization objectives.
#
#################################################################################

#
# Instructions for this assignment:
#   1. Implement Individual.dominates method
#   2. Implement Individual.compareRankAndCrowding method
#   3. Implement Population.computeFrontRanks method
#   4. Implement Population.binaryTournament method
#   5. Run experiments and examine resulting data plots, do they make sense?
#

from random import Random
import math
import matplotlib.pyplot as plt
import copy

#
# Individual class
#
class Individual:
    def __init__(self,state=[],objectives=[],frontRank=None,crowdDist=None):
        """
        Individual Ctor
        """
        self.state=state
        self.objectives=objectives
        self.numObj=len(self.objectives)
        self.frontRank=frontRank
        self.crowdDist=crowdDist
        
    def dominates(self, other):# Include [obj1,obj2]
        """"
        Multi-objective domination comparison between self and other
        
          if self dominates other: return 1
          elif other dominates self: return -1
          else: return 0
        """
        #print('1', self.objectives, '2',other.objectives)
        if(self.objectives[0] > other.objectives[0] and self.objectives[1] <= other.objectives[1]) or (self.objectives[0] >= other.objectives[0] and self.objectives[1] < other.objectives[1]):
            return 1
        elif(self.objectives[0] > other.objectives[0] and self.objectives[1] > other.objectives[1]) or (self.objectives[0] > other.objectives[0] and self.objectives[1] < other.objectives[1]):
            return -1
        else:
            return 0
    
    def compareRankAndCrowding(self, other):
        """"
        Compare two individuals using frontRank and crowding distance metric
        
        Note: Compare by front-rank first. If front-ranks are equal, then compare using crowding metric.
        
        if self better than other: return 1
        elif self worse than other: return -1
        else: return 0
        """
        if self.frontRank > other.frontRank:
            return 1
        elif self.frontRank < other.frontRank:
            return -1
        else:
            if self.crowdDist > other.crowdDist:
                return 1
            elif self.crowdDist < other.crowdDist:
                return -1
            else:
                return 0

    def distance(self, other, normalizationVec=[None]):
        """
        Compute distance between self & other in objective space
        """
        # check if self vs self
        if self is other:
            return 0.0
        
        #set default normalization to 1.0, if not specified
        if normalizationVec[0] == None:
            normalizationVec=[1.0]*self.numObj
            
        # compute normalized Euclidian distance
        distance=0
        i=0
        while i < self.numObj:
            tmp=(self.objectives[i]-other.objectives[i])/normalizationVec[i]
            distance+=(tmp*tmp)
            i+=1
            
        distance=math.sqrt(distance)
        
        return distance  
              
    def __str__(self):
        """
        Stringify magic method
        """
        s=''
        s+='state     : ' + str(self.state) + '\n'
        s+='objectives: ' + str(self.objectives) + '\n'
        s+='frontRank : ' + str(self.frontRank) + '\n'
        s+='crowdDist : ' + str(self.crowdDist) + '\n'
        return s

    
#
# Population class
#
class Population:
    def __init__(self, pop=None):
        """
        Population Ctor
        """
        if pop is None:
            self.pop=[]
        else:
            self.pop=pop
        
    def computeCrowding(self):
        """
        Compute crowding metric using k-th nearest-neighbor w/ normalized distance.
        """
        
        if len(self.pop) == 0:
            return #nothing to do
        
        # if single objective, set all densities to zero then return
        if self.pop[0].numObj == 1:
            for ind in self.pop:
                ind.crowdDist=0.0
            return
                
        # compute k for knn density estimate
        kdist=int(math.sqrt(len(self.pop)))
        
        # compute normalization vector
        maxObj=self.pop[0].objectives.copy()
        minObj=self.pop[0].objectives.copy()
        for ind in self.pop:
            for i in range(ind.numObj):
                if ind.objectives[i] < minObj[i]: minObj[i]=ind.objectives[i]
                if ind.objectives[i] > maxObj[i]: maxObj[i]=ind.objectives[i]
        
        normVec=[]        
        for min,max in zip(minObj,maxObj):
            norm=math.fabs(max-min)
            if norm == 0: norm=1.0 #watch out for possible divide by zero problems
            normVec.append(norm)    
        
        # init distance matrix
        distanceMatrix=[]
        for i in range(len(self.pop)):
            distanceMatrix.append([0.0]*len(self.pop))

        # compute distance matrix
        # (matrix is diagonally symmetric so only need to compute half, then reflect)
        for i in range(len(self.pop)):
            for j in range(i+1):
                distanceMatrix[i][j]=self.pop[i].distance(self.pop[j],normVec)
                distanceMatrix[j][i]=distanceMatrix[i][j]
                      
        # sort the rows by distance
        for row in distanceMatrix:
            row.sort()
        
        # find the crowding distance using knn index
        i=0
        for ind in self.pop:
            ind.crowdDist=distanceMatrix[i][kdist]
            i+=1            

    def computeFrontRanks(self):
        """
        Compute non-dominated front ranks using NSGA-II front-ranking scheme
        """
        collect=[]
        for menber in self.pop:
            rank = 0
            for other in self.pop:
                rank += menber.dominates(other)
                collect.append(rank)
            menber.frontRank = rank

    def updateRanking(self):
        """
        Update front-rank and crowding distance for entire population
        """
        self.computeFrontRanks()
        self.computeCrowding()
    

    def binaryTournament(self,prng):
        
        """
        Multi-objective binary tournament operator based on non-domination front-ranking scheme.
        
        Input Parameters:
          prng: Random number generator (i.e., random.Random object)
    
        Note: Similar to single-objective implementation, 
          - Tournament pairs should be randomly selected
          - All individuals from initial population should participate in exactly 2 tournaments   
        """
            # generate random binary tournament pairs
        indexList1=list(range(len(self.pop)))
        indexList2=list(range(len(self.pop)))
        prng.shuffle(indexList1)
        prng.shuffle(indexList2)
        
        # do not allow self competition
        for i in range(len(self.pop)):
            if indexList1[i] == indexList2[i]:
                temp_2=indexList2[i]
                if i == 0:
                    
                    indexList2[i]=indexList2[-1]
                    indexList2[-1]=temp_2
                else:
                    indexList2[i]=indexList2[i-1]
                    indexList2[i-1]=temp_2
        
        #compete
        newPop=[]        
        for index1,index2 in zip(indexList1,indexList2):
            temp_1 = self.pop[index1]
            result_of_compare = temp_1.compareRankAndCrowding(self.pop[index2])
            if  result_of_compare == 1:
                
                newPop.append(self.pop[index1])
            elif result_of_compare == -1:
                newPop.append(self.pop[index2])
            else:
                rn=prng.random()
                if rn > 0.5:
                    newPop.append(self.pop[index1])
                else:
                    
                    newPop.append(self.pop[index2])
        #It's your job to implement this function!
        #
        
        # overwrite old pop with newPop (i.e., the selected pop)   
        self.pop=newPop  
    
    def generatePlots(self,title=None,showScreen=True,saveToFile=False,fileName=None):
        """
        Generate plots for:
          1. Individuals in state space
          2. Individuals in objective space
          3. Non-dominated ranked fronts in objective space
        
          Note: This method only uses the first two dimensions
                 in both state & objective space.  If state or objective
                 space dimensionality < 2, exception will be thrown
        """
        
        #first, make sure state & objective space have at least 2 dimensions, pop size at least 1
        if len(self.pop) < 1:
            raise Exception('showPlots error: Population size must be >= 1 !')
        if (len(self.pop[0].state) < 2) or (len(self.pop[0].objectives) < 2):
            raise Exception('showPlots error: State & objective spaces must have at least 2 dimensions!')
    
        #if front ranking has not been computed, then skip
        # the front-rank plot
        if self.pop[0].frontRank is None: plotOrder=[121,122,000]
        else: plotOrder=[131,132,133]
        
        #top-level attributes for collection of subplots
        if title is not None:
            fig, axs = plt.subplots(13)
            fig.suptitle(title)
        plt.subplots_adjust(wspace=0.75) #increase spacing between plots a bit
        
        #individuals in state space
        plt.subplot(plotOrder[0])
        x=[ind.state[0] for ind in self.pop]
        y=[ind.state[1] for ind in self.pop]
        plt.scatter(x,y)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('State Space')
        
        
        #individuals in objective space
        plt.subplot(plotOrder[1])
        x=[ind.objectives[0] for ind in self.pop]
        y=[ind.objectives[1] for ind in self.pop]
        plt.scatter(x,y)
        plt.xlabel('f1')
        plt.ylabel('f2')
        plt.title('Objective Space')
        
        
        #Note: If front ranks have not been computed, then
        #      skip the frontRank plot...
        if self.pop[0].frontRank is not None:
            #non-dominated ranked fronts in objective space
            plt.subplot(plotOrder[2])   
            
            #first, let's find highest front rank
            maxRank=0
            for ind in self.pop:
                if ind.frontRank > maxRank: maxRank=ind.frontRank
            
            rank=0
            while rank <= maxRank:
                xy=[ind.objectives for ind in self.pop if ind.frontRank == rank]
                xy.sort(key=lambda obj: obj[0]) #need to sort in 1st dim to make connected line plots look sensible!
                x=[obj[0] for obj in xy]
                y=[obj[1] for obj in xy]
                plt.plot(x,y,marker='o',label=str(rank))
                rank+=1
                
            plt.xlabel('f1')
            plt.ylabel('f2')
            plt.title('Ranked Fronts')
    
        #write plots to file?
        if saveToFile:
            plt.savefig(fileName)
        
        #display on screen?
        if showScreen:
            plt.show()
    
    def __str__(self):
        """
        Stringify magic method
        """
        s=''
        for ind in self.pop:
            s+=str(ind)+'\n'
        return s
      
#
#Random Population initializer based on Deb's MinEx benchmark
#  State: 2-D real, Objectives: 2-D min/min
#
def minExInitializer(popSize,prng):
    #x1, x2 range limits
    x1_min=0.1
    x1_max=1.0
    
    x2_min=0.0
    x2_max=5.0
    
    #generate random population of Individuals
    # based on MinEx benchmark
    population=[]
    
    i=0
    while i < popSize:
        x1=prng.uniform(x1_min,x1_max)
        x2=prng.uniform(x2_min,x2_max)
        
        obj1=x1
        obj2=(1+x2)/x1
        
        population.append(Individual([x1,x2], [obj1,obj2]))
        
        i+=1
        #print(population)
    return population


#
#Let's test our binary selection operator by iteratively
# applying it to an initial random population.  Examine
# the effects by plotting the state space, objective space & ranked fronts
#
print('--- Starting Binary Tournaments ---')
popSize = 500
numGenerations = 10
prng=Random()
prng.seed(456)

pop=Population(minExInitializer(popSize, prng))
pop.updateRanking()
pop.generatePlots(title='Generation: Init')

for generation in range(numGenerations):
    print('Generation: ' + str(generation))
    pop.binaryTournament(prng)
    pop.updateRanking()
    pop.generatePlots(title='Generation: ' + str(generation))
    
print('--- Finished Binary Tournaments ---\n\n')

