#
# Population.py
#
#
from autochess_sys import *
import copy
import math
from operator import attrgetter
from Individual_autochess import *
import random
import numpy as np
'''
The auto chess game begins with eight individuals,each of them have a initial state:
    money, hand_cards(chess),strength of hand_cards,level, the population class have to build the whole
    game process, with the help of autochess_sys.
'''

class Population:
    """
    Population
    """
    
    def __init__(self, populationSize, total_epochs):
        """
        Population constructor
        """
        self.population=[]
        for i in range(populationSize):
            self.population.append(Individual())    #this part should be modified to load the individuals                                                                                                                                    
        self.total_epochs = total_epochs
    
    def game_on(self):
        for epoch in range(1,self.total_epochs+1):
            for player in self.population:
                player.choicetime(epoch)
        
    def __len__(self):
        return len(self.population)
    
    def __getitem__(self,key):
        return self.population[key]
    
    def __setitem__(self,key,newValue):
        self.population[key]=newValue
        
    def copy(self):
        return copy.deepcopy(self)
        
    def crossover(self):
        offspring = []
        for _ in range((pop_size - len(self.population)) // 2):
            parent1 = random.choice(agents)
            parent2 = random.choice(agents)
            child1 = Agent(network)
            child2 = Agent(network)
            
            shapes = [a.shape for a in parent1.neural_network.weights]
            
            genes1 = np.concatenate([a.flatten() for a in parent1.neural_network.weights])
            genes2 = np.concatenate([a.flatten() for a in parent2.neural_network.weights])
            
            split = random.ragendint(0,len(genes1)-1)
            child1_genes = np.asrray(genes1[0:split].tolist() + genes2[split:].tolist())
            child2_genes = np.array(genes1[0:split].tolist() + genes2[split:].tolist())
            
            child1.neural_network.weights = unflatten(child1_genes,shapes)
            child2.neural_network.weights = unflatten(child2_genes,shapes)
            
            offspring.append(child1)
            offspring.append(child2)
        agents.extend(offspring)
        return agents
    def mutation(self):
        for agent in agents:
            if random.uniform(0.0, 1.0) <= 0.1:
                weights = agent.neural_network.weights
                shapes = [a.shape for a in weights]
            flattened = np.concatenate([a.flatten() for a in weights])
            randint = random.randint(0,len(flattened)-1)
            flattened[randint] = np.random.randn()
        newarray = []
        indeweights = 0
        for shape in shapes:
            size = np.product(shape)
            newarray.append(flattened[indeweights : indeweights + size].reshape(shape))
            indeweights += size
            agent.neural_network.weights = newarray
        return agents             
    def conductTournament(self):#choose the better networks 
        # binary tournament
        indexList1=list(range(len(self)))
        indexList2=list(range(len(self)))
        
        self.uniprng.shuffle(indexList1)
        self.uniprng.shuffle(indexList2)
        
        # do not allow self competition
        for i in range(len(self)):
            if indexList1[i] == indexList2[i]:
                temp=indexList2[i]
                if i == 0:
                    indexList2[i]=indexList2[-1]
                    indexList2[-1]=temp
                else:
                    indexList2[i]=indexList2[i-1]
                    indexList2[i-1]=temp
        
        #compete
        newPop=[]        
        for index1,index2 in zip(indexList1,indexList2):
            if self[index1].fit > self[index2].fit:
                newPop.append(copy.deepcopy(self[index1]))
            elif self[index1].fit < self[index2].fit:
                newPop.append(copy.deepcopy(self[index2]))
            else:
                rn=self.uniprng.random()
                if rn > 0.5:
                    newPop.append(copy.deepcopy(self[index1]))
                else:
                    newPop.append(copy.deepcopy(self[index2]))
        
        # overwrite old pop with newPop    
        self.population=newPop        


        
       
