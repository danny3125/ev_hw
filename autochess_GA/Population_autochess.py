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
import matplotlib.pyplot as plt
'''
The auto chess game begins with eight individuals,each of them have a initial state:
    money, hand_cards(chess),strength of hand_cards,level, the population class have to build the whole
    game process, with the help of autochess_sys.
'''

class Population:
    """
    Population
    """
    uniprng=None
    crossoverFraction=None
    playtimes = None
    def __init__(self, populationSize, total_epochs):
        """
        Population constructor
        """
        self.population=[]
        for i in range(populationSize):
            self.population.append(Individual())  #this part should be modified to load the individuals                                                                                                                                    
        self.total_epochs = total_epochs
        self.pointlist = []
        self.goodpointlist = []
        self.best_ten_list = []
        self.best_list = []
    def game_on(self):
        for epoch in range(1,self.total_epochs+1):
            for player in self.population:
                player.choicetime(epoch)
                player.strength = None
        for player in self.population:
            player.money = 0
            player.level = 1
    def __len__(self):
        return len(self.population)
    
    def __getitem__(self,key):
        return self.population[key]
    
    def __setitem__(self,key,newValue):
        self.population[key]=newValue
        
    def evaluateFitness(self):
        for individual in self.population: individual.evaluateFitness()
    def suming_Fitness(self):
        for individual in self.population: 
            individual.suming_Fitness += individual.strength
    def Clean_Fitness(self):
        for individual in self.population:
            individual.strength = individual.suming_Fitness / self.playtimes
            individual.suming_Fitness = 0
    def copy(self):
        return copy.deepcopy(self)
        
    def crossover(self):
        indexList1=list(range(len(self)))
        indexList2=list(range(len(self)))
        self.uniprng.shuffle(indexList1)
        self.uniprng.shuffle(indexList2)
        if self.crossoverFraction == 1.0:    
            for index1,index2 in zip(indexList1,indexList2):
                self[index1].crossover(self[index2])
        else:
            for index1,index2 in zip(indexList1,indexList2):
                rn=self.uniprng.random()
                if rn < self.crossoverFraction:
                    self[index1].crossover(self[index2])

    def mutate(self):     
        for individual in self.population:
            individual.mutation()           
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
            if self[index1].strength > self[index2].strength:
                newPop.append(copy.deepcopy(self[index1]))
            elif self[index1].strength < self[index2].strength:
                newPop.append(copy.deepcopy(self[index2]))
                
            else:
                if np.sum(self[index1].hand_cards) > np.sum(self[index2].hand_cards):
                    newPop.append(copy.deepcopy(self[index2]))
                elif np.sum(self[index1].hand_cards) < np.sum(self[index2].hand_cards):
                    newPop.append(copy.deepcopy(self[index1]))
                else:
                    rn=self.uniprng.random()
                    if rn > 0.5:
                        newPop.append(copy.deepcopy(self[index1]))
                    else:
                        newPop.append(copy.deepcopy(self[index2]))      
        # overwrite old pop with newPop    
        self.population=newPop    
    def combinePops(self,otherPop):
        self.population.extend(otherPop.population)

    def clean_hand(self):
        for player in self.population:
            player.cleanhands()
    def truncateSelect(self,newPopSize):
        #sort by fitness
        avg_strength = 0
        self.population.sort(key=attrgetter('strength'),reverse=True)
        for item in self.population:
            avg_strength += item.strength
        avg_strength /= len(self.population)
        print('all_avg:',avg_strength)
        avg_strength_quarter = 0
        for item in self.population[:int(len(self.population)/4)]:
            avg_strength_quarter += item.strength
        avg_strength_quarter /= (len(self.population)/4)
        avg_ten_strength = 0
        for item in self.population[:10]:
            avg_ten_strength += item.strength
        avg_ten_strength/=10
        print('best_quarter avg:',avg_strength_quarter)
        print('best_avg_ten_strength:',avg_ten_strength)
        print('best_strength:',self.population[0].strength)
        self.pointlist.append(avg_strength)
        self.goodpointlist.append(avg_strength_quarter)
        self.best_ten_list.append(avg_ten_strength)
        self.best_list.append(self.population[0].strength)
        plt.plot(self.pointlist, 'k--+')   # black dashed line, with "+" markers
        plt.plot(self.goodpointlist, 'gd')     # green dimonds (no line)
        plt.plot(self.best_ten_list, 'r:')     # red dotted line (no marker)
        plt.plot(self.best_list, 'b:')     
        plt.show()        #then truncate the bottom
        self.population=self.population[:newPopSize] 


        
       
