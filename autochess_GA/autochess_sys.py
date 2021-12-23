#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 21:45:49 2021

@author: wangyiren
"""
import copy
import numpy as np
class system:
    faction_threshold = None
    faction_strength_table = None
    chesstypelist = None
    maxlevel = None
    table_length = None
    maxchess_num = None
    def __init__(self):
        self.proba_table = [0] * len(chesstypelist)
        self.proba_table[0] = 1
        self.proba_matrix = []
        self.strength_table = [i for i in range(1,len(chesstypelist)+1)]
        self.faction_strength_table = faction_strength_table # waiting for a file call
        self.chesstypelist = chesstypelist # chesstypelist = [level1_types,level2....] len = max chess level
        self.faction_threshold = faction_threshold # waiting for a file call
        self.maxlevel = maxlevel
        self.table_length = table_length
        self.maxchess_num = maxchess_num
        
        for level in range(1,self.maxlevel+1):
            self.proba_matrix.append(copy.deepcopy(self.proba_table))
            for i in range(len(self.proba_table)):
                if self.proba_table[0] > 0.3 and self.proba_table[i] > 0.1:
                    if i == 0 :
                        self.proba_table[0] = 0.8**level
                        spread = 1 - (self.proba_table[0])
                    if i < (len(self.proba_table) - 1):
                        self.proba_table[i+1] = spread * 0.8 
                        spread = spread - self.proba_table[i+1]
                elif self.proba_table[i] < 0.1:
                    self.proba_table[i] += spread
                    
                else:
                    break
        
        
    
    def chessoffer(self, level):
        temp = self.proba_matrix[level-1]
        choice = np.random.choice(len(temp),self.table_length,p = temp)
        final_choice = []
        for item in choice :
            type_inlevel = np.random.randint(0,self.typelist[int(item)],1)
            final_choice.append([item,int(type_inlevel[0])])
        return final_choice #final_choice is like [[1,0],[0,2],..] ,length = table_length
    
    def calculate_strength(self,chess_set):#chess set = [[1,0],[0,2],......]
        bonus_strength = []
        strength = 0
        for item in chess_set:
            strength += self.strength_table[item[0]]
            bonus_strength.append(self.faction_strength_table[int(item[0])][int(item[1])])
            print(bonus_strength)
            maxfrac = max(bonus_strength)
            bonus_strength = np.array(bonus_strength)
            for frac in range(maxfrac):
                bonus_strength-=1
                count = 0
                for i in bonus_strength:
                    if i == 0:
                        count+=1
                if count >= self.faction_threshold[0][frac]:
                    strength += self.faction_threshold[1][frac]
            bonus_strength += maxfrac
            bonus_strength = bonus_strength.tolist()
        return strength
    def money_offer(self, money, epoch):
        if epoch > 5:
            epoch = 5
        return epoch + int(money/10)  
    def chess_sell(self, chess): #chess like [0,3,2] => level,rank,type of chess 
        if chess[1] == 0:
            return (chess[0]*2 + 1)
        else:
            return (chess[1]+1)*(chess[0]*2+1)
        
            
    
                

        
        
        
        