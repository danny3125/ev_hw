#
# Individual.py
#
#
from autochess_sys import system
import math
import random
import numpy as np
def activation(acti_type,x):
    if acti_type == 'sigmoid':
        return 1/(1+np.exp(-x))
    elif acti_type == 'binaryStep':
        return np.heaviside(x,1)
    elif acti_type == 'tanh':
        return np.tanh(x)
    elif acti_type == 'RELU':
        x1=[]
        for i in x:
            if i < 0:
                x1.append(0)
            else:
                x1.append(i)
        return x1
    elif acti_type == 'softmax':
        return np.exp(x) / np.sum(np.exp(x), axis=0)
    else :
        print('wrong type of activate function!!')
def unflatten(flattened,shapes):
    newarray = []
    index = 0
    for shape in shapes:
        size = np.product(shape)
        newarray.append(flattened[index : index + size].reshape(shape))
        index += size
    return newarray
class Individual:
    """
    Individual
    """  
    def __init__(self):
        class neural_network:
            def __init__(self,network): # network be like [[3,10,sigmoid],[None,1,sigmoid]]
                self.weights = []
                self.activations = []
                for layer in network:
                    if layer[0] != None:
                        input_size = layer[0]
                    else:
                        input_size = network[network.index(layer) - 1][1] 
                        # hidden layer's input size = former layer's output size
                    output_size = layer[1]
                    activation = layer[2]
                    self.weights.append(np.random.randn(input_size,output_size))
                    self.activations.append(activation)
            def binaryStep(self,x):
                ''' It returns '0' is the input is less then zero otherwise it returns one '''
                return np.heaviside(x,1)
            def tanh(self,x):
                ''' It returns the value (1-exp(-2x))/(1+exp(-2x)) and the value returned will be lies in between -1 to 1.'''
                return np.tanh(x)
            def sigmoid(self,x):
                return 1/(1+np.exp(-x))
            def RELU(self,x):
                x1=[]
                for i in x:
                    if i < 0:
                        x1.append(0)
                    else:
                        x1.append(i)
                return x1
            def softmax(self,x):
                return np.exp(x) / np.sum(np.exp(x), axis=0)
            def propagate(self,data):
                input_data = data
                for i in range(len(self.weights)):
                    z = np.dot(input_data,self.weights[i])
                    a = activation(self.activations[i],z)
                    input_data = a
                yhat = a
                return yhat
        self.strength= None
        chesstypelist = system.chesstypelist
        level = len(chesstypelist)
        types = max(chesstypelist)
        nn_output = sum(chesstypelist)
        self.hand_cards = np.zeros((3,level,types))
        self.money = 0
        self.level = 1
        self.maxchess = system.maxchess_num
        self.P_chess_nn = neural_network([[self.maxchess*3,20,'sigmoid'],[None,20,'sigmoid'],[None,20,'sigmoid'],[None,nn_output,'softmax']])
    def hand_cards(self,chess):
        self.hand_cards[0][chess[0]][chess[1]]+=1
    def refreshtable(self):
        self.money = self.money - 2
        return system.chessoffer(system,self.level)
                
    def state_check(self):# state_check should be a binary tree
        self.hand_cards[1] = self.hand_cards[0]/3
        self.hand_cards[1] = self.hand_cards[1].astype(int)
        self.hand_cards[1] = self.hand_cards[1].astype(float)
        self.hand_cards[0] = self.hand_cards[0]%3
        self.hand_cards[2] = self.hand_cards[1]/3
        self.hand_cards[2] = self.hand_cards[2].astype(int)
        self.hand_cards[2] = self.hand_cards[2].astype(float)
        self.hand_cards[1] = self.hand_cards[1]%3
    def picking_chess(self,level_distri):
        count_pro = 0
        for chess in self.chess_table:
            if level_distri[chess[0]][chess[1]] >= 1/sum(system.chesstypelist):
                self.money -= chess[0] + 1
                self.hand_cards[0][chess[0]][chess[1]]+=1
                self.state_check()
                count_pro += level_distri[chess[0]][chess[1]] * 2
                if self.money <= 0:
                    break
            else:
                count_pro += level_distri[chess[0]][chess[1]]
        if count_pro < 2/sum(system.chesstypelist):
            return False
        else:
            return True
    def choicetime(self, epoch):
        # in a single epoch, a player should go through three part : get money, buying, end epoch
        # buying is the most tricky part of this game, it needs a 'brain'
        if epoch % 5 == 0 and self.level < system.maxlevel:
            self.level += 1
        self.money += system.money_offer(self.money, epoch)
        #now I have chess table and money ,what chould I do ?
        #what I want is to build at least two nn for an individual, one nn look at the chess set it has,
        #and decide how much it wants for all type of chesses.
        #another focus on whether it should buy the chess on the chess table, based on the money it has.
        self.state_check()
        self.chess_table = system.chessoffer(system,self.level)
        #deciding whether to sell or not sell a chess 
        cards_inhand = np.nonzero(self.hand_cards) # cards_inhand is a len = 3 tuple
        all_cards_num = np.sum(self.hand_cards)
        while (all_cards_num > self.maxchess):
            self.money += system.chess_sell([cards_inhand[0][0],cards_inhand[1][0],cards_inhand[2][0]])
            self.hand_cards[cards_inhand[0][0]][cards_inhand[1][0]][cards_inhand[2][0]] -=1
            cards_inhand = np.nonzero(self.hand_cards)
            all_cards_num -= 1
        nn_input = [0]*(self.maxchess*3)    
        if len(cards_inhand[0]) > 0:
            index = 0
            for card in zip(cards_inhand[0],cards_inhand[1],cards_inhand[2]):
                for i in range(int(self.hand_cards[card[0]][card[1]][card[2]])):
                    nn_input[index], nn_input[index+1], nn_input[index+2] = card[0], card[1], card[2]
                    index += 3
        nn_input = np.array(nn_input)
        softmax_distri = self.P_chess_nn.propagate(nn_input)
        level_distri = []
        temp_index = 0
        for level in system.chesstypelist:
            level_distri.append(softmax_distri[temp_index:level+temp_index])
            temp_index += level
        # Now we have the softmax probability of all the chess type in the game, so, how do I use
        # this imformation ?
        while(True):
            if (self.picking_chess(level_distri) or self.money <= 0):
                break
            self.refreshtable()
        all_cards_num = np.sum(self.hand_cards)
        while (all_cards_num > self.maxchess):
            self.money += system.chess_sell([cards_inhand[0][0],cards_inhand[1][0],cards_inhand[2][0]])
            self.hand_cards[cards_inhand[0][0]][cards_inhand[1][0]][cards_inhand[2][0]] -=1
            cards_inhand = np.nonzero(self.hand_cards)
            all_cards_num -= 1
        if epoch == 20: 
            self.money = 0
            self.level = 1
        #if self.money > 60 or all_cards_num < self.level:
            #self.refreshtable()
    def evaluateFitness(self):
        if self.strength == None :
            self.strength = system.calculate_strength(system,self.hand_cards)
            chesstypelist = system.chesstypelist
            level = len(chesstypelist)
            types = max(chesstypelist)
            self.hand_cards = np.zeros((3,level,types))
    def crossover(self, other):
        shapes = [a.shape for a in self.P_chess_nn.weights]
        genes1 = np.concatenate([a.flatten() for a in self.P_chess_nn.weights])
        genes2 = np.concatenate([a.flatten() for a in other.P_chess_nn.weights])
        split = random.randint(0,len(genes1)-1)
        child1_genes = np.array(genes1[0:split].tolist() + genes2[split:].tolist())
        child2_genes = np.array(genes2[0:split].tolist() + genes1[split:].tolist())
        self.P_chess_nn.weights = unflatten(child1_genes,shapes)
        other.P_chess_nn.weights = unflatten(child2_genes,shapes)
        self.strength = None
        other.strength = None
    def mutation(self):
        if random.uniform(0.0, 1.0) <= 0.1:
            weights = self.P_chess_nn.weights
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
                self.P_chess_nn.weights = newarray 
        self.strength = None
            
                        
            
            
            
            
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
