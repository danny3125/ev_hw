#
# Individual.py
#
#
from autochess_sys import *
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

class Individual:
    """
    Individual
    """  
    def __init__(self):
        class neural_network:
            def __init__(self,network): # network be like [[3,10,sigmoid],[None,1,sigmoid]]
                self.weight = []
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
                    a = self.activations[i](z)
                    input_data = a
                yhat = a
                return yhat
        self.strength= 0
        chesstypelist = system.chesstypelist
        level = len(chesstypelist)
        types = max(chesstypelist)
        nn_output = sum(chesstypelist)
        self.hand_cards = np.zeros((3,level,types))
        self.money = 0
        self.level = 1
        self.fit=None
        self.chess_table = system.chessoffer(self.level)
        self.maxchess = system.maxchess_num
        self.P_chess_nn = neural_network([[self.maxchess*3,10,'RELU'],[None,10,'RELU'],[None,nn_output,'softmax']])
    def hand_cards(self,chess):
        self.hand_cards[0][chess[0]][chess[1]]+=1
    def refreshtable(self):
        self.money = self.money - 2
        return system.chessoffer(self.level)
                
    def state_check(self):# state_check should be a binary tree
        self.hand_cards[1] = self.hand_cards[0]/3
        self.hand_cards[1] = self.hand_cards[1].astype(int)
        self.hand_cards[1] = self.hand_cards[1].astype(float)
        self.hand_cards[2] = self.hand_cards[1]/3
        self.hand_cards[2] = self.hand_cards[2].astype(int)
        self.hand_cards[2] = self.hand_cards[2].astype(float)
    def picking_chess(self,level_distri):
        for chess in self.chess_table:
            if level_distri[chess[0]][chess[1]] > 1/sum(system.chesstypelist):
                self.money -= chess[0]
                self.hand_cards[0][chess[0]][chess[1]]+=1
                self.state_check()
            else:
                pass
    def choicetime(self, epoch):
        # in a single epoch, a player should go through three part : get money, buying, end epoch
        # buying is the most tricky part of this game, it needs a 'brain'
        self.money += system.money_offer(self.money, epoch)
        #now I have chess table and money ,what chould I do ?
        #what I want is to build at least two nn for an individual, one nn look at the chess set it has,
        #and decide how much it wants for all type of chesses.
        #another focus on whether it should buy the chess on the chess table, based on the money it has.
        self.state_check()
        chess_table = systen.chessoffer(self.level)
        #deciding whether to sell or not sell a chess 
        cards_inhand = np.nonzero(self.hand_cards) # cards_inhand is a len = 3 tuple
        all_cards_num = sum(self.hand_cards)
        
        while all_cards_num > self.maxchess:
            self.money += system.chess_sell([cards_inhand[0][0],cards_inhand[1][0],cards_inhand[2][0]])
            self.hand_cards[cards_inhand[0][0]][cards_inhand[1][0]][cards_inhand[2][0]] -=1
            cards_inhand = np.nonzero(self.hand_cards)
            
        nn_input = [0]*(self.maxchess*3)    
        for i in range(len(cards_inhand[0])):
            nn_input[3*i] = cards_inhand[0][i]
            nn_input[3*i+1] = cards_inhand[1][i]
            nn_input[3*i+2] = cards_inhand[2][i]
        temp_hand_cards = self.hand_cards - 1
        cards_inhand = np.nonzero(temp_hand_cards)
        for i in range(len(cards_inhand[0])):
            nn_input[3*i] = cards_inhand[0][i]
            nn_input[3*i+1] = cards_inhand[1][i]
            nn_input[3*i+2] = cards_inhand[2][i]
        nn_input = np.araray(nn_input)
        softmax_distri = self.P_chess_nn.prpagate(nn_input)
        while(self.money > 0):
            level_distri = []
            temp_index = 0
            for level in system.chesstypelist:
                level_distri.append(softmax_distri[temp_index:level+temp_index])
                temp_index += level
            # Now we have the softmax probability of all the chess type in the game, so, how do I use
            # this imformation ?
            self.picking_chess()
            if self.money > 60 or all_cards_num < self.maxchess:
                self.refreshtable(level_distri)
            
                        
            
            
            
            
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
