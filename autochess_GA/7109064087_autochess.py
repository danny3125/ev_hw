#
# ev3.py: An elitist (mu+mu) generational-with-overlap EA
#
#
# To run: python ev3.py --input ev3_example.cfg
#         python ev3.py --input my_params.cfg
#
# Basic features of ev3:
#   - Supports self-adaptive mutation
#   - Uses binary tournament selection for mating pool
#   - Uses elitist truncation selection for survivors
#
'''
So, we begin on a set of chesses with max rank 2, three types of rank 1 chess, one type of rank 2, 
with a serie of bonus strangth if you combine a rank 2 chess with a particular rank 1 chess,
8 players are included in one game, with health like 10, with max player level like 5,
the state we have included the chess table, unsorted, the 'strength value' your set will become if you 
take an action like 'take it all',or 'just pick one'
there is no limit on the number of fight rounds, but every five rounds, there is a 'strength test' for
every player, if the player lose, they will loss health, 
the imformation the players will get on each round includes: health, set strangth, chess table, money.  
the action the player can choose is to decide the money you want to keep in this round,
and how to use the money you decide to spend.
'''

import optparse
import sys
import yaml
import math
from random import Random
from Population_autochess import *
from autochess_sys import *


#EV3 Config class 
class EV3_Config:
    """
    EV3 configuration class
    """
    # class variables
    sectionName='EV3'
    options={'populationSize': (int,True),
             'generationCount': (int,True),
             'randomSeed': (int,True),
             'crossoverFraction': (float,True),
             'interactionEnergyMatrix':(list,True),
             'faction_strength_table':(list,True),
             'faction_threshold':(list,True),
             'maxlevel':(int,True)
             'chesstypelist':(list,True)
             }
     
    #constructor
    def __init__(self, inFileName):
        #read YAML config and get EV3 section
        infile=open(inFileName,'r',encoding="utf-8")
        ymlcfg=yaml.safe_load(infile)
        infile.close()
        eccfg=ymlcfg.get(self.sectionName,None)
        if eccfg is None: raise Exception('Missing {} section in cfg file'.format(self.sectionName))
         
        #iterate over options
        for opt in self.options:
            if opt in eccfg:
                optval=eccfg[opt]
 
                #verify parameter type
                if type(optval) != self.options[opt][0]:
                    raise Exception('Parameter "{}" has wrong type'.format(opt))
                 
                #create attributes on the fly
                setattr(self,opt,optval)
            else:
                if self.options[opt][1]:
                    raise Exception('Missing mandatory parameter "{}"'.format(opt))
                else:
                    setattr(self,opt,None)
     
    #string representation for class data    
    def __str__(self):
        return str(yaml.dump(self.__dict__,default_flow_style=False))
         


#Print some useful stats to screen
def printStats(pop,gen):
    print('Generation:',gen)
    avgval=0
    minval=pop[0].fit
    sigma=pop[0].sigma
    for ind in pop:
        avgval+=ind.fit
        if ind.fit < minval:
            minval=ind.fit
            sigma=ind.sigma
    print('Min energy cost',-minval)
    print('Sigma',sigma)
    print('Avg energy cost',-(avgval/len(pop)))
    print('')
def fitfunc()

#EV3:
#            
def ev3(cfg):
    #set static params on classes
    # (probably not the most elegant approach, but let's keep things simple...)
    system.faction_strength_table = cfg.faction_strength_table
    system.faction_threshold = cfg.faction_threshold
    system.chesstypelist = cfg.chesstypelist
    system.maxlevel = cfg.maxlevel
    #create initial Population (random initialization)
    population=Population(cfg.populationSize)
    #print initial pop stats    
    #printStats(population,0)

    #evolution main loop
    for i in range(cfg.generationCount):
        #create initial offspring population by copying parent pop
        offspring=population.copy()

        #select mating pool
        offspring.conductTournament()

        #perform crossover
        offspring.crossover()

        #random mutation
        offspring.mutate()

        #update fitness values
        offspring.evaluateFitness()
        #survivor selection: elitist truncation using parents+offspring
        population.combinePops(offspring)
        population.truncateSelect(cfg.populationSize)
        
        #print population stats    
        printStats(population,i+1)
        
        
#
# Main entry point
#
def main(argv=None):
    if argv is None:
        argv = sys.argv
        
    try:
        #
        # get command-line options
        #
        parser = optparse.OptionParser()
        parser.add_option("-i", "--input", action="store", dest="inputFileName", help="input filename", default=None)
        parser.add_option("-q", "--quiet", action="store_true", dest="quietMode", help="quiet mode", default=False)
        parser.add_option("-d", "--debug", action="store_true", dest="debugMode", help="debug mode", default=False)
        (options, args) = parser.parse_args(argv)
        
        #validate options
        if options.inputFileName is None:
            raise Exception("Must specify input file name using -i or --input option.")
        
        #Get EV3 config params
        cfg=EV3_Config(options.inputFileName)
        
        #print config params
        print(cfg)      
        #run EV3
        ev3(cfg)
        
        if not options.quietMode:                    
            print('EA autochess Completed!')    
    
    except Exception as info:
        if 'options' in vars() and options.debugMode:
            from traceback import print_exc
            print_exc()
        else:
            print(info)
    

if __name__ == '__main__':
    main()
    
