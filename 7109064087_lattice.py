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

import optparse
import sys
import yaml
import math
from random import Random
from Population_lattice import *


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
             'minLimit': (float,True),
             'maxLimit': (float,True),
             'selfEnergyVector':(list,True),
             'interactionEnergyMatrix':(list,True),
             'latticeLength': (int,True),
             'numParticleTypes':(int,True)}
     
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
         

#Simple fitness function example: 1-D Rastrigin function
#        
def fitnessFunc(ball_line,matrix,v_vector):
    fit_value = 0
    print(ball_line)
    for i in range(len(ball_line)):
        if (i == 0):
            fit_value += v_vector[ball_line[i]] + matrix[ball_line[i]][ball_line[i+1]]
        elif (i == (len(ball_line) - 1)):
            fit_value += v_vector[ball_line[i]] + matrix[ball_line[i-1]][ball_line[i]]            
        else:
            fit_value += v_vector[ball_line[i]] + matrix[ball_line[i-1]][ball_line[i]] + matrix[ball_line[i]][ball_line[i+1]]
        
    return (-fit_value)


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


#EV3:
#            
def ev3(cfg):
    #start random number generators
    uniprng=Random()
    uniprng.seed(cfg.randomSeed)
    normprng=Random()
    normprng.seed(cfg.randomSeed+101)

    #set static params on classes
    # (probably not the most elegant approach, but let's keep things simple...)
    Individual.minLimit=cfg.minLimit
    Individual.maxLimit=cfg.maxLimit
    Individual.fitFunc=fitnessFunc
    Individual.uniprng=uniprng
    Individual.normprng=normprng
    Individual.latticeLength=cfg.latticeLength
    Individual.numParticleTypes = cfg.numParticleTypes
    Individual.selfEnergyVector = cfg.selfEnergyVector
    Individual.interactionEnergyMatrix = cfg.interactionEnergyMatrix
    Population.uniprng=uniprng
    Population.crossoverFraction=cfg.crossoverFraction
    
    
    #create initial Population (random initialization)
    population=Population(cfg.populationSize)
    #print initial pop stats    
    printStats(population,0)

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
        print('here')
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
        print('hello')  
        print(cfg.interactionEnergyMatrix)          
        #run EV3
        ev3(cfg)
        
        if not options.quietMode:                    
            print('EV3 Completed!')    
    
    except Exception as info:
        if 'options' in vars() and options.debugMode:
            from traceback import print_exc
            print_exc()
        else:
            print(info)
    

if __name__ == '__main__':
    main()
    
