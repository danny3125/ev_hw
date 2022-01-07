
import optparse
import sys
import yaml
import math
from random import Random
from Population_autochess import *
from autochess_sys import *
import multiprocessing as mp
from functools import partial


#EV3 Config class 
class EV3_Config:
    """
    EV3 configuration class
    """
    # class variables
    sectionName='EV3'
    options={'populationSize': (int,True),
             'total_epochs': (int,True),
             'generationCount': (int,True),
             'crossoverFraction':(float,True),
             'faction_strength_table':(list,True),
             'proba_matrix':(list,True),
             'faction_threshold':(list,True),
             'maxlevel':(int,True),
             'chesstypelist':(list,True),
             'maxchess_num':(int,True),
             'table_length':(int,True),
             'playtimes':(int,True),
             'randomSeed':(int,True)
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
         

        
def ev3(cfg):
    uniprng=Random()
    uniprng.seed(cfg.randomSeed)
    normprng=Random()
    normprng.seed(cfg.randomSeed+101)
    #set static params on classes
    # (probably not the most elegant approach, but let's keep things simple...)
    system.faction_strength_table = cfg.faction_strength_table
    system.faction_threshold = cfg.faction_threshold
    system.chesstypelist = cfg.chesstypelist
    system.maxlevel = cfg.maxlevel
    system.table_length = cfg.table_length
    system.maxchess_num = cfg.maxchess_num
    system.proba_matrix = cfg.proba_matrix
    Individual.uniprng=uniprng
    Individual.normprng=normprng
    #create initial Population (random initialization)
    Population.uniprng=uniprng
    Population.crossoverFraction=cfg.crossoverFraction
    Population.playtimes = cfg.playtimes
    population=Population(cfg.populationSize,cfg.total_epochs)
    
    #print initial pop stats    
    #printStats(population,0)

    #evolution main loop
    for i in range(cfg.generationCount):
        #create initial offspring population by copying parent pop
        print('generation:',i)
        offspring=population.copy()
        offspring.evaluateFitness()
        offspring.conductTournament()
        offspring.clean_hand()
        offspring.crossover()
        offspring.mutate()
        offspring.game_on()
        offspring.evaluateFitness()
        population.evaluateFitness()
        population.game_on()
        population.evaluateFitness()
        population.combinePops(offspring)
        population.truncateSelect(cfg.populationSize)
        
def island(cfg):
    pool = mp.Pool(processes = (mp.cpu_count() - 6))
    task = partial(ev3, cfg = cfg)
    result = pool.map(task, [1,2])
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
    main(['-i','autochess.cfg', '-d'])
    
