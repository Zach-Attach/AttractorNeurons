from CTRNN import CTRNN
from stochsearch import EvolSearch
import numpy as np
import matplotlib.pyplot as plt

# evolve a net that is capable of switching to two modes from an external manipulation

# CONSTANTS
WAVELENGTH: float = 2*np.pi # 2pi is the wavelength of a sin/cos wave
INIT_STATE: float = 0.5 # initial state of all neurons
STEP_SIZE: float = 0.01 # step size for the euler integration
NUM_GENS = 100 # number of generations to run the evolutionary search
GENE_SCALES: list = [1,16,16] # scales each gene for the taus, biases, and weights
MODES: list = [np.sin, np.cos] # the two modes of behavior

FITNESS_FUNCS: dict = {
  'diff': lambda states, yhat: np.abs(states-yhat).sum(), # difference between the predicted and expected output
  'derivative': lambda states, yhat: np.abs(np.diff(states)-np.diff(yhat)).sum(), # difference between the derivative of the predicted and expected output
}

# Helper Functions

def adjustWave(x: np.ndarray[np.float64]): # adjust the wave to be between 0 and 1
  return (x+1)/2

def getOutputs(cns, input, steps=200): # get the outputs of the last neuron in the CTRNN after inputing an input value to the 1st neuron
  states = np.array([])
  inputArr = np.zeros(cns.size)
  inputArr[0] = input

  for _ in range(steps):
    cns.euler_step(inputArr)
    states = np.append(states, cns.outputs[-1])
  return states

def fitness(cns, f, steps=200):
  fitness = 0
  samplePoints = np.linspace(0, WAVELENGTH, steps)

  for i, measure in enumerate(MODES):
    yhat = adjustWave(measure(samplePoints)) # get the expected output

    cns.states = np.full(cns.size, INIT_STATE) # reset the states of all neurons
    states = getOutputs(cns, i, steps) # get the outputs of the CTRNN

    fitness += FITNESS_FUNCS[f](states, yhat)
  
  return fitness, states, yhat, samplePoints
    
def plot(cns, steps=200):
  _, ypred, yhat, x = fitness(cns, 'diff', steps)

  plt.plot(x, yhat, 'r', label='Expected')
  plt.plot(x, ypred, 'b', label='CTRNN')
  plt.show()


def makeCTRNN(genome, size=3):
  cns = CTRNN(size,step_size=STEP_SIZE) # create a CTRNN of {size} neurons

  genes = np.split(genome, [size,2*size]) # split the genome
  cns.taus, cns.biases, weights = [GENE_SCALES[i]*genes[i] for i in range(3)] # scale genes values to appropriate ranges
  cns.weights = np.reshape(weights, (size,size)) # reshape the weights to be a square matrix

  return cns

def eval(genome):
  cns = makeCTRNN(genome)
  f = fitness(cns, 'diff')[0]
  return 0 if np.isnan(f) else 100/f

# defining the parameters for the evolutionary search
EVOL_PARAMS: dict = {
    'num_processes' : 4, # (optional) number of proccesses for multiprocessing.Pool
    'pop_size' : 100,    # population size
    'genotype_size': 15, # dimensionality of solution
    'fitness_function': eval, # custom function defined to evaluate fitness of a solution
    'elitist_fraction': 0.04, # fraction of population retained as is between generations
    'mutation_variance': 0.2 # mutation noise added to offspring.
}

es = EvolSearch(EVOL_PARAMS)

bestList = [] # list of best fitnesses
meanList = [] # list of mean fitnesses

for g in range(NUM_GENS):
    es.step_generation()

    bestFit = es.get_best_individual_fitness()
    meanFit = es.get_mean_fitness()

    bestList.append(bestFit)
    meanList.append(meanFit)

    print(f'Gen #${g} Best Fitness = ${bestFit}')

# print results
print('Max fitness of population = ',es.get_best_individual_fitness(),
      '\nBest individual in population = ',es.get_best_individual())

# plot results
plt.figure()
plt.plot(bestList)
plt.plot(meanList)
plt.xlabel('Generations')
plt.ylabel('Fitness')
plt.legend(['best fitness', 'avg. fitness'])
plt.show()

plot(makeCTRNN(es.get_best_individual()))