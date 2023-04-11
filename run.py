from CTRNN import CTRNN
from stochsearch import EvolSearch
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter('ignore', RuntimeWarning) # ignore divide by zero warnings

# evolve a net that is capable of switching to two modes from an external manipulation

# CONSTANTS
WAVELENGTH: float = 4*np.pi # 2pi is the wavelength of a sin/cos wave
STEP_SIZE: float = 0.01 # step size for the euler integration
NUM_GENS: int = 100 # number of generations to run the evolutionary search
NUM_TIME_UNITS: float = 40. # number of time units to run the CTRNN for
NUM_STEPS: int = int(NUM_TIME_UNITS/STEP_SIZE) # number of steps to run the CTRNN for
POP_SIZE: int = 100 # population size
NUM_NODES = 4 # number of neurons in the CTRNN
GENE_SCALES: list[int] = [1,32,32] # scales each gene for the taus, biases, and weights
CHOSEN_FUNC: str = 'derivative' # the fitness function to use
def wave(x): return np.cos(x * 3) # the wave to be predicted
MODES: list = [np.sin, np.cos] #wave] # the two modes of behavior

X_AXIS: np.ndarray[float] = np.linspace(0., WAVELENGTH, NUM_STEPS)
INIT_STATE: float = 0.5 # initial state of all neurons


FITNESS_FUNCS: dict = {
  'diff': lambda states, yhat: np.abs(states-yhat).sum(), # difference between the predicted and expected output
  'derivative': lambda states, yhat: np.abs(np.diff(states[int(-.9*NUM_STEPS):])-np.diff(yhat[int(-.9*NUM_STEPS):])).sum(), # difference between the derivative of the predicted and expected output
}

# defining the parameters for the evolutionary search
EVOL_PARAMS: dict = {
    'num_processes' : 8, # (optional) number of proccesses for multiprocessing.Pool
    'pop_size' : POP_SIZE,    # population size
    'genotype_size': NUM_NODES*2+NUM_NODES**2, # dimensionality of solution
    'elitist_fraction': 0.04, # fraction of population retained as is between generations
    'mutation_variance': 0.2 # mutation noise added to offspring.
}

# Helper Functions

def adjustWave(x: np.ndarray[np.float64]) -> np.ndarray[np.float64]: # adjust the wave to be between 0 and 1
  return (x+1.)/2.

def getOutputs(cns: CTRNN, input: int) -> np.ndarray[np.float64]: # get the outputs of the last neuron in the CTRNN after inputing an input value to the 1st neuron
  states: np.ndarray = np.array([])
  inputArr: np.ndarray[np.int32] = np.zeros(NUM_NODES)
  inputArr[0] = input

  for _ in range(NUM_STEPS):
    cns.euler_step(inputArr)
    states: np.ndarray[np.float64] = np.append(states, cns.outputs[-1])
  return states

def fitness(cns: CTRNN, f: str) -> list:
  fitness: float = 0.

  states: list[np.ndarray[np.float64]] = []
  yhats: list[np.ndarray] = []

  for i, measure in enumerate(MODES):
    yhats.append(adjustWave(measure(X_AXIS))) # get the expected output

    cns.states = np.full(NUM_NODES, INIT_STATE) # reset the states of all neurons
    states.append(getOutputs(cns, i)) # get the outputs of the CTRNN

    fitness += FITNESS_FUNCS[f](states[-1], yhats[-1])
  
  return fitness, states, yhats

def geno2pheno(genome) -> list[np.ndarray]:
  genes: list[np.ndarray] = np.split(genome, [NUM_NODES,2*NUM_NODES]) # split the genome
  taus, biases, weights = [GENE_SCALES[i]*genes[i] for i in range(3)] # scale genes values to appropriate ranges
  biases -= GENE_SCALES[1]/2 # shift the biases to be between -16 and 16
  weights -= GENE_SCALES[2]/2 # shift the weights to be between -16 and 16
  weights = np.reshape(weights, (NUM_NODES,NUM_NODES)) # reshape the weights to be a square matrix
  return taus, biases, weights

def makeCTRNN(genome):
  cns = CTRNN(NUM_NODES,step_size=STEP_SIZE) # create a CTRNN of {size} neurons

  cns.taus, cns.biases, cns.weights = geno2pheno(genome) # set the taus, biases, and weights of the CTRNN``

  return cns

def eval(genome: np.ndarray):
  cns: CTRNN = makeCTRNN(genome)
  f = fitness(cns, CHOSEN_FUNC)[0]
  return 0 if np.isnan(f) else 1./f

EVOL_PARAMS['fitness_function'] = eval # custom function defined to evaluate fitness of a solution

def plot(cns: CTRNN):
  _, ypred, yhat = fitness(cns, CHOSEN_FUNC)

  for i in range(len(MODES)):
    plt.plot(X_AXIS, yhat[i], 'r', label='Expected')
    plt.plot(X_AXIS, ypred[i], 'b', label='CTRNN')
    plt.show()

es = EvolSearch(EVOL_PARAMS)

bestList = [] # list of best fitnesses
meanList = [] # list of mean fitnesses

for g in range(NUM_GENS):
    es.step_generation()

    bestFit = es.get_best_individual_fitness()
    meanFit = es.get_mean_fitness()

    bestList.append(bestFit)
    meanList.append(meanFit)

    print(f'Gen #{g} Best Fitness = {bestFit}')

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