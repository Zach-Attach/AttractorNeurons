from CTRNN import CTRNN
from stochsearch import EvolSearch
import numpy as np
import jax.numpy as jnp
from jax import jit
import matplotlib.pyplot as plt

# evolve a net that is capable of switching to two modes from an external manipulation

# CONSTANTS
WAVELENGTH: float = 4*jnp.pi # 2pi is the wavelength of a sin/cos wave
INIT_STATE: float = 0.5 # initial state of all neurons
STEP_SIZE: float = 0.01 # step size for the euler integration
NUM_GENS: int = 100 # number of generations to run the evolutionary search
NUM_TIME_UNITS: float = 40. # number of time units to run the CTRNN for
NUM_STEPS: int = int(NUM_TIME_UNITS/STEP_SIZE) # number of steps to run the CTRNN for
POP_SIZE: int = 100 # population size
NUM_NODES = 4 # number of neurons in the CTRNN
GENE_SCALES: list = [1,32,32] # scales each gene for the taus, biases, and weights
CHOSEN_FUNC: str = 'derivative' # the fitness function to use
# GENE_SCALES: list = [1,16,16] # scales each gene for the taus, biases, and weights
def wave(x): return jnp.cos(x * 3) # the wave to be predicted
MODES: list = [jnp.sin, jnp.cos] #wave] # the two modes of behavior


FITNESS_FUNCS: dict = {
  'diff': lambda states, yhat: jnp.abs(states-yhat).sum(), # difference between the predicted and expected output
  'derivative': lambda states, yhat: jnp.abs(jnp.diff(states[-500:])-jnp.diff(yhat[-500:])).sum(), # difference between the derivative of the predicted and expected output
}

# Helper Functions

@jit
def adjustWave(x): # adjust the wave to be between 0 and 1
  return (x+1.)/2.

# @jit
def getOutputs(cns, input): # get the outputs of the last neuron in the CTRNN after inputing an input value to the 1st neuron
  states = jnp.array([])
  inputArr = jnp.zeros(NUM_NODES)
  # inputArr[0] = input
  inputArr = inputArr.at[0].set(input)

  for _ in range(NUM_STEPS):
    cns.euler_step(inputArr)
    states = jnp.append(states, cns.outputs[-1])
  return states

# @jit
def fitness(cns, f:str = None):
  fitness = 0.
  samplePoints = jnp.linspace(0., WAVELENGTH, NUM_STEPS)
  states: list = []
  yhats = []

  for i, measure in enumerate(MODES):
    yhats.append(adjustWave(measure(samplePoints))) # get the expected output

    cns.states = jnp.full(NUM_NODES, INIT_STATE) # reset the states of all neurons
    states.append(getOutputs(cns, i)) # get the outputs of the CTRNN

    if f:
      x = FITNESS_FUNCS[f](states[-1], yhats[-1])
      # print('TYPE of X: ', type(x), x)
      fitness += FITNESS_FUNCS[f](states[-1], yhats[-1])
  
  return fitness, states, yhats, samplePoints

# @jit
def plot(cns):
  _, ypred, yhat, x = fitness(cns)

  for i in range(len(MODES)):
    plt.plot(x.block_until_ready(), yhat[i].block_until_ready(), 'r', label='Expected')
    plt.plot(x.block_until_ready(), ypred[i].block_until_ready(), 'b', label='CTRNN')
    plt.show()

@jit
def geno2pheno(genome):
  genes = jnp.split(genome, [NUM_NODES,2*NUM_NODES]) # split the genome
  taus, biases, weights = [GENE_SCALES[i]*genes[i] for i in range(3)] # scale genes values to appropriate ranges
  biases -= 16 # shift the biases to be between -16 and 16
  weights -= 16 # shift the weights to be between -16 and 16
  weights = jnp.reshape(weights, (NUM_NODES,NUM_NODES)) # reshape the weights to be a square matrix
  return taus, biases, weights

# @jit
def makeCTRNN(genome):
  cns = CTRNN(NUM_NODES,step_size=STEP_SIZE) # create a CTRNN of {size} neurons

  cns.taus, cns.biases, cns.weights = geno2pheno(genome) # set the taus, biases, and weights of the CTRNN

  return cns

def eval(genome):
  cns = makeCTRNN(genome)
  f = float(fitness(cns, CHOSEN_FUNC)[0])
  # print('FITNESS: ', type(f), f)
  return 0. if jnp.isnan(f) else 1./f

# defining the parameters for the evolutionary search
EVOL_PARAMS: dict = {
    'num_processes' : 8, # (optional) number of proccesses for multiprocessing.Pool
    'pop_size' : POP_SIZE,    # population size
    'genotype_size': NUM_NODES*2+NUM_NODES**2, # dimensionality of solution
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

    # print(f'Gen #{g} Best Fitness = {bestFit}')

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