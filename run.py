from CTRNN import CTRNN
from stochsearch import EvolSearch
import numpy as np
import matplotlib.pyplot as plt

# one network with two modes of behavior

# modes = ['limit cycle', 'EP']
modes = {'sin':0, 'cos':1}

# evolve a net that is capable of switching to two modes from an external manipulation

def fitness(cns, steps=200):
  #Assume 3 node ctrnn
  fitness = 0
  for k, v in modes.items():
    cns.states = np.zeros(cns.size) + 0.5
    states = np.array([])
    inputs = np.zeros(cns.size)
    inputs[0] = v
    for _ in range(steps):
      cns.euler_step(inputs) # list of the inputs for each neuron
      states = np.append(states, cns.outputs[-1])
      # if (states==np.NaN).any(): 
      #   return np.Inf
    measure = np.cos if k == 'cos' else np.sin
    fitness += np.mean(np.abs(states - measure(np.linspace(0, 2*np.pi, steps))))
  if np.isnan(fitness):
    return np.Inf
  
  return fitness
    
def makeCTRNN(genome, size=3):
  index = 0
  taus = genome[index:size] * 5
  index += size
  biases = genome[index:index+size] * 16
  index += size
  weights = np.reshape(genome[index:], (size,size))* 16
  cns = CTRNN(size,step_size=0.01*np.pi)
  cns.taus = taus
  cns.biases = biases
  cns.weights = weights
  return cns

def eval(genome):
  cns = makeCTRNN(genome)
  f = fitness(cns)
  return 100/f

# defining the parameters for the evolutionary search
evol_params = {
    'num_processes' : 4, # (optional) number of proccesses for multiprocessing.Pool
    'pop_size' : 100,    # population size
    'genotype_size': 15, # dimensionality of solution
    'fitness_function': eval, # custom function defined to evaluate fitness of a solution
    'elitist_fraction': 0.04, # fraction of population retained as is between generations
    'mutation_variance': 0.2 # mutation noise added to offspring.
}

es = EvolSearch(evol_params)

best_fit = []
mean_fit = []
num_gen = 0
max_num_gens = 100
desired_fitness = 100
#while es.get_best_individual_fitness() < desired_fitness and num_gen < max_num_gens:
while num_gen < max_num_gens:
    print('Gen #'+str(num_gen)+' Best Fitness = '+str(es.get_best_individual_fitness()))
    es.step_generation()
    best_fit.append(es.get_best_individual_fitness())
    mean_fit.append(es.get_mean_fitness())
    num_gen += 1

# print results
print('Max fitness of population = ',es.get_best_individual_fitness())
print('Best individual in population = ',es.get_best_individual())

# plot results
plt.figure()
plt.plot(best_fit)
plt.plot(mean_fit)
plt.xlabel('Generations')
plt.ylabel('Fitness')
plt.legend(['best fitness', 'avg. fitness'])
plt.show()