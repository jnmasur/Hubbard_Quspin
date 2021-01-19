from deap import base, creator, tools
import numpy as np
import random
from minimization_methods import *
from tools import parameter_instantiate as hhg
from multiprocessing import Pool

"""Hubbard model Parameters"""
L = 10  # system size
N_up = L // 2 + L % 2  # number of fermions with spin up
N_down = L // 2  # number of fermions with spin down
N = N_up + N_down  # number of particles
t0 = 0.52  # hopping strength
pbc = True

"""Laser pulse parameters"""
field = 32.9  # field angular frequency THz
F0 = 10  # Field amplitude MV/cm

# target parameters
target_U = 1 * t0
target_a = 4
target_x = [target_U, target_a]

# add all parameters to the class and create the basis
params = Parameters(L, N_up, N_down, t0, field, F0, pbc)
params.set_basis()

J_target = current_expectation_power_spectrum(target_x, params)

# bounds for variables
U_upper = 10 * params.t
U_lower = 0
a_upper = 10
a_lower = 0
bounds = ((U_lower, U_upper), (a_lower, a_upper))

num_threads = 2


def init_individual(container, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    randU = np.random.random() * (U_upper - U_lower)
    randa = np.random.random() * (a_upper - a_lower)
    return container([randU, randa])


def evaluator(individual, J, p):
    return minimize(objective, individual, args=(J,p), bounds=bounds).fval,


def mutator(individual, mu=None, sigma=None, indpb=.8, seed=None):
    if mu is None:
        mu = (individual[0], individual[1])
    if sigma is None:
        mu = ((U_upper-U_lower)*.15, (a_upper-a_lower)*.15)
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)  # used in tools.mutGaussian()
    tools.mutGaussian(individual, mu, sigma, indpb)
    # if parameters are out of bounds, this guaranteed that they will not be any more
    if individual[0] < U_lower:
        individual[0] = U_lower + (np.random.random() * abs(U_lower - individual[0]))
    elif individual[0] > U_upper:
        individual[0] = U_upper - (np.random.random() * abs(U_upper - individual[0]))
    if individual[1] < a_lower:
        individual[1] = a_lower + (np.random.random() * abs(a_lower - individual[1]))
    elif individual[1] > a_upper:
        individual[1] = a_upper - (np.random.random() * abs(a_upper - individual[1]))


def cloner(individual, container, nclones=1):
    if nclones == 1:
        new_ind = container([individual[0], individual[1]])
        new_ind.fitness.values = individual.fitness.values
        return new_ind
    elif nclones > 1:
        ret = []
        for i in range(nclones):
            ret.append(container(individual[0], individual[1]))
            ret[i].fitness.values = individual.fitness.values
        return ret


def repopulator(spop, popsize, container, clone_func, mut_func):
    """
    Populate new_pop with individuals of spop and their mutants, so that there
    are popsize individuals in new_pop
    :param spop: population containing the best individuals from an iteration
    :param popsize: number of total individuals we want in the population
    :param container: the container for the individuals
    :param clone_func: function that performs cloning (see cloner)
    :param mut_func: the function that performs mutation (see mutator)
    :return: new_pop which contains all individuals in spop and mutants of them
    """
    nclones = (popsize - len(spop)) // len(spop)
    new_pop = []
    # populate with mutants
    for ind in spop:
        new_pop.append(ind)
        mutants = clone_func(ind, container, nclones)
        for mutant in mutants:
            mut_func(mutant)
            del mutant.fitness.values
        new_pop.extend(mutants)
    for i in range(popsize % len(spop)):
        mutant = clone_func(spop[i], container)
        mut_func(mutant)
        del mutant.fitness.values
        new_pop.append(mutant)
    return new_pop


# create fitness and individual that will have a fitness
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# define a bunch of functions for the algorithm
toolbox = base.Toolbox()
toolbox.register("individual", init_individual, creator.Individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=num_threads)
toolbox.register("evaluate", evaluator, J=J_target, p=params)  # parameter: x (individual)
toolbox.register("mutate", mutator)  # parameters: individual, low, up, indpb, seed(optional)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("select", tools.selTournament)
toolbox.register("clone", cloner, container=toolbox.Individual)
toolbox.register("repopulate", popsize=num_threads, container=creator.Individual, clone_func=toolbox.clone,
                 mut_func=toolbox.mutate)

"""Actual start of algorithm"""
pop = toolbox.population()
hof = tools.HallOfFame(10)

# evaluate individuals in pop
with Pool(num_threads) as pool:
    fitnesses = pool.map(toolbox.evaluate, pop)
    for i in range(len(pop)):
        pop[i].fitness.values = fitnesses[i]

NSURVIVORS = 10
TSIZE = 20
NGEN = 30
for gen in range(NGEN):

    # mate individuals
    for ind1, ind2 in zip(pop[::2], pop[1::2]):
        child1, child2 = toolbox.clone(ind1), toolbox.clone(ind2)
        toolbox.mate(child1, child2)
        del child1.fitness.values
        del child2.fitness.values
        pop.append(child1)
        pop.append(child2)

    # find all invalid fitnesses, remove them from pop, add to invalid_pop
    invalid_pop = []
    for i in range(len(pop)):
        if not pop[i].fitness.valid:
            ind = pop.pop(i)
            invalid_pop.append(ind)
    # evaluate all individuals with invalid fitness
    with Pool() as pool:
        invalid_fitnesses = pool.map(toolbox.evaluate, invalid_pop)
        for i in range(len(invalid_pop)):
            invalid_pop[i].fitness.values = invalid_fitnesses[i]

    pop.extend(invalid_pop)

    new_pop = toolbox.select(pop, NSURVIVORS, TSIZE)  # select best individuals
    hof.update(new_pop)  # update hall of fame
    pop = toolbox.repopulate(new_pop)  # repopulate by mutating from individuals in new_pop

# afterwards processing can be done on hof which contains the best individuals
