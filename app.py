import numpy as np
import matplotlib.pyplot as plt
from ypstruct import structure
import ga
import random
import math


# Sphere Test Function
def sphere(x):
    return sum(x ** 2)


def rastrigin(x):
    fitness = 10 * len(x)
    for i in range(len(x)):
        fitness += x[i] ** 2 - (10 * math.cos(2*math.pi*x[i]))
    return fitness


min = [-5.12] * 30
max = [5.12] * 30

# Problem Definition
problem = structure()
problem.costfunc = rastrigin
# problem.nvar = 5
# problem.varmin = [-10, -10, -1, -5,  4]
# problem.varmax = [ 10,  10,  1,  5, 10]

problem.nvar = 30
problem.varmin = min
problem.varmax = max

# GA Parameters
params = structure()
params.maxit = 1000
params.npop = 100
params.beta = 1
params.pc = 1
params.gamma = 0.1
params.mu = 0.01
params.sigma = 0.1

# Run GA
out = ga.run(problem, params)

# Results
plt.plot(out.bestcost)
# plt.semilogy(out.bestcost)
plt.xlim(0, params.maxit)
plt.xlabel('Iterations')
plt.ylabel('Best Cost')
plt.title('Genetic Algorithm (GA)')
plt.grid(True)
plt.show()
