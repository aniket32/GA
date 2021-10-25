import numpy as np
import matplotlib.pyplot as plt
from ypstruct import structure
import ga
import random
import math


# Sphere Test Function
def a(x):
    return sum(x ** 2)


# Rastrigin Test Function
def b(x):
    fitness = 10 * len(x)
    for i in range(len(x)):
        fitness += x[i] ** 2 - (10 * math.cos(2*math.pi*x[i]))
    return fitness

# First try to create 30 constraints for 30 dimensionality
# min = [random.randint(-10, 10) for i in range(0, 30)]
# max = [random.randint(1, 10) for j in range(0, 30)]

min = [-5.12] * 30
max = [5.12] * 30

# Problem Definition
problem = structure()
# Change value between a and b to change the fitness function
problem.costfunc = a
# problem.nvar = 5
# problem.varmin = [-10, -10, -1, -5,  4]
# problem.varmax = [ 10,  10,  1,  5, 10]
# Total dimensionality
problem.nvar = 30
problem.varmin = min
problem.varmax = max

# GA Parameters
params = structure()
#  Mx number of iteration
params.maxit = 1000
# Total population size
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
