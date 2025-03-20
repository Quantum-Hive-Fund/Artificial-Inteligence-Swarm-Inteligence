Artificial Intelligence - Swarm Intelligence

One of the areas I find extremely interesting in Artificial Intelligence is Swarm Intelligence.

Swarm Intelligence is a field of artificial intelligence based on the collective behavior of decentralized and self-organized systems. There are different types of Swarm Intelligence, each with specific applications. Here are the main ones:


---

Particle Swarm Optimization (PSO)

What it is:

An algorithm inspired by the behavior of bird flocks and fish schools for global function optimization.

Applications:

Hyperparameter optimization in machine learning

Solving complex optimization problems (e.g., mathematical functions)

Dynamic system control


How it works:

Each particle represents a possible solution and moves towards the best-known position based on its own experience and that of the group.

Python Implementation:

import random
import math

class Particle:
    def __init__(self, dim, minx, maxx):
        self.position = [random.uniform(minx, maxx) for _ in range(dim)]
        self.velocity = [0.0 for _ in range(dim)]
        self.best_pos = self.position.copy()
        self.best_score = float('inf')

class PSO:
    def __init__(self, dim, num_particles, iterations):
        self.dim = dim
        self.num_particles = num_particles
        self.iterations = iterations
        self.swarm = [Particle(dim, -5, 5) for _ in range(num_particles)]
        self.global_best = [0.0]*dim
        self.global_best_score = float('inf')

    def fitness(self, position):
        return sum(x**2 for x in position)

    def optimize(self):
        for _ in range(self.iterations):
            for particle in self.swarm:
                current_score = self.fitness(particle.position)
                
                if current_score < particle.best_score:
                    particle.best_score = current_score
                    particle.best_pos = particle.position.copy()
                
                if current_score < self.global_best_score:
                    self.global_best_score = current_score
                    self.global_best = particle.position.copy()

            for particle in self.swarm:
                for i in range(self.dim):
                    w = 0.5  # Inertia
                    c1 = 1   # Cognitive
                    c2 = 2   # Social

                    new_velocity = (w * particle.velocity[i] +
                                    c1 * random.random() * (particle.best_pos[i] - particle.position[i]) +
                                    c2 * random.random() * (self.global_best[i] - particle.position[i]))
                    
                    particle.velocity[i] = new_velocity
                    particle.position[i] += particle.velocity[i]

        return self.global_best

# Execution
pso = PSO(dim=2, num_particles=30, iterations=100)
result = pso.optimize()
print("Best solution found:", result)
print("Objective function value:", sum(x**2 for x in result))

Go Implementation:

package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

type Particle struct {
	position  []float64
	velocity  []float64
	bestPos   []float64
	bestScore float64
}

type PSO struct {
	dim           int
	numParticles  int
	iterations    int
	swarm         []Particle
	globalBest    []float64
	globalBestVal float64
}

func NewPSO(dim, numParticles, iterations int) *PSO {
	rand.Seed(time.Now().UnixNano())
	pso := &PSO{
		dim:           dim,
		numParticles:  numParticles,
		iterations:    iterations,
		globalBestVal: math.Inf(1),
	}
	
	pso.swarm = make([]Particle, numParticles)
	for i := range pso.swarm {
		particle := Particle{
			position:  make([]float64, dim),
			velocity:  make([]float64, dim),
			bestPos:   make([]float64, dim),
			bestScore: math.Inf(1),
		}
		for j := range particle.position {
			particle.position[j] = rand.Float64()*10 - 5
		}
		pso.swarm[i] = particle
	}
	return pso
}

func (pso *PSO) fitness(position []float64) float64 {
	sum := 0.0
	for _, x := range position {
		sum += x * x
	}
	return sum
}

func (pso *PSO) Optimize() []float64 {
	for iter := 0; iter < pso.iterations; iter++ {
		for i := range pso.swarm {
			currentScore := pso.fitness(pso.swarm[i].position)
			
			if currentScore < pso.swarm[i].bestScore {
				pso.swarm[i].bestScore = currentScore
				copy(pso.swarm[i].bestPos, pso.swarm[i].position)
			}
			
			if currentScore < pso.globalBestVal {
				pso.globalBestVal = currentScore
				pso.globalBest = make([]float64, pso.dim)
				copy(pso.globalBest, pso.swarm[i].position)
			}
		}

		for i := range pso.swarm {
			for j := range pso.swarm[i].position {
				w := 0.5  // Inertia
				c1 := 1.0 // Cognitive
				c2 := 2.0 // Social

				newVelocity := w*pso.swarm[i].velocity[j] +
					c1*rand.Float64()*(pso.swarm[i].bestPos[j]-pso.swarm[i].position[j]) +
					c2*rand.Float64()*(pso.globalBest[j]-pso.swarm[i].position[j])

				pso.swarm[i].velocity[j] = newVelocity
				pso.swarm[i].position[j] += pso.swarm[i].velocity[j]
			}
		}
	}
	return pso.globalBest
}

func main() {
	pso := NewPSO(2, 30, 100)
	result := pso.Optimize()
	fmt.Printf("Best solution found: %v\n", result)
	fmt.Printf("Objective function value: %f\n", pso.fitness(result))
}


---

Artificial Bee Colony (ABC)

What it is:

Inspired by the foraging behavior of honeybee colonies, where bees optimize food source selection.

Applications:

Numerical optimization

Training of neural networks

Software engineering (testing and optimization)


How it works:

The population of bees is divided into "employed bees", "onlooker bees", and "scout bees", which explore and exploit different solution sources.


---

Python Implementation:

import random
import math

class Bee:
    def __init__(self, dim, minx, maxx):
        self.position = [random.uniform(minx, maxx) for _ in range(dim)]
        self.fitness = float('inf')
        self.trials = 0

class ABC:
    def __init__(self, dim, colony_size=20, limit=10, max_iter=100):
        self.dim = dim
        self.colony_size = colony_size
        self.limit = limit
        self.max_iter = max_iter
        self.minx, self.maxx = -5, 5
        self.best_solution = None
        self.best_fitness = float('inf')
        self.employed = [Bee(dim, self.minx, self.maxx) for _ in range(colony_size//2)]
        self.onlookers = [Bee(dim, self.minx, self.maxx) for _ in range(colony_size//2)]

    def calculate_fitness(self, position):
        return sum(x**2 for x in position)

    def optimize(self):
        for _ in range(self.max_iter):
            # Employed bees phase
            for bee in self.employed:
                new_solution = bee.position.copy()
                j = random.randint(0, self.dim-1)
                phi = random.uniform(-1, 1)
                new_solution[j] += phi * (new_solution[j] - random.choice(self.employed).position[j])
                new_solution[j] = max(min(new_solution[j], self.maxx), self.minx)
                
                new_fitness = self.calculate_fitness(new_solution)
                if new_fitness < bee.fitness:
                    bee.position = new_solution
                    bee.fitness = new_fitness
                    bee.trials = 0
                else:
                    bee.trials += 1

            # Onlooker bees phase
            total = sum(math.exp(-bee.fitness) for bee in self.employed)
            for bee in self.onlookers:
                r = random.uniform(0, total)
                cumulative = 0.0
                for emp in self.employed:
                    cumulative += math.exp(-emp.fitness)
                    if cumulative >= r:
                        j = random.randint(0, self.dim-1)
                        phi = random.uniform(-1, 1)
                        new_solution = emp.position.copy()
                        new_solution[j] += phi * (new_solution[j] - random.choice(self.employed).position[j])
                        new_solution[j] = max(min(new_solution[j], self.maxx), self.minx)
                        
                        new_fitness = self.calculate_fitness(new_solution)
                        if new_fitness < bee.fitness:
                            bee.position = new_solution
                            bee.fitness = new_fitness
                            bee.trials = 0
                        else:
                            bee.trials += 1
                        break

            # Scout bees phase
            all_bees = self.employed + self.onlookers
            for bee in all_bees:
                if bee.trials >= self.limit:
                    bee.position = [random.uniform(self.minx, self.maxx) for _ in range(self.dim)]
                    bee.fitness = self.calculate_fitness(bee.position)
                    bee.trials = 0

                if bee.fitness < self.best_fitness:
                    self.best_fitness = bee.fitness
                    self.best_solution = bee.position.copy()

        return self.best_solution

# Usage:
abc = ABC(dim=2, colony_size=20, limit=10, max_iter=100)
result = abc.optimize()
print("Best solution:", result)
print("Fitness:", sum(x**2 for x in result))


---

Stochastic Diffusion Search (SDS)

What it is:

An algorithm based on indirect communication between agents to solve distributed search problems.

Applications:

Pattern detection

Distributed search

Robust optimization in noisy environments


How it works:

Agents perform independent searches and share information about good solutions, increasing the efficiency of the collective search.


---

Python Implementation:

import numpy as np
import random

class SDSAgent:
    def __init__(self, search_space):
        self.position = random.uniform(search_space[0], search_space[1])  # Initial agent position
        self.active = False  # Active/inactive state

    def evaluate(self, function):
        """Evaluates the function at the agent's position"""
        return function(self.position)

    def test_hypothesis(self, function, threshold=0.1):
        """Probabilistic test to keep an agent active"""
        self.active = np.random.rand() < threshold

    def communicate(self, other_agent):
        """If inactive, copy the position of an active agent"""
        if not self.active:
            self.position = other_agent.position

class StochasticDiffusionSearch:
    def __init__(self, function, search_space, num_agents=20, iterations=100):
        self.function = function
        self.search_space = search_space
        self.agents = [SDSAgent(search_space) for _ in range(num_agents)]
        self.iterations = iterations

    def optimize(self):
        """Executes Stochastic Diffusion Search"""
        for _ in range(self.iterations):
            # Hypothesis testing
            for agent in self.agents:
                agent.test_hypothesis(self.function)
            
            # Information diffusion
            for agent in self.agents:
                if not agent.active:
                    other = random.choice(self.agents)
                    if other.active:
                        agent.communicate(other)
            
            # Random position update for exploration
            for agent in self.agents:
                if random.random() < 0.2:  # Small chance of exploration
                    agent.position = random.uniform(self.search_space[0], self.search_space[1])
        
        # Best solution found
        best_agent = max(self.agents, key=lambda a: a.evaluate(self.function))
        return best_agent.position, self.function(best_agent.position)

# Define objective function
def objective_function(x):
    return np.sin(x) + np.cos(2*x)

# Run SDS
sds = StochasticDiffusionSearch(objective_function, search_space=(-10, 10))
best_x, best_value = sds.optimize()
print(f"Best solution found: x = {best_x:.4f}, f(x) = {best_value:.4f}")


---



