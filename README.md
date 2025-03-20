# Inteligência Artificial - Swarm Intelligence

Uma das áreas que eu acho super interessante da Inteligência Artificial é o Swarm Inteligence, conheça seus algoritmos:

Swarm Intelligence (Inteligência de Enxame) é um campo da inteligência artificial baseado no comportamento coletivo de sistemas descentralizados e auto-organizados. Existem diferentes tipos de Swarm Intelligence, cada um com aplicações específicas. Aqui estão os principais:

## Particle Swarm Optimization (PSO)

O que é:
Algoritmo inspirado no comportamento de bandos de pássaros e cardumes de peixes para otimização global de funções.

Aplicações:
Otimização de hiperparâmetros em machine learning
Resolução de problemas de otimização complexos (ex.: funções matemáticas)
Controle de sistemas dinâmicos

Como funciona:
Cada partícula representa uma possível solução e se move em direção à melhor posição conhecida com base em sua experiência e na experiência do grupo.


```py
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
                    w = 0.5  # Inércia
                    c1 = 1   # Cognitivo
                    c2 = 2   # Social

                    new_velocity = (w * particle.velocity[i] +
                                    c1 * random.random() * (particle.best_pos[i] - particle.position[i]) +
                                    c2 * random.random() * (self.global_best[i] - particle.position[i]))
                    
                    particle.velocity[i] = new_velocity
                    particle.position[i] += particle.velocity[i]

        return self.global_best

# Execução
pso = PSO(dim=2, num_particles=30, iterations=100)
result = pso.optimize()
print("Melhor solução encontrada:", result)
print("Valor na função objetivo:", sum(x**2 for x in result))
```

Exemplo em Go:

```go
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
				w := 0.5  // Inércia
				c1 := 1.0 // Cognitivo
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
	fmt.Printf("Melhor solução encontrada: %v\n", result)
	fmt.Printf("Valor na função objetivo: %f\n", pso.fitness(result))
}
```
## Ant Colony Optimization (ACO)

O que é:
Inspirado no comportamento das formigas na busca por alimentos usando feromônios para encontrar caminhos eficientes.

Aplicações:
Problemas de roteamento (ex.: TSP – Problema do Caixeiro Viajante)
Otimização de redes (roteamento de pacotes na internet)
Planejamento logístico

Como funciona:
As formigas artificiais depositam feromônios nos caminhos percorridos, reforçando rotas mais curtas e eficientes ao longo do tempo.

Exemplo em Python:

```py
import random
import math

class ACO_TSP:
    def __init__(self, cities, num_ants=10, iterations=100, decay=0.5, alpha=1, beta=2):
        self.cities = cities
        self.num_ants = num_ants
        self.iterations = iterations
        self.decay = decay
        self.alpha = alpha  # Importância do feromônio
        self.beta = beta    # Importância da distância
        self.pheromone = [[1.0 for _ in cities] for _ in cities]

    def distance(self, city1, city2):
        return math.hypot(city1[0]-city2[0], city1[1]-city2[1])

    def run(self):
        best_path = None
        best_distance = float('inf')

        for _ in range(self.iterations):
            paths = []
            for ant in range(self.num_ants):
                visited = [False]*len(self.cities)
                path = [random.randint(0, len(self.cities)-1)]
                visited[path[0]] = True

                while len(path) < len(self.cities):
                    current = path[-1]
                    probs = []
                    total = 0.0

                    for city in range(len(self.cities)):
                        if not visited[city]:
                            pheromone = self.pheromone[current][city] ** self.alpha
                            heuristic = (1.0 / self.distance(self.cities[current], self.cities[city])) ** self.beta
                            probs.append((city, pheromone * heuristic))
                            total += pheromone * heuristic

                    # Seleção probabilística
                    r = random.uniform(0, total)
                    upto = 0.0
                    for city, prob in probs:
                        upto += prob
                        if upto >= r:
                            path.append(city)
                            visited[city] = True
                            break

                # Atualiza melhor caminho
                total_dist = sum(self.distance(self.cities[path[i]], self.cities[path[i+1]]) for i in range(len(path)-1))
                if total_dist < best_distance:
                    best_distance = total_dist
                    best_path = path

            # Atualiza feromônios
            for i in range(len(self.cities)):
                for j in range(len(self.cities)):
                    self.pheromone[i][j] *= self.decay

            # Adiciona feromônio do melhor caminho
            for i in range(len(best_path)-1):
                a = best_path[i]
                b = best_path[i+1]
                self.pheromone[a][b] += 1.0 / best_distance

        return best_path, best_distance

# Uso:
cities = [(0,0), (1,2), (3,1), (4,3)]
aco = ACO_TSP(cities, num_ants=15, iterations=50)
path, distance = aco.run()
print("Melhor caminho:", path, "Distância:", distance)
```

Exemplo em Go:

```go
package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

type ACO struct {
	cities     [][2]float64
	numAnts    int
	iterations int
	decay      float64
	alpha      float64
	beta       float64
	pheromone  [][]float64
}

func NewACO(cities [][2]float64, numAnts, iterations int, decay, alpha, beta float64) *ACO {
	aco := &ACO{
		cities:     cities,
		numAnts:    numAnts,
		iterations: iterations,
		decay:      decay,
		alpha:      alpha,
		beta:       beta,
	}
	aco.pheromone = make([][]float64, len(cities))
	for i := range aco.pheromone {
		aco.pheromone[i] = make([]float64, len(cities))
		for j := range aco.pheromone[i] {
			aco.pheromone[i][j] = 1.0
		}
	}
	return aco
}

func (aco *ACO) distance(city1, city2 [2]float64) float64 {
	return math.Hypot(city1[0]-city2[0], city1[1]-city2[1])
}

func (aco *ACO) Run() ([]int, float64) {
	rand.Seed(time.Now().UnixNano())
	bestPath := make([]int, len(aco.cities))
	bestDistance := math.Inf(1)

	for iter := 0; iter < aco.iterations; iter++ {
		for ant := 0; ant < aco.numAnts; ant++ {
			visited := make([]bool, len(aco.cities))
			path := []int{rand.Intn(len(aco.cities))}
			visited[path[0]] = true

			for len(path) < len(aco.cities) {
				current := path[len(path)-1]
				var probs []struct {
					city int
					prob float64
				}
				total := 0.0

				for city := range aco.cities {
					if !visited[city] {
						pheromone := math.Pow(aco.pheromone[current][city], aco.alpha)
						heuristic := math.Pow(1.0/aco.distance(aco.cities[current], aco.cities[city]), aco.beta)
						prob := pheromone * heuristic
						probs = append(probs, struct{ city int; prob float64 }{city, prob})
						total += prob
					}
				}

				// Seleção probabilística
				r := rand.Float64() * total
				upto := 0.0
				for _, p := range probs {
					upto += p.prob
					if upto >= r {
						path = append(path, p.city)
						visited[p.city] = true
						break
					}
				}
			}

			// Calcula distância
			totalDist := 0.0
			for i := 0; i < len(path)-1; i++ {
				totalDist += aco.distance(aco.cities[path[i]], aco.cities[path[i+1]])
			}

			if totalDist < bestDistance {
				bestDistance = totalDist
				copy(bestPath, path)
			}
		}

		// Atualiza feromônios
		for i := range aco.pheromone {
			for j := range aco.pheromone[i] {
				aco.pheromone[i][j] *= aco.decay
			}
		}

		// Adiciona feromônio do melhor caminho
		for i := 0; i < len(bestPath)-1; i++ {
			a := bestPath[i]
			b := bestPath[i+1]
			aco.pheromone[a][b] += 1.0 / bestDistance
		}
	}

	return bestPath, bestDistance
}

func main() {
	cities := [][2]float64{{0, 0}, {1, 2}, {3, 1}, {4, 3}}
	aco := NewACO(cities, 15, 50, 0.5, 1, 2)
	path, dist := aco.Run()
	fmt.Printf("Melhor caminho: %v\nDistância: %.2f\n", path, dist)
}
```

## Artificial Bee Colony (ABC)

O que é:
Inspirado no comportamento de colônias de abelhas na busca e otimização de fontes de alimento.

Aplicações:
Otimização numérica
Treinamento de redes neurais
Engenharia de software (testes e otimização)

Como funciona:
Divide a população de abelhas em "operárias", "observadoras" e "batedoras", que exploram e exploram diferentes fontes de solução.

Exemplo em Python:

```pyimport random
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
            # Fase das abelhas empregadas
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

            # Fase das abelhas observadoras
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

            # Fase de scout
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

# Uso:
abc = ABC(dim=2, colony_size=20, limit=10, max_iter=100)
result = abc.optimize()
print("Melhor solução:", result)
print("Fitness:", sum(x**2 for x in result))

```

Exemplo em Go:

```go
package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

type Bee struct {
	position []float64
	fitness  float64
	trials   int
}

type ABC struct {
	dim         int
	colonySize  int
	limit       int
	maxIter     int
	minx, maxx  float64
	bestSolution []float64
	bestFitness  float64
	employed    []Bee
	onlookers   []Bee
}

func NewABC(dim, colonySize, limit, maxIter int) *ABC {
	rand.Seed(time.Now().UnixNano())
	abc := &ABC{
		dim:        dim,
		colonySize: colonySize,
		limit:      limit,
		maxIter:    maxIter,
		minx:       -5,
		maxx:       5,
		bestFitness: math.Inf(1),
	}

	half := colonySize / 2
	abc.employed = make([]Bee, half)
	abc.onlookers = make([]Bee, half)

	for i := range abc.employed {
		abc.employed[i] = newBee(dim, abc.minx, abc.maxx)
	}
	for i := range abc.onlookers {
		abc.onlookers[i] = newBee(dim, abc.minx, abc.maxx)
	}

	return abc
}

func newBee(dim int, minx, maxx float64) Bee {
	position := make([]float64, dim)
	for i := range position {
		position[i] = rand.Float64()*(maxx-minx) + minx
	}
	return Bee{
		position: position,
		fitness:  math.Inf(1),
	}
}

func (abc *ABC) calculateFitness(position []float64) float64 {
	sum := 0.0
	for _, x := range position {
		sum += x * x
	}
	return sum
}

func (abc *ABC) Optimize() []float64 {
	for iter := 0; iter < abc.maxIter; iter++ {
		// Fase empregada
		for i := range abc.employed {
			newPos := make([]float64, abc.dim)
			copy(newPos, abc.employed[i].position)

			j := rand.Intn(abc.dim)
			phi := rand.Float64()*2 - 1
			partner := abc.employed[rand.Intn(len(abc.employed))]
			newPos[j] += phi * (newPos[j] - partner.position[j])
			newPos[j] = math.Max(math.Min(newPos[j], abc.maxx), abc.minx)

			newFit := abc.calculateFitness(newPos)
			if newFit < abc.employed[i].fitness {
				abc.employed[i].position = newPos
				abc.employed[i].fitness = newFit
				abc.employed[i].trials = 0
			} else {
				abc.employed[i].trials++
			}
		}

		// Fase observadora
		total := 0.0
		for _, emp := range abc.employed {
			total += math.Exp(-emp.fitness)
		}

		for i := range abc.onlookers {
			r := rand.Float64() * total
			cumulative := 0.0
			var emp Bee
			for e := range abc.employed {
				cumulative += math.Exp(-abc.employed[e].fitness)
				if cumulative >= r {
					emp = abc.employed[e]
					break
				}
			}

			newPos := make([]float64, abc.dim)
			copy(newPos, emp.position)

			j := rand.Intn(abc.dim)
			phi := rand.Float64()*2 - 1
			partner := abc.employed[rand.Intn(len(abc.employed))]
			newPos[j] += phi * (newPos[j] - partner.position[j])
			newPos[j] = math.Max(math.Min(newPos[j], abc.maxx), abc.minx)

			newFit := abc.calculateFitness(newPos)
			if newFit < abc.onlookers[i].fitness {
				abc.onlookers[i].position = newPos
				abc.onlookers[i].fitness = newFit
				abc.onlookers[i].trials = 0
			} else {
				abc.onlookers[i].trials++
			}
		}

		// Fase scout
		allBees := append(abc.employed, abc.onlookers...)
		for i := range allBees {
			if allBees[i].trials >= abc.limit {
				newBee := newBee(abc.dim, abc.minx, abc.maxx)
				newBee.fitness = abc.calculateFitness(newBee.position)
				allBees[i] = newBee
			}

			if allBees[i].fitness < abc.bestFitness {
				abc.bestFitness = allBees[i].fitness
				abc.bestSolution = make([]float64, abc.dim)
				copy(abc.bestSolution, allBees[i].position)
			}
		}
	}

	return abc.bestSolution
}

func main() {
	abc := NewABC(2, 20, 10, 100)
	result := abc.Optimize()
	fmt.Printf("Melhor solução: %v\n", result)
	fmt.Printf("Fitness: %f\n", abc.calculateFitness(result))
}
```

## Stochastic Diffusion Search (SDS)

O que é:
Algoritmo baseado na comunicação indireta entre agentes para resolver problemas de busca distribuída.

Aplicações:
Detecção de padrões
Pesquisa distribuída
Otimização robusta em ambientes ruidosos

Como funciona:
Os agentes realizam buscas individuais e compartilham informações sobre boas soluções, aumentando a eficiência da busca coletiva.

Exemplo em Python:

```py
import numpy as np
import random

class SDSAgent:
    def __init__(self, search_space):
        self.position = random.uniform(search_space[0], search_space[1])  # Posição inicial do agente
        self.active = False  # Estado ativo/inativo do agente

    def evaluate(self, function):
        """Avalia a função objetivo na posição do agente"""
        return function(self.position)

    def test_hypothesis(self, function, threshold=0.1):
        """Teste probabilístico para manter um agente ativo"""
        self.active = np.random.rand() < threshold

    def communicate(self, other_agent):
        """Se estiver inativo, copia a posição de outro agente ativo"""
        if not self.active:
            self.position = other_agent.position

class StochasticDiffusionSearch:
    def __init__(self, function, search_space, num_agents=20, iterations=100):
        self.function = function
        self.search_space = search_space
        self.agents = [SDSAgent(search_space) for _ in range(num_agents)]
        self.iterations = iterations

    def optimize(self):
        """Executa a busca estocástica"""
        for _ in range(self.iterations):
            # Teste da hipótese
            for agent in self.agents:
                agent.test_hypothesis(self.function)
            
            # Difusão da informação
            for agent in self.agents:
                if not agent.active:
                    other = random.choice(self.agents)
                    if other.active:
                        agent.communicate(other)
            
            # Atualização das posições aleatoriamente para exploração
            for agent in self.agents:
                if random.random() < 0.2:  # Pequena chance de explorar
                    agent.position = random.uniform(self.search_space[0], self.search_space[1])
        
        # Melhor solução encontrada
        best_agent = max(self.agents, key=lambda a: a.evaluate(self.function))
        return best_agent.position, self.function(best_agent.position)

# Definição da função objetivo
def objective_function(x):
    return np.sin(x) + np.cos(2*x)

# Execução do SDS
sds = StochasticDiffusionSearch(objective_function, search_space=(-10, 10))
best_x, best_value = sds.optimize()
print(f"Melhor solução encontrada: x = {best_x:.4f}, f(x) = {best_value:.4f}")
```

Exemplo em Go:

```go
package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// SDSAgent representa um agente no algoritmo Stochastic Diffusion Search
type SDSAgent struct {
	position float64
	active   bool
}

// Evaluate avalia a função objetivo na posição do agente
func (a *SDSAgent) Evaluate() float64 {
	return math.Sin(a.position) + math.Cos(2*a.position)
}

// TestHypothesis define um teste probabilístico para manter um agente ativo
func (a *SDSAgent) TestHypothesis(threshold float64) {
	a.active = rand.Float64() < threshold
}

// Communicate permite que um agente inativo copie a posição de outro ativo
func (a *SDSAgent) Communicate(other *SDSAgent) {
	if !a.active {
		a.position = other.position
	}
}

// StochasticDiffusionSearch representa o algoritmo SDS
type StochasticDiffusionSearch struct {
	agents      []*SDSAgent
	searchSpace [2]float64
	iterations  int
}

// NewSDS inicializa o SDS
func NewSDS(numAgents int, searchSpace [2]float64, iterations int) *StochasticDiffusionSearch {
	rand.Seed(time.Now().UnixNano())
	agents := make([]*SDSAgent, numAgents)
	for i := range agents {
		agents[i] = &SDSAgent{
			position: searchSpace[0] + rand.Float64()*(searchSpace[1]-searchSpace[0]),
			active:   false,
		}
	}
	return &StochasticDiffusionSearch{
		agents:      agents,
		searchSpace: searchSpace,
		iterations:  iterations,
	}
}

// Optimize executa o SDS para otimização da função objetivo
func (sds *StochasticDiffusionSearch) Optimize() (float64, float64) {
	for i := 0; i < sds.iterations; i++ {
		// Teste da hipótese
		for _, agent := range sds.agents {
			agent.TestHypothesis(0.1)
		}

		// Difusão da informação
		for _, agent := range sds.agents {
			if !agent.active {
				other := sds.agents[rand.Intn(len(sds.agents))]
				if other.active {
					agent.Communicate(other)
				}
			}
		}

		// Atualização aleatória para exploração
		for _, agent := range sds.agents {
			if rand.Float64() < 0.2 { // Exploração ocasional
				agent.position = sds.searchSpace[0] + rand.Float64()*(sds.searchSpace[1]-sds.searchSpace[0])
			}
		}
	}

	// Encontrar a melhor solução
	bestAgent := sds.agents[0]
	for _, agent := range sds.agents {
		if agent.Evaluate() > bestAgent.Evaluate() {
			bestAgent = agent
		}
	}
	return bestAgent.position, bestAgent.Evaluate()
}

func main() {
	sds := NewSDS(20, [2]float64{-10, 10}, 100)
	bestX, bestValue := sds.Optimize()
	fmt.Printf("Melhor solução encontrada: x = %.4f, f(x) = %.4f\n", bestX, bestValue)
}
```


## Glowworm Swarm Optimization (GSO)

O que é:
Baseado no comportamento de vaga-lumes que ajustam sua intensidade luminosa para atrair parceiros.

Aplicações:
Otimização multiobjetivo
Detecção de clusters em big data
Reconhecimento de padrões

Como funciona:
Cada vaga-lume ajusta sua luz de acordo com a qualidade da solução encontrada e segue os mais brilhantes.

Exemplo em Python:

```py
import numpy as np
import random

class Glowworm:
    def __init__(self, search_space):
        self.position = random.uniform(search_space[0], search_space[1])  # Posição inicial
        self.luciferin = 0.0  # Intensidade inicial de luz

    def update_luciferin(self, function, decay=0.4, enhancement=0.6):
        """Atualiza a intensidade da luz (luciferina) com base na função objetivo"""
        self.luciferin = (1 - decay) * self.luciferin + enhancement * function(self.position)

    def move_towards(self, other, step_size=0.1):
        """Move-se na direção de outro vaga-lume mais brilhante"""
        if self.luciferin < other.luciferin:
            direction = np.sign(other.position - self.position)
            self.position += direction * step_size

class GlowwormSwarmOptimization:
    def __init__(self, function, search_space, num_agents=20, iterations=100):
        self.function = function
        self.search_space = search_space
        self.agents = [Glowworm(search_space) for _ in range(num_agents)]
        self.iterations = iterations

    def optimize(self):
        """Executa o GSO"""
        for _ in range(self.iterations):
            # Atualizar luciferina
            for agent in self.agents:
                agent.update_luciferin(self.function)

            # Movimento baseado em intensidade de luz
            for agent in self.agents:
                brighter_neighbors = [other for other in self.agents if other.luciferin > agent.luciferin]
                if brighter_neighbors:
                    best_neighbor = max(brighter_neighbors, key=lambda a: a.luciferin)
                    agent.move_towards(best_neighbor)

        # Melhor solução encontrada
        best_agent = max(self.agents, key=lambda a: self.function(a.position))
        return best_agent.position, self.function(best_agent.position)

# Definição da função objetivo
def objective_function(x):
    return np.sin(x) + np.cos(2*x)

# Execução do GSO
gso = GlowwormSwarmOptimization(objective_function, search_space=(-10, 10))
best_x, best_value = gso.optimize()
print(f"Melhor solução encontrada: x = {best_x:.4f}, f(x) = {best_value:.4f}")
```

Exemplo em Go:

```go
package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// Glowworm representa um agente no GSO
type Glowworm struct {
	position   float64
	luciferin  float64
}

// UpdateLuciferin atualiza a intensidade da luz (luciferina)
func (g *Glowworm) UpdateLuciferin(function func(float64) float64, decay, enhancement float64) {
	g.luciferin = (1 - decay)*g.luciferin + enhancement*function(g.position)
}

// MoveTowards movimenta o vaga-lume em direção a outro mais brilhante
func (g *Glowworm) MoveTowards(other *Glowworm, stepSize float64) {
	if g.luciferin < other.luciferin {
		direction := math.Copysign(1, other.position-g.position)
		g.position += direction * stepSize
	}
}

// GlowwormSwarmOptimization representa o algoritmo GSO
type GlowwormSwarmOptimization struct {
	agents      []*Glowworm
	searchSpace [2]float64
	iterations  int
}

// NewGSO inicializa o GSO
func NewGSO(numAgents int, searchSpace [2]float64, iterations int) *GlowwormSwarmOptimization {
	rand.Seed(time.Now().UnixNano())
	agents := make([]*Glowworm, numAgents)
	for i := range agents {
		agents[i] = &Glowworm{
			position:  searchSpace[0] + rand.Float64()*(searchSpace[1]-searchSpace[0]),
			luciferin: 0.0,
		}
	}
	return &GlowwormSwarmOptimization{
		agents:      agents,
		searchSpace: searchSpace,
		iterations:  iterations,
	}
}

// Optimize executa o GSO
func (gso *GlowwormSwarmOptimization) Optimize(function func(float64) float64) (float64, float64) {
	for i := 0; i < gso.iterations; i++ {
		// Atualizar luciferina
		for _, agent := range gso.agents {
			agent.UpdateLuciferin(function, 0.4, 0.6)
		}

		// Movimento baseado na intensidade da luz
		for _, agent := range gso.agents {
			var bestNeighbor *Glowworm
			for _, other := range gso.agents {
				if other.luciferin > agent.luciferin {
					if bestNeighbor == nil || other.luciferin > bestNeighbor.luciferin {
						bestNeighbor = other
					}
				}
			}
			if bestNeighbor != nil {
				agent.MoveTowards(bestNeighbor, 0.1)
			}
		}
	}

	// Encontrar a melhor solução
	bestAgent := gso.agents[0]
	for _, agent := range gso.agents {
		if function(agent.position) > function(bestAgent.position) {
			bestAgent = agent
		}
	}
	return bestAgent.position, function(bestAgent.position)
}

// Função objetivo
func objectiveFunction(x float64) float64 {
	return math.Sin(x) + math.Cos(2*x)
}

func main() {
	gso := NewGSO(20, [2]float64{-10, 10}, 100)
	bestX, bestValue := gso.Optimize(objectiveFunction)
	fmt.Printf("Melhor solução encontrada: x = %.4f, f(x) = %.4f\n", bestX, bestValue)
}
```

## Firefly Algorithm (FA)

O que é:
Inspirado no comportamento de vaga-lumes na natureza, onde indivíduos se atraem com base na intensidade da luz.

Aplicações:
Otimização em engenharia elétrica e eletrônica
Redes neurais e aprendizado de máquina
Problemas de otimização de engenharia

Como funciona:
Vaga-lumes com soluções melhores brilham mais e atraem outros vaga-lumes para suas soluções.

Exemplo em Python:

```python
import numpy as np

class Firefly:
    def __init__(self, dim, minx, maxx):
        self.position = np.random.uniform(minx, maxx, dim)
        self.intensity = 0.0

    def update_intensity(self, function):
        self.intensity = function(self.position)

class FireflyAlgorithm:
    def __init__(self, function, dim, num_fireflies, max_iter, alpha=0.5, beta=0.2, gamma=1.0):
        self.function = function
        self.dim = dim
        self.num_fireflies = num_fireflies
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.minx = -5
        self.maxx = 5
        self.fireflies = [Firefly(dim, self.minx, self.maxx) for _ in range(num_fireflies)]

    def optimize(self):
        for _ in range(self.max_iter):
            for firefly in self.fireflies:
                firefly.update_intensity(self.function)

            for i in range(self.num_fireflies):
                for j in range(self.num_fireflies):
                    if self.fireflies[i].intensity < self.fireflies[j].intensity:
                        distance = np.linalg.norm(self.fireflies[i].position - self.fireflies[j].position)
                        beta = self.beta * np.exp(-self.gamma * distance ** 2)
                        self.fireflies[i].position += beta * (self.fireflies[j].position - self.fireflies[i].position) + \
                                                      self.alpha * (np.random.rand(self.dim) - 0.5)

                        self.fireflies[i].position = np.clip(self.fireflies[i].position, self.minx, self.maxx)

        best_firefly = min(self.fireflies, key=lambda f: f.intensity)
        return best_firefly.position, best_firefly.intensity

def objective_function(x):
    return np.sum(x ** 2)

fa = FireflyAlgorithm(objective_function, dim=2, num_fireflies=20, max_iter=100)
best_pos, best_intensity = fa.optimize()
print(f"Melhor posição: {best_pos}, Melhor intensidade: {best_intensity}")
```

Exemplo em Go:

```go
package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// Firefly representa um agente no Firefly Algorithm
type Firefly struct {
	position  float64
	intensity float64
}

// UpdateIntensity atualiza a intensidade de brilho do vaga-lume
func (f *Firefly) UpdateIntensity(function func(float64) float64) {
	f.intensity = function(f.position)
}

// MoveTowards move o vaga-lume em direção a outro mais brilhante
func (f *Firefly) MoveTowards(other *Firefly, beta, alpha float64) {
	if f.intensity < other.intensity {
		randFactor := (rand.Float64()*2 - 1) * alpha
		f.position += beta*(other.position-f.position) + randFactor
	}
}

// FireflyAlgorithm representa o algoritmo FA
type FireflyAlgorithm struct {
	fireflies   []*Firefly
	searchSpace [2]float64
	iterations  int
}

// NewFA inicializa o Firefly Algorithm
func NewFA(numFireflies int, searchSpace [2]float64, iterations int) *FireflyAlgorithm {
	rand.Seed(time.Now().UnixNano())
	fireflies := make([]*Firefly, numFireflies)
	for i := range fireflies {
		fireflies[i] = &Firefly{
			position:  searchSpace[0] + rand.Float64()*(searchSpace[1]-searchSpace[0]),
			intensity: 0.0,
		}
	}
	return &FireflyAlgorithm{
		fireflies:   fireflies,
		searchSpace: searchSpace,
		iterations:  iterations,
	}
}

// Optimize executa o FA
func (fa *FireflyAlgorithm) Optimize(function func(float64) float64) (float64, float64) {
	for i := 0; i < fa.iterations; i++ {
		// Atualizar a intensidade de cada vaga-lume
		for _, firefly := range fa.fireflies {
			firefly.UpdateIntensity(function)
		}

		// Movimento dos vaga-lumes baseados na intensidade
		for i := 0; i < len(fa.fireflies); i++ {
			for j := 0; j < len(fa.fireflies); j++ {
				if fa.fireflies[i].intensity < fa.fireflies[j].intensity {
					fa.fireflies[i].MoveTowards(fa.fireflies[j], 1.0, 0.2)
				}
			}
		}
	}

	// Encontrar a melhor solução
	bestFirefly := fa.fireflies[0]
	for _, firefly := range fa.fireflies {
		if firefly.intensity > bestFirefly.intensity {
			bestFirefly = firefly
		}
	}
	return bestFirefly.position, bestFirefly.intensity
}

// Função objetivo
func objectiveFunction(x float64) float64 {
	return math.Sin(x) + math.Cos(2*x)
}

func main() {
	fa := NewFA(20, [2]float64{-10, 10}, 100)
	bestX, bestValue := fa.Optimize(objectiveFunction)
	fmt.Printf("Melhor solução encontrada: x = %.4f, f(x) = %.4f\n", bestX, bestValue)
}
```


## Bacterial Foraging Optimization (BFO)

O que é:
Algoritmo inspirado na forma como as bactérias procuram alimento e evitam substâncias tóxicas.

Aplicações:
Controle de sistemas dinâmicos
Bioinformática e otimização de proteínas
Sistemas de energia renovável

Como funciona:
Simula processos biológicos de natação, reorientação e eliminação/reprodução para explorar o espaço de soluções.

Exemplo em Python:

```py
import numpy as np
import random

class Bacterium:
    def __init__(self, search_space):
        self.position = random.uniform(search_space[0], search_space[1])  # Posição inicial
        self.cost = float('inf')  # Custo inicial da função objetivo

    def evaluate(self, function):
        """Avalia a função objetivo na posição da bactéria"""
        self.cost = function(self.position)

    def tumble(self, step_size=0.1):
        """Movimento aleatório da bactéria"""
        self.position += random.uniform(-1, 1) * step_size

    def swim(self, best_neighbor, step_size=0.1):
        """Movimento em direção à melhor solução"""
        if self.cost > best_neighbor.cost:
            self.position += np.sign(best_neighbor.position - self.position) * step_size

class BFO:
    def __init__(self, function, search_space, num_bacteria=20, iterations=100):
        self.function = function
        self.search_space = search_space
        self.bacteria = [Bacterium(search_space) for _ in range(num_bacteria)]
        self.iterations = iterations

    def optimize(self):
        """Executa o algoritmo BFO"""
        for _ in range(self.iterations):
            # Quimiotaxia: Avaliação e movimento aleatório
            for bacterium in self.bacteria:
                bacterium.evaluate(self.function)
                bacterium.tumble()

            # Movimentação em direção às melhores soluções
            best_bacteria = sorted(self.bacteria, key=lambda b: b.cost, reverse=True)
            for bacterium in self.bacteria:
                bacterium.swim(best_bacteria[0])

            # Reprodução: As melhores bactérias se dividem
            self.bacteria.sort(key=lambda b: b.cost, reverse=True)
            self.bacteria = self.bacteria[:len(self.bacteria) // 2] * 2

            # Eliminação e Dispersão: Algumas bactérias são eliminadas e redistribuídas
            for i in range(len(self.bacteria)):
                if random.random() < 0.1:  # Probabilidade de dispersão
                    self.bacteria[i] = Bacterium(self.search_space)

        # Melhor solução encontrada
        best_bacterium = min(self.bacteria, key=lambda b: b.cost)
        return best_bacterium.position, best_bacterium.cost

# Definição da função objetivo
def objective_function(x):
    return np.sin(x) + np.cos(2*x)

# Execução do BFO
bfo = BFO(objective_function, search_space=(-10, 10))
best_x, best_value = bfo.optimize()
print(f"Melhor solução encontrada: x = {best_x:.4f}, f(x) = {best_value:.4f}")

```
Exemplo em Go:

```go
package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// Bacterium representa uma bactéria no BFO
type Bacterium struct {
	position float64
	cost     float64
}

// Evaluate avalia a função objetivo na posição da bactéria
func (b *Bacterium) Evaluate(function func(float64) float64) {
	b.cost = function(b.position)
}

// Tumble realiza um movimento aleatório da bactéria
func (b *Bacterium) Tumble(stepSize float64) {
	b.position += (rand.Float64()*2 - 1) * stepSize
}

// Swim move a bactéria em direção à melhor solução
func (b *Bacterium) Swim(bestNeighbor *Bacterium, stepSize float64) {
	if b.cost > bestNeighbor.cost {
		direction := math.Copysign(1, bestNeighbor.position-b.position)
		b.position += direction * stepSize
	}
}

// BFO representa o algoritmo Bacterial Foraging Optimization
type BFO struct {
	bacteria    []*Bacterium
	searchSpace [2]float64
	iterations  int
}

// NewBFO inicializa o BFO
func NewBFO(numBacteria int, searchSpace [2]float64, iterations int) *BFO {
	rand.Seed(time.Now().UnixNano())
	bacteria := make([]*Bacterium, numBacteria)
	for i := range bacteria {
		bacteria[i] = &Bacterium{
			position: searchSpace[0] + rand.Float64()*(searchSpace[1]-searchSpace[0]),
			cost:     math.Inf(1),
		}
	}
	return &BFO{
		bacteria:    bacteria,
		searchSpace: searchSpace,
		iterations:  iterations,
	}
}

// Optimize executa o BFO
func (bfo *BFO) Optimize(function func(float64) float64) (float64, float64) {
	for i := 0; i < bfo.iterations; i++ {
		// Quimiotaxia: avaliação e movimento aleatório
		for _, bacterium := range bfo.bacteria {
			bacterium.Evaluate(function)
			bacterium.Tumble(0.1)
		}

		// Movimentação em direção às melhores soluções
		bestBacteria := make([]*Bacterium, len(bfo.bacteria))
		copy(bestBacteria, bfo.bacteria)
		// Ordena as bactérias pelo custo (maior valor primeiro)
		for i := 1; i < len(bestBacteria); i++ {
			for j := i; j > 0 && bestBacteria[j].cost > bestBacteria[j-1].cost; j-- {
				bestBacteria[j], bestBacteria[j-1] = bestBacteria[j-1], bestBacteria[j]
			}
		}

		for _, bacterium := range bfo.bacteria {
			bacterium.Swim(bestBacteria[0], 0.1)
		}

		// Reprodução: As melhores bactérias se duplicam
		mid := len(bfo.bacteria) / 2
		bfo.bacteria = append(bfo.bacteria[:mid], bfo.bacteria[:mid]...)

		// Eliminação e dispersão: algumas bactérias são redistribuídas
		for i := range bfo.bacteria {
			if rand.Float64() < 0.1 { // Probabilidade de dispersão
				bfo.bacteria[i] = &Bacterium{
					position: bfo.searchSpace[0] + rand.Float64()*(bfo.searchSpace[1]-bfo.searchSpace[0]),
					cost:     math.Inf(1),
				}
			}
		}
	}

	// Encontrar a melhor solução
	bestBacterium := bfo.bacteria[0]
	for _, bacterium := range bfo.bacteria {
		if bacterium.cost < bestBacterium.cost {
			bestBacterium = bacterium
		}
	}
	return bestBacterium.position, bestBacterium.cost
}

// Função objetivo
func objectiveFunction(x float64) float64 {
	return math.Sin(x) + math.Cos(2*x)
}

func main() {
	bfo := NewBFO(20, [2]float64{-10, 10}, 100)
	bestX, bestValue := bfo.Optimize(objectiveFunction)
	fmt.Printf("Melhor solução encontrada: x = %.4f, f(x) = %.4f\n", bestX, bestValue)
}
```
 






## Cuckoo Search (CS)

O que é: Baseado no comportamento de certas espécies de cucos que colocam seus ovos nos ninhos de outras aves.

Aplicações:
Otimização de engenharia
Otimização de redes neurais
Problemas financeiros e econômicos

Como funciona:
Gera novas soluções inspiradas no mecanismo de busca de ninhos ideais e elimina soluções ruins.

Exemplo em Python:

```py
import numpy as np
import random

def levy_flight(beta=1.5):
    """Gera um passo de voo de Lévy"""
    sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
             (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma)
    v = np.random.normal(0, 1)
    step = u / abs(v) ** (1 / beta)
    return step

class Cuckoo:
    def __init__(self, search_space):
        self.position = random.uniform(search_space[0], search_space[1])  # Posição inicial
        self.fitness = float('-inf')  # Inicialmente ruim

    def evaluate(self, function):
        """Avalia a função objetivo"""
        self.fitness = function(self.position)

    def perform_levy_flight(self, best_nest, alpha=0.01):
        """Executa um voo de Lévy em direção ao melhor ninho"""
        step = levy_flight() * alpha
        self.position += step * (self.position - best_nest.position)

class CuckooSearch:
    def __init__(self, function, search_space, num_nests=20, iterations=100, pa=0.25):
        self.function = function
        self.search_space = search_space
        self.nests = [Cuckoo(search_space) for _ in range(num_nests)]
        self.iterations = iterations
        self.pa = pa  # Taxa de abandono de ninhos

    def optimize(self):
        """Executa o algoritmo Cuckoo Search"""
        for _ in range(self.iterations):
            # Avaliação das soluções
            for nest in self.nests:
                nest.evaluate(self.function)

            # Encontrar o melhor ninho
            best_nest = max(self.nests, key=lambda n: n.fitness)

            # Voos de Lévy para explorar novas soluções
            for nest in self.nests:
                if random.random() > self.pa:
                    nest.perform_levy_flight(best_nest)

            # Substituição de soluções ruins
            for nest in self.nests:
                if random.random() < self.pa:
                    nest.position = random.uniform(self.search_space[0], self.search_space[1])

        # Melhor solução encontrada
        best_nest = max(self.nests, key=lambda n: n.fitness)
        return best_nest.position, best_nest.fitness

# Definição da função objetivo
def objective_function(x):
    return np.sin(x) + np.cos(2*x)

# Execução do CS
cs = CuckooSearch(objective_function, search_space=(-10, 10))
best_x, best_value = cs.optimize()
print(f"Melhor solução encontrada: x = {best_x:.4f}, f(x) = {best_value:.4f}")
```

Exemplo em Go:

```go
package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// Cuckoo representa um ninho no Cuckoo Search
type Cuckoo struct {
	position float64
	fitness  float64
}

// Evaluate avalia a função objetivo na posição do ninho
func (c *Cuckoo) Evaluate(function func(float64) float64) {
	c.fitness = function(c.position)
}

// LevyFlight gera um passo de voo de Lévy
func LevyFlight(beta float64) float64 {
	sigma := math.Pow((math.Gamma(1+beta) * math.Sin(math.Pi*beta/2)) /
		(math.Gamma((1+beta)/2) * beta * math.Pow(2, (beta-1)/2)), 1/beta)
	u := rand.NormFloat64() * sigma
	v := rand.NormFloat64()
	return u / math.Pow(math.Abs(v), 1/beta)
}

// PerformLevyFlight executa um voo de Lévy em direção ao melhor ninho
func (c *Cuckoo) PerformLevyFlight(bestNest *Cuckoo, alpha float64) {
	step := LevyFlight(1.5) * alpha
	c.position += step * (c.position - bestNest.position)
}

// CuckooSearch representa o algoritmo CS
type CuckooSearch struct {
	nests      []*Cuckoo
	searchSpace [2]float64
	iterations int
	pa         float64 // Taxa de abandono de ninhos
}

// NewCS inicializa o Cuckoo Search
func NewCS(numNests int, searchSpace [2]float64, iterations int, pa float64) *CuckooSearch {
	rand.Seed(time.Now().UnixNano())
	nests := make([]*Cuckoo, numNests)
	for i := range nests {
		nests[i] = &Cuckoo{
			position: searchSpace[0] + rand.Float64()*(searchSpace[1]-searchSpace[0]),
			fitness:  math.Inf(-1),
		}
	}
	return &CuckooSearch{
		nests:      nests,
		searchSpace: searchSpace,
		iterations: iterations,
		pa:         pa,
	}
}

// Optimize executa o Cuckoo Search
func (cs *CuckooSearch) Optimize(function func(float64) float64) (float64, float64) {
	for i := 0; i < cs.iterations; i++ {
		// Avaliação das soluções
		for _, nest := range cs.nests {
			nest.Evaluate(function)
		}

		// Encontrar o melhor ninho
		bestNest := cs.nests[0]
		for _, nest := range cs.nests {
			if nest.fitness > bestNest.fitness {
				bestNest = nest
			}
		}

		// Voos de Lévy para explorar novas soluções
		for _, nest := range cs.nests {
			if rand.Float64() > cs.pa {
				nest.PerformLevyFlight(bestNest, 0.01)
			}
		}

		// Substituição de ninhos ruins
		for _, nest := range cs.nests {
			if rand.Float64() < cs.pa {
				nest.position = cs.searchSpace[0] + rand.Float64()*(cs.searchSpace[1]-cs.searchSpace[0])
			}
		}
	}

	// Encontrar a melhor solução
	bestNest := cs.nests[0]
	for _, nest := range cs.nests {
		if nest.fitness > bestNest.fitness {
			bestNest = nest
		}
	}
	return bestNest.position, bestNest.fitness
}

// Função objetivo
func objectiveFunction(x float64) float64 {
	return math.Sin(x) + math.Cos(2*x)
}

func main() {
	cs := NewCS(20, [2]float64{-10, 10}, 100, 0.25)
	bestX, bestValue := cs.Optimize(objectiveFunction)
	fmt.Printf("Melhor solução encontrada: x = %.4f, f(x) = %.4f\n", bestX, bestValue)
}
```

## Grey Wolf Optimizer (GWO)

O que é:
Inspirado na hierarquia e estratégias de caça de lobos cinzentos.

Aplicações:
Otimização de processos industriais
Controle de robôs e sistemas autônomos
Ajuste de parâmetros de redes neurais

Como funciona:
Os lobos são divididos em líderes (alfa, beta, delta) e seguidores (ômega), e juntos convergem para soluções ótimas.

Exemplo em Python:

```py
import numpy as np
import random

class GreyWolf:
    def __init__(self, search_space):
        self.position = random.uniform(search_space[0], search_space[1])  # Posição inicial
        self.fitness = float('-inf')  # Avaliação inicial da função objetivo

    def evaluate(self, function):
        """Avalia a função objetivo"""
        self.fitness = function(self.position)

class GreyWolfOptimizer:
    def __init__(self, function, search_space, num_wolves=20, iterations=100):
        self.function = function
        self.search_space = search_space
        self.wolves = [GreyWolf(search_space) for _ in range(num_wolves)]
        self.iterations = iterations

    def optimize(self):
        """Executa o Grey Wolf Optimizer"""
        alpha, beta, delta = None, None, None

        for _ in range(self.iterations):
            # Avaliar os lobos
            for wolf in self.wolves:
                wolf.evaluate(self.function)

            # Atualizar α (melhor), β (segundo melhor) e δ (terceiro melhor)
            sorted_wolves = sorted(self.wolves, key=lambda w: w.fitness, reverse=True)
            alpha, beta, delta = sorted_wolves[:3]

            # Atualizar as posições dos lobos
            a = 2 - 2 * (_ / self.iterations)  # Componente de exploração/exploração

            for wolf in self.wolves:
                if wolf not in [alpha, beta, delta]:
                    A1, A2, A3 = (2 * a * random.random() - a for _ in range(3))
                    C1, C2, C3 = (2 * random.random() for _ in range(3))

                    X1 = alpha.position - A1 * abs(C1 * alpha.position - wolf.position)
                    X2 = beta.position - A2 * abs(C2 * beta.position - wolf.position)
                    X3 = delta.position - A3 * abs(C3 * delta.position - wolf.position)

                    wolf.position = (X1 + X2 + X3) / 3  # Atualização da posição

        # Melhor solução encontrada
        best_wolf = max(self.wolves, key=lambda w: w.fitness)
        return best_wolf.position, best_wolf.fitness

# Definição da função objetivo
def objective_function(x):
    return np.sin(x) + np.cos(2*x)

# Execução do GWO
gwo = GreyWolfOptimizer(objective_function, search_space=(-10, 10))
best_x, best_value = gwo.optimize()
print(f"Melhor solução encontrada: x = {best_x:.4f}, f(x) = {best_value:.4f}")
```

Exemplo em Go:

```go
package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// GreyWolf representa um lobo na otimização
type GreyWolf struct {
	position float64
	fitness  float64
}

// Evaluate avalia a função objetivo
func (w *GreyWolf) Evaluate(function func(float64) float64) {
	w.fitness = function(w.position)
}

// GreyWolfOptimizer representa o GWO
type GreyWolfOptimizer struct {
	wolves      []*GreyWolf
	searchSpace [2]float64
	iterations  int
}

// NewGWO inicializa o GWO
func NewGWO(numWolves int, searchSpace [2]float64, iterations int) *GreyWolfOptimizer {
	rand.Seed(time.Now().UnixNano())
	wolves := make([]*GreyWolf, numWolves)
	for i := range wolves {
		wolves[i] = &GreyWolf{
			position: searchSpace[0] + rand.Float64()*(searchSpace[1]-searchSpace[0]),
			fitness:  math.Inf(-1),
		}
	}
	return &GreyWolfOptimizer{
		wolves:      wolves,
		searchSpace: searchSpace,
		iterations:  iterations,
	}
}

// Optimize executa o Grey Wolf Optimizer
func (gwo *GreyWolfOptimizer) Optimize(function func(float64) float64) (float64, float64) {
	var alpha, beta, delta *GreyWolf

	for iter := 0; iter < gwo.iterations; iter++ {
		// Avaliar os lobos
		for _, wolf := range gwo.wolves {
			wolf.Evaluate(function)
		}

		// Encontrar os três melhores lobos (α, β, δ)
		alpha, beta, delta = gwo.selectBestWolves()

		// Atualizar as posições dos lobos
		a := 2 - 2*float64(iter)/float64(gwo.iterations) // Fator de exploração

		for _, wolf := range gwo.wolves {
			if wolf != alpha && wolf != beta && wolf != delta {
				A1, A2, A3 := 2*a*rand.Float64()-a, 2*a*rand.Float64()-a, 2*a*rand.Float64()-a
				C1, C2, C3 := 2*rand.Float64(), 2*rand.Float64(), 2*rand.Float64()

				X1 := alpha.position - A1*math.Abs(C1*alpha.position-wolf.position)
				X2 := beta.position - A2*math.Abs(C2*beta.position-wolf.position)
				X3 := delta.position - A3*math.Abs(C3*delta.position-wolf.position)

				wolf.position = (X1 + X2 + X3) / 3
			}
		}
	}

	// Melhor solução encontrada
	bestWolf := alpha
	return bestWolf.position, bestWolf.fitness
}

// selectBestWolves encontra os três melhores lobos
func (gwo *GreyWolfOptimizer) selectBestWolves() (*GreyWolf, *GreyWolf, *GreyWolf) {
	wolves := gwo.wolves
	alpha, beta, delta := wolves[0], wolves[1], wolves[2]

	for _, wolf := range wolves {
		if wolf.fitness > alpha.fitness {
			delta = beta
			beta = alpha
			alpha = wolf
		} else if wolf.fitness > beta.fitness {
			delta = beta
			beta = wolf
		} else if wolf.fitness > delta.fitness {
			delta = wolf
		}
	}
	return alpha, beta, delta
}

// Função objetivo
func objectiveFunction(x float64) float64 {
	return math.Sin(x) + math.Cos(2*x)
}

func main() {
	gwo := NewGWO(20, [2]float64{-10, 10}, 100)
	bestX, bestValue := gwo.Optimize(objectiveFunction)
	fmt.Printf("Melhor solução encontrada: x = %.4f, f(x) = %.4f\n", bestX, bestValue)
}
```

Cada um desses algoritmos de Swarm Intelligence tem aplicações específicas, mas todos compartilham a característica de descentralização e auto-organização. Dependendo do seu problema (por exemplo, otimização de portfólio, trading algorítmico, ajuste de hiperparâmetros), um desses métodos pode ser mais adequado.

## Utilização pelo Quantum Hive Fund
Por hora estou utilizando o PSO e o GWO para otimização de parâmetros de estratégias de trading, com ótimos resultados, muito superior aos Algoritmos Genéticos. Bem na verdade eu uso um fluxo iniciando com os Algoritmos Genéticos, depois uso o melhor resultado no PSO e por final uso o QAOA (Quantum Approximate Optimization Algorithm) para garantir a máxima otimização.

#ficadica
