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

## Glowworm Swarm Optimization (GSO)

O que é:
Baseado no comportamento de vaga-lumes que ajustam sua intensidade luminosa para atrair parceiros.

Aplicações:
Otimização multiobjetivo
Detecção de clusters em big data
Reconhecimento de padrões

Como funciona:
Cada vaga-lume ajusta sua luz de acordo com a qualidade da solução encontrada e segue os mais brilhantes.

## Firefly Algorithm (FA)

O que é:
Inspirado no comportamento de vaga-lumes na natureza, onde indivíduos se atraem com base na intensidade da luz.

Aplicações:
Otimização em engenharia elétrica e eletrônica
Redes neurais e aprendizado de máquina
Problemas de otimização de engenharia

Como funciona:
Vaga-lumes com soluções melhores brilham mais e atraem outros vaga-lumes para suas soluções.

## Bacterial Foraging Optimization (BFO)

O que é:
Algoritmo inspirado na forma como as bactérias procuram alimento e evitam substâncias tóxicas.

Aplicações:
Controle de sistemas dinâmicos
Bioinformática e otimização de proteínas
Sistemas de energia renovável

Como funciona:
Simula processos biológicos de natação, reorientação e eliminação/reprodução para explorar o espaço de soluções.

## Cuckoo Search (CS)

O que é: Baseado no comportamento de certas espécies de cucos que colocam seus ovos nos ninhos de outras aves.

Aplicações:
Otimização de engenharia
Otimização de redes neurais
Problemas financeiros e econômicos

Como funciona:
Gera novas soluções inspiradas no mecanismo de busca de ninhos ideais e elimina soluções ruins.

## Grey Wolf Optimizer (GWO)

O que é:
Inspirado na hierarquia e estratégias de caça de lobos cinzentos.

Aplicações:
Otimização de processos industriais
Controle de robôs e sistemas autônomos
Ajuste de parâmetros de redes neurais

Como funciona:
Os lobos são divididos em líderes (alfa, beta, delta) e seguidores (ômega), e juntos convergem para soluções ótimas.

Cada um desses algoritmos de Swarm Intelligence tem aplicações específicas, mas todos compartilham a característica de descentralização e auto-organização. Dependendo do seu problema (por exemplo, otimização de portfólio, trading algorítmico, ajuste de hiperparâmetros), um desses métodos pode ser mais adequado.

Por hora estou utilizando o PSO e o GWO para otimização de parâmetros de estratégias de trading, com ótimos resultados, muito superior aos Algoritmos Genéticos. Bem na verdade eu uso um fluxo iniciando com os Algoritmos Genéticos, depois uso o melhor resultado no PSO e por final uso o QAOA (Quantum Approximate Optimization Algorithm) para garantir a máxima otimização.

#ficadica
