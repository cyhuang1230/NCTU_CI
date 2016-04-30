import random
from enum import IntEnum
import heapq

def random_ints(count: int, limit: int = 1) -> [int]:
    """
    :param count:
    :param limit: inclusive
    :return:
    """
    random.seed()
    return [random.randint(0, limit-1) for _ in range(count)]


def print_custom(l: list):
    print('-- begin ---')
    for i in l:
        print(i.__str__())
    print('-- end ---')


class Individual:

    def __init__(self, k: int, genes: [int], data=None):
        """
        :param k:
        :param genes: a list of int indicating the cluster each point belongs to
        """
        self.k = k
        self.gene_size = len(genes)
        self.genes = genes
        self.fitness = None
        self.centroids = None
        self.data = data

    def get_fitness(self) -> float:
        if self.fitness is None:
            self.fitness = 0.0
            centroids = self.get_centroids()
            for i in range(self.gene_size):
                self.fitness += pow(self.data[i] - centroids[self.genes[i]], 2)
        return self.fitness

    def get_centroids(self) -> [float]:
        if self.centroids is None:
            num_of_points_in_cluster = [0] * self.k
            sum_of_coordinate_in_cluster = [0.0] * self.k
            for i in range(self.gene_size):
                cluster_num = self.genes[i]
                coordinate = self.data[i]
                num_of_points_in_cluster[cluster_num] += 1
                sum_of_coordinate_in_cluster[cluster_num] += coordinate
            self.centroids = [sum_of_coordinate_in_cluster[i]/num_of_points_in_cluster[i] for i in range(self.k)]
        return self.centroids

    # only valid if there's at least one point in each cluster
    def is_valid(self) -> bool:
        ret = [False for _ in range(self.k)]
        for g in self.genes:
            ret[g] = True
        for r in ret:
            if not r:
                return False
        return True

    def __str__(self):
        return "k = %d, data = %s, genes = %s, fitness = %.3f, centroids = %s" \
               % (self.k, self.data if self.data else 'None', self.genes.__str__(),
                  self.fitness if self.fitness else self.get_fitness(),
                  self.get_centroids().__str__())

    def __lt__(self, other):
        return self.get_fitness() < other.get_fitness()


class ParentSelectionMethod(IntEnum):
    FITNESS_BASED = 1   # @TODO
    TOURNAMENT_SELECTION = 2


class CrossoverMethod(IntEnum):
    MEAN = 1
    ONE_POINT_ = 2  # @TODO
    TWO_POINT = 3   # @TODO
    UNIFORM = 4     # @TODO


class EvolutionaryAlgorithm:
    POPULATION_SIZE = 30
    PARENT_SELECTION = ParentSelectionMethod.TOURNAMENT_SELECTION
    CROSSOVER = CrossoverMethod.MEAN

    def __init__(self, k: int, data: [int]):
        self.k = k
        self.data_size = len(data)
        self.data = data
        self.population = []
        self.best_individual = None

        # generate enough `valid` individuals
        kount = 0
        while kount < self.POPULATION_SIZE:
            an_idv = Individual(k, random_ints(self.data_size, self.k), data)
            if not an_idv.is_valid():
                continue
            self.population.append(an_idv)
            kount += 1

        # make population a heap to improve performance
        heapq.heapify(self.population)
        self.best_individual = self.choose_best_individual()

    def __str__(self):
        ret = ''
        for p in self.population:
            ret += p.__str__() + '\n'
        return ret

    def do(self, generation_count: int):

        # Parent selection
        parents = self.choose_parent()

        # Crossover
        children = self.do_crossover(parents)

        # Mutation & Survivor selection
        for child in children:
            child = self.do_mutation(child)
            heapq.heappushpop(self.population, child)

    def choose_parent(self) -> [Individual]:
        if self.PARENT_SELECTION == ParentSelectionMethod.FITNESS_BASED:
            pass
        elif self.PARENT_SELECTION == ParentSelectionMethod.TOURNAMENT_SELECTION:
            parents = []
            for _ in range(2):
                participants = [self.population[i] for i in random.sample(range(self.POPULATION_SIZE), 2)]
                participants.sort(reverse=True)
                prob = random.uniform(0, 1)
                # if prob in [0,0.8], choose the winner (i.e., participant[0])
                # else if prob in (0.8,1], choose the loser (i.e., participant[1])
                parents.append(participants[int(prob/0.8)])
            return parents

    def choose_best_individual(self):
        return heapq.nlargest(1, self.population)[0]

    def do_crossover(self, parents: [Individual]) -> [Individual]:
        pass

    def do_mutation(self, an_idv: Individual) -> Individual:
        pass

if __name__ == '__main__':

    # generate number
    # x = random_ints(50, 100)
    x = [i for i in range(4)]
    ea = EvolutionaryAlgorithm(2, x)
    print(ea)
    ea.do(10)
