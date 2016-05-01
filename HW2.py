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
            self.update_fitness()
        return self.fitness

    def get_centroids(self) -> [float]:
        if self.centroids is None:
            self.update_centroids()
        return self.centroids

    def update(self):
        self.update_centroids()
        self.update_fitness()

    def update_fitness(self):
        self.fitness = 0.0
        centroids = self.get_centroids()
        for i in range(self.gene_size):
            self.fitness += pow(self.data[i] - centroids[self.genes[i]], 2)

    def update_centroids(self):
        num_of_points_in_cluster = [0] * self.k
        sum_of_coordinate_in_cluster = [0.0] * self.k
        for i in range(self.gene_size):
            cluster_num = self.genes[i]
            coordinate = self.data[i]
            num_of_points_in_cluster[cluster_num] += 1
            sum_of_coordinate_in_cluster[cluster_num] += coordinate
        self.centroids = [sum_of_coordinate_in_cluster[i] / num_of_points_in_cluster[i] for i in range(self.k)]

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
        # in our case, fitness is the lower the better
        return self.get_fitness() > other.get_fitness()


class ParentSelectionMethod(IntEnum):
    FITNESS_BASED = 1   # @TODO
    TOURNAMENT_SELECTION = 2


class CrossoverMethod(IntEnum):
    MEAN = 1
    ONE_POINT = 2
    TWO_POINT = 3
    UNIFORM = 4


class EvolutionaryAlgorithm:
    POPULATION_SIZE = 100
    PARENT_SELECTION = ParentSelectionMethod.TOURNAMENT_SELECTION
    CROSSOVER = CrossoverMethod.MEAN
    MUTATE_PROBABILITY = 0.4

    def __init__(self, k: int, data: [int]):
        self.k = k
        self.data_size = len(data)
        self.data = data
        self.population = []
        self.result = []

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

        self.record_result()

    def __str__(self):
        ret = ''
        for p in self.population:
            ret += p.__str__() + '\n'
        return ret

    def do(self, generation: int):

        for i in range(generation):
            # Parent selection
            parents = self.choose_parent()
            print('--- parents ---')
            print_custom(parents)

            # Crossover
            while True:
                crossover_children = self.do_crossover(parents)
                # Proceed only when children are all valid
                if not EvolutionaryAlgorithm.is_valid_individuals(crossover_children):
                    continue

                # Mutation
                mutation_children = [self.do_mutation(child) for child in crossover_children]
                # Proceed only when children are all valid
                if EvolutionaryAlgorithm.is_valid_individuals(mutation_children):
                    break

            print('--- children ---')
            print_custom(mutation_children)

            # Survivor selection
            for child in mutation_children:
                heapq.heappushpop(self.population, child)

            self.record_result()

    def choose_parent(self) -> [Individual]:
        parents = []
        if self.PARENT_SELECTION == ParentSelectionMethod.FITNESS_BASED:
            parents = self.get_best_individual(2)
        elif self.PARENT_SELECTION == ParentSelectionMethod.TOURNAMENT_SELECTION:
            for _ in range(2):
                participants = [self.population[i] for i in random.sample(range(self.POPULATION_SIZE), 2)]
                participants.sort(reverse=True)
                prob = random.uniform(0, 1)
                # if prob in [0,0.8], choose the winner (i.e., participant[0])
                # else if prob in (0.8,1], choose the loser (i.e., participant[1])
                parents.append(participants[int(prob/0.8)])
        return parents

    def get_best_individual(self, kount = 1):
        # get one with least fitness
        # however, we implement the __lt__ in the reversed way, so we choose the largest one here
        ret = heapq.nlargest(kount, self.population)
        return ret[0] if kount == 1 else ret

    def do_crossover(self, parents: [Individual]) -> [Individual]:
        assert len(parents) == 2
        assert parents[0].gene_size == parents[1].gene_size

        size = parents[0].gene_size
        children = []

        if self.CROSSOVER == CrossoverMethod.MEAN:
            new_gene = [int((parents[0].genes[i]+parents[1].genes[i])/2) for i in range(size)]
            children.append(Individual(self.k, new_gene, self.data))
        elif self.CROSSOVER == CrossoverMethod.ONE_POINT or self.CROSSOVER == CrossoverMethod.TWO_POINT:
            num_of_crossover_points = 0
            if self.CROSSOVER == CrossoverMethod.ONE_POINT:
                num_of_crossover_points = 1
            elif self.CROSSOVER == CrossoverMethod.TWO_POINT:
                num_of_crossover_points = 2

            # only get crossover_point between (1, size-1) to avoid get same children as parents
            crossover_points = random.sample(range(1, size-1), num_of_crossover_points)

            child_one_gene = []
            child_two_gene = []
            should_swap = False
            for i in range(size):
                if i in crossover_points:
                    should_swap = not should_swap
                child_one_gene.append(parents[0 if not should_swap else 1].genes[i])
                child_two_gene.append(parents[1 if not should_swap else 0].genes[i])
            children.append(Individual(self.k, child_one_gene, self.data))
            children.append(Individual(self.k, child_two_gene, self.data))
        elif self.CROSSOVER == CrossoverMethod.UNIFORM:
            child_one_gene = []
            child_two_gene = []
            for i in range(size):
                prob = random.uniform(0, 1)
                # if prob in [0,0.5) -> child_one gets parent[0], child_two gets parent[1] => i.e. should not swap
                # else if prob in [0.5,1] -> child_one gets parent[1], child_two gets parent[0] => i.e. should swap
                child_one_gene.append(parents[int(prob/0.5)].genes[i])
                child_two_gene.append(parents[(int(prob/0.5)+1)%2].genes[i])
            children.append(Individual(self.k, child_one_gene, self.data))
            children.append(Individual(self.k, child_two_gene, self.data))
        return children

    def do_mutation(self, an_idv: Individual) -> Individual:
        for i in range(an_idv.gene_size):
            should_mutate = random.uniform(0, 1)
            if should_mutate > self.MUTATE_PROBABILITY:
                continue
            an_idv.genes[i] = random.randint(0, self.k-1)
        if an_idv.is_valid():
            an_idv.update()
        return an_idv

    def record_result(self):
        best = self.get_best_individual()
        self.result.append({"genes": best.genes, "fitness": best.get_fitness(), "centroids": best.get_centroids()})

    @staticmethod
    def is_valid_individuals(idvs: [Individual]):
        for idv in idvs:
            if not idv.is_valid():
                return False
        return True


if __name__ == '__main__':

    # generate number
    # x = random_ints(50, 100)
    x = [i for i in range(100)]
    ea = EvolutionaryAlgorithm(10, x)
    ea.PARENT_SELECTION = ParentSelectionMethod.FITNESS_BASED
    ea.CROSSOVER = CrossoverMethod.UNIFORM
    print(ea)
    ea.do(10)
    print(ea)

    print(ea.get_best_individual())
    print(ea.result)
