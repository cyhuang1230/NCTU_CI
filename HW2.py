import random


def random_ints(count: int, limit: int = 1) -> [int]:
    """
    :param count:
    :param limit: inclusive
    :return:
    """
    random.seed()
    return [random.randint(0, limit-1) for _ in range(count)]


class Individual:

    def __init__(self, k:int, genes: [int]):
        """
        :param k:
        :param genes: a list of int indicating the cluster each point belongs to
        """
        self.k = k
        self.gene_size = len(genes)
        self.genes = genes
        self.fitness = None

    def get_fitness(self, data: [int]) -> float:
        if self.fitness is None:
            self.fitness = 0.0
            centroids = self.get_centroids(data)
            for i in range(self.gene_size):
                self.fitness += pow(data[i] - centroids[self.genes[i]], 2)
        return self.fitness

    def get_centroids(self, data: [int]) -> [float]:
        num_of_points_in_cluster = [0] * self.k
        sum_of_coordinate_in_cluster = [0.0] * self.k
        for i in range(self.gene_size):
            cluster_num = self.genes[i]
            coordinate = data[i]
            num_of_points_in_cluster[cluster_num] += 1
            sum_of_coordinate_in_cluster[cluster_num] += coordinate
        return [sum_of_coordinate_in_cluster[i]/num_of_points_in_cluster[i] for i in range(self.k)]

    # only valid if there's at least one point in each cluster
    def is_valid(self) -> bool:
        ret = [False for _ in range(self.k)]
        for g in self.genes:
            ret[g] = True
        for r in ret:
            if not r:
                return False
        return True

    def __str__(self, data=None):
        return "k = %d, data = %s, genes = %s, fitness = %.3f, centroids = %s" \
               % (self.k, data if data else 'None', self.genes.__str__(),
                  self.fitness if self.fitness else self.get_fitness(data) if data else -1,
                  self.get_centroids(data).__str__())

class EvolutionaryAlgorithm:
    POPULATION_SIZE = 100

    def __init__(self, k: int, data: [int]):
        self.k = k
        self.data_size = len(data)
        self.data = data
        self.population = []
        
        kount = 0
        while kount < self.POPULATION_SIZE:
            anIdv = Individual(k, random_ints(self.data_size, self.k))
            if not anIdv.is_valid():
                continue
            self.population.append(anIdv)
            kount += 1

    def __str__(self):
        ret = ''
        for p in self.population:
            ret += p.__str__(self.data) + '\n'
        return ret

    def do(self, generation_count: int):
        pass


if __name__ == '__main__':

    # generate number
    # x = random_ints(50, 100)
    x = [i for i in range(4)]
    ea = EvolutionaryAlgorithm(2, x)
    print(ea)
