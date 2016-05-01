import random
import heapq
import matplotlib.pyplot as plt
from enum import IntEnum
import datetime
import pickle


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
        self.gene = genes
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

    def update(self) -> None:
        self.update_centroids()
        self.update_fitness()

    def update_fitness(self) -> None:
        self.fitness = 0.0
        centroids = self.get_centroids()
        for i in range(self.gene_size):
            self.fitness += pow(self.data[i] - centroids[self.gene[i]], 2)

    def update_centroids(self) -> None:
        num_of_points_in_cluster = [0] * self.k
        sum_of_coordinate_in_cluster = [0.0] * self.k
        for i in range(self.gene_size):
            cluster_num = self.gene[i]
            coordinate = self.data[i]
            num_of_points_in_cluster[cluster_num] += 1
            sum_of_coordinate_in_cluster[cluster_num] += coordinate
        self.centroids = [sum_of_coordinate_in_cluster[i] / num_of_points_in_cluster[i] for i in range(self.k)]

    # only valid if there's at least one point in each cluster
    def is_valid(self) -> bool:
        ret = [False for _ in range(self.k)]
        for g in self.gene:
            ret[g] = True
        for r in ret:
            if not r:
                return False
        return True

    def __str__(self) -> str:
        return "k = %d, data = %s, gene = %s, fitness = %.3f, centroids = %s" \
               % (self.k, self.data if self.data else 'None', self.gene.__str__(),
                  self.fitness if self.fitness else self.get_fitness(),
                  self.get_centroids().__str__())

    def __lt__(self, other) -> bool:
        # in our case, fitness is the lower the better
        return self.get_fitness() > other.get_fitness()


class ParentSelectionMethod(IntEnum):
    FITNESS_BASED = 1
    TOURNAMENT_SELECTION = 2


class CrossoverMethod(IntEnum):
    MEAN = 1
    ONE_POINT = 2
    TWO_POINT = 3
    UNIFORM = 4


class EvolutionaryAlgorithm:
    POPULATION_SIZE = 1000
    PARENT_SELECTION = ParentSelectionMethod.TOURNAMENT_SELECTION
    CROSSOVER = CrossoverMethod.MEAN
    MUTATE_PROBABILITY = 0.6

    def __init__(self, k: int, data: [int]):
        self.k = k
        self.data_size = len(data)
        self.data = data
        self.population = []
        self.result = {"gene":[], "fitness": [], "centroids": []}
        self.generation = 1

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

    def __str__(self) -> str:
        ret = ''
        for p in self.population:
            ret += p.__str__() + '\n'
        return ret

    def do(self, generation: int) -> None:
        self.generation += generation

        for i in range(generation):
            # Parent selection
            parents = self.choose_parent()
            # print('--- parents ---')
            # print_custom(parents)

            # to ensure the validity of children
            while True:
                # Crossover
                crossover_children = self.do_crossover(parents)
                # Proceed only when children are all valid
                if not EvolutionaryAlgorithm.is_valid_individuals(crossover_children):
                    continue

                # Mutation
                mutation_children = [self.do_mutation(child) for child in crossover_children]
                # Proceed only when children are all valid
                if EvolutionaryAlgorithm.is_valid_individuals(mutation_children):
                    break

            # print('--- children ---')
            # print_custom(mutation_children)

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
                child_one_gene.append(parents[0 if not should_swap else 1].gene[i])
                child_two_gene.append(parents[1 if not should_swap else 0].gene[i])
            children.append(Individual(self.k, child_one_gene, self.data))
            children.append(Individual(self.k, child_two_gene, self.data))
        elif self.CROSSOVER == CrossoverMethod.UNIFORM:
            child_one_gene = []
            child_two_gene = []
            for i in range(size):
                prob = random.uniform(0, 1)
                # if prob in [0,0.5) -> child_one gets parent[0], child_two gets parent[1] => i.e. should not swap
                # else if prob in [0.5,1] -> child_one gets parent[1], child_two gets parent[0] => i.e. should swap
                child_one_gene.append(parents[int(prob/0.5)].gene[i])
                child_two_gene.append(parents[(int(prob/0.5)+1)%2].gene[i])
            children.append(Individual(self.k, child_one_gene, self.data))
            children.append(Individual(self.k, child_two_gene, self.data))
        return children

    def do_mutation(self, an_idv: Individual) -> Individual:
        for i in range(an_idv.gene_size):
            should_mutate = random.uniform(0, 1)
            if should_mutate > self.MUTATE_PROBABILITY:
                continue
            an_idv.gene[i] = random.randint(0, self.k - 1)
        if an_idv.is_valid():   # in case we mutate sth invalid
            an_idv.update()
        return an_idv

    def record_result(self) -> None:
        best = self.get_best_individual()
        self.result["gene"].append(best.gene)
        self.result["fitness"].append(best.get_fitness())
        self.result["centroids"].append(best.get_centroids())

    def draw(self, timestamp=None) -> None:
        figure_filename = '%u-%dk-%dg-%ddata-%dpopu-%dPS-%dCO-%.1fMP' \
                          % (timestamp if timestamp else datetime.datetime.now().timestamp(),
                             self.k, self.generation, self.data_size, len(self.population),
                             self.PARENT_SELECTION, self.CROSSOVER, self.MUTATE_PROBABILITY)

        figure_title = '%dk, %dgen, %ddata, %dpopu, PS%d, CO%d, %.1fMP' \
                          % (self.k, self.generation, self.data_size, len(self.population),
                             self.PARENT_SELECTION, self.CROSSOVER, self.MUTATE_PROBABILITY)

        figure = plt.figure()
        x = list(range(self.generation))
        fitness = self.result["fitness"]
        y_min = min(fitness) * 0.98
        y_max = max(fitness) * 1.01
        plt.plot(x, fitness)
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        figure.canvas.set_window_title(figure_filename)
        ax = figure.add_subplot(111)
        ax.set_title(figure_title)
        # plt.show()
        plt.savefig(figure_filename + '.png')

    @staticmethod
    def is_valid_individuals(idvs: [Individual]) -> bool:
        for idv in idvs:
            if not idv.is_valid():
                return False
        return True


if __name__ == '__main__':


    # generate number
    input_filename = 'input.pickle'
    # x = random_ints(1000, 10000)
    # pickle.dump(x, open(input_filename, 'wb'))
    x = pickle.load(open(input_filename, 'rb'))
    for k in range(1, 11):
        start_time = datetime.datetime.now()
        timestamp = start_time.timestamp()
        ea = EvolutionaryAlgorithm(k, x)
        ea.PARENT_SELECTION = ParentSelectionMethod.FITNESS_BASED
        ea.CROSSOVER = CrossoverMethod.UNIFORM
        ea.do(10000)

        # print(ea.get_best_individual())
        # print(ea.result['fitness'])
        ea.draw(timestamp)
        end_time = datetime.datetime.now()

        log_file_name = '%u_log.txt' % timestamp
        log_file = open(log_file_name, 'w+')
        print('start_time = %u, end_time = %u, duration = %s'
              % (timestamp, end_time.timestamp(), end_time-start_time), file=log_file)
        print('k = %d, %d generations, %d data points , population = %d, %.1f mutation probability'\
                          % (ea.k, ea.generation, ea.data_size, len(ea.population), ea.MUTATE_PROBABILITY), file=log_file)
        parent_selection_method = ['', 'Fitness-based', 'Tournament selection']
        crossover_method = ['', 'arithmetic mean', '1-point crossover', '2-point crossover', 'Uniform crossover']
        print('ParentSelectionMethod = %s' % parent_selection_method[ea.PARENT_SELECTION], file=log_file)
        print('CrossoverMethod = %s' % crossover_method[ea.CROSSOVER], file=log_file)
        print('data:\n', x, file=log_file)
        print('best_individual:\n', ea.get_best_individual(), file=log_file)
        print('fitness_result:\n', ea.result["fitness"], file=log_file)
        log_file.close()

        print('No. %d Done.' %k)

    print('ALL Done.')
