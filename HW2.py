import random


# limit is exclusive
def random_ints(count: int, limit: int = 1) -> [int]:
    random.seed()
    return [random.randint(0, limit-1) for _ in range(count)]


class EvolutionAlgorithm:
    POPULATION_SIZE = 100

    def __init__(self, k: int, data: [int]):

        # Init
        self.k = k
        self.data_size = len(data)
        self.data = data
        self.population = [random_ints(self.data_size, self.k) for _ in range(self.POPULATION_SIZE)]


if __name__ == '__main__':
    ea = EvolutionAlgorithm(4, [1,2,3])

