import datetime
import pickle
import random
import math
from enum import IntEnum
from collections import deque
import matplotlib.pyplot as plt
import itertools
import sys
import os.path


class Path:
    def __init__(self, origin: int, dest: int, distance: float):
        self._origin = origin
        self._dest = dest
        self._distance = distance
        self._pheromone = 0.0

    def __str__(self):
        return '%s --(D:%.2f,P:%.2f)--> %s' % (self._origin, self._distance, self._pheromone, self._dest)

    @property
    def pheromone(self):
        return self._pheromone

    @pheromone.setter
    def pheromone(self, p):
        self._pheromone = p

    @property
    def distance(self):
        return self._distance


class City:
    def __init__(self, id: int, x: float, y: float):
        self.id = id
        self.x = x
        self.y = y
        self.paths = {}


class AntStatus(IntEnum):
    IN_CITY = 1
    ON_PATH = 2
    FINISHED = 3


class Ant:
    __slots__ = ['_id', '_init_city', '_cur_city', '_next_city','_status',
                 '_visited_city', '_left_distance', '_total_city_count', '_route', '_moved_distance']
    MOVING_SPEED = 0.01
    ALPHA = 1
    BETA = 1

    def __init__(self, id: int, init_city: int, total_city: int):
        self._id = id
        self._init_city = init_city
        self._cur_city = init_city
        self._next_city = -1
        self._visited_city = deque()
        self._status = AntStatus.IN_CITY
        self._left_distance = 0.0
        self._total_city_count = total_city
        self._moved_distance = 0.0

    def move(self, city_path: {int: Path}=None):
        # # print('Move:', self)
        assert self._status != AntStatus.FINISHED
        if self._status == AntStatus.IN_CITY:
            # if in city => choose next city
            self._visited_city.append(self._cur_city)
            self._status, self._next_city, self._left_distance = self.choose_next_city(city_path)
            self._moved_distance += self._left_distance

        elif self._status == AntStatus.ON_PATH:
            # if on the path => keep moving
            self._left_distance -= self.MOVING_SPEED
            if self._left_distance <= 0.0:  # if arrived
                self._left_distance = 0.0
                self._status = AntStatus.IN_CITY
                self._cur_city = self._next_city
                self._next_city = -1

    def choose_next_city(self, city_path: {int: Path}) -> (AntStatus, int, float):
        print(self._id, [c.__str__() for _, c in city_path.items()], file=log)

        # if finished
        if len(self._visited_city) == self._total_city_count:
            if self.cur_city == self._init_city:    # finally return to init node
                self._visited_city.appendleft(self._init_city)
                self._visited_city.append(self._init_city)
                return AntStatus.FINISHED, self._init_city, 0.0
            else:   # when ant is at last node
                self._visited_city.popleft()

        # calculate probability
        prob = [(city_path[i].pheromone ** self.ALPHA) / (city_path[i].distance ** self.BETA)
                if i not in self._visited_city else 0.0
                for i in range(self._total_city_count)]
        sum_ = sum(prob)
        choice = None
        if sum_ == 0:
            choice = random.choice([i for i in range(self._total_city_count) if i not in self._visited_city])
        else:
            # choose one city
            prob = [p/sum_ for p in prob]
            choice_prob = random.random()
            for i, p in enumerate(prob):
                choice_prob -= p
                if choice_prob <= 0.0:
                    choice = i
                    break

        assert choice is not None
        return AntStatus.ON_PATH, choice, city_path[choice].distance

    def get_route(self, undirected=True) -> deque:
        ret = deque()
        # in the end, there will be _total_city_count+1 element in _visited_city
        for i in range(self._total_city_count):
            origin = self._visited_city[i]
            dest = self._visited_city[i+1]
            ret.append((origin, dest))
            if undirected:
                ret.append((dest, origin))
        return ret

    @property
    def status(self):
        return self._status

    @property
    def cur_city(self):
        return self._cur_city

    @property
    def moved_distance(self):
        return self._moved_distance

    def __str__(self):
        return '#%d, cur:%d, next:%d, left_dist:%.2f, visited:%s' \
               % (self._id, self._cur_city, self._next_city, self._left_distance, self._visited_city.__str__())


class AntColonyProblem:

    MAKE_ANT_INTERVAL = 10
    DELTA = 0.2  # [0.1,0.3]
    Q = 1

    def __init__(self, population: int, city_count: int, filename=None):
        self.t = 0
        self._theoretical_distance = None
        self._theoretical_route = None
        self.cities = None
        self.population = population
        self.ants = deque()
        self.last_id = 0
        self.ant_count = 0
        self.init_city = 0
        self.total_city_count = city_count
        self.fitness = deque([0.0])
        self.get_graph(filename)
        self._theoretical_fitness = 1/self._theoretical_distance

    def run(self, max_time: int):
        while self.t < max_time and self.fitness[-1] != self._theoretical_fitness:

            print('[%d] t = %d begin, cur_fitness: %.2f' % (self.total_city_count, self.t, self.fitness[-1]), file=log)
            print('[%d] t = %d begin, cur_fitness: %.2f' % (self.total_city_count, self.t, self.fitness[-1]), end='\r')

            # make new ant if necessary
            if self.ant_count == 0 or \
                    (self.ant_count < self.population and not self.t % self.MAKE_ANT_INTERVAL):
                self.make_ant()

            # move every ant
            has_finished_ants = deque()
            best_fitness = self.fitness[-1]
            for ant in self.ants:
                ant.move(self.cities[ant.cur_city].paths if ant.status == AntStatus.IN_CITY else None)
                if ant.status == AntStatus.FINISHED:
                    has_finished_ants.append(ant)
                    fitness = 1/ant.moved_distance
                    if fitness > best_fitness:
                        best_fitness = fitness

            # remove finished ants
            deltas = self.remove_ants(has_finished_ants)

            # update pheromone
            self.update_pheromone(deltas)

            self.fitness.append(best_fitness)
            self.t += 1

    def make_ant(self):
        print('Making ant #%d' % self.last_id, file=log)
        ant = Ant(self.last_id, self.init_city, self.total_city_count)
        self.ants.append(ant)
        self.last_id += 1
        self.ant_count += 1

    def remove_ants(self, ants: deque):
        deltas = {}
        for ant in ants:
            print('Finished ant #%d, route = %s' % (ant._id, ant.get_route().__str__()), file=log)
            # calculate delta
            delta = self.Q / ant.moved_distance
            self.ants.remove(ant)
            self.ant_count -= 1
            for p in ant.get_route():
                deltas[p] = deltas.get(p, 0.0) + delta
        return deltas

    def update_pheromone(self, deltas):
        # update every paths' pheromone
        for i in range(self.total_city_count):
            for j in range(i+1, self.total_city_count):
                new_pheromone = (1 - self.DELTA) * self.cities[i].paths[j].pheromone + deltas.get((i, j), 0.0)
                self.cities[j].paths[i].pheromone = self.cities[i].paths[j].pheromone = new_pheromone

    def get_graph(self, filename=None) -> None:

        if filename is None: #or not os.path.exists(filename):
            # generate cities
            random.seed()
            cities = [City(i, random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(self.total_city_count)]

            # generate paths
            for i in range(self.total_city_count):
                for j in range(i+1, self.total_city_count):
                    distance = math.sqrt((cities[i].x - cities[j].x) ** 2 + (cities[i].y - cities[j].y) ** 2) \
                               * random.uniform(1, 2)
                    cities[i].paths[j] = Path(i, j, distance)
                    cities[j].paths[i] = Path(j, i, distance)

            self.cities = cities

            # get best route
            self._theoretical_distance, self._theoretical_route = self.theoretical_distance()

            # save to file
            pickle.dump((cities, self._theoretical_distance, self._theoretical_route),
                        open('input3-%d.pickle' % self.total_city_count, 'wb'))

        else:
            self.cities, self._theoretical_distance, self._theoretical_route = pickle.load(open(filename, 'rb'))

        assert len(self.cities) == self.total_city_count

    def theoretical_distance(self) -> (float, [int]):
        if self._theoretical_distance is not None and self._theoretical_route is not None:
            return self._theoretical_distance, self._theoretical_route
        city_count = len(self.cities)
        self._theoretical_route = None
        self._theoretical_distance = sys.maxsize
        numbers = [i for i in range(city_count)]
        numbers.remove(self.init_city)
        permutations = itertools.permutations(numbers)
        for p in permutations:
            route = [self.init_city] + list(p) + [self.init_city]
            cur_distance = 0.0
            can_be_solution = True
            for i in range(city_count):
                cur_distance += self.cities[route[i]].paths[route[i+1]].distance
                if cur_distance > self._theoretical_distance:
                    can_be_solution = False
                    break
            print("Finished running route %s, distance = %.2f" % (route.__str__(), cur_distance))
            if can_be_solution:
                self._theoretical_distance = cur_distance
                self._theoretical_route = route
        return self._theoretical_distance, self._theoretical_route

if __name__ == '__main__':

    for city_count in range(5, 11, 1):
        now = datetime.datetime.now()
        max_time = 100000
        figure_title = '(%u) %d cities, %d maxtime, 0.01 rate' % (now.timestamp(), city_count, max_time)
        figure = plt.figure()
        color = ['r', 'g', 'navy', 'black', 'gray']
        for population in range(10000, 50001, 10000):
            # population = 10000
            # city_count = 10

            log = open('log-%u.txt' % now.timestamp(), 'w+')

            ac = AntColonyProblem(population, city_count, filename='input3-%d.pickle' % city_count)
            cities, best_distance, best_route = ac.cities, ac._theoretical_distance, ac._theoretical_route
            # for c in cities:
            #     print(c.id, c.x, c.y, [(pp.__str__()) for p,pp in c.paths.items()])

            ac.run(max_time)
            end_time = datetime.datetime.now()
            print('Duration: %s' % (end_time-now), file=log)

            # print('-----')
            # for c in cities:
            #     print(c.id, c.x, c.y, [(pp.__str__()) for p, pp in c.paths.items()])
            result = []
            map(lambda x, result=result: result.append(x) if x not in result else None, ac.fitness)
            # print(result, file=log)
            log.close()

            # figure_filename = '%u-%dcity-%dant-%dt' % (now.timestamp(), city_count, population, max_time)
            # figure_title = '(%u) %d cities, %d ants, %d/%d time' \
            #                % (now.timestamp(), city_count, population, ac.t, max_time)
            # figure = plt.figure()
            # figure.canvas.set_window_title(figure_filename)
            ax = figure.add_subplot(111)
            ax.set_title(figure_title)
            x = [i for i in range(len(ac.fitness))]
            if population == 10000:
                plt.axhline(1/best_distance, c='blue', label='Best Fitness')
            plt.plot(x, ac.fitness, color[int(population/10000%5)], label="%d (%d)" % (population, len(ac.fitness)))
        plt.xlabel("t")
        plt.ylabel("Fitness")
        plt.legend(loc='best', ncol=1, fancybox=True, shadow=True)
        plt.savefig("%u_result.png" % now.timestamp())
        # plt.show()
        plt.clf()

