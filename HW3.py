import datetime
import pickle
import random
import math
from enum import IntEnum
from collections import deque

class Path:
    def __init__(self, origin: int, dest: int, distance: float):
        self.origin = origin
        self.dest = dest
        self.distance = distance
        self.pheromone = 0.0

    def __str__(self):
        return '%s --(%.2f)--> %s' % (self.origin, self.distance, self.dest)


class City:
    def __init__(self, id: int, x: float, y: float):
        self.id = id
        self.x = x
        self.y = y
        self.paths = {}


class AntMovingStatus(IntEnum):
    IN_CITY = 1
    ON_PATH = 2


class Ant:
    __slots__ = ['id', 'init_city', 'cur_city', 'next_city', 'status', 'visited_city', 'left_distance']
    MOVING_SPEED = 0.01

    def __init__(self, id: int, init_city: int):
        self.id = id
        self.init_city = init_city
        self.cur_city = init_city
        self.next_city = -1
        # put init_city in visited so no need to check every time when choosing next city
        self.visited_city = [init_city]
        self.status = AntMovingStatus.IN_CITY
        self.left_distance = 0.0

    # @TODO
    def move(self, city_path: {}):
        if self.status == AntMovingStatus.IN_CITY:
            # if in city => choose next city
            pass
        elif self.status == AntMovingStatus.ON_PATH:
            # if on the path => keep moving
            pass


class AntColonyProblem:

    def __init__(self, population: int):
        self.t = 0
        self.cities = AntColonyProblem.get_graph()
        self.population = population
        self.ants = deque()
        self.last_id = 0
        self.ant_kount = 0
        self.init_city = 0

    # @TODO
    def run(self, max_time: int):
        while self.t < max_time:
            for ant in self.ants:
                pass

    # @TODO
    def make_ant(self):
        ant = Ant(self.last_id, self.init_city)
        self.ants.append(ant)
        self.last_id += 1
        self.ant_kount += 1

    @staticmethod
    def get_graph(get_new=True, city_count=10, filename='input3.pickle'):

        if get_new:
            # generate cities
            random.seed()
            cities = [City(i, random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(city_count)]

            # generate paths
            for i in range(city_count):
                for j in range(i+1, city_count):
                    distance = math.sqrt((cities[i].x - cities[j].x) ** 2 + (cities[i].y - cities[j].y) ** 2) \
                               * random.uniform(1, 2)
                    cities[i].paths[j] = Path(i, j, distance)
                    cities[j].paths[i] = Path(j, i, distance)

            # save to file
            # pickle.dump(cities, open('input3-%u.pickle' % datetime.datetime.now().timestamp(), 'wb'))

        else:
            cities = pickle.load(open(filename, 'rb'))

        return cities


if __name__ == '__main__':
    ac = AntColonyProblem(10000)
    cities = ac.get_graph()
    for c in cities:
        print(c.id, c.x, c.y, [(pp.__str__()) for _,pp in c.paths.items()])
