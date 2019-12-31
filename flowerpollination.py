from random import random, sample
import numpy as np
from levy import get_levy_flight_array

class FlowerPollinationAlgorithm:
    def __init__(self, obj_fun, iterations_num, pop_size, p=0.5, low=0, high=1 ):
        """
           obj_fun, We want to maximize this function because we use fitness function. 
           iterations_num, Number of iteration or max generation, in this code we use max generation to stop iteration
           pop_size, size of population
           p, switch propbability between 'global_pollination' and 'local_pollination'
           low and high, each values to generate population will be choosen between this number		   
        """
        self.obj_fun = obj_fun
        self.iterations_num = iterations_num
        self.pop_size = pop_size
        self.population = None
        self.best = (0, None)
        self.p = p
        self.obj_vals = np.zeros(pop_size)
        self.low = low
        self.high = high

    def find_solution(self,x_train, y_train):
        self.dim = x_train.shape[1]+1
        self.initialize_population()
        self.calculate_obj(x_train, y_train)
        self.avg = []
        self.bests = []
        for i in range(self.iterations_num):
            self.find_best()
            for i in range(self.pop_size):
                if random() < self.p:
                    new_flower = self.global_pollination(i)
                else:
                    new_flower = self.local_pollination(i)
				
                new_obj_val = self.obj_fun(new_flower, x_train, y_train)
                # we use fitness function, so we want to maximize this function. if you use error function you can switch the sign '>' to '<'
                if new_obj_val > self.obj_vals[i]:
                    self.population[i] = new_flower
                    self.obj_vals[i] = new_obj_val

            self.avg.append(np.mean(self.obj_vals))
            self.bests.append(max(self.obj_vals))
 
    def initialize_population(self):
        self.population = [np.random.uniform(low=self.low,high=self.high, size=self.dim) for _ in range(self.pop_size)]

    def calculate_obj(self,x_train, y_train):
        for i, solution in enumerate(self.population):
            self.obj_vals[i] = self.obj_fun(solution,x_train, y_train)

    def find_best(self):
        # we use fitness function, the biger the better. if you use error function you can switch the sign '>' to '<'
        if max(self.obj_vals) > self.best[0]:
            ind = np.argmax(self.obj_vals)
            self.best = (self.obj_vals[ind], self.population[ind])
			
    def local_pollination(self, i):
        e = np.random.rand(self.dim)
        indexes = sample(range(0, self.pop_size), 2)
        sol1 = self.population[indexes[0]]
        sol2 = self.population[indexes[1]]
        tmp = e * (sol1 - sol2)
        new_solution = self.population[i] + tmp
        return new_solution

    def global_pollination(self, i):
        levy_vector = get_levy_flight_array(self.dim)
        tmp = levy_vector * (self.best[1] - self.population[i])
        new_solution = self.population[i] + tmp
        return new_solution