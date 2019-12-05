import random as rand
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class GeneticOptimization:
    def __init__(self, population_size: int, fun, mutation_rate: float, boundaries):
        self.__population_size = population_size
        self.__fun = fun
        self.__mutation_rate = mutation_rate
        self.__boundaries = boundaries
        self.__population = self.__generate_population()

    def __generate_population(self):
        return [(rand.uniform(self.__boundaries[0][0], self.__boundaries[0][1]),
                 rand.uniform(self.__boundaries[1][0], self.__boundaries[1][1]))
                for _ in range(self.__population_size)]

    def calculate_fittnes(self, population):
        return [self.__fun(*x) for x in population]

    def calculate_selection(self, fitness):
        max_fitness = sum(fitness)
        sel_max = [[index, (fit/max_fitness)] for index, fit in enumerate(fitness)]
        sorted_sel_max = sorted(sel_max, key=lambda val:val[1])
        tab_middle = (int)(math.floor(len(sorted_sel_max)/2))
        for left, right in zip(sorted_sel_max[:tab_middle], sorted_sel_max[:tab_middle - 1:-1]):
            temp = left[0]
            left[0] = right[0]
            right[0] = temp

        return sorted_sel_max



    def perform_selection(self, selection):
        """
        Based on roulette wheel selection
        :param fitness:
        :return:
        """
        sorted_selections = sorted(selection, key=lambda val: val[1])
        to_ret = []
        for _ in range(len(selection)):
            sum = 0
            rand_num = rand.uniform(0, 0.99)
            for index, value in sorted_selections:
                sum += value
                if(rand_num < sum):
                    to_ret.append(index)
                    break

        return to_ret

    def crossover(self, parent1, parent2):
        rand_split = rand.random()
        new_child = [par1_val * rand_split + par2_val * (1 - rand_split)
                     for par1_val, par2_val in zip(parent1, parent2)]

        return new_child

    def mutate(self, child):
        for dimension, child_val in zip(range(len(child)), child):
            random_val = rand.uniform(*self.__boundaries[dimension])
            random_val *= self.__mutation_rate
            child[dimension] += random_val

    def find(self, steps_num: int, accuracy: float):
        plt.ion()
        fig = plt.figure()
        plot_3d = fig.gca(projection='3d')

        selection_values = None
        best_child = None
        for _ in range(steps_num):
            actual_fitness = self.calculate_fittnes(self.__population)
            selection_values = self.calculate_selection(actual_fitness)
            selection_indexes = self.perform_selection(selection_values)
            # generating new population
            new_population = []
            for _ in range(self.__population_size):
                # getting random parents
                parent1 = self.__population[selection_indexes[rand.randint(0, self.__population_size-1)]]
                parent2 = self.__population[selection_indexes[rand.randint(0, self.__population_size-1)]]
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                new_population.append(child)

            plot_3d.clear()
            x, y = zip(*self.__population)
            x_boundaries = (self.__boundaries[0][0], self.__boundaries[0][1])
            y_boundaries = (self.__boundaries[1][0], self.__boundaries[1][1])
            z_boundaries = [self.__fun(*val) for val in zip(x_boundaries, y_boundaries)]
            plot_3d.scatter(x, y, actual_fitness, c="r", marker="x")
            plot_3d.scatter(x_boundaries, y_boundaries, z_boundaries, c="black", marker="o")

            self.__population = new_population

            act_best_child = self.__population[max(selection_values, key=lambda val: val[1])[0]]
            print(act_best_child)
            if best_child is None:
                best_child = act_best_child
            else:
                act_accuracy = max([old - new for old, new in zip(best_child, act_best_child)])
                best_child = act_best_child
                if accuracy is not None and act_accuracy <= accuracy:
                    break
            plt.pause(1)
        # getting best child

        return best_child


if __name__ == "__main__":
    f = lambda x, y: 20 + (x**2 - 10*math.cos(2 * math.pi * x)) + (y**2 - 10*math.cos(2 * math.pi * y))
    gen = GeneticOptimization(1000, f, 0, ((-5.12, 5.12), (-5.12, 5.12)))
    print(gen.find(20, None))