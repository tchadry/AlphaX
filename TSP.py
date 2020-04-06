import numpy as np
from math import hypot
from itertools import permutations


class TSP:

    def __init__(self, n, d=2, points="random_euclidean"):
        self.n, self.d = n, d
        if isinstance(points, (list, np.ndarray)):
            self.points = points
        elif points == "random_euclidean":
            self.points = random_euclidean_tsp(n, d)
        else:
            raise ValueError("Invalid points argument to TSP.")

        self.distances = [[0 for x in range(self.n)] for y in range(self.n)]

        for i in range(self.n):
            for j in range(i):
                x_i, y_i = self.points[i]
                x_j, y_j = self.points[j]
                weight = hypot(x_i - x_j, y_i - y_j)
                self.distances[i][j] = weight
                self.distances[j][i] = weight

    def tour_length(self, tour):
        """Compute the length of the given tour.
        Arguments:
            tour {list(int, n)} -- a permutation of the nodes representing a tour
        Returns:
            float -- tour length
        """
        points = self.points[tour]
        diffs = np.diff(points, axis=0)
        tour_len = np.linalg.norm(diffs, axis=1, ord=2).sum()
        return tour_len


    def get_optimal_length(self):

        if self.n > 10:
            print('WARNING: too many nodes to compute optimal length.')
            return -1

        optimal_value = 100000000000

        for path in permutations(range(self.n)):
            # length = self.get_path_length(path=path)

            length = 0

            for node_index in range(self.n - 1):
                length += self.distances[path[node_index]][path[node_index + 1]]

            length += self.distances[path[0]][path[self.n - 1]]

            if (length < optimal_value):
                optimal_value = length

        return optimal_value

    def payoff(self, tour):
        """Compute the payoff of the given tour, a mapping of the tour length to
        [0,1] where 1 is a better tour.
        Arguments:
            tour {list(int, n)} -- a permutation of the nodes representing a tour
        Returns:
            float -- tour length
        """
        return ((2 * self.n) - self.tour_length(tour)) / (2 * self.n)

def random_euclidean_tsp(n, d=2):
    points = np.random.rand(n, d)
    return points