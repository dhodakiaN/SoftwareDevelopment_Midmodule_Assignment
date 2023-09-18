
import unittest
import timeit
import random
import os

FOLDER_PATH = r'C:\\Users\\NDhod\\Desktop\\Practical Assessment_Programming'

def listDir(dir):
    fileNames = os.listdir(dir)
    for fileName in fileNames:  
        print('File Name:' + fileName)
        print('Folder Path: ' + os.path.abspath(os.path.join(dir, fileName)))  # Use os.path.join correctly

if __name__ == '__main__':  
    listDir(FOLDER_PATH)


###################################
# Number of vertices in the graph
V = 4

# Define infinity as the large enough value.
# This value will be used for vertices not connected to each other
INF = 99999

def floydWarshallIterative(graph):
    """ dist[][] will be the output matrix that will finally have the shortest distances
    between every pair of vertices """

    dist = list(map(lambda i: list(map(lambda j: j, i)), graph))

    # Perform iterative calculations for all pairs of vertices
    for k in range(V):
        for i in range(V):
            for j in range(V):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    print("Iterative Floyd-Warshall Result:")
    printSolution(dist)

def floydWarshallRecursive(graph):
    """ dist[][] will be the output matrix that will finally have the shortest distances
    between every pair of vertices """
    """ initializing the solution matrix same as input graph matrix
    OR we can say that the initial values of shortest distances are based on shortest
    paths considering no intermediate vertices """

    def recursiveFloydWarshall(k, i, j):
        # Base case: If k is 0, return the value in the original graph
        if k == 0:
            return graph[i][j]

        without_k = recursiveFloydWarshall(k - 1, i, j)
        with_k = recursiveFloydWarshall(k - 1, i, k - 1) + recursiveFloydWarshall(k - 1, k - 1, j)

        # Update the distance matrix
        return min(without_k, with_k)

    dist = [[recursiveFloydWarshall(V - 1, i, j) for j in range(V)] for i in range(V)]

    print("Recursive Floyd-Warshall Result:")
    printSolution(dist)

# A utility function to print the solution
def printSolution(dist):
    print("Following matrix shows the shortest distances between every pair of vertices")
    for i in range(V):
        for j in range(V):
            if(dist[i][j] == INF):
                print("%7s" % ("INF"), end=" ")
            else:
                print("%7d\t" % (dist[i][j]), end=' ')
            if j == V-1:
                print()

# Driver's code
if __name__ == "__main__":
    graph = [[0, 5, INF, 10],
            [INF, 0, 3, INF],
            [INF, INF, 0, 1],
            [INF, INF, INF, 0]
            ]
    # Call both iterative and recursive functions
    floydWarshallIterative(graph)
    print("\n")
    floydWarshallRecursive(graph)
# Number of vertices in the graph
V = 4

# Define infinity as a large enough value.
# This value will be used for vertices not connected to each other
INF = 99999


def floyd_warshall_iterative(graph):
    """Solve all-pair shortest path via Floyd Warshall Algorithm"""
    dist = [list(map(lambda i: list(map(lambda j: j, i)), graph))]

    for k in range(V):
        for i in range(V):
            for j in range(V):
                dist[0][i][j] = min(dist[0][i][j], dist[0][i][k] + dist[0][k][j])

    return dist[0]


def floyd_warshall_recursive(graph):
    """dist[][] will be the output matrix that will finally have the shortest distances
    between every pair of vertices"""
    """initializing the solution matrix same as the input graph matrix
    OR we can say that the initial values of shortest distances are based on shortest
    paths considering no intermediate vertices"""

    def recursive_floyd_warshall(k, i, j):
        # Base case: If k is 0, return the value in the original graph
        if k == 0:
            return graph[i][j]

        without_k = recursive_floyd_warshall(k - 1, i, j)
        with_k = recursive_floyd_warshall(k - 1, i, k - 1) + recursive_floyd_warshall(k - 1, k - 1, j)

        # Update the distance matrix
        return min(without_k, with_k)

    dist = [list(map(lambda i: list(map(lambda j: j, i)), graph))]

    # Perform recursive calculations for all pairs of vertices
    for k in range(1, V):
        for i in range(V):
            for j in range(V):
                dist[0][i][j] = recursive_floyd_warshall(k, i, j)

    return dist[0]


# A utility function to print the solution
def print_solution(dist):
    print("Following matrix shows the shortest distances between every pair of vertices")
    for i in range(V):
        for j in range(V):
            if dist[i][j] == INF:
                print("%7s" % ("INF"), end=" ")
            else:
                print("%7d\t" % (dist[i][j]), end=' ')
            if j == V - 1:
                print()


class TestFloydWarshall(unittest.TestCase):
    def test_unit_iterative(self):
        graph = [[0, 5, INF, 10],
                 [INF, 0, 3, INF],
                 [INF, INF, 0, 1],
                 [INF, INF, INF, 0]
                 ]

        dist_iterative = floyd_warshall_iterative(graph)
        self.assertTrue(self.check_distance_matrix(dist_iterative))

    def test_unit_recursive(self):
        graph = [[0, 5, INF, 10],
                 [INF, 0, 3, INF],
                 [INF, INF, 0, 1],
                 [INF, INF, INF, 0]
                 ]

        dist_recursive = floyd_warshall_recursive(graph)
        self.assertTrue(self.check_distance_matrix(dist_recursive))

    def test_integration(self):
        # Integration test: Compare results between the iterative and recursive versions
        graph = [[0, 5, INF, 10],
                 [INF, 0, 3, INF],
                 [INF, INF, 0, 1],
                 [INF, INF, INF, 0]
                 ]

        dist_iterative = floyd_warshall_iterative(graph)
        dist_recursive = floyd_warshall_recursive(graph)

        for i in range(V):
            for j in range(V):
                self.assertAlmostEqual(dist_iterative[i][j], dist_recursive[i][j], delta=0.001)

    def test_performance_iterative(self):
        graph = [[random.randint(0, 10) if i != j else 0 for j in range(100)] for i in range(100)]

        def run_floyd_warshall_iterative():
            floyd_warshall_iterative(graph)

        execution_time_iterative = timeit.timeit(run_floyd_warshall_iterative, number=1000)
        print(f"Performance Test (Iterative): Executed in {execution_time_iterative:.3f} seconds for 1000 iterations.")

    def test_performance_recursive(self):
        graph = [[random.randint(0, 10) if i != j else 0 for j in range(100)] for i in range(100)]

        def run_floyd_warshall_recursive():
            floyd_warshall_recursive(graph)

        execution_time_recursive = timeit.timeit(run_floyd_warshall_recursive, number=1000)
        print(f"Performance Test (Recursive): Executed in {execution_time_recursive:.3f} seconds for 1000 iterations.")

    def check_distance_matrix(self, dist_matrix):
        # Validate the distance matrix.
        # Check if the distances meet specific criteria.
        # For example, you can check if all distances are non-negative.
        for i in range(V):
            for j in range(V):
                if dist_matrix[i][j] < 0:
                    return False
        return True

    def test_big_o_notation(self):
        # Measure the execution time for larger graphs to evaluate Big O Notation.
        # We'll increase the number of vertices exponentially.
        for n in [10, 20, 40, 80]:
            graph = [[random.randint(0, 10) if i != j else 0 for j in range(n)] for i in range(n)]

            def run_floyd_warshall_iterative():
                floyd_warshall_iterative(graph)

            def run_floyd_warshall_recursive():
                floyd_warshall_recursive(graph)

            execution_time_iterative = timeit.timeit(run_floyd_warshall_iterative, number=1)
            execution_time_recursive = timeit.timeit(run_floyd_warshall_recursive, number=1)

            print(f"Big O Test (n={n}): Iterative Version executed in {execution_time_iterative:.3f} seconds.")
            print(f"Big O Test (n={n}): Recursive Version executed in {execution_time_recursive:.3f} seconds.")


if __name__ == "__main__":
    unittest.main()
