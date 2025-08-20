import networkx as nx
import numpy as np


def calculation(G, init_nodes):
    adjacencyMatrix = nx.adjacency_matrix(G).todense()
    N = len(G.nodes)

    # One step transition probability matrix
    P = np.zeros((N, N))
    for row in range(N):
        sumOverRow = np.sum(adjacencyMatrix[row])
        for col in range(N):
            P[row, col] = adjacencyMatrix[row, col] / sumOverRow

    # Reproductive value
    Pi = np.zeros(N)
    W = np.sum(adjacencyMatrix)
    for i in range(N):
        Pi[i] = np.sum(adjacencyMatrix[i]) / W

    A = np.zeros((N * N, N * N))
    B = np.zeros((N * N, 1))

    # initiation
    Xi = np.zeros(N)
    samples = init_nodes
    for sample in samples:
        Xi[sample] = 1
    weighted_xi = Pi @ Xi

    # eta_ii
    identical_index = [i * N + i for i in range(N)]

    for row in range(N * N - 1):
        if row in identical_index:
            i = int(row / N)
            B[row, 0] = N * (weighted_xi - Xi[i])
            for j in identical_index:
                if int(j / N) == i:
                    A[row, j] = 1 - P[i, int(j / N)]
                else:
                    A[row, j] = -P[i, int(j / N)]
        else:
            i = int(row / N)
            j = row % N
            B[row, 0] = N / 2 * (weighted_xi - Xi[i] * Xi[j])
            for k in range(i * N, (i + 1) * N):
                if k % N == j:
                    A[row, k] = 1 - 0.5 * P[j, k % N]
                else:
                    A[row, k] = -0.5 * P[j, k % N]
            for k in range(j, N ** 2 + j, N):
                if int(k / N) == i:
                    A[row, k] = 1 - 0.5 * P[i, int(k / N)]
                else:
                    A[row, k] = -0.5 * P[i, int(k / N)]

    # add the constraint that the sum over pi_i*eta_ii equals 0
    for row in identical_index:
        i = int(row / N)
        A[N * N - 1, row] = Pi[i]

    X = np.linalg.solve(A, B)
    Eta = X.reshape(N, N)
    eta_1 = np.sum(Pi.T @ (P * Eta))
    eta_2 = np.sum(Pi.T @ (P @ P * Eta))
    eta_3 = np.sum(Pi.T @ (P @ P @ P * Eta))

    if eta_3 == eta_1:
        print(np.inf)
        return np.inf
    else:
        threshold = eta_2 / (eta_3 - eta_1)
    return threshold


# Use scale-free network as one example
BA = nx.barabasi_albert_graph(50, 3, seed=0)
# Calculate the critical ratio, where node 0 is initialized as cooperator and others are defectors
print(calculation(BA, [0]))
