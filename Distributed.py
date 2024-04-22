import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

np.random.seed(0)


def build_random_graph(required_probability=0.999):
    """
    Builds a random geometric graph with a given required probability.

    Parameters:
    - required_probability (float): The required probability for the graph connectivity. Default is 0.999.

    Returns:
    - num_nodes (int): The number of nodes in the graph.
    - G (networkx.Graph): The generated random geometric graph.
    - A (numpy.ndarray): The adjacency matrix of the graph.
    - pos (dict): The positions of the nodes in the graph.
    """
    # we are working in 2 dimensions
    num_nodes = int(np.ceil(np.sqrt(1 / (1 - required_probability))))

    num_nodes = 200
    r_c = np.sqrt(np.log(num_nodes) / num_nodes)

    pos = {i: (np.random.uniform(low=0, high=100), np.random.uniform(low=0, high=100)) for i in range(num_nodes)}

    G = nx.random_geometric_graph(n=num_nodes, radius=r_c * 100, pos=pos)

    A = nx.adjacency_matrix(G).toarray()
    return num_nodes, G, A, pos

def W_construct_rand_gossip(i, j, n):
    """
    Constructs a weight matrix W for a distributed signal processing system.

    Parameters:
    i (int): Index of the first element.
    j (int): Index of the second element.
    n (int): Number of elements in the system.

    Returns:
    numpy.ndarray: Weight matrix W.

    """
    e_i = np.zeros([n, 1])
    e_j = np.zeros([n, 1])
    e_i[i] = 1
    e_j[j] = 1
    W = np.identity(n)
    W -= 0.5 * (e_i - e_j) * np.transpose(e_i - e_j)
    return W

def generate_temp_field(num_nodes, var, true_temp):
    """
    Generate a temperature field for a given number of nodes.

    Parameters:
    - num_nodes (int): The number of nodes in the field.
    - var (float): The variance of the temperature values.
    - true_temp (float): The true temperature value.

    Returns:
    - temperature (numpy.ndarray): An array of shape (num_nodes, 1) containing the generated temperature values.
    """
    temperature = np.zeros([num_nodes, 1])

    for i, val in enumerate(temperature):
        temperature[i][0] += np.random.normal(true_temp, np.sqrt(var))

    return temperature


def random_gossip(temperature, A, tolerance=0.00001):
    """
    Perform random gossip algorithm to update the temperature values.

    Parameters:
    temperature (numpy.ndarray): The initial temperature values for each node.
    num_nodes (int): The total number of nodes in the network.
    A (numpy.ndarray): The adjacency matrix representing the network connections.
    tolerance (float, optional): The convergence tolerance. Defaults to 0.00001.

    Returns:
    numpy.ndarray: The loss values at each iteration until convergence.

    """
    num_nodes = np.shape(A)[0]
    converged = False
    loss = np.array([])
    while not converged:

        node_i = int(np.random.uniform(low=0, high=num_nodes))
        i_neigh = np.transpose(np.nonzero(A[node_i, :]))
        j_index = int(np.random.uniform(low=0, high=np.shape(i_neigh)[0]))
        node_j = i_neigh[j_index][0]

        # # update equation
        # W = W_construct_rand_gossip(node_i, node_j, num_nodes)
        # temperature = np.dot(W, temperature)
        avg=(temperature[node_i]+temperature[node_j])/2
        temperature[node_i]=avg
        temperature[node_j]=avg

        if np.std(temperature) < tolerance:
            loss = np.append(loss, np.std(temperature))
            converged = True
        else:
            loss = np.append(loss, np.std(temperature))
    return loss,temperature


def async_distr_averaging(temperature,A,tolerance):
    """
    Perform asynchronous distributed averaging algorithm.

    Returns:
    numpy.ndarray: The loss values at each iteration until convergence.

    """
    num_nodes = np.shape(A)[0]
    converged = False
    loss_a = np.array([])
    while not converged:

        node_i = int(np.random.uniform(low=0, high=num_nodes))
        i_neigh = np.transpose(np.nonzero(A[node_i, :]))
        num_neigh = np.shape(i_neigh)[0]
        
        # update equation
        avg_val= (np.sum(temperature[i_neigh])+temperature[node_i]) / (num_neigh+1)
        temperature[node_i] = avg_val
        temperature[i_neigh] = avg_val

        if np.std(temperature) < tolerance:
            loss_a = np.append(loss_a, np.std(temperature))
            converged = True
        else:
            loss_a = np.append(loss_a, np.std(temperature))

    return loss_a,temperature


def plot_log_convergence(losses, legend):
    for loss in losses:
        plt.plot(range(1, len(loss)+1), loss)
    plt.xlabel('Iterations')
    plt.ylabel('Loss (standard deviation)')
    plt.title('Loss vs Iterations')
    plt.yscale('log')
    plt.legend(legend)
    plt.show()
    
required_probability=0.9999
num_nodes, G,A,pos=build_random_graph(required_probability)
print("num_nodes:",num_nodes)

#now to generate measured values for the temperature sensors ins some flat 3d field
temperature=generate_temp_field(num_nodes,5,25)

#select a node i at random (uniformly) and contact neigbouring node j at random(uniformly)
# nx.draw(G, pos=pos, with_labels=True)
# plt.show()

tolerance=10**-5
loss_random,temperature_rand=random_gossip(temperature.copy(),A,tolerance)
loss_async, temperature_async =async_distr_averaging(temperature.copy(),A,tolerance)

plot_log_convergence([loss_async,loss_random],['Asynchronous Distributed Averaging','Random Gossip'])

# print("temperature_rand:",temperature_rand[0:10])
print("temperature_async:",loss_async[0:10])
