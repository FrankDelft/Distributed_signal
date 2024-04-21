import numpy as np
import matplotlib.pyplot as plt
import networkx as nx



def W_construct(i, j, n):
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

    # num_nodes = 32
    r_c = np.sqrt(np.log(num_nodes) / num_nodes)

    pos = {i: (np.random.uniform(low=0, high=100), np.random.uniform(low=0, high=100)) for i in range(num_nodes)}

    G = nx.random_geometric_graph(n=num_nodes, radius=r_c * 100, pos=pos)

    A = nx.adjacency_matrix(G).toarray()
    return num_nodes, G, A, pos


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


def random_gossip(temperature, num_nodes, A, tolerance=0.00001):
    num_iter=0
    converged=False
    loss=np.array([])
    while(not converged):
        # for var in range(100000):
        num_iter+=1
        node_i=int(np.random.uniform(low=0, high=num_nodes))
        i_neigh=np.transpose(np.nonzero(A[node_i,:]))
        j_index=int(np.random.uniform(low=0, high=np.shape(i_neigh)[0]))
        node_j=i_neigh[j_index][0]

        #update equation
        W=W_construct(node_i,node_j,num_nodes)
        temperature=np.dot(W,temperature)

        if np.std(temperature) < tolerance:
            loss=np.append(loss,np.std(temperature))
            converged=True
        else:
            loss=np.append(loss,np.std(temperature))
    return loss

required_probability=0.9999
num_nodes, G,A,pos=build_random_graph(required_probability)
print("num_nodes:",num_nodes)
#now to generate measured values for the temperature sensors ins some flat 3d field
temperature=generate_temp_field(num_nodes,5,25)

#select a node i at random (uniformly) and contact neigbouring node j at random(uniformly)
nx.draw(G, pos=pos, with_labels=True)
plt.show()



temp_og=temperature
tolerance=0.00001
loss_random=random_gossip(temperature,num_nodes,A,tolerance)

plt.plot(range(1, len(loss_random)+1), loss_random)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss vs Iterations')
plt.yscale('log')
plt.show()

# print("new",np.transpose(temperature),"\nold",np.transpose(temp_og),"\n numiter:",num_iter)


