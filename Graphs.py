import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Erdos-Renyi ===========================================================================================================
# =======================================================================================================================

def erdos_renyi_graph(C, N, plot_degree_distribution=False):

    G = nx.fast_gnp_random_graph(N, C/N, seed=42)
    alpha = nx.to_scipy_sparse_array(G, format="csr")

    if plot_degree_distribution:
        deg = np.array(alpha.sum(axis=1)).flatten()
        plt.hist(deg, bins=range(int(deg.min()), int(deg.max()) + 2), density=True, align='left')
        plt.xlabel("Grado k")
        plt.ylabel("P(k)")
        plt.title("Connectivity distribution of bimodal graph")
        plt.grid(True)
        plt.show()

    return alpha

# Chung-Lu Bimodal ======================================================================================================
# =======================================================================================================================

def bimodal_graph(delta_bimodal, C, N, plot_degree_distribution=False, seed = 42):

    k2 = int(C + delta_bimodal/2)
    k1 = int(C - delta_bimodal/2)

    degrees = [k1 if np.random.rand() < 0.5 else k2 for _ in range(N)]
    G = nx.expected_degree_graph(degrees, selfloops=False)
    alpha = nx.to_scipy_sparse_array(G, format='csr')

    if plot_degree_distribution:
        deg = np.array(alpha.sum(axis=1)).flatten()
        plt.hist(deg, bins=range(int(deg.min()), int(deg.max()) + 2), density=True, align='left')
        plt.xlabel("Grado k")
        plt.ylabel("P(k)")
        plt.title("Connectivity distribution of bimodal graph")
        plt.grid(True)
        plt.show()

    return alpha, degrees

# Configuration graph on Bimodal ========================================================================================
# =======================================================================================================================

def hard_bimodal_graph(delta_bimodal, C, N, plot_degree_distribution=False):

    k2 = int(C + delta_bimodal/2)
    k1 = int(C - delta_bimodal/2)

    degrees = [k1 if np.random.rand() < 0.5 else k2 for _ in range(N)]
    G = nx.configuration_model(degrees)
    G.remove_edges_from(nx.selfloop_edges(G))
    alpha = nx.to_scipy_sparse_array(G, format='csr')

    if plot_degree_distribution:
        deg = np.array(alpha.sum(axis=1)).flatten()
        plt.hist(deg, bins=range(int(deg.min()), int(deg.max()) + 2), density=True, align='left')
        plt.xlabel("Grado k")
        plt.ylabel("P(k)")
        plt.title("Connectivity distribution of bimodal graph")
        plt.grid(True)
        plt.show()

    return alpha, degrees

# Chung-Lu Exponential ==================================================================================================
# =======================================================================================================================

def exponential_graph(C, N, plot_degree_distribution=False):

    degrees = np.random.exponential(scale=C, size=N).astype(int)
    G = nx.expected_degree_graph(degrees, selfloops=False)
    alpha = nx.to_scipy_sparse_array(G, format='csr')

    if plot_degree_distribution:
        deg = np.array(alpha.sum(axis=1)).flatten()
        plt.hist(deg, bins=range(int(deg.min()), int(deg.max()) + 2), density=True, align='left')
        plt.xlabel("Grado k")
        plt.ylabel("P(k)")
        plt.title("Connectivity distribution of Exponential graph")
        plt.grid(True)
        plt.show()

    return alpha, degrees


# Chung-Lu Uniformal ====================================================================================================
# =======================================================================================================================

def uniform_graph(C, N, delta_uniform, plot_degree_distribution=False):

    degrees = np.random.uniform(C- delta_uniform/2, C+ delta_uniform/2, size=N).astype(int)
    G = nx.expected_degree_graph(degrees, selfloops=False)
    alpha = nx.to_scipy_sparse_array(G, format='csr')

    if plot_degree_distribution:
        deg = np.array(alpha.sum(axis=1)).flatten()
        plt.hist(deg, bins=range(int(deg.min()), int(deg.max()) + 2), density=True, align='left')
        plt.xlabel("Grado k")
        plt.ylabel("P(k)")
        plt.title("Connectivity distribution of Uniform graph")
        plt.grid(True)
        plt.show()

    return alpha, degrees

# Chung-Lu Polynomial ===================================================================================================
# =======================================================================================================================

def sample_powerlaw_discrete(N, alpha, size=1):
    x = np.arange(1, N+1)
    p = x**(-alpha)
    p /= p.sum()
    return np.random.choice(x, size=size, p=p)

def polynomial_graph(C, N, alpha, plot_degree_distribution=False):

    degrees = sample_powerlaw_discrete(N, alpha, size=N)
    print("Average degree: ", np.mean(degrees))
    G = nx.expected_degree_graph(degrees, selfloops=False)
    alpha = nx.to_scipy_sparse_array(G, format='csr')

    if plot_degree_distribution:
        deg = np.array(alpha.sum(axis=1)).flatten()
        plt.hist(deg, bins=range(int(deg.min()), int(deg.max()) + 2), density=True, align='left')
        plt.xlabel("Grado k")
        plt.ylabel("P(k)")
        plt.title("Connectivity distribution of bimodal graph")
        plt.grid(True)
        plt.show()

    return alpha, degrees