import networkx as nx
import matplotlib.pyplot as plt
from torch.utils.data import Dataset



class GraphsDataset(Dataset):
    def __init__(self, graphs, labels):
        assert len(graphs) == len(labels)
        self.graphs = graphs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]



def break_into_kmers(sequence: str, k: int = 4):
    return [sequence[i:i + k] for i in range(0, len(sequence) - k + 1)]

def build_DBG(*sequences: str, k: int = 4):
    dbg = nx.DiGraph()
    for seq in sequences:
        k_mers = break_into_kmers(seq, k)
        k_mers_set = list(set(k_mers))

        for i in range(len(k_mers_set)):
            for j in range(i, len(k_mers_set)):
                if k_mers_set[i][:-1] == k_mers_set[j][1:]:
                    # print(f'adding edge between {k_mers_set[j]} and {k_mers_set[i]}')
                    dbg.add_edge(k_mers_set[j], k_mers_set[i])
                elif k_mers_set[j][:-1] == k_mers_set[i][1:]:
                    # print(f'adding edge between {k_mers_set[i]} and {k_mers_set[j]}')
                    dbg.add_edge(k_mers_set[i], k_mers_set[j])

        

    return dbg

def visualize_de_bruijn_graph(graph):
    pos = nx.random_layout(graph)
    plt.figure(figsize=(12, 8))

    nx.draw_networkx_nodes(graph, pos, node_size=700, node_color='skyblue')
    nx.draw_networkx_edges(graph, pos, edgelist=graph.edges(), edge_color='black', arrowsize=15, node_size=700)
    nx.draw_networkx_labels(graph, pos, font_size=10, font_color='black')

    plt.title('De Bruijn Graph')
    plt.show()
 



