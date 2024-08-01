import networkx as nx
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import sqlite3
import tqdm
import gzip
import os
import pickle
import json

GRAPH_EXT = '.dbg'

def get_dbgs_from_dir(indir: str, infile_ext: str = GRAPH_EXT) -> list[object]:
    dbgs = []
    for filename in os.listdir(indir):
        if os.path.splitext(filename)[1] == infile_ext:
            dbg = pickle.load(filename)
            dbgs.append(dbg)
    return dbgs

def get_reads_from_fq(fq_path: str) -> list[str]:
    reads = []
    with open(fq_path, 'r') as f:
        fastq_reads = f.readlines()
        for i in range(0, len(fastq_reads), 4):
            reads.append(str(fastq_reads[i + 1].rstrip()))
    return reads

def get_reads_from_gzed_fq(gzed_fq_path: str) -> list[str]:
    reads = []
    with gzip.open(gzed_fq_path, 'r') as f:
        fastq_reads = f.readlines()
        for i in range(0, len(fastq_reads), 4):
            reads.append(str(fastq_reads[i + 1].rstrip()))
    return reads


def parse_train_labels(data_path: str, outdir: str = None, 
                       data_exts: tuple[str] = ('.gz', '.fastq', '.fasta', '.fq', '.fa')) -> tuple[dict, dict]:
    unique_codes = set()

    for sample in os.listdir(data_path):
        sample_filename = os.path.basename(sample)
        sample_ext = os.path.splitext(sample_filename)[1]
        if sample_ext not in data_exts:
            continue

        # Example: CAMDA20_MetaSUB_CSD16_BCN_012_1_kneaddata_subsampled_20_percent.fastq
        city_id = list(sample_filename.split('_'))[3]

        unique_codes.add(city_id)

    unique_codes_list = list(unique_codes)


    id_to_code = {}
    code_to_id = {}
    for idx, code in enumerate(unique_codes_list):
        id_to_code[idx] = code
        code_to_id[code] = idx

    if outdir:
        id_to_code_outfile = os.path.join(outdir, 'id_to_code.json')
        code_to_id_outfile = os.path.join(outdir, 'code_to_id.json')
    else:
        id_to_code_outfile = 'id_to_code.json'
        code_to_id_outfile = 'code_to_id.json'


    with open(id_to_code_outfile, 'w') as f:
        json.dump(id_to_code, f)

    with open(code_to_id_outfile, 'w') as f:
        json.dump(code_to_id, f)

    return id_to_code, code_to_id
    
            


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
    return [sequence[i:i + k] for i in range(len(sequence) - k + 1)]

def build_DBG(sequences: list[str], k: int = 4):
    dbg = nx.DiGraph()
    print(f'{len(sequences) = }')
    for i in tqdm.tqdm(range(0, len(sequences)), desc='Reads in sample'):
        seq = sequences[i]
        k_mers = break_into_kmers(seq, k)
        # print(f'{len(k_mers) = }')

        for i in range(len(k_mers) - 1):
            parent = k_mers[i]
            child = k_mers[i + 1]
            dbg.add_edge(parent, child)

    return dbg

def visualize_de_bruijn_graph(graph):
    pos = nx.random_layout(graph)
    plt.figure(figsize=(12, 8))

    nx.draw_networkx_nodes(graph, pos, node_size=700, node_color='skyblue')
    nx.draw_networkx_edges(graph, pos, edgelist=graph.edges(), edge_color='black', arrowsize=15, node_size=700)
    nx.draw_networkx_labels(graph, pos, font_size=10, font_color='black')

    plt.title('De Bruijn Graph')
    plt.show()
 



