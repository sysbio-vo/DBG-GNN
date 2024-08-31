import os
import datetime
import json
import pickle
import gzip
import functools
import logging
import tqdm

import networkx as nx
import matplotlib.pyplot as plt

from torch.utils.data import Dataset


GRAPH_EXT = '.labeled_dbg'
LOGGER_CONFIGURATION = {
    'format': '%(asctime)s %(levelname)-8s %(funcName)s %(message)s',
    'datefmt': '%d %h %Y %H:%M:%S'
}

def get_verbosity_level(int_level: int) -> int:
    # reverse order based on the number of `v`s provided in command line
    if int_level == 0:
        return logging.WARNING
    elif int_level == 1:
        return logging.INFO
    elif int_level == 2:
        return logging.DEBUG
    return logging.DEBUG

def get_dbgs_from_dir(indir: str, infile_ext: str = GRAPH_EXT) -> list[object]:
    dbgs = []
    for filename in os.listdir(indir):
        if os.path.splitext(filename)[1] == infile_ext:
            file_abs = os.path.abspath(os.path.join(indir, filename))
            with open(file_abs, 'rb') as f:
                dbg = pickle.load(f)
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


def parse_train_labels(data_path: str, outdir: str = None, save_to_json=True,
                       data_exts: tuple[str] = ('fastq.gz', 'fastq', 'fasta', 'fq', 'fa')) -> tuple[dict, dict]:
    unique_codes = set()

    for sample in os.listdir(data_path):
        sample_filename = os.path.basename(sample)
        # sample_ext = os.path.splitext(sample_filename)[1]
        sample_ext = sample_filename.split(os.extsep, 1)[-1]
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

    if save_to_json:
        with open(id_to_code_outfile, 'w') as f:
            json.dump(id_to_code, f)

        with open(code_to_id_outfile, 'w') as f:
            json.dump(code_to_id, f)

    return id_to_code, code_to_id
    

def timestamps(logger: logging.Logger):
    """
    Decorator factory to log function start and finish timestamps.

    Parameters
    ----------
    logger : logging.Logger
        Logger to use.

    Returns
    -------
    timestamps_decorator : func
        Decorator that uses the provided logger to output timestamps. 
    """
    def timestamps_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f'{datetime.datetime.now().strftime("%d %h %Y %H:%M:%S")} started {func.__name__}')
            res = func(*args, **kwargs)
            logger.info(f'{datetime.datetime.now().strftime("%d %h %Y %H:%M:%S")} finished {func.__name__}')
            return res
        return wrapper
    return timestamps_decorator


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
    for i in tqdm.tqdm(range(0, len(sequences)), desc='Reads in sample'):
        seq = sequences[i]
        k_mers = break_into_kmers(seq, k)

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
