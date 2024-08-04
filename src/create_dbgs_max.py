import random
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils.convert import from_networkx
import random
import networkx as nx
from collections import Counter, defaultdict
from torch_geometric.loader import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import utils.utils as ut
import os
import pickle
import datetime
from multiprocessing import Process, Pool

# Function to generate k-mers from a sequence
def generate_kmers(sequence, k):
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

def kmer_to_index(kmer):
    """Converts a kmer (string) to an index."""
    base_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    index = 0
    for char in kmer:
        index = 5 * index + base_to_index[char]
    return index

def subkmer_frequencies_in_kmer(kmer, subkmer_length):
    """Calculates the frequency of each subkmer in a kmer."""
    subkmer_counts = Counter(kmer[i:i+subkmer_length] for i in range(len(kmer) - subkmer_length + 1))
    frequencies = np.zeros(5**subkmer_length)
    for subkmer, count in subkmer_counts.items():
        index = kmer_to_index(subkmer)
        frequencies[index] = count
    return frequencies

id_to_code, code_to_id = ut.parse_train_labels(data_path='/home/sysbio/camda2020/camda2020_tiny/filtered_subsampled', save_to_json=False)

num_classes = len(code_to_id)

print(f'{datetime.datetime.now().strftime("%d %h %Y %H:%M:%S")} {num_classes = }')

def get_labeled_reads_from_dir_with_samples(indir: str, filesize_lim_mb = 500) -> dict:
    reads_for_samples = {} # dict
    id_to_code, code_to_id = ut.parse_train_labels(data_path=indir, save_to_json=False)
    files_in_dir = os.listdir(indir)
    for file in files_in_dir:
        print(f'{datetime.datetime.now().strftime("%d %h %Y %H:%M:%S")} processing file {file}')
        if os.path.splitext(file)[1] != '.fastq' or os.path.getsize(os.path.join(indir, file)) / (1024 * 1024.0) > filesize_lim_mb:
            print(f'{datetime.datetime.now().strftime("%d %h %Y %H:%M:%S")} skipping {file}')
            continue
        city_code = os.path.basename(file).split('_')[3]
        sample_name = os.path.splitext(os.path.basename(file))[0]
        int_label = int(code_to_id[city_code])
        print(f'{datetime.datetime.now().strftime("%d %h %Y %H:%M:%S")} {city_code = } ; {int_label = }')
        print(f'{datetime.datetime.now().strftime("%d %h %Y %H:%M:%S")} getting reads')
        reads = ut.get_reads_from_fq(os.path.join(indir, file))
        print(f'{datetime.datetime.now().strftime("%d %h %Y %H:%M:%S")} saving labelled reads')
        if sample_name not in reads_for_samples:
            reads_for_samples[sample_name] = [int_label, reads]
        else:
            reads_for_samples[sample_name][1].extend(reads)
    return reads_for_samples
        

genome_sequences = get_labeled_reads_from_dir_with_samples('/home/sysbio/camda2020/camda2020_tiny/filtered_subsampled',
                                                          700)

outdir = '/home/sysbio/DBG-GNN/data/camda2020_tiny_max'
if not os.path.exists(outdir):
    os.makedirs(outdir)

graphs = []
kmer_len = 4
subkmer_len = 2
num_features = 5**subkmer_len

def build_graph_max(dict_item):
    sample_name, code_and_reads = dict_item
    print(f'{datetime.datetime.now().strftime("%d %h %Y %H:%M:%S")} processing {sample_name}')
    city_code, seqs = code_and_reads
    print(f'{datetime.datetime.now().strftime("%d %h %Y %H:%M:%S")} {city_code = }')
    G = nx.DiGraph()
    kmers = set()
    print(f'{datetime.datetime.now().strftime("%d %h %Y %H:%M:%S")} getting k-mers from {len(seqs)} reads')
    transition_counts = defaultdict(int)
    for idx, seq in enumerate(seqs):
        if idx % 1_000_000 == 0:
            print(f'processed {idx} reads')
        kmers_in_read = generate_kmers(seq, kmer_len)
        kmers = kmers.union(set(kmers_in_read))
        for kk in range(len(kmers_in_read) - 1):
            transition_counts[(kmers_in_read[kk], kmers_in_read[kk + 1])] += 1
    nodes = []
    print(f'{datetime.datetime.now().strftime("%d %h %Y %H:%M:%S")} adding nodes to graph')
    for kmer in kmers:
        nodes.append((kmer, {"x": torch.as_tensor(subkmer_frequencies_in_kmer(kmer, subkmer_len)/(kmer_len-1), dtype=torch.float32)}))
    G.add_nodes_from(nodes)
    # edges = []
    # transition_counts = defaultdict(int)
    # print(f'{datetime.datetime.now().strftime("%d %h %Y %H:%M:%S")} calculating transition counts')
    # for i in range(len(kmers)-1):
    #     transition_counts[(kmers[i], kmers[i+1])] += 1
    max_count = max(transition_counts.values())
    print(f'{datetime.datetime.now().strftime("%d %h %Y %H:%M:%S")} adding edges')
    for key in transition_counts.keys():
        G.add_edge(key[0], key[1], weight=transition_counts[key]/max_count)
    #     edges.append((kmers[i], kmers[i+1]))
    # G.add_edges_from(edges)
    print(f'{datetime.datetime.now().strftime("%d %h %Y %H:%M:%S")} saving as torch graph')
    torch_graph = from_networkx(G)
    torch_graph['y'] = torch.tensor([city_code])

    print(f'{datetime.datetime.now().strftime("%d %h %Y %H:%M:%S")} saving graph for sample {sample_name}')
    outfile_graph_name = os.path.join(outdir, sample_name + '.labeled_graph_max')
    with open(outfile_graph_name, 'wb') as f:
        pickle.dump(torch_graph, f)


if __name__ == '__main__':
    with Pool(processes = 5) as p:
        build_graph_max_result = p.map(build_graph_max, genome_sequences.items())

