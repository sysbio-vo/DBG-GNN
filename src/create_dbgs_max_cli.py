import random
import functools
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

import argparse
import pathlib
import logging


DNA_ALPHABET = ('A', 'T', 'G', 'C')
DNA5_ALPHABET = ('A', 'T', 'G', 'C', 'N')

def get_args():
    '''
    Get args from the command line
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--kmer_len', type=int, default=4,
        help='k-mer length to build the de Bruijn Graph')
    parser.add_argument('-s', '--subkmer_len', type=int, default=2,
        help='sub k-mer length to initialize node features')
    parser.add_argument('-o', '--outdir', type=pathlib.Path,
        help='output directory to store DBGs', action='store_true')
    parser.add_argument('-i', '--indir', type=pathlib.Path,
        help='directory with samples', action='store_true')
    parser.add_argument('-N', '--keep_N',
        help='keep k-mers with N', action='store_true')
    parser.add_argument('-v', '--verbose', 
        help='verbosity level', action='count', default=0)
    parser.add_argument('-n', '--normalization_method', choices=['avg', 'max'],
            help='edge weight normalization method')

    args = parser.parse_args()
    return args


# Function to generate k-mers from a sequence
# TODO: optimize
def generate_kmers(sequence, k, skip_N=True):
    kmers =  [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    if skip_N:
        filtered_kmers = []
        for kmer in kmers:
            if 'N' not in kmer:
                filtered_kmers.append(kmer)
        return filtered_kmers
    return kmers


def kmer_to_index(kmer, skip_N=True):
    """Converts a kmer (string) to an index."""
    if skip_N:
        alphabet = DNA_ALPHABET
    else:
        alphabet = DNA5_ALPHABET

    base_to_index = {k: v for v, k in enumerate(alphabet)}

    index = 0
    num_bases = len(alphabet)
    for char in kmer:
        index = num_bases * index + base_to_index[char]
    return index

def subkmer_frequencies_in_kmer(kmer, subkmer_length, skip_N=True):
    """Calculates the frequency of each subkmer in a kmer."""
    subkmer_counts = Counter(kmer[i:i + subkmer_length] for i in range(len(kmer) - subkmer_length + 1))
    if skip_N:
        frequencies = np.zeros(len(DNA_ALPHABET)**subkmer_length)
    else:
        frequencies = np.zeros(len(DNA5_ALPHABET)**subkmer_length)

    for subkmer, count in subkmer_counts.items():
        index = kmer_to_index(subkmer)
        frequencies[index] = count
    return frequencies

# num_classes = len(code_to_id)

# print(f'{datetime.datetime.now().strftime("%d %h %Y %H:%M:%S")} {num_classes = }')

def get_labeled_reads_from_dir_with_samples(indir: str, filesize_lim_mb = 500) -> dict:
    reads_for_samples = {} # dict
    id_to_code, code_to_id = ut.parse_train_labels(data_path=indir, save_to_json=False)
    files_in_dir = os.listdir(indir)
    for file in files_in_dir:
        print(f'{datetime.datetime.now().strftime("%d %h %Y %H:%M:%S")} processing file {file}')
        # if os.path.splitext(file)[1] != '.fastq' or os.path.getsize(os.path.join(indir, file)) / (1024 * 1024.0) > filesize_lim_mb:
        if os.path.splitext(file)[1] != '.fastq':
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

graphs = []
kmer_len = 6
subkmer_len = 2
num_features = 4**subkmer_len

def build_graph_max(dict_item, skip_N=True):
    sample_name, code_and_reads = dict_item
    print(f'{datetime.datetime.now().strftime("%d %h %Y %H:%M:%S")} processing {sample_name}')
    city_code, seqs = code_and_reads
    print(f'{datetime.datetime.now().strftime("%d %h %Y %H:%M:%S")} {city_code = }')
    G = nx.DiGraph()
    kmers = set()
    print(f'{datetime.datetime.now().strftime("%d %h %Y %H:%M:%S")} getting k-mers from {len(seqs)} reads')
    transition_counts = defaultdict(int)
    for idx, seq in enumerate(seqs):
        if idx % 1_000_00 == 0:
            print(f'processed {idx} reads')
        kmers_in_read = generate_kmers(seq, kmer_len, skip_N)
        kmers = kmers.union(set(kmers_in_read))
        for kk in range(len(kmers_in_read) - 1):
            transition_counts[(kmers_in_read[kk], kmers_in_read[kk + 1])] += 1
    nodes = []
    print(f'{datetime.datetime.now().strftime("%d %h %Y %H:%M:%S")} adding nodes to graph')
    if skip_N:
        print(f'{datetime.datetime.now().strftime("%d %h %Y %H:%M:%S")} skipping padding nucleotide')
        for kmer in kmers:
            if 'N' not in kmer:
                nodes.append((kmer, {"x": torch.as_tensor(subkmer_frequencies_in_kmer(kmer, subkmer_len, skip_N)/(kmer_len-1), dtype=torch.float32)}))
    else:
        print(f'{datetime.datetime.now().strftime("%d %h %Y %H:%M:%S")} not skipping padding nucleotide')
        for kmer in kmers:
            nodes.append((kmer, {"x": torch.as_tensor(subkmer_frequencies_in_kmer(kmer, subkmer_len, skip_N)/(kmer_len-1), dtype=torch.float32)}))
    G.add_nodes_from(nodes)
    # edges = []
    # transition_counts = defaultdict(int)
    # print(f'{datetime.datetime.now().strftime("%d %h %Y %H:%M:%S")} calculating transition counts')
    # for i in range(len(kmers)-1):
    #     transition_counts[(kmers[i], kmers[i+1])] += 1


    # normalization_val = max(transition_counts.values())
    
    print(f'{datetime.datetime.now().strftime("%d %h %Y %H:%M:%S")} normalizing by sum transition count')
    normalization_val = np.sum(np.array(list(transition_counts.values())))
    print(f'{datetime.datetime.now().strftime("%d %h %Y %H:%M:%S")} adding edges')
    for key in transition_counts.keys():
        G.add_edge(key[0], key[1], weight=transition_counts[key]/normalization_val)
    #     edges.append((kmers[i], kmers[i+1]))
    # G.add_edges_from(edges)
    print(f'{datetime.datetime.now().strftime("%d %h %Y %H:%M:%S")} saving as torch graph')
    torch_graph = from_networkx(G)
    torch_graph['y'] = torch.tensor([city_code])

    print(f'{datetime.datetime.now().strftime("%d %h %Y %H:%M:%S")} saving graph for sample {sample_name}')
    outfile_graph_name = os.path.join(outdir, sample_name + '.labeled_graph_max')
    with open(outfile_graph_name, 'wb') as f:
        pickle.dump(torch_graph, f)



def main():

    args = get_args()
    logging.basicConfig()
    logger = logging.getLogger("create_dbgs")
    logger.setLevel(args.verbose * 10)

    # outdir = '/home/nepotlet/camda2020/camda2020/subsampled_no_kneaddata/dbgs_no_kneaddata_skip_N_normalized_sum_kmerlen_6_subkmerlen_2'


    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)


    # genome_sequences = get_labeled_reads_from_dir_with_samples(
    #     '/home/nepotlet/camda2020/camda2020/subsampled_no_kneaddata', 
    #     700)


    genome_sequences = get_labeled_reads_from_dir_with_samples(
        args.indir, 
        700)



    with Pool(processes = 6) as p:
        build_graph_max_wrapper = functools.partial(build_graph_max, skip_N=True)
        build_graph_max_result = p.map(build_graph_max_wrapper, genome_sequences.items())


if __name__ == '__main__':
    main()

