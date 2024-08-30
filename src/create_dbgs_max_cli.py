import os
import datetime
import pickle
import functools
import argparse
import pathlib
import logging
from collections import Counter, defaultdict
from collections.abc import Iterable

import numpy as np
import networkx as nx

import utils.utils as ut

import torch
from torch_geometric.utils.convert import from_networkx

from multiprocessing import Pool


DNA_ALPHABET = ('A', 'T', 'G', 'C')
DNA5_ALPHABET = ('A', 'T', 'G', 'C', 'N')

def get_args():
    '''
    Get args from the command line
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--indir', type=pathlib.Path,
        help='directory with samples')
    parser.add_argument('-k', '--kmer_len', type=int, default=4,
        help='k-mer length to build the de Bruijn Graph')
    parser.add_argument('-s', '--subkmer_len', type=int, default=2,
        help='sub k-mer length to initialize node features')
    parser.add_argument('-N', '--skip_N',
        help='skip k-mers with padding nucleotide N', action='store_true')
    parser.add_argument('-o', '--outdir', type=pathlib.Path,
        help='output directory to store DBGs')
    parser.add_argument('-n', '--normalization_method', choices=['avg', 'max'],
        help='edge weight normalization method', default='max')
    parser.add_argument('-v', '--verbose', 
        help='verbosity level', action='count', default=0)

    args = parser.parse_args()
    return args


# Function to generate k-mers from a sequence
# TODO: optimize
def generate_kmers(sequence: str, k: int, skip_N: bool = True) -> list:
    kmers =  [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    if skip_N:
        filtered_kmers = []
        for kmer in kmers:
            if 'N' not in kmer:
                filtered_kmers.append(kmer)
        return filtered_kmers
    return kmers


def kmer_to_index(kmer: int, skip_N: bool = True) -> int:
    """Converts a kmer (string) to an index.
    """
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

def subkmer_frequencies_in_kmer(kmer: int, subkmer_length: int, skip_N: bool = True) -> np.array:
    """Calculate the frequency of each sub-k-mer in a k-mer.
    """
    subkmer_counts = Counter(kmer[i:i + subkmer_length] for i in range(len(kmer) - subkmer_length + 1))
    if skip_N:
        frequencies = np.zeros(len(DNA_ALPHABET)**subkmer_length)
    else:
        frequencies = np.zeros(len(DNA5_ALPHABET)**subkmer_length)

    for subkmer, count in subkmer_counts.items():
        index = kmer_to_index(subkmer, skip_N=skip_N)
        frequencies[index] = count
    return frequencies

def get_labeled_reads_from_dir_with_samples(indir: str, filesize_lim_mb: int = None) -> dict:
    """
    """
    reads_for_samples = {} # dict
    id_to_code, code_to_id = ut.parse_train_labels(data_path=indir, save_to_json=False) # id_to_code not used here
    num_classes = len(code_to_id)
    print(f'{datetime.datetime.now().strftime("%d %h %Y %H:%M:%S")} {num_classes = }')
    files_in_dir = os.listdir(indir)

    # TODO: use tqdm
    # TODO: improve logging
    for file in files_in_dir:
        print(f'{datetime.datetime.now().strftime("%d %h %Y %H:%M:%S")} processing file {file}')
        skip_based_on_filesize = False
        if filesize_lim_mb:
            skip_based_on_filesize = os.path.getsize(os.path.join(indir, file)) / (1024.0 * 1024.0) > filesize_lim_mb

        if os.path.splitext(file)[1] != '.fastq' or skip_based_on_filesize:
            print(f'{datetime.datetime.now().strftime("%d %h %Y %H:%M:%S")} skipping {file}')
            continue

        city_code = os.path.basename(file).split('_')[3] # TODO: specific to CAMDA dataset
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


# TODO: move to util
def get_normalization_val(data: Iterable[int], method: str = 'avg') -> float:
    data_np = np.array(list(data))
    if method == 'avg':
        return np.sum(data_np)
    elif method == 'max':
        return np.max(data_np)

def build_graph_max(dict_item, 
                    skip_N: bool = True, outdir: str = None, 
                    kmer_len: int = 4,
                    subkmer_len: int = 2,
                    normalization_method: str = 'max'
                    ) -> None:
    """Build a DBG from an entry of form `(sample_name, [int_city_code, [read_1, read_2, ...]])`.

    Parameters
    ----------
    dict_item : tuple[int, list[str]]
        Tuple with read sequences in a single sample. The first element is the name of the sample.
        Second element is a list consisting of 2 elements: integer city code of the sample and list of read sequences.
    
    skip_N : bool, default: True
        Flag to either keep or leave out k-mers containing padding nucleotide **N**.

    kmer_len : int, default: 4
        Length of the k-mers to build DBG on.

    subkmer_len : int, default: 2
        Length of the sub-k-mers whose frequences are used to initialize node (k-mer) features in a DBG.

    outdir : str, default: None
        Directory to save constructed DBGs to. Saves to the current folder if not specified.

    Returns
    -------
    None
    """
    sample_name, code_and_reads = dict_item
    print(f'{datetime.datetime.now().strftime("%d %h %Y %H:%M:%S")} processing {sample_name}')
    city_code, seqs = code_and_reads
    print(f'{datetime.datetime.now().strftime("%d %h %Y %H:%M:%S")} {city_code = }')

    G = nx.DiGraph()
    kmers = set()
    print(f'{datetime.datetime.now().strftime("%d %h %Y %H:%M:%S")} getting k-mers from {len(seqs)} reads')
    transition_counts = defaultdict(int)

    # TODO: use tqdm
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
                nodes.append(
                    (
                    kmer, 
                    {"x": torch.as_tensor(subkmer_frequencies_in_kmer(kmer, subkmer_len, skip_N) / (kmer_len - 1), dtype=torch.float32)}
                    )
                )
    else:
        print(f'{datetime.datetime.now().strftime("%d %h %Y %H:%M:%S")} not skipping padding nucleotide')
        for kmer in kmers:
            nodes.append(
                (
                kmer, 
                {"x": torch.as_tensor(subkmer_frequencies_in_kmer(kmer, subkmer_len, skip_N) / (kmer_len - 1), dtype=torch.float32)}
                )
            )
    G.add_nodes_from(nodes)

    print(f'{datetime.datetime.now().strftime("%d %h %Y %H:%M:%S")} normalizing by transition count by {normalization_method}')
    normalization_val = get_normalization_val(transition_counts.values(), method=normalization_method)

    print(f'{datetime.datetime.now().strftime("%d %h %Y %H:%M:%S")} adding edges')
    for key in transition_counts.keys():
        G.add_edge(key[0], key[1], weight=transition_counts[key] / normalization_val)

    print(f'{datetime.datetime.now().strftime("%d %h %Y %H:%M:%S")} saving as torch graph')
    torch_graph = from_networkx(G)
    torch_graph['y'] = torch.tensor([city_code])

    print(f'{datetime.datetime.now().strftime("%d %h %Y %H:%M:%S")} saving graph for sample {sample_name}')

    outfile_graph_name = (
        os.path.join(outdir, sample_name + '.labeled_graph_max') 
        if outdir 
        else sample_name + '.labeled_graph_max'
    )

    with open(outfile_graph_name, 'wb') as f:
        pickle.dump(torch_graph, f)

def main():

    args = get_args()

    # TODO: replace vanilla prints with proper logging
    logging.basicConfig()
    logger = logging.getLogger("create_dbgs")
    logger.setLevel(args.verbose * 10)

    print(f'Creating {args.outdir = } DEBUG')
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    genome_sequences = get_labeled_reads_from_dir_with_samples(args.indir)

    with Pool(processes = 6) as p:
        build_graph_max_wrapper = functools.partial(build_graph_max, skip_N=args.skip_N, outdir=args.outdir,
                                                    kmer_len=args.kmer_len, subkmer_len=args.subkmer_len, 
                                                    normalization_method=args.normalization_method)
        build_graph_max_result = p.map(build_graph_max_wrapper, genome_sequences.items()) # not used

if __name__ == '__main__':
    main()

