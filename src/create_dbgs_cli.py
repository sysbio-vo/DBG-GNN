import os
import json
import torch
import pickle
import logging
import pathlib
import argparse
import functools

import numpy as np
import networkx as nx
import utils.utils as ut

# from tqdm import tqdm # TODO: use correctly with multiprocessing
from multiprocessing import Pool
from collections.abc import Iterable
from collections import Counter, defaultdict
from torch_geometric.utils.convert import from_networkx


logging.basicConfig(**ut.LOGGER_CONFIGURATION)
logger = logging.getLogger(__name__)

# TODO: move constants to utils
DNA_ALPHABET = ('A', 'T', 'G', 'C')
DNA5_ALPHABET = ('A', 'T', 'G', 'C', 'N')

def get_args():
    """Get DBG construction args from the command line using argparse.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--indir', type=pathlib.Path,
        help='directory with samples')
    parser.add_argument('-k', '--kmer_len', type=int, default=4,
        help='k-mer length to build the de Bruijn Graph')
    parser.add_argument('-s', '--subkmer_len', type=int, default=2,
        help='sub k-mer length to initialize node features')
    parser.add_argument('-N', '--skip_N',
        help='skip k-mers with padding nucleotide N', action='store_true')
    parser.add_argument('-n', '--normalization_method', choices=['sum', 'max'],
        help='edge weight normalization method', default='max')
    parser.add_argument('-f', '--node_feature_method', choices=['subkmer_freq', 'subkmer_freq_positional'],
        help='node feature initialization method', default='subkmer_freq')
    parser.add_argument('-m', '--normalize_node_features',
        help='normalize node features', action='store_true')
    parser.add_argument('-t', '--threads', type=int, default=4,
        help='number of threads to build graphs in parallel')
    parser.add_argument('-o', '--outdir', type=pathlib.Path,
        help='output directory to store DBGs')
    parser.add_argument('-v', '--verbose', 
        help='verbosity level', action='count', default=0) # TODO: add different levels of logging

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


def kmer_to_index(kmer: str, skip_N: bool = True) -> int:
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

# TODO: move to a separate module
def node_feature_method_selector(method_name: str, *args, **kwargs):

    # define different feature calculation methods here

    # "subkmer_freq"
    def subkmer_frequencies_in_kmer(kmer: str, subkmer_len: int, skip_N: bool = True, 
    normalize: bool = True) -> np.array:
        """Calculate the frequency of each sub-k-mer in a k-mer.
        """
        subkmer_counts = Counter(kmer[i:i + subkmer_len] for i in range(len(kmer) - subkmer_len + 1))
        if skip_N:
            frequencies = np.zeros(len(DNA_ALPHABET)**subkmer_len)
        else:
            frequencies = np.zeros(len(DNA5_ALPHABET)**subkmer_len)

        for subkmer, count in subkmer_counts.items():
            index = kmer_to_index(subkmer, skip_N=skip_N)
            frequencies[index] = count

        if normalize:
            frequencies = frequencies / (len(kmer) - 1)

        return frequencies
    
    # "subkmer_freq_positional"
    def subkmer_frequencies_in_kmer_positioned(kmer: str, subkmer_len: int, skip_N: bool = True, 
                                            normalize: bool = False) -> np.array:
        """Test naive prototype implementation of sub-k-mer frequencies enhanced with their positional information
        as initial node embeddings.

        The idea is the following. Consier an example of `k = 5`, `sub_k = 2`.
        Take this k-mer: **ATGGG**
        Let's assume for simplicity the following index mapping of the sub-kmers: 
        {   'AA': 0,
            'AC': 1,
            'AG': 2,
            'AT': 3,
            'CA': 4,
            'CC': 5,
            'CG': 6,
            'CT': 7,
            'GA': 8,
            'GC': 9,
            'GT': 10,
            'GG': 11,
            'TA': 12,
            'TC': 13,
            'TT': 14,
            'TG': 15
        }

        ATGGG -> AT(3), TG(15), GG(11), GG(11)

        Usual subkmer-frequency-based embedding for the k-mer in this case is: 
        [0, 0, 1, 0, ..., 2, ..., 1, 0 ]
            ^          ^       ^
            |          |       |
            idx=3        11      15

        
        I propose doing the following:
        positional_information = np.arange(1, k) = [1, 2, 3, 4]

        Then we multiple each bitvector representing a sub-kmer by the corresponding positional number
        to obtain the following embedding:
        positionally_resolved_AT = [0, 0, 1, 0, ..., 0] * 1 = [0, 0, 1, 0, ..., 0]
        positionally_resolved_TG = [0, 0, 0, 0, ..., 1, 0] * 2 = [0, 0, 0, 0, ..., 2, 0]
        positionally_resolved_GG = [0, ..., 1, ..., 0, 0] * 3 + [0, ..., 1, ..., 0, 0] * 4 = [0, ..., 7, ..., 0, 0]

        So that the final k-mer embedding looks as follows:
        [0, 0, 1, 0, ..., 7, ..., 2, 0]
            ^          ^       ^
            |          |       |
            idx=3        11      15

        TODO: try out normalizing the resulting array

        """
        # TODO: add sub-kmer position information to the embedding
        if skip_N:
            positionally_resolved_frequenies = np.zeros(len(DNA_ALPHABET)**subkmer_len)
        else:
            positionally_resolved_frequenies = np.zeros(len(DNA5_ALPHABET)**subkmer_len)

        positional_weights = np.arange(1, len(kmer))
        for i in range(len(kmer) - subkmer_len + 1):
            subkmer = kmer[i:i + subkmer_len]
            subkmer_idx = kmer_to_index(subkmer)
            positionally_resolved_frequenies[subkmer_idx] += positional_weights[i]

        if normalize:
            positionally_resolved_frequenies = positionally_resolved_frequenies / np.linalg.norm(positionally_resolved_frequenies)
        return positionally_resolved_frequenies

    # return needed method
    picked_method = subkmer_frequencies_in_kmer # default

    match method_name:
        case 'subkmer_freq':
            picked_method = subkmer_frequencies_in_kmer
        case 'subkmer_freq_positional':
            picked_method = subkmer_frequencies_in_kmer_positioned

    return functools.partial(picked_method, *args, **kwargs)

        


def get_labeled_reads_from_dir_with_samples(indir: str, filesize_lim_mb: int = None, 
                                            supported_sample_exts: tuple=('.fastq',)) -> dict:
    """
    """
    reads_for_samples = {} # dict
    id_to_code, code_to_id = ut.parse_train_labels(data_path=indir, save_to_json=False) # id_to_code not used here
    num_classes = len(code_to_id)
    logger.info(f'{num_classes = }')
    
    files_in_dir = os.listdir(indir)

    # TODO: use tqdm
    # TODO: improve logging
    # TODO: run in parallel
    for file in files_in_dir:
        logger.info(f'processing file {file}')
        skip_based_on_filesize = False
        if filesize_lim_mb:
            skip_based_on_filesize = os.path.getsize(os.path.join(indir, file)) / (1024.0 * 1024.0) > filesize_lim_mb
        
        if os.path.splitext(file)[1] not in supported_sample_exts or skip_based_on_filesize:
            logger.info(f'skipping {file} because not a sample')
            continue

        city_code = os.path.basename(file).split('_')[3] # TODO: specific to CAMDA dataset
        sample_name = os.path.splitext(os.path.basename(file))[0]
        logger.info(f'{sample_name = } DEBUG')

        int_label = int(code_to_id[city_code])

        logger.info(f'{city_code = } ; {int_label = }')
        logger.info(f'getting reads')

        reads = ut.get_reads_from_fq_or_gzed_fq(os.path.join(indir, file))
        logger.info(f'saving labelled reads')

        reads_for_samples[sample_name] = [int_label, reads]

    return reads_for_samples

def samples_from_indir(indir: str, supported_sample_exts: tuple=('.fastq',)) -> tuple[list, dict]:
    """Get list of sample files and city label to its integer id map.
    """
    id_to_code, code_to_id = ut.parse_train_labels(data_path=indir, save_to_json=False) # id_to_code not used here
    num_classes = len(code_to_id)
    logger.info(f'{num_classes = }')
    
    sample_files = []
    files_in_dir = os.listdir(indir)
    for file in files_in_dir:
        logger.info(f'processing file {file}')

        if os.path.splitext(file)[1] not in supported_sample_exts:
            logger.info(f'skipping {file} because not a sample')
            continue

        sample_files.append(os.path.join(indir, file))
    
    return sample_files, code_to_id

def filter_out_built_graphs(sample_paths: list[str], outdir: str):

    # DEBUG function

    filtered_samples = []
    for sample in sample_paths:
        sample_basename = os.path.basename(sample)
        sample_no_ext = os.path.splitext(sample_basename)[0]
        built_graph_path = sample_no_ext + '.labeled_dbg'
        built_graph_full_path = os.path.join(outdir, built_graph_path)

        print(f'{built_graph_full_path = } DEBUG')
        if os.path.exists(built_graph_full_path):
            print(f'{built_graph_full_path} skipping')
            continue
        print(f'{built_graph_full_path} using')
        filtered_samples.append(sample)

    return filtered_samples

def get_labelled_reads_from_file(sample: str, code_to_id: dict):
    """Get reads from a fastq or gzed fastq sample and get sample location integer id.  
    """
    logger.info(f'processing file {sample}')
    city_code = os.path.basename(sample).split('_')[3] # TODO: specific to CAMDA dataset
    sample_name = os.path.splitext(os.path.basename(sample))[0]
    logger.info(f'{sample_name = } DEBUG')

    int_label = int(code_to_id[city_code])

    reads = ut.get_reads_from_fq_or_gzed_fq(sample)
    logger.info(f'returning labelled reads')

    return sample_name, int_label, reads


def get_labelled_reads_and_build_graph(sample: str, code_to_id: dict, 
                                       skip_N: bool = True, outdir: str = None, 
                                       kmer_len: int = 4,
                                       subkmer_len: int = 2,
                                       normalization_method: str = 'max',
                                       savefile_ext: str = None,
                                       log_every_n_reads: int = 100_000,
                                       edge_weight_dtype: torch.dtype = torch.float32,
                                       node_feature_method: str = 'subkmer_freq',
                                       normalize_node_features: bool = True,
                                    ):
    """Get reads from a sample and build a DBG labelled with a city ID.
    """
    sample_name, city_code, reads = get_labelled_reads_from_file(sample, code_to_id)

    build_graph_max(sample_name=sample_name, city_code=city_code, seqs=reads, 
                    skip_N=skip_N, outdir=outdir,
                    kmer_len=kmer_len, subkmer_len=subkmer_len, normalization_method=normalization_method, 
                    savefile_ext=savefile_ext, log_every_n_reads=log_every_n_reads, 
                    edge_weight_dtype=edge_weight_dtype,
                    node_feature_method=node_feature_method, 
                    normalize_node_features=normalize_node_features)

# TODO: move to util
def get_normalization_val(data: Iterable[int], method: str = 'sum') -> float:
    data_np = np.array(list(data))
    if method == 'sum':
        return np.sum(data_np)
    elif method == 'max':
        return np.max(data_np)
    else:
        raise ValueError(f'Normalization method {method} is not recognized.')

def build_graph_max(sample_name: str, city_code: int, seqs: list[str],
                    skip_N: bool = True, outdir: str = None, 
                    kmer_len: int = 4,
                    subkmer_len: int = 2,
                    normalization_method: str = 'max',
                    savefile_ext: str = None,
                    log_every_n_reads: int = 100_000,
                    edge_weight_dtype: torch.dtype = torch.float32,
                    node_feature_method: str = 'subkmer_freq',
                    normalize_node_features: bool = True,
                    ) -> None:
    """Build a DBG from an entry of form `(sample_name, [int_city_code, [read_1, read_2, ...]])`.

    Parameters
    ----------
    dict_item : tuple[str, list[str]]
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

    edge_weight_dtype : torch.dtype, default: torch.float32
        Data type for the edge weight. Allows you to control precision. **Currently only torch.float32 is supported.**
        TODO: make tunable. 

    node_feature_method : str, default: 'subkmer_freq'
        Node feature initialization method.
    
    normalize_node_features : bool, default : True


    Returns
    -------
    None
    """

    # select node feature calculation method
    logger.info(f'using node feature calculation method: {node_feature_method}')
    feature_method = node_feature_method_selector(node_feature_method, subkmer_len=subkmer_len, 
    skip_N=skip_N, normalize=normalize_node_features)

    logger.info(f'processing {sample_name}')
    logger.info(f'{city_code = }')

    G = nx.DiGraph()
    kmers = set()
    logger.info(f'{sample_name}: getting k-mers from {len(seqs)} reads')
    transition_counts = defaultdict(int)

    # TODO: use tqdm
    for idx, seq in enumerate(seqs):
        if idx % log_every_n_reads == 0:
            logger.info(f'{sample_name}: processed {idx} reads')
        kmers_in_read = generate_kmers(seq, kmer_len, skip_N)
        kmers = kmers.union(set(kmers_in_read))
        for kk in range(len(kmers_in_read) - 1):
            transition_counts[(kmers_in_read[kk], kmers_in_read[kk + 1])] += 1
    nodes = []
    logger.info(f'{sample_name}: adding nodes to graph')
    if skip_N:
        logger.info(f'{sample_name}: skipping padding nucleotide')
        for kmer in kmers:
            if 'N' not in kmer:
                nodes.append(
                    (
                    kmer, 
                    {"x": torch.as_tensor(feature_method(kmer), dtype=torch.float32)}
                    )
                )
    else:
        logger.info(f'{sample_name}: not skipping padding nucleotide')
        for kmer in kmers:
            nodes.append(
                (
                kmer, 
                {"x": torch.as_tensor(feature_method(kmer), dtype=torch.float32)}
                )
            )
    G.add_nodes_from(nodes)

    logger.info(f'{sample_name}: normalizing transition count by {normalization_method}')
    # TODO: normalize only by the weights aggregated from a source node, not from the whole graph
    normalization_val = get_normalization_val(transition_counts.values(), method=normalization_method)

    logger.info(f'{sample_name}: adding edges with edge weights of precision {edge_weight_dtype}')
    for key in transition_counts.keys():
        G.add_edge(key[0], key[1], 
                   weight=torch.as_tensor(transition_counts[key] / normalization_val, dtype=edge_weight_dtype))

    logger.info(f'{sample_name}: saving as torch graph')
    torch_graph = from_networkx(G)
    torch_graph['y'] = torch.tensor([city_code])
    torch_graph['sample_name'] = sample_name

    logger.info(f'{sample_name}: saving graph for sample {sample_name}')

    savefile_ext = savefile_ext if savefile_ext else ut.GRAPH_EXT
    outfile_graph_name = (
        os.path.join(outdir, sample_name + savefile_ext) 
        if outdir 
        else sample_name + savefile_ext
    )

    with open(outfile_graph_name, 'wb') as f:
        pickle.dump(torch_graph, f)



def main():

    args = get_args()

    ut.save_run_params_to_json(args, args.outdir, 'run_parameters.json')

    logger.setLevel(ut.get_verbosity_level(args.verbose))

    logger.info(f'getting samples from {args.indir}')
    sample_paths, city_code_to_id = samples_from_indir(args.indir)

    logger.info(f'filtering out samples with already built graphs')
    sample_paths = filter_out_built_graphs(sample_paths, args.outdir)

    logger.info(f'building graph in parallel with {args.threads} threads')
    # TODO: use tqdm here instead of inside of functions
    with Pool(processes = args.threads) as p:

        get_reads_and_build_graph_wrapper = functools.partial(get_labelled_reads_and_build_graph,
                                                    code_to_id=city_code_to_id,
                                                    skip_N=args.skip_N, outdir=args.outdir,
                                                    kmer_len=args.kmer_len, subkmer_len=args.subkmer_len, 
                                                    normalization_method=args.normalization_method, 
                                                    node_feature_method=args.node_feature_method, 
                                                    normalize_node_features=args.normalize_node_features)
        build_graph_max_result = p.map(get_reads_and_build_graph_wrapper, sample_paths) # output list not used

    logger.info(f'Finished building graphs. Goodbye :)')

if __name__ == '__main__':
    main()

