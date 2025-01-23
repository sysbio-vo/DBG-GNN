
import functools
import numpy as np
from enum import IntEnum
from collections import Counter
from typing import Any
from create_dbgs_cli import DNA_ALPHABET, DNA5_ALPHABET, kmer_to_index


class EmbeddingMethod(IntEnum):
    SUBKMER = 0
    SUBKMER_POS = 1

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
    def subkmer_frequencies_in_kmer_positional(kmer: str, subkmer_len: int, skip_N: bool = True, 
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
            ^             ^       ^
            |             |       |
            idx=3         11      15

        
        I propose doing the following:
        positional_information = np.arange(1, k) = [1, 2, 3, 4]

        Then we multiple each bitvector representing a sub-kmer by the corresponding positional number
        to obtain the following embedding:
        positionally_resolved_AT = [0, 0, 1, 0, ..., 0] * 1 = [0, 0, 1, 0, ..., 0]
        positionally_resolved_TG = [0, 0, 0, 0, ..., 1, 0] * 2 = [0, 0, 0, 0, ..., 2, 0]
        positionally_resolved_GG = [0, ..., 1, ..., 0, 0] * 3 + [0, ..., 1, ..., 0, 0] * 4 = [0, ..., 7, ..., 0, 0]

        So that the final k-mer embedding looks as follows:
        [0, 0, 1, 0, ..., 7, ..., 2, 0]
            ^             ^       ^
            |             |       |
            idx=3         11      15

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
            picked_method = subkmer_frequencies_in_kmer_positional

    return functools.partial(picked_method, *args, **kwargs)



def get_k_mer_embedding(k_mer: str, method: str | None = None, *args, **kwargs) -> np.array:
    # TODO: replace method with the enum class
    if method is None:
        method = 'subkmer_freq'
    embedding_method = node_feature_method_selector(method, *args, **kwargs)

    return embedding_method(k_mer)
