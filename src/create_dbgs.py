import utils.utils as ut
import utils.utils_adopted as utad
import time
import os
from multiprocessing import Process, Pool
import pickle
import functools


def build_dbg_adopted_with_label(infile: str, k:int = 10, city_code_to_int: dict = None) -> object:
    print(f'Getting reads from file {infile}')
    # Example: CAMDA20_MetaSUB_CSD16_DOH_013_1_kneaddata_subsampled_30_percent.fastq 
    infile_ext = os.path.splitext(infile)[1]
    start_fq = time.time()
    if infile_ext in ('.fq', '.fastq'):
        reads = ut.get_reads_from_fq(infile)
    elif infile_ext in ('.gz'):
        reads = ut.get_reads_from_gzed_fq(infile)
    end_fq = time.time()
    print(f'Got reads from {infile} in {end_fq - start_fq} s')

    print("Building DBG")
    start = time.time()
    _, pair_frequency = utad.weighted_directed_edges(
        reads,
        k=k,
        stride=1,
        inlcudeUNK=False,
        disable_tqdm=False,
    )
    if city_code_to_int:
        infile_city_code = os.path.basename(infile).split('_')[3]
        int_label = int(city_code_to_int[infile_city_code])

    graph = utad.build_deBruijn_graph(
        pair_frequency,
        normalise=True,
        remove_N=True,
        create_all_kmers=False,
        disable_tqdm=False,
        y_label=int_label
    )
    end = time.time()
    print(f'Built DBG from {infile} in {end - start} s')

    return graph

def build_dbg_and_save(infile, k: int = 10, city_code_to_int: dict = None, outdir: str = None):
    dbg = build_dbg_adopted_with_label(infile, k, city_code_to_int)
    infile_basename = os.path.basename(infile)
    infile_no_ext = os.path.splitext(infile_basename)[0]
    if outdir:
        outfile = os.path.join(outdir, infile_no_ext + '.labeled_dbg')
    else:
        outfile = infile_no_ext + '.labeled_dbg'
    with open(outfile, 'wb') as f:
        pickle.dump(dbg, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    input_data_path = '/home/sysbio/camda2020/camda2020_tiny/filtered_subsampled/'
    id_to_code, code_to_id = ut.parse_train_labels(data_path=input_data_path, 
                                                outdir='/home/sysbio/camda2020/camda2020_tiny/filtered_subsampled/')

    input_fastqs = []
    for file in os.listdir(input_data_path):
        filename = os.path.basename(file)
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in ('.gz', '.fastq', '.fasta', '.fq', '.fa'):
            continue

        input_fastqs.append(os.path.abspath(os.path.join(input_data_path, file)))

    print(f'{input_fastqs = } DEBUG') # DEBUG
    graphs_outdir = '/home/sysbio/camda2020/camda2020_tiny/filtered_subsampled/dbgs/'
    if not os.path.exists(graphs_outdir):
        os.makedirs(graphs_outdir)

    with Pool(processes=4) as p:
        create_dbgs_from_dir = functools.partial(build_dbg_and_save, k=4, 
                                                 city_code_to_int=code_to_id, outdir=graphs_outdir)
        built_graphs_result = p.map(create_dbgs_from_dir, input_fastqs)


