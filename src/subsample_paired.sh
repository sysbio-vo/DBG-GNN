#! /bin/bash

# MODIFY: set your paths here
CONDAPATH=/home/lbombini/micromamba/bin
CONDASTARTPATH=/home/lbombini/micromamba/etc/profile.d
source $CONDASTARTPATH/micromamba.sh


# MODIFY: activate env with seqkit
echo "activating conda env"
micromamba activate gnn


# MODIFY: set proportion of reads to subsample
export PROPORTION_TO_TAKE=0.2
# MODIFY: set outdir with a unique name
export OUTPATH=subsampled_vk11

mkdir -p $OUTPATH

# MODIFY: (optionally) 
# set datetime after which downloaded samples will be ignored 
# (to not process partially downloaded files)
export DATETIME="2024-08-08 00:00:00"

process_sample() {
    fwd_path="$1"
    fwd=$(basename "$fwd_path")
    fwd_out=${OUTPATH}/${fwd%.fastq.gz}_subsampled_20_percent.fastq

    dir=$(dirname "$fwd_path")

    rev="${fwd/_1.fastq.gz/_2.fastq.gz}"
    rev_path="${dir}/${rev}"
    rev_out=${OUTPATH}/${rev%.fastq.gz}_subsampled_20_percent.fastq

    # check if the valid output already exists
    if [ -f "$fwd_out" ] && [ -s "$fwd_out" ] && [ -f "$rev_out" ] && [ -s "$rev_out" ]; then
        echo "${fwd_out} and its pair already exist. Skipping the pair."
        return 0
    fi

    # check if the pair exists
    if [ ! -f "$rev" ]; then
        echo "R2 for ${fwd} has not been found. Skipping the pair"
        return 1
    fi

    # check if not empty
    if [ ! -s "$fwd" ] || [ ! -s "$rev" ]; then
        echo "Either $fwd or its pair is empty. Skipping the pair"
        return 1
    fi

    seed=$RANDOM
    echo "Subsampling $fwd  Proportion: $PROPORTION_TO_TAKE  Seed: ${seed}"
    seqkit sample -p $PROPORTION_TO_TAKE $fwd_path -o $fwd_out -s $seed
    
    echo "Subsampling $rev  Proportion: $PROPORTION_TO_TAKE  Seed: ${seed}"
    seqkit sample -p $PROPORTION_TO_TAKE $rev_path -o $rev_out -s $seed
}

export -f process_sample

# process fastq.gz files from the current dir
find . -name "*_1.fastq.gz" -type f | parallel process_sample


# # build graphs
# echo "activating gnn env"
# conda activate gnn
# pushd /home/nepotlet/DBG-GNN/src/
# python create_dbgs_max.py
# popd
