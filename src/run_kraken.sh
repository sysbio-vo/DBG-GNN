#! /bin/bash

# MODIFY: set your paths here
CONDAPATH=/home/nepotlet/miniconda3/bin
CONDASTARTPATH=/home/nepotlet/miniconda3/etc/profile.d
source $CONDASTARTPATH/conda.sh

echo "activating conda env"
CONDAENV="gnn"
conda activate $CONDAENV

# INPUTS
export INDIR="/scratch2/sysbio/fastped_data"
export INFILE_SUFFIX="_trimmed"
export OUTDIR="../kraken_vk11"
export OUTFILE_SUFFIX="_filtered"

# kraken params
export KRAKEN_DB="kraken2_human_db/"
export THREADS=4


mkdir -p $OUTDIR

run_kraken() {
    fwd_path="$1"
    fwd=$(basename "$fwd_path")
    indir=$(dirname "$fwd_path")

    rev="${fwd/_1${INFILE_SUFFIX}.fastq.gz/_2${INFILE_SUFFIX}.fastq.gz}"
    rev_path="${indir}/${rev}"

    out_filtered="${fwd/_1${INFILE_SUFFIX}.fastq.gz/_#${INFILE_SUFFIX}${OUTFILE_SUFFIX}.fastq}" # `#` is a kraken2-specific wildcard that expands to _1 or _2 for the paired read case
    out_filtered_path="${OUTDIR}/${out_filtered}"

    # full outfile paths are still needed for checks
    fwd_filtered="${fwd/_1${INFILE_SUFFIX}.fastq.gz/_1${INFILE_SUFFIX}${OUTFILE_SUFFIX}.fastq}"
    fwd_filtered_path="${OUTDIR}/${fwd_filtered}"
    rev_filtered="${rev/_2${INFILE_SUFFIX}.fastq.gz/_2${INFILE_SUFFIX}${OUTFILE_SUFFIX}.fastq}"
    rev_filtered_path="${OUTDIR}/${rev_filtered}"   

    # check if the valid output already exists
    if [ -f "$fwd_filtered_path" ] && [ -s "$fwd_filtered_path" ] && [ -f "$rev_filtered_path" ] && [ -s "$rev_filtered_path" ]; then
        echo "${fwd_filtered_path} and its pair already exist. Skipping the pair."
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

    echo "Running kraken on ${fwd} and ${rev}"

    kraken2 --paired --gzip-compressed \
        --db ${KRAKEN_DB} --threads ${THREADS} \
        --unclassified-out ${out_filtered_path} \
        ${fwd_path} ${rev_path}

}

export -f run_kraken
find $INDIR -name "*_1${INFILE_SUFFIX}.fastq.gz" -type f | parallel run_kraken