#! /bin/bash

# MODIFY: set conda start path
CONDASTARTPATH=/home/lbombini/micromamba/etc/profile.d
# MODIFY: set the conda env name with kraken installed
CONDAENV="gnn"
# MODIFY: set in sample suffix (e.g. pop_1{SUFFIX}.fastq)
export SUFFIX="_trimmed"
# MODIFY: set outdir with a unique name
export OUTDIR="../kraken_vk11"

# kraken params
export KRAKEN_DB="kraken2_human_db/"
export THREADS=4

source $CONDASTARTPATH/micromamba.sh
micromamba activate ${CONDAENV}
mkdir -p ${OUTDIR}

run_kraken() {
    fwd_path="$1"
    fwd=$(basename "$fwd_path")
    indir=$(dirname "$fwd_path")

    rev="${fwd/_1${SUFFIX}.fastq.gz/_2${SUFFIX}.fastq.gz}"
    rev_path="${indir}/${rev}"

    out_filtered="${fwd/_1${SUFFIX}.fastq.gz/_#${SUFFIX}_filtered.fastq}"
    out_filtered_path="${OUTDIR}/${out_filtered}"

    # full outfile paths are still needed for checks
    fwd_filtered="${fwd/_1${SUFFIX}.fastq.gz/_1${SUFFIX}_filtered.fastq}"
    fwd_filtered_path="${OUTDIR}/${fwd_filtered}"
    rev_filtered="${rev/_2${SUFFIX}.fastq.gz/_2${SUFFIX}_filtered.fastq}"
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
find . -name "*_1${SUFFIX}.fastq.gz" -type f | parallel run_kraken