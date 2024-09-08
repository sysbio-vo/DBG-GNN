#! /bin/bash

# MODIFY: set conda start path
CONDASTARTPATH=/home/lbombini/micromamba/etc/profile.d
# MODIFY: set the conda env name with fastp installed
CONDAENV="gnn"
# MODIFY: set outdir with a unique name
export OUTDIR="../fastp_vk11"

source $CONDASTARTPATH/micromamba.sh
micromamba activate ${CONDAENV}
mkdir -p ${OUTDIR}

run_fastp() {
    fwd_path="$1"
    fwd=$(basename "$fwd_path")
    indir=$(dirname "$fwd_path")

    rev="${fwd/_1.fastq.gz/_2.fastq.gz}"
    rev_path="${indir}/${rev}"

    fwd_trimmed="${fwd/_1.fastq.gz/1_trimmed.fastq.gz}"
    fwd_trimmed_path="${OUTDIR}/${fwd_trimmed}"
    rev_trimmed="${rev/_2.fastq.gz/2_trimmed.fastq.gz}"
    rev_trimmed_path="${OUTDIR}/${rev_trimmed}"

    # check if the valid output already exists
    if [ -f "$fwd_trimmed_path" ] && [ -s "$fwd_trimmed_path" ] && [ -f "$rev_trimmed_path" ] && [ -s "$rev_trimmed_path" ]; then
        echo "${fwd_trimmed_path} and its pair already exist. Skipping the pair."
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

    echo "Running fastp on ${fwd} and ${rev}"
    micromamba run -n gnn fastp -i $fwd_path -I $rev_path -o $fwd_trimmed_path -O $rev_trimmed_path

}

export -f run_fastp
find . -name "*_1.fastq.gz" -type f | parallel run_fastp