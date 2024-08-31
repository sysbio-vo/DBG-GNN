# DBG-GNN
This repo provides a toolset for building de Bruijn graphs enhanced with various edge and node features as well as versatile graph learning options.

> [!WARNING]
> The code here is under development. Bugs are possible.

# Usage

Before running the scripts, create and activate the environment with a PyTorch version suitable for your machine. You can try to use provided `yaml` file, but no guarantees that would work for you:

```bash
conda env create -f path/to/DBG-GNN/envs/environment.yaml
conda activate GNNs
```

## DBG construction
To build de Bruijn graphs supplied with node and edge features, run:
```bash
python create_dbgs_cli.py \
--indir /path/to/dir/with/samples \
--outdir /path/to/outdir \
--kmer_len 4 \
--subkmer_len 2 \
--skip_N \
--normalization_method max \
--threads 4 \
--verbose
```
See more info on arguments by running `python create_dbgs_max_cli.py --help`.

## Model training
Use this script to train the model
```bash
python train_gnn_cli.py \
--indir /path/to/dir/with/graphs \
--outfile /path/to/model/savefile \
--plots_outdir /path/to/dir/to/save/plots \
--verbose
```
Explore other parameters via `python train_gnn_cli.py --help`.
