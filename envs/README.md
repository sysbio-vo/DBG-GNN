There is a weird problem with torch install via pip, it doesn't want to install packages in order, resulting in missing torch import for packages that need it. So, the solution is to install pip packages one by one rather than using whole requirements.txt file:

1. Install conda env

```bash
conda env create -n gnn --file gnn_rgb.yaml
conda activate gnn
```

2. Install pip packages

```bash
cat req_rgb.txt | xargs -n 1 -L 1 pip install
```
