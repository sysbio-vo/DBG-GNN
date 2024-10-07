import os
import json
import torch
import pickle
import random
import logging
import pathlib
import argparse

import model_zoo as mz
import utils.utils as ut
import utils.plotting_utils as plut

from multiprocessing import Pool
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import to_networkx, from_networkx


os.environ['TORCH'] = torch.__version__

logging.basicConfig(**ut.LOGGER_CONFIGURATION)
logger = logging.getLogger(__name__)

logger.debug(f'Running torch version {torch.__version__}')
logger.debug(f'CUDA is available: {torch.cuda.is_available()}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def get_args():
    """Get training args from the command line using argparse.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--indir', type=pathlib.Path,
        help='directory with built de Bruijn graphs')

    parser.add_argument('-ts', '--train_split', type=float, default=0.8,
        help='proportion of samples to use for training')

    parser.add_argument('-tbs', '--train_batch_size', type=int, default=64,
        help='train batch size for pytorch dataloader')
    parser.add_argument('-st', '--shuffle_train',
        help='shuffle train batch', action='store_true')

    parser.add_argument('-vls', '--validation_batch_size', type=int, default=1,
        help='validation batch size for pytorch dataloader')
    parser.add_argument('-sv', '--shuffle_validation',
        help='shuffle validation batch', action='store_true')
    
    parser.add_argument('-sd', '--seed', type=int, default=123,
        help='torch seed for training')

    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01,
        help="optimizer's learning rate")
    parser.add_argument('-n', '--n_epochs', type=int, default=3_000,
        help='number of training epochs')
    parser.add_argument('-pn', '--log_every_n_epochs', type=int, default=100,
        help='print loss values every num of epochs')
    parser.add_argument('-sn', '--save_every_n_epochs', type=int, default=1_000,
        help='save (update) model every num of epochs')
    
    parser.add_argument('-o', '--outfile', type=pathlib.Path,
        help='path to save pickled model to')
    
    parser.add_argument('-plo', '--plots_outdir', type=pathlib.Path,
        help='path to save training plots to')
    
    parser.add_argument('-v', '--verbose', 
        help='verbosity level', action='count', default=0)  # TODO: add different levels of logging

    args = parser.parse_args()
    return args


def get_num_node_features(data) -> int:
    """Get number of node features from a graph stored as a torch_geometric.data.data.Data object.
    """
    return data.x.shape[1]


def read_graphs_from_dir(indir: str, regex_pattern: str = 'CAMDA') -> tuple[list[object], set]:
    """Read graph objects from a directory in a list. Consider only files matching pattern.
    """
    graphs = []
    cities_in_dataset = set() # city codes available in dataset
    for file in os.listdir(indir):
        # TODO: make real regex
        if not file.startswith(regex_pattern):
            continue

        logger.info(f'Reading {file}')
        city_code = os.path.splitext(file)[0].split('_')[3] # specific for CAMDA files
        cities_in_dataset.add(city_code)

        with open(os.path.join(indir, file), 'rb') as f:
            graph = pickle.load(f) # FIXME: address torch FutureWarning on loading pickled models with `weights_only=False` by default
        graphs.append(graph)

    return graphs, cities_in_dataset

def train(model, optimizer, criterion, train_loader):
    model.train()
    total_loss = 0
    correct = 0
    for data in train_loader:
        # use GPU
        data = data.to(device)

        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.weight, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()

    return total_loss / len(train_loader), correct / len(train_loader.dataset)
    

def validate(model, validation_loader):
    model.eval()
    correct = 0
    for data in validation_loader:
        # use GPU
        data = data.to(device)

        with torch.no_grad():
            out = model(data.x, data.edge_index, data.weight, data.batch)
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
    return correct / len(validation_loader.dataset)
    


def main():
    args = get_args()
    params_out = os.path.join(args.outfile.parent, "parameters.json")
    os.makedirs(args.outfile.parent, exist_ok=True)
    with open(params_out, "w") as out:

        def to_str_if_posix_path(val):
            if isinstance(val, pathlib.Path):
                return str(val)
            return val

        args_str = {k: to_str_if_posix_path(v) for k, v in vars(args).items()} 
        json.dump(args_str, out, indent=4)
    logger.setLevel(ut.get_verbosity_level(args.verbose))


    logger.info(f'Loading DBGs from folder {args.indir}')
    graphs, city_codes_in_dataset = read_graphs_from_dir(args.indir) # TODO: unite with similar function from utils

    graphs_num = len(graphs)

    # TODO: use cli args
    random.shuffle(graphs) # TODO: use seed
    split_value = args.train_split

    train_dataset = graphs[:int(graphs_num * split_value)]
    val_dataset = graphs[int(graphs_num * split_value):]

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=args.shuffle_train, 
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.validation_batch_size, shuffle=args.shuffle_validation, 
                            pin_memory=True)

    num_node_features = get_num_node_features(graphs[0])
    num_classes = len(city_codes_in_dataset)

    model = mz.GCN_MLP(num_features=num_node_features, num_classes=num_classes, 
                       mlp_layers=4, mlp_hidden_channels=128, 
                       hidden_channels=32, seed=args.seed)
    
    model.to(device) # load model to GPU
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # training loop
    model_savefile = args.outfile
    os.makedirs(os.path.dirname(model_savefile), exist_ok=True) # create parent dirs for the model savefile
    train_loss_record = []
    train_acc_record = []
    val_acc_record = []
    num_epochs = args.n_epochs
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, optimizer, criterion, train_loader)
        val_acc = validate(model, val_loader)

        train_loss_record.append(train_loss)
        train_acc_record.append(train_acc)
        val_acc_record.append(val_acc)

        if epoch % args.log_every_n_epochs == 0:
            logger.info(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, \
Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

        if (epoch % args.save_every_n_epochs == 0) and (epoch != 0):
            logger.info(f'Saving updated model at epoch {epoch}')
            with open(model_savefile, 'wb') as f:
                pickle.dump(model, f)

    logger.info(f'Saving loss plots to {args.plots_outdir}')
    os.makedirs(args.plots_outdir, exist_ok=True)
    plot_filename = 'losses.png'
    plot_filename = os.path.join(args.plots_outdir, plot_filename)

    losses_fig = plut.plot_in_row([train_loss_record, train_acc_record, val_acc_record], 
                                  titles=['Train Loss', 'Train Accuracy', 'Validation Accuracy'])
    losses_fig.savefig(plot_filename)

    logger.info(f'Finished training GNN. Goodbye :)')

if __name__ == '__main__':
    main()
