import utils.utils as ut
import utils.utils_adopted as utad
import networkx as nx
import os
import torch
import pickle
import model_zoo as mz
import json
import random
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import to_networkx, from_networkx
import datetime
from sklearn.preprocessing import OneHotEncoder

os.environ['TORCH'] = torch.__version__

print(f'Running torch version {torch.__version__}')
print(f'CUDA is available: {torch.cuda.is_available()}')

num_node_features = 100

# load dataset from folder

data_folder = '../data/camda2020_tiny'
print(f'{datetime.datetime.now().strftime("%d %h %Y %H:%M:%S")} Loading DBGs from folder {data_folder}')

dbgs = ut.get_dbgs_from_dir(data_folder)
train_labels = [grph.graph['y_label'] for grph in dbgs]
unique_train_labels = list(set(train_labels))


# initialize node features

# print(f'{datetime.datetime.now().strftime("%d %h %Y %H:%M:%S")} Initializing node features with one-hot encoding')
# for dbg in dbgs:

#     num_nodes = dbg.number_of_nodes()
#     all_nodes = list(dbg.nodes)
#     encoder = OneHotEncoder(sparse_output=False)
#     encoded_labels = encoder.fit_transform([[node_label] for node_label in all_nodes])
#     for idx, node in enumerate(dbg.nodes()):
#         dbg.nodes[node]['x'] = torch.tensor(encoded_labels[idx], dtype=torch.float)

print(f'{datetime.datetime.now().strftime("%d %h %Y %H:%M:%S")} Initializing node features with const vectors')
for dbg in dbgs:
    for node in dbg.nodes():
        dbg.nodes[node]['x'] = torch.ones(num_node_features, dtype=torch.float)

print(f'{datetime.datetime.now().strftime("%d %h %Y %H:%M:%S")} Initializing node features with one-hot encoding')

# convert to torch geometric data

print(f'{datetime.datetime.now().strftime("%d %h %Y %H:%M:%S")} Converting to torch geometric dataset')

random.shuffle(dbgs)
train_data_torch = []
for gr in dbgs:

    graph_label = torch.tensor([gr.graph['y_label']], dtype=torch.uint8)
    gr_torch = from_networkx(gr)
    gr_torch.y = graph_label

    train_data_torch.append(gr_torch)


# split train test

print(f'{datetime.datetime.now().strftime("%d %h %Y %H:%M:%S")} Splitting in train test')

train_dataset_torch = train_data_torch[:10]
test_dataset_torch = train_data_torch[10:]

train_loader = DataLoader(train_dataset_torch, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset_torch, batch_size=1, shuffle=False)

# select model from the model zoo

model = mz.GCN(num_node_features=num_node_features, hidden_channels=128, num_classes=len(unique_train_labels))

# train model

print(f'{datetime.datetime.now().strftime("%d %h %Y %H:%M:%S")} Training model')

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
         out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
         loss = criterion(out, data.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.

def test(loader):
     model.eval()

     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.batch)  
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.


train_acc_record = []
test_acc_record = []
model_savefile = 'gcn.model'
for epoch in range(1, 200):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)

    train_acc_record.append(train_acc)
    test_acc_record.append(test_acc)

    print(f'Epoch: {epoch: 03d}, Train Acc: {train_acc: .4f}, Test Acc: {test_acc: .4f}')

    if epoch % 20 == 0:
        print(f'{datetime.datetime.now().strftime("%d %h %Y %H:%M:%S")} Saving updated model at epoch {epoch}')
        with open(model_savefile, 'wb') as f:
            pickle.dump(model, f)
    


