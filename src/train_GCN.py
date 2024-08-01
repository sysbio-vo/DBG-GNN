import utils.utils as ut
import utils.utils_adopted as utad
import networkx as nx
import os
import torch
import pickle
import model_zoo as mz
import json
from torch_geometric.loader import DataLoader

os.environ['TORCH'] = torch.__version__

print(f'Running torch version {torch.__version__}')
print(f'CUDA is available: {torch.cuda.is_available()}')

# load dataset from folder

data_folder = 'data/'

dbgs = ut.get_dbgs_from_dir(data_folder)
train_labels = None

# split train test

train_loader = None
test_loader = None

# select model from the model zoo

model = mz.GCN(num_node_features=1, hidden_channels=64, num_classes = len(train_labels))

# train model

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
for epoch in range(1, 100):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)

    train_acc_record.append(train_acc)
    test_acc_record.append(test_acc)

    print(f'Epoch: {epoch: 03d}, Train Acc: {train_acc: .4f}, Test Acc: {test_acc: .4f}')


