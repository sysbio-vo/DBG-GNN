import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

class GCN_MLP(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels: int = 32, 
                 mlp_layers: int = 2, mlp_hidden_channels: int = 64, seed: int = 12345):
        super(GCN_MLP, self).__init__()
        torch.manual_seed(seed)
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        
        self.mlp = self.build_mlp(hidden_channels, mlp_hidden_channels, mlp_layers, num_classes)

    def build_mlp(self, input_dim, hidden_dim, num_layers, output_dim):
        layers = []
        for i in range(num_layers):
            layers.append(Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
        layers.append(Linear(hidden_dim, output_dim))
        return ModuleList(layers)

    def forward(self, x, edge_index, edge_weight, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weight)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_weight)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        for layer in self.mlp:
            x = layer(x)
        
        return x


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels: int = 32, seed: int = 12345):
        super(GCN, self).__init__()
        torch.manual_seed(seed)
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_weight, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weight)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_weight)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x
    
class GAT(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(GAT, self).__init__()
        torch.manual_seed(112233)
        self.conv1 = GATConv(num_node_features, hidden_channels, edge_dim=1)
        self.conv2 = GATConv(hidden_channels, hidden_channels, edge_dim=1)
        self.conv3 = GATConv(hidden_channels, hidden_channels, edge_dim=1)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        graph_embedding = x

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x, graph_embedding


