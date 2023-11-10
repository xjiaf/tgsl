import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import DenseGraphConv
from torch_geometric.utils import dense_to_sparse, to_dense_adj


class VariationalInformationBottleneck(nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
        super().__init__()
        self.fc_mu = nn.Linear(input_dim, bottleneck_dim)
        self.fc_log_var = nn.Linear(input_dim, bottleneck_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var

class GraphModelWithIB(nn.Module):
    def __init__(self, num_nodes, num_features, bottleneck_dim, out_features):
        super().__init__()
        self.ib_layer = VariationalInformationBottleneck(num_features, bottleneck_dim)
        self.edge_weights = nn.Parameter(torch.randn(num_nodes, num_nodes))
        self.conv1 = DenseGraphConv(bottleneck_dim, out_features)

    def forward(self, data):
        x, adj = data.x, to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]

        z, mu, log_var = self.ib_layer(x)
        edge_mask = torch.sigmoid(self.edge_weights)
        masked_adj = adj * edge_mask

        sparse_adj = dense_to_sparse(masked_adj)[0]
        out = self.conv1(z, sparse_adj)
        return out, mu, log_var, edge_mask

    def loss(self, prediction, target, mu, log_var, edge_mask, beta=1.0):
        task_loss = F.nll_loss(prediction, target)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        edge_loss = F.mse_loss(edge_mask, torch.zeros_like(edge_mask), reduction='sum')
        return task_loss + beta * kl_loss + edge_loss

# Assume data is a PyG Data object with x and edge_index fields
num_nodes = data.num_nodes
num_features = data.num_node_features
bottleneck_dim = 32  # or any appropriate bottleneck dimension
out_features = data.y.max().item() + 1  # for classification tasks

model = GraphModelWithIB(num_nodes, num_features, bottleneck_dim, out_features)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):  # adjust epochs according to your dataset
    optimizer.zero_grad()
    out, mu, log_var, edge_mask = model(data)
    loss = model.loss(F.log_softmax(out, dim=1), data.y, mu, log_var, edge_mask, beta=1.0)
    loss.backward()
    optimizer.step()
