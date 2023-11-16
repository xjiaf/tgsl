import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import TemporalEncoding

from datasets.temporal_graph import TemporalGraph
from modules.dgn import DGN


class TemporalGraphInformationBottleneck(nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
        super().__init__()
        self.fc_mu = nn.Linear(input_dim, bottleneck_dim)
        self.fc_log_var = nn.Linear(input_dim, bottleneck_dim)
        self.dgn = nn.Sequential(
            DGN(input_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.ReLU()
            )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, target_node_indice, x, edge_index, edge_time,
                node_time, edge_weight=None) -> torch.Tensor:
        node_emb = self.dgn(x=x, edge_index=edge_index,
                            edge_time=edge_time,
                            node_time=node_time,
                            edge_weight=edge_weight)

        target_node_emb = []
        for emb, idx in zip(node_emb, target_node_indice):
            target_node_emb.append(emb[idx])
        else:
            target_node_emb = torch.stack(target_node_emb, dim=0)

        mu = self.fc_mu(target_node_emb)
        log_var = self.fc_log_var(target_node_emb)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var

    def loss(self, mu, log_var, edge_mask, beta=1.0):
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        edge_loss = F.mse_loss(
            edge_mask, torch.zeros_like(edge_mask), reduction='sum')
        return beta * kl_loss + edge_loss


class GraphModelWithIB(nn.Module):
    def __init__(self, temporal_enc_dim, edge_emb_dim, node_embed_dim):
        super().__init__()
        self.te = TemporalEncoding(out_channels=temporal_enc_dim)
        self.lin1 = nn.Linear(2 * node_embed_dim + temporal_enc_dim + 1,
                              edge_emb_dim)
        self.mask = nn.Sequential(nn.Linear(edge_emb_dim, 1), nn.Sigmoid())

    def forward(self, aug_graph: TemporalGraph) -> TemporalGraph:
        # processed by attention module
        # It is the concatenation of node emb from DGN after IB and edge_weight
        edge_attr = aug_graph.edge_attr
        edge_emb = self.generate_edge_emb(edge_attr, aug_graph.edge_time)
        edge_mask = self.mask(edge_emb)
        aug_graph.edge_weight *= edge_mask

        return aug_graph

    def generate_edge_emb(self, edge_attr, edge_time):
        # Get node embeddings for source and destination nodes
        # node_embs_src = node_emb[edge_time, edge_index[0, :], :]
        # node_embs_dst = node_emb[edge_time, edge_index[1, :], :]

        te = self.te(edge_time)
        edge_emb = self.lin1(torch.cat([edge_attr, te], dim=1))
        return edge_emb

    def loss(self, prediction, target, mu, log_var, edge_mask, beta=1.0):
        task_loss = F.nll_loss(prediction, target)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        edge_loss = F.mse_loss(
            edge_mask, torch.zeros_like(edge_mask), reduction='sum')
        return task_loss + beta * kl_loss + edge_loss
