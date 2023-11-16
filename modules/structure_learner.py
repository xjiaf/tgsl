import numpy as np
import torch
from torch import nn, LongTensor, FloatTensor
from torch.nn import (
    Module,
    MultiheadAttention,
    functional as F
    )
from torch_geometric.nn import TemporalEncoding

from utils.tools import get_neighbors
from datasets.temporal_graph import TemporalGraph


class StructureLearner(Module):
    def __init__(self, graph: TemporalGraph, num_candidates,
                 candidate_num_neighbors: list, te_emb_dim, num_heads: int = 1,
                 threshold: float = 0.2, tau: float = 1, dropout: float = 0.2,
                 directed: bool = False):
        super().__init__()
        self.graph = graph
        self.num_candidates = num_candidates
        self.candidate_num_neighbors = candidate_num_neighbors
        self.directed = directed
        self.te_encoder = TemporalEncoding(out_channels=te_emb_dim)
        self.gag = GraphAttentionGumbel(
            features_dim=graph.num_node_features + te_emb_dim,
            num_heads=num_heads,
            threshold=threshold,
            tau=tau,
            dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(graph.num_node_features + te_emb_dim + 1, 1),
            nn.LeakyReLU())

    def forward(self, target_node_ids: LongTensor,
                target_node_times: LongTensor) -> TemporalGraph:
        candidate_seed_nodes = self.get_candidate_seed_nodes(
            self.graph, target_node_ids, num_candidates=self.num_candidates)

        for target_node, target_time, candidate_nodes in zip(
                target_node_ids, target_node_times, candidate_seed_nodes):
            # get the candidate subgraph
            sampler_output = get_neighbors(
                self.graph, self.candidate_num_neighbors, candidate_nodes,
                target_time)

            # get the target node and its temporal encoding
            target_x = self.graph.x[target_node]
            target_te = self.te_encoder(target_time.float())
            target_emb = torch.cat([target_x, target_te], dim=1)

            # get the candidate nodes and their temporal encodings
            x = self.graph.x[sampler_output.node]
            edge_time = self.graph.edge_time[sampler_output.edge]
            edge_weight = self.graph.edge_weight[sampler_output.edge]
            candidate_x = torch.cat((x[sampler_output.row],
                                     x[sampler_output.col]), dim=0)
            candidate_te = self.te_encoder(
                torch.cat((edge_time, edge_time), dim=0).float())
            candidate_weight = torch.cat(
                (edge_weight, edge_weight), dim=0)
            candidate_emb = torch.cat((candidate_x, candidate_te), dim=1)

            # graph attention gumbel
            attn_output, candidate_mask = self.gag(
                target_emb, candidate_emb)

            # Select the candidate nodes
            selected_candidate_nodes = sampler_output.node[candidate_mask]
            selected_candidate_weight = candidate_weight[candidate_mask]

            # graph structure update
            self.graph_structure_update(
                target_node, target_time, selected_candidate_nodes,
                selected_candidate_weight, attn_output)
        else:
            return self.graph

    def graph_structure_update(self, target_node: int,
                               target_time: int,
                               selected_candidate_nodes: LongTensor,
                               selected_candidate_weight: FloatTensor,
                               attn_output: FloatTensor):
        new_edge_weight = self.mlp(torch.cat((
            attn_output.repeat(selected_candidate_weight.size(0), 1),
            selected_candidate_weight.unsqueeze(1)), dim=1))
        new_edge_weight.squeeze_(1)
        new_edge_index = torch.stack(
            (selected_candidate_nodes, torch.ones_like(
                selected_candidate_nodes) * target_node), dim=0).long()
        new_edge_time = torch.randint(
            0, target_time, (selected_candidate_nodes.size(0),)).long()
        new_graph = TemporalGraph(edge_index=new_edge_index,
                                  edge_time=new_edge_time,
                                  edge_weight=new_edge_weight,
                                  directed=self.directed)
        self.graph.update(new_graph)

    def get_candidate_seed_nodes(self, target_node_ids: LongTensor,
                                 num_candidates: int,
                                 sample_size: int = None) -> LongTensor:
        if sample_size is None:
            sample_size = target_node_ids.size(0) * 10

        # Randomly select 'sample_size' candidate nodes
        # from [0, num_nodes-1] without replacement
        candidate_seed_list = torch.LongTensor(
            np.random.choice(self.graph.num_nodes, sample_size, replace=False))

        # Get expend both node features tensors for broadcasting
        candidate_features = self.graph.x[candidate_seed_list].unsqueeze(0)
        target_features = self.graph.x[target_node_ids].unsqueeze(1)

        # Compute cosine similarity
        similarities = F.cosine_similarity(
            target_features, candidate_features, dim=2)

        # Select the top 'num_candidates' seed nodes based on similarity
        _, top_indices = torch.topk(similarities, num_candidates, dim=1)

        # Convert candidate node indices to actual node IDs
        seed_nodes = candidate_seed_list[top_indices.squeeze(1)]

        return seed_nodes


class GraphAttentionGumbel(Module):
    def __init__(self, features_dim: int,
                 num_heads: int = 1, threshold: float = 0.2,
                 tau: float = 1, dropout: float = 0.2):
        super().__init__()
        self.threshold = threshold
        self.tau = tau
        self.attention_layer = MultiheadAttention(
            embed_dim=features_dim, num_heads=num_heads, dropout=dropout)

    def forward(self, target_emb: FloatTensor, candidate_emb: FloatTensor):

        # graph attention
        attn_output, attn_output_weights = self.attention_layer(
            target_emb, candidate_emb, candidate_emb)

        # gumbel softmax
        candidate_mask = F.gumbel_softmax(
            attn_output_weights, tau=self.tau, hard=False) > self.threshold

        if candidate_mask.size(0) == 1:
            candidate_mask.squeeze_(0)

        if attn_output.size(0) == 1:
            attn_output.squeeze_(0)

        return attn_output, candidate_mask
