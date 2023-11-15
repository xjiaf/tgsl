import torch
from torch.nn import MultiheadAttention
from torch import LongTensor, Tensor

from torch_geometric.nn import TemporalEncoding

from utils.tools import gumbel_softmax, get_neighbors


class StructureLearner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass


class GraphAttentionGumbel(torch.nn.Module):
    def __init__(self, candidate_num_neighbors: list, feature_dim: int,
                 num_heads: int = 1, threshold: float = 0.5,
                 temperature: float = 0.5):
        super().__init__()
        self.candidate_num_neighbors = candidate_num_neighbors
        self.threshold = threshold
        self.temperature = temperature
        self.te_encoder = TemporalEncoding(out_channels=feature_dim)
        self.attention_layer = MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads)

    def forward(self, candidate_x: Tensor, candidate_time: LongTensor,
                target_x, target_time):

        # get the neighbor sampler for the given graph and node list
        node_te = self.te_encoder(candidate_time.float())
        x_with_te = torch.cat([candidate_x, node_te], dim=1)

        # get the target node and its temporal encoding
        target_te = self.te_encoder(target_time.float())
        target_x_with_te = torch.cat([target_x, target_te], dim=1)

        # reshape the input for the attention layer
        x_with_te_reshaped = x_with_te.unsqueeze(1)  # (L, N, E)
        target_x_with_te_reshaped = target_x_with_te.unsqueeze(0)  # (L, N, E)

        # attention layer
        attn_output, attn_output_weights = self.attention_layer(
            target_x_with_te_reshaped, x_with_te_reshaped, x_with_te_reshaped)

        candidate_mask = gumbel_softmax(
            attn_output_weights, self.temperature) > self.threshold

        return candidate_mask, attn_output
