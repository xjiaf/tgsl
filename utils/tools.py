import torch
from torch import LongTensor
from torch_geometric.sampler import (
    NeighborSampler,
    NodeSamplerInput
    )

from datasets.temporal_graph import TemporalGraph


def gumbel_softmax(logits, temperature=1.0, eps=1e-10):
    """
    Apply the Gumbel-Softmax trick for a differentiable
        approximation of a discrete distribution.

    :param logits: Tensor of logits, representing raw scores before softmax.
    :param temperature: Controls the smoothness of the output distribution.
                        Lower values make the distribution more discrete.
    :param eps: A small number to prevent numerical issues.
    :return: Softmax-applied, temperature-scaled logits
        for soft selection probabilities.
    """

    # Generate uniform random numbers for Gumbel noise
    # directly in the shape of logits
    uniform_random = torch.rand(logits.shape)

    # Transform the uniform random numbers into Gumbel noise
    gumbel_noise = -torch.log(-torch.log(uniform_random + eps) + eps)

    # Combine the logits with Gumbel noise, then scale by temperature
    scaled_logits = (logits + gumbel_noise) / temperature

    # Apply softmax to convert the scaled logits into probabilities
    return torch.softmax(scaled_logits, dim=-1)


def get_neighbors(graph: TemporalGraph, num_neighbors: int, node_list:
                  LongTensor, node_time: LongTensor, device):
    """
    Get the neighbor sampler for the given graph and node list.

    Returns:
        sampler_output: SamplerOutput
    """
    sampler = NeighborSampler(
        data=graph,
        num_neighbors=num_neighbors,
        temporal_strategy='uniform',
        time_attr='edge_time'
    )

    # sampling
    sampler_output = sampler.sample_from_nodes(
        NodeSamplerInput(input_id=None, node=node_list, time=node_time)
    )

    edge_index = torch.stack([sampler_output.row, sampler_output.col],
                             dim=0, device=device)
    x = graph.x[sampler_output.node].to(device)
    edge_time = graph.edge_time[sampler_output.edge].to(device)
    edge_weight = graph.edge_weight[sampler_output.edge].to(device)

    return edge_index, x, edge_time, edge_weight
