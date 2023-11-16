from __future__ import annotations
import os
from collections import defaultdict
from collections import OrderedDict
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import torch
from torch import Tensor, LongTensor, FloatTensor
from torch_geometric.data import Data


class TemporalGraph(Data):
    def __init__(self, edge=None, x=None, edge_time=None,
                 edge_weight=None, directed=False, **kwargs):
        """
        Parameters:
        - edge : the edge data file_path or np/tensor object.
            Expected format of loaded data is a 2D array where
            each row represents an edge with columns:
            [source_node_id (long), target_node_id (long),
                    time_stamp (long), edge_weight (float)].

        - x: Path to the node feature file or np/tensor object.
            Expected format of loaded data is a 2D array
            where each row represents a node with the row index being
            node_id (long) and columns are feature values.

        - directed (bool): Indicates whether the graph is
            directed or undirected. Default is False.

        Attributes:
        - neighbor_sequence (defaultdict of list): Maps each
            node to a list of tuples, where each tuple represents a
            (historical neighbors, time, weight). the historical
            neighbors of each node ordered by ascending
            Format: {node_1: [(neighbor_node_a1, time_1a,
                edge_weight_1a), ...], node_2:[...],...}.

        - stream_graph (defaultdict of defaultdict of list):
            Maps each node to a dictionary where keys are
            time stamps (ascending ordered) and values are lists of
            neighboring nodes at that time. Format:
            {node_1: {time_1: [(neighbor_node_a, edge_weight_a), ...],
                time_2: [(neighbor_node_b, edge_weight_b), ...], ...}, ...}.
        """
        super().__init__(directed=directed, edge_time=edge_time,
                         edge_weight=edge_weight,
                         neighbor_sequence=defaultdict(list),
                         stream_graph={}, **kwargs)

        # Load and process data
        if edge is not None:
            self.edge_index, self.edge_time, self.edge_weight \
                = self.process_edge(edge)
        else:
            self.edge_index, self.edge_time, self.edge_weight = to_undirected(
                self.edge_index, self.edge_time, self.edge_weight
            )

        if x is not None:
            self.x = self._load_node_features(x)

        # update neighbor_sequence and stream_graph
        # need to improve efficiency via node sampler
        # self.update_further_graph(
        #     self.edge_index, self.edge_time, self.edge_weight)
        logging.info("finish temporal graph init")

    def process_edge(self, edge) -> (Tensor, Tensor, Tensor):
        edge_tuple = self._load_edge(edge)
        if not self.directed:
            edge_tuple = to_undirected(*edge_tuple)
        return remove_duplicate_edges(*edge_tuple)

    def update_further_graph(self, new_edge_index: Tensor,
                             new_edge_time: Tensor,
                             new_edge_weight=None):
        """
        Update the TemporalGraph with new edges, timestamps, and weights.

        Parameters:
        - new_edge_index: A 2xN tensor where N is the number of edges.
                        The first row contains source nodes,
                        and the second row contains target nodes.
        - new_edge_time: A 1D tensor of length N, representing
                        the timestamps for each edge.
        - new_edge_weight: A 1D tensor of length N,
                        representing the weights for each edge.

        This function will:
        1. Update the neighbor_sequence dictionary with new edges.
        2. Update the stream_graph dictionary to maintain
                edges in a temporal order.
        """
        # Determine the iterators based on the presence of new_edge_weight
        iterables = [new_edge_index[0], new_edge_index[1], new_edge_time]
        if new_edge_weight is not None:
            iterables.append(new_edge_weight)

        for data in zip(*iterables):
            source_node, target_node, time_stamp = data[:3]
            edge_weight = data[3].item() if len(data) > 3 else None

            # Update the neighbor sequence
            if edge_weight is None:
                self.neighbor_sequence.setdefault(
                    source_node.item(), []).append((
                        target_node.item(), time_stamp.item()))
            else:
                self.neighbor_sequence.setdefault(
                    source_node.item(), []).append((
                        target_node.item(), time_stamp.item(), edge_weight))

            # Update the stream graph
            if source_node not in self.stream_graph:
                self.stream_graph[source_node] = OrderedDict()

            if time_stamp not in self.stream_graph[source_node]:
                # Insert the time_stamp in sorted order
                self.stream_graph[source_node][time_stamp] = []
                # Sort the keys (time_stamps) of the inner dictionary
                self.stream_graph[source_node] = OrderedDict(sorted(
                    self.stream_graph[source_node].items()))

            # Decide whether to add edge_weight or not
            if edge_weight is None:
                self.stream_graph[source_node][time_stamp].append(target_node)
            else:
                self.stream_graph[source_node][time_stamp].append((
                    target_node, edge_weight))

        # Sort the neighbor sequence
        for source_node, hist in self.neighbor_sequence.items():
            sorted_hist = sorted(hist, key=lambda x: x[1])
            self.neighbor_sequence[source_node] = sorted_hist

    def update(self, new_graph: TemporalGraph):
        """
        Update the TemporalGraph with new edge_index, timestamps, and weights.
        The input can be either a combined new edge tensor or
            new edge_index, new_edge_time and new_edge_weight tensors.
        """

        # Append the new_edge_index to the existing edge_index
        edge_index = torch.cat([self.edge_index, new_graph.edge_index], dim=1)
        edge_time = torch.cat([self.edge_time, new_graph.edge_time])
        if new_graph.edge_weight is not None:
            if self.edge_weight is None:
                raise ValueError(
                    "Cannot add edge weights to a graph without edge weights.")
            edge_weight = torch.cat([self.edge_weight, new_graph.edge_weight])

        self.edge_index, self.edge_time, \
            self.edge_weight = remove_duplicate_edges(
                edge_index, edge_time, edge_weight)

    @property
    def max_time(self):
        return self.edge_time.max()

    @property
    def num_nodes(self) -> int:
        return self.x.shape[0]

    def _adaptive_load(self, filepath) -> Tensor:
        """
        Load .npy or .pt file based on its extension and return as Tensor.
        """
        _, file_extension = os.path.splitext(filepath)
        if file_extension == '.npy':
            return Tensor(np.load(filepath))
        elif file_extension == '.pt':
            return torch.load(filepath)
        elif file_extension == '.pkl':
            return Tensor(pd.read_pickle(filepath))
        else:
            raise ValueError(
                f"Unsupported file extension: {file_extension}. "
                f"Supported extensions are .pkl, .npy and .pt."
            )

    def _load_node_features(self, node_feat) -> FloatTensor:
        logging.info('load node features')
        x = self._adaptive_load(node_feat) if isinstance(
            node_feat, (str, Path)) else FloatTensor(node_feat)
        # x = Tensor(StandardScaler().fit_transform(x))
        logging.info('finish load node features')
        return x.float()

    def _load_edge(self, edge) -> (LongTensor, LongTensor, FloatTensor):
        """
        Load edge data from the given path. Depending on the
            number of columns in the file:
        - 1 and 2 columns: Assumes edge indices.
        - 3 columns: Assumes edge indices followed by timestamps.
        - 4 columns: Assumes edge indices followed by timestamps and weights.

        Args:
        - edge (str or tensor): Path or edge tensor
            to the file containing edge data.

        Returns:
        - edge_index (Tensor): Edge indices.
        - edge_time (Tensor or None): Timestamps for the edges.
            It must be present in the file.
        - edge_weight (Tensor or None): Weights for the edges.
            Returns None if not present in the file.
        """
        logging.info("load edge")
        edge_data = self._adaptive_load(edge) if isinstance(
            edge, (str, Path)) else Tensor(edge)
        if edge_data.shape[1] < 3:  # min columns: source, target, time
            raise ValueError("The edge data must have at"
                             "least 3 columns: source, target, and time.")
        edge_index = edge_data[:, :2].T

        # need to be long type for temporal linker exampler, maybe a bug of PyG
        edge_time = edge_data[:, 2]

        # Check for the weight column
        if edge_data.shape[1] > 3:
            edge_weight = edge_data[:, 3]
        else:
            edge_weight = None
        return edge_index.long(), edge_time.long(), edge_weight.float()


def remove_duplicate_edges(edge_index: LongTensor,
                           edge_time: LongTensor = None,
                           edge_weight: FloatTensor = None):
    """
    Remove duplicate edges from the given
        edge_index, edge_time, and edge_weight.
    """
    # Create a unique hash for each edge based on its attributes
    if edge_time is not None:
        edge = torch.cat([edge_index.T, edge_time.unsqueeze(-1)], dim=-1)
    if edge_weight is not None:
        edge = torch.cat([edge_index.T, edge_weight.unsqueeze(-1)], dim=-1)
    if edge_time is not None and edge_weight is not None:
        edge = torch.cat([edge_index.T, edge_time.unsqueeze(-1),
                          edge_weight.unsqueeze(-1)], dim=-1)
    else:
        edge = edge_index.T

    # Get the unique indices based on the hashes
    edge = torch.unique(edge, sorted=False, dim=0)

    # Update the edge attributes based on unique indices
    edge_index = edge[:, :2].T.contiguous().long()
    if edge_time is not None and edge_weight is not None:
        edge_time = edge[:, 2].long()
        edge_weight = edge[:, 3].float()
    if edge_weight is not None and edge_time is None:
        edge_weight = edge[:, 2].float()
    if edge_time is not None and edge_weight is None:
        edge_time = edge[:, 2].long()

    return edge_index.contiguous(), edge_time, edge_weight


def to_undirected(edge_index, edge_time, edge_weight=None):
    # Ensure that edge_time and edge_weight have the same length
    assert edge_index.shape[1] == edge_time.shape[0]
    assert edge_weight is None or edge_index.shape[1] == edge_weight.shape[0]

    # Create tensors for reversed edges directly during concatenation
    all_edge_index = torch.cat(
        [edge_index, torch.flip(edge_index, dims=[0])], dim=1)
    all_edge_time = torch.cat([edge_time, edge_time])
    if edge_weight is not None:
        all_edge_weight = torch.cat([edge_weight, edge_weight])

    # Remove duplicates
    return remove_duplicate_edges(
        all_edge_index, all_edge_time, all_edge_weight)
