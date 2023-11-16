import json
import numpy as np
import pandas as pd
import torch
from torch import LongTensor
from torch_geometric.sampler import (
    NeighborSampler,
    NodeSamplerInput,
    SamplerOutput
    )

from datasets.temporal_graph import TemporalGraph


def dataframe_to_tensor(item_df: pd.DataFrame) -> torch.Tensor:
    """
    Convert a DataFrame into a PyTorch tensor.

    Args:
    - item_df (pd.DataFrame): Input DataFrame.

    Returns:
    - torch.Tensor: The resulting tensor.
    """
    tensors = []

    for column in item_df.columns:
        data = item_df[column].values

        # judge the type
        first_element = data[0]

        # PyTorch tensor
        if isinstance(first_element, (list, np.ndarray)):
            try:
                tensor_data = torch.from_numpy(np.array(data.tolist()))
                tensors.append(tensor_data)
            except ValueError:
                print(f"Failed to convert column {column}"
                      "to tensor. Skipping...")
                continue
        else:
            tensors.append(torch.tensor(data).reshape(-1, 1))

    final_tensor = torch.cat(tensors, dim=1)

    # Check if the tensor has only one row, if so, reshape to 1D
    if final_tensor.shape[0] == 1:
        final_tensor = final_tensor.squeeze(0)

    return final_tensor


def large_df_to_hdf5(df, h5_save_path, min_itemsize: dict = None,
                     chunk_size=10000):
    from pandas.api.types import is_list_like
    store = pd.HDFStore(h5_save_path)
    obj_list = []
    for col in df.columns:
        if is_list_like(df[col]):
            obj_list.append(col)

    for start in range(0, df.shape[0], chunk_size):
        end = min(start + chunk_size, df.shape[0])
        df_chunk = df.iloc[start:end].copy()

        for col in obj_list:
            df_chunk[col] = df_chunk[col].apply(json.dumps)

        # append to store
        store.append('data', df_chunk, format='table', data_columns=True,
                     index=False, min_itemsize=min_itemsize)

    store.close()


def get_neighbors(graph: TemporalGraph, num_neighbors: list, node_list:
                  LongTensor, node_time: LongTensor) -> SamplerOutput:
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

    return sampler_output
