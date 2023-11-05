import numpy as np
import torch
import torch.utils.data
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, coalesce, remove_self_loops,sort_edge_index, one_hot
from sklearn.cluster import KMeans, SpectralClustering

import torch_geometric.typing as pyg_typing
from torch_geometric.utils.sparse import index2ptr
from torch_geometric.utils.map import map_index



def to_inductive(data):
    inductive_data = data.clone()
    inductive_data.x = data.x[data.train_mask]
    inductive_data.y = data.y[data.train_mask]
    inductive_data.edge_index, _ = subgraph(data.train_mask, data.edge_index, None, True, data.x.size(0))
    inductive_data.num_nodes = data.train_mask.sum().item()
    inductive_data.train_mask = data.train_mask[data.train_mask]
    inductive_data.val_mask = torch.tensor([0 for _ in range(inductive_data.num_nodes)], dtype=torch.bool)
    inductive_data.test_mask = torch.tensor([0 for _ in range(inductive_data.num_nodes)], dtype=torch.bool)

    return inductive_data


def GCN_adj(edge_index, self_loops=True):
    adj = SparseTensor(row=edge_index[0], col=edge_index[1])
    adj = adj.set_diag() if self_loops else adj

    deg = adj.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    adj_sym = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
    adj_sym = adj_sym.to_scipy(layout='csr')

    return adj_sym


def metis_divide(edge_index, num_nodes: int, num_parts: int, recursive: bool):
    print('Computing METIS partitioning...')
    row, col = sort_edge_index(edge_index, num_nodes=num_nodes)
    rowptr = index2ptr(row, size=num_nodes)
    cluster = None

    if pyg_typing.WITH_TORCH_SPARSE:
        try:
            cluster = torch.ops.torch_sparse.partition(
                rowptr.cpu(), col.cpu(), None, num_parts, recursive).to(edge_index.device)
        except (AttributeError, RuntimeError):
            pass

    if cluster is None and pyg_typing.WITH_METIS:
        cluster = pyg_typing.pyg_lib.partition.metis(
            rowptr.cpu(), col.cpu(), num_parts, recursive=recursive).to(edge_index.device)

    cluster = cluster.numpy()

    subgraph_idx = []
    for i in range(num_parts):
        subgraph_idx.append(((np.where(cluster == i))[0]).tolist())

    print('Done!')

    return subgraph_idx


def kmeans_divide(data_x, num_cluster):
    print('Computing KMeans partitioning...')
    clustering_method = KMeans(n_clusters=num_cluster, n_init=20)

    pred = clustering_method.fit_predict(data_x)

    subgraph_idx = []
    for i in range(num_cluster):
        subgraph_idx.append(((np.where(pred == i))[0]).tolist())

    print('Done!')

    return subgraph_idx


def SC_divide(data_x, num_cluster):
    print('Computing Spectral Clustering partitioning...')
    clustering_method = SpectralClustering(n_clusters=num_cluster)

    pred = clustering_method.fit_predict(data_x)

    subgraph_idx = []
    for i in range(num_cluster):
        subgraph_idx.append(((np.where(pred == i))[0]).tolist())

    print('Done!')

    return subgraph_idx


def get_subgraph(data, nodes_index, relabel_nodes):
    sub_x = data.x[nodes_index]
    sub_y = data.y[nodes_index]
    sub_train_mask = data.train_mask[nodes_index]
    sub_val_mask = data.val_mask[nodes_index]
    sub_test_mask = data.test_mask[nodes_index]
    sub_edge_index, _ = subgraph(nodes_index, data.edge_index, None, relabel_nodes, data.x.size(0))

    induced_subgraph = Data(x=sub_x, y=sub_y, edge_index=sub_edge_index,
                            train_mask=sub_train_mask, val_mask=sub_val_mask, test_mask=sub_test_mask)

    return induced_subgraph


def GPPool(data, divide_list, unpooled_parts, y_setting):
    pooled_parts, unpooled_nodes = [], []
    for i, part in enumerate(divide_list):
        if i in unpooled_parts:
            unpooled_nodes.extend(part)
        else:
            pooled_parts.append(part)

    cluster = torch.zeros(data.x.size(0), dtype=torch.long)
    cluster[unpooled_nodes] = torch.tensor(range(len(unpooled_nodes)), dtype=torch.long)

    retain_x = data.x[unpooled_nodes]
    x2 = [retain_x]

    data_y = one_hot(data.y, data.num_classes)
    retain_y = torch.softmax(data_y[unpooled_nodes], dim=1) if y_setting == "softmax" else data_y[unpooled_nodes]
    y2 = [retain_y]

    retain_mask = data.train_mask[unpooled_nodes]
    train_mask2 = [retain_mask]

    for i, part in enumerate(pooled_parts):
        cluster[part] = torch.tensor(i + len(unpooled_nodes), dtype=torch.long)

        part_x = torch.mean(data.x[part], dim=0).unsqueeze(0)
        norm = torch.sqrt(torch.tensor(len(part)))
        x2.append(part_x * norm)

        mask = (data.train_mask | data.val_mask)[part]
        if mask.sum() > 0:
            part_y = data_y[part]
            part_y[~mask] = torch.Tensor([0 for _ in range(data.num_classes)])

            if y_setting == "argmax":
                part_mask = torch.tensor(1, dtype=torch.bool).view(1)
                label = torch.argmax(torch.sum(part_y, dim=0))
                part_y = one_hot(label.view(1), data.num_classes)
            elif y_setting == "same":
                part_mask = (torch.sum(torch.sum(part_y, dim=0).bool()) == 1).view(1)
                label = torch.argmax(torch.sum(part_y, dim=0))
                part_y = one_hot(label.view(1), data.num_classes)
            elif y_setting == "softmax":
                part_mask = torch.tensor(1, dtype=torch.bool).view(1)
                label = torch.sum(part_y, dim=0).view(1, -1)
                part_y = torch.softmax(label, dim=-1)
        else:
            part_y = torch.tensor([[0 for _ in range(data.num_classes)]])
            part_mask = torch.tensor(0, dtype=torch.bool).view(1)

        y2.append(part_y)
        train_mask2.append(part_mask)

    x2 = torch.cat(x2, dim=0)
    y2 = torch.cat(y2, dim=0)
    train_mask2 = torch.cat(train_mask2, dim=0)

    edge_index2 = cluster[data.edge_index.view(-1)].view(2, -1)
    edge_index2, _ = remove_self_loops(edge_index2)
    edge_index2 = coalesce(edge_index2)

    pooled_g = Data(x=x2, y=y2, edge_index=edge_index2, train_mask=train_mask2)

    return pooled_g
