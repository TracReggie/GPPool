import torch
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, CitationFull, Flickr, Reddit2, AmazonProducts
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.transforms import RandomNodeSplit, ToUndirected


def return_dataset(dataset_name: str):
    if dataset_name == 'cora':
        dataset = Planetoid(root='/home/zq2/data', name='Cora')
        data = dataset[0]

    if dataset_name == 'citeseer':
        dataset = Planetoid(root='/home/zq2/data', name='CiteSeer')
        data = dataset[0]

    if dataset_name == 'pubmed':
        dataset = Planetoid(root='/home/zq2/data', name='PubMed')
        data = dataset[0]

    if dataset_name == 'photo':
        T = RandomNodeSplit('test_rest', num_train_per_class=20, num_val=240)
        dataset = Amazon(root='/home/zq2/data', name='Photo', transform=T)
        data = dataset[0]

    if dataset_name == 'dblp':
        T = RandomNodeSplit('test_rest', num_train_per_class=20, num_val=120)
        dataset = CitationFull(root='/home/zq2/data/CitationFull', name='DBLP', transform=T)
        data = dataset[0]

    if dataset_name == 'cs':
        T = RandomNodeSplit('test_rest', num_train_per_class=20, num_val=450)
        dataset = Coauthor(root='/home/zq2/data', name='CS', transform=T)
        data = dataset[0]

    if dataset_name == 'physics':
        T = RandomNodeSplit('test_rest', num_train_per_class=20, num_val=150)
        dataset = Coauthor(root='/home/zq2/data', name='Physics', transform=T)
        data = dataset[0]

    if dataset_name == 'flickr':
        T = RandomNodeSplit('test_rest', num_train_per_class=20, num_val=210)
        dataset = Flickr(root='/home/zq2/data/Flickr', transform=T)
        data = dataset[0]

    if dataset_name == 'arxiv':
        dataset = PygNodePropPredDataset('ogbn-arxiv', '/home/zq2/data', ToUndirected())
        data = dataset[0]
        data.y = data.y.squeeze(1)
        trans = RandomNodeSplit('test_rest', num_train_per_class=20, num_val=1200)
        data = trans(data)

    data.num_classes = dataset.num_classes

    return data
