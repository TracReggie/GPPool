import argparse
import torch
from torch_geometric.datasets import CitationFull
from torch_geometric.transforms import RandomNodeSplit

from experiments import GCN_experiments, APPNP_experiments, SAGE_experiments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_name', type=str, default="dblp")

    parser.add_argument('--inductive', type=bool, default=False)
    parser.add_argument('--num_subgraphs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.00005)

    parser.add_argument('--y_setting', type=str, default="argmax")  # argmax, same, softmax
    parser.add_argument('--gppool', type=bool, default=True)
    parser.add_argument('--divide_method', type=str, default="metis") # metis, kmeans, SC
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=3)
    args = parser.parse_args()
    print(args)

    device = torch.device('cuda:1')

    T = RandomNodeSplit('test_rest', num_train_per_class=20, num_val=120)
    dataset = CitationFull(root='/home/zq2/data/CitationFull', name='DBLP', transform=T)
    data = dataset[0]
    data.num_classes = dataset.num_classes

    GCN_experiments(args, data, device)
    # APPNP_experiments(args, data, device)
    # SAGE_experiments(args, data, device)

    print('============================================')
    print(args)


if __name__ == "__main__":
    main()
