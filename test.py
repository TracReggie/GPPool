import argparse
from torch_geometric.datasets import Planetoid
from experiments import run_experiments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default="GCN")  # GCN, APPNP, SAGE
    parser.add_argument('--divide_method', type=str, default="metis") # metis, kmeans, SC
    parser.add_argument('--num_subgraphs', type=int, default=70) # 70, 70, 70
    parser.add_argument('--y_setting', type=str, default="argmax")  # argmax, same, softmax

    parser.add_argument('--gppool', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.00005)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--cuda', type=int, default=2)
    args = parser.parse_args()
    print(args)

    dataset = Planetoid(root='/home/zq2/data', name='Cora')
    data = dataset[0]

    run_experiments(
        data, dataset.num_classes, args.backbone, args.divide_method, args.num_subgraphs, args.y_setting,
        args.gppool, args.lr, args.weight_decay, args.epochs, args.eval_steps, args.runs, args.cuda)


    print('============================================')
    print(args)


if __name__ == "__main__":
    main()
