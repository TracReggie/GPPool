import argparse
from experiments import run_experiments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="cora")
    parser.add_argument('--backbone', type=str, default="GCN")  # GCN, APPNP, SAGE
    parser.add_argument('--divide_method', type=str, default="metis") # metis, kmeans, SC
    parser.add_argument('--num_subgraphs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--y_setting', type=str, default="argmax")  # argmax, same, softmax

    parser.add_argument('--gppool', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.00005)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--cuda', type=int, default=1)
    args = parser.parse_args()
    print(args)

    run_experiments(args)

    print('============================================')
    print(args)


if __name__ == "__main__":
    main()