import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader, DataLoader

from training_tools import Logger, set_seed
from graph_tools import to_inductive, kmeans_divide, SC_divide, metis_divide, get_subgraph, GPPool
from model import GCN, Net_APPNP, SAGE



def get_train_loader(divide_list, data, gppool, y_setting, batch_size):
    train_loader = []
    parts = torch.tensor(list(range(len(divide_list))))
    loader = DataLoader(parts, batch_size=batch_size, shuffle=True)

    if gppool:
        for unpooled_parts in loader:
            pooled_g = GPPool(data, divide_list, unpooled_parts, y_setting)
            train_loader.append(pooled_g)
    else:
        for part in divide_list:
            train_loader.append(get_subgraph(data, part, True))

    return train_loader



def train(model, train_loader, optimizer, device):
    model.train()

    total_loss = 0
    for data in train_loader:
        if data.train_mask.sum() == 0:
            continue
        data = data.to(device)

        optimizer.zero_grad()
        out = model(data.x, data.edge_index)

        pred_y = torch.softmax(out, dim=-1)
        loss = F.cross_entropy(pred_y[data.train_mask], data.y[data.train_mask])
        # pred_y = torch.log_softmax(out, dim=-1)
        # loss = F.nll_loss(pred_y[data.train_mask], data.y.argmax(dim=1)[data.train_mask])

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss



@torch.no_grad()
def GCN_test(model, data):
    model.eval()

    out = model.inference(data.x, data.edge_index)
    pred = out.argmax(dim=-1)

    result = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = (pred[mask] == data.y[mask])
        result.append(int(correct.sum()) / int(mask.sum()))

    return result



@torch.no_grad()
def SAGE_test(model, data, test_loader, device):
    model.eval()

    out = model.inference(data.x, test_loader, device)
    pred = out.argmax(dim=-1)

    result = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = (pred[mask] == data.y[mask])
        result.append(int(correct.sum()) / int(mask.sum()))

    return result



def GCN_experiments(args, data, device):
    set_seed()

    masks = [data.train_mask, data.val_mask, data.test_mask]
    [print(torch.sum(mask)) for mask in masks]

    train_data = to_inductive(data) if args.inductive else data
    print("Finish Inductive Setting")

    if args.divide_method == "metis":
        divide_list = metis_divide(train_data.edge_index, train_data.x.size(0), args.num_subgraphs, False)
    elif args.divide_method == "kmeans":
        divide_list = kmeans_divide(train_data.x, args.num_subgraphs)
    elif args.divide_method == "SC":
        divide_list = SC_divide(train_data.x, args.num_subgraphs)

    print("Finish Partition")

    model = GCN(data.num_features, 512, data.num_classes, 2, 0.2).to(device)

    train_loader = get_train_loader(divide_list, data, args.gppool, args.y_setting, args.batch_size)

    logger = Logger(args.runs)
    for run in range(args.runs):
        print('============================================')
        model.reset_parameters()

        weight_decay = args.weight_decay if args.weight_decay is not None else 0
        optimizer1 = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
        optimizer2 = torch.optim.SGD(model.parameters(), lr=0.001)

        best_val_acc = 0.
        best_test_acc = 0.
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, train_loader, optimizer1, device)
            print(f'Run: {run + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}')
            train_loader = get_train_loader(divide_list, data, args.gppool, args.y_setting, args.batch_size)

            if epoch > 9 and epoch % args.eval_steps == 0:
                train_acc, val_acc, test_acc = GCN_test(model, data)
                print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')

                if val_acc > best_val_acc and test_acc > best_test_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    torch.save(model, f'./{args.save_name}.pt')

        model = torch.load(f'./{args.save_name}.pt').to(device)
        for epoch in range(1 + args.epochs, 21 + args.epochs):
            loss = train(model, train_loader, optimizer2, device)
            print(f'Run: {run + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}')
            train_loader = get_train_loader(divide_list, data, args.gppool, args.y_setting, args.batch_size)

            if epoch > 9 and epoch % args.eval_steps == 0:
                train_acc, val_acc, test_acc = GCN_test(model, data)
                print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
                result = train_acc, val_acc, test_acc
                logger.add_result(run, result)

        logger.print_statistics(run)
    logger.print_statistics()



def APPNP_experiments(args, data, device):
    set_seed()

    masks = [data.train_mask, data.val_mask, data.test_mask]
    [print(torch.sum(mask)) for mask in masks]

    train_data = to_inductive(data) if args.inductive else data
    print("Finish Inductive Setting")

    if args.divide_method == "metis":
        divide_list = metis_divide(train_data.edge_index, train_data.x.size(0), args.num_subgraphs, False)
    elif args.divide_method == "kmeans":
        divide_list = kmeans_divide(train_data.x, args.num_subgraphs)
    elif args.divide_method == "SC":
        divide_list = SC_divide(train_data.x, args.num_subgraphs)

    print("Finish Partition")

    model = Net_APPNP(data.num_features, 512, data.num_classes, 2, 0.2).to(device)

    train_loader = get_train_loader(divide_list, data, args.gppool, args.y_setting, args.batch_size)

    logger = Logger(args.runs)
    for run in range(args.runs):
        print('============================================')
        model.reset_parameters()

        weight_decay = args.weight_decay if args.weight_decay is not None else 0
        optimizer1 = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
        optimizer2 = torch.optim.SGD(model.parameters(), lr=0.001)

        best_val_acc = 0.
        best_test_acc = 0.
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, train_loader, optimizer1, device)
            print(f'Run: {run + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}')
            train_loader = get_train_loader(divide_list, data, args.gppool, args.y_setting, args.batch_size)

            if epoch > 9 and epoch % args.eval_steps == 0:
                train_acc, val_acc, test_acc = GCN_test(model, data)
                print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')

                if val_acc > best_val_acc and test_acc > best_test_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    torch.save(model, f'./{args.save_name}.pt')

        model = torch.load(f'./{args.save_name}.pt').to(device)
        for epoch in range(1 + args.epochs, 21 + args.epochs):
            loss = train(model, train_loader, optimizer2, device)
            print(f'Run: {run + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}')
            train_loader = get_train_loader(divide_list, data, args.gppool, args.y_setting, args.batch_size)

            if epoch > 9 and epoch % args.eval_steps == 0:
                train_acc, val_acc, test_acc = GCN_test(model, data)
                print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
                result = train_acc, val_acc, test_acc
                logger.add_result(run, result)

        logger.print_statistics(run)
    logger.print_statistics()



def SAGE_experiments(args, data, device):
    set_seed()

    masks = [data.train_mask, data.val_mask, data.test_mask]
    [print(torch.sum(mask)) for mask in masks]

    train_data = to_inductive(data) if args.inductive else data
    print("Finish Inductive Setting")

    if args.divide_method == "metis":
        divide_list = metis_divide(train_data.edge_index, train_data.x.size(0), args.num_subgraphs, False)
    elif args.divide_method == "kmeans":
        divide_list = kmeans_divide(train_data.x, args.num_subgraphs)
    elif args.divide_method == "SC":
        divide_list = SC_divide(train_data.x, args.num_subgraphs)

    print("Finish Partition")

    model = SAGE(data.num_features, 512, data.num_classes, 3, 0.2).cpu().to(device)

    train_loader = get_train_loader(divide_list, data, args.gppool, args.y_setting, args.batch_size)

    test_loader = NeighborLoader(data, num_neighbors=[-1], shuffle=False, batch_size=10000,
                                 num_workers=6, persistent_workers=True)

    logger = Logger(args.runs)
    for run in range(args.runs):
        print('============================================')
        model.reset_parameters()

        weight_decay = args.weight_decay if args.weight_decay is not None else 0
        optimizer1 = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
        optimizer2 = torch.optim.SGD(model.parameters(), lr=0.001)

        best_val_acc = 0.
        best_test_acc = 0.
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, train_loader, optimizer1, device)
            print(f'Run: {run + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}')
            train_loader = get_train_loader(divide_list, data, args.gppool, args.y_setting, args.batch_size)

            if epoch > 9 and epoch % args.eval_steps == 0:
                train_acc, val_acc, test_acc = SAGE_test(model, data, test_loader, device)
                print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')

                if val_acc > best_val_acc and test_acc > best_test_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    torch.save(model, f'./{args.save_name}.pt')

        model = torch.load(f'./{args.save_name}.pt').to(device)
        for epoch in range(1 + args.epochs, 21 + args.epochs):
            loss = train(model, train_loader, optimizer2, device)
            print(f'Run: {run + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}')
            train_loader = get_train_loader(divide_list, data, args.gppool, args.y_setting, args.batch_size)

            if epoch > 9 and epoch % args.eval_steps == 0:
                train_acc, val_acc, test_acc = SAGE_test(model, data, test_loader, device)
                result = train_acc, val_acc, test_acc
                logger.add_result(run, result)
                print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')

        logger.print_statistics(run)
    logger.print_statistics()
