import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader, DataLoader

from training_tools import Logger, set_seed
from graph_tools import kmeans_divide, SC_divide, metis_divide, get_subgraph, GPPool
from model import GCN, Net_APPNP, SAGE
from prepare_data import return_dataset



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
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss



@torch.no_grad()
def test(model, data, backbone, test_loader, device):
    model.eval()

    if backbone == "SAGE":
        out = model.inference(data.x, test_loader, device)
    else:
        out = model.inference(data.x, data.edge_index)

    pred = out.argmax(dim=-1)

    result = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = (pred[mask] == data.y[mask])
        result.append(int(correct.sum()) / int(mask.sum()))

    return result



def run_experiments(
        dataset, backbone, divide_method, num_subgraphs, y_setting,
        gppool, lr, weight_decay, epochs, eval_steps, runs, cuda):

    set_seed()
    data = return_dataset(f'{dataset}')
    device = torch.device(f'cuda:{cuda}')

    masks = [data.train_mask, data.val_mask, data.test_mask]
    [print(torch.sum(mask)) for mask in masks]

    if divide_method == "metis":
        divide_list = metis_divide(data.edge_index, data.x.size(0), num_subgraphs, False)
    elif divide_method == "kmeans":
        divide_list = kmeans_divide(data.x, num_subgraphs)
    elif divide_method == "SC":
        divide_list = SC_divide(data.x, num_subgraphs)

    if backbone == "GCN":
        model = GCN(data.num_features, 512, data.num_classes, 2, 0.2).to(device)
    elif backbone == "APPNP":
        model = Net_APPNP(data.num_features, 512, data.num_classes, 2, 0.2).to(device)
    elif backbone == "SAGE":
        model = SAGE(data.num_features, 512, data.num_classes, 3, 0.2).to(device)

    train_loader = get_train_loader(divide_list, data, gppool, y_setting, int(num_subgraphs / 4))
    test_loader = NeighborLoader(data, num_neighbors=[-1], shuffle=False, batch_size=15000,
                                 num_workers=16, persistent_workers=True)

    logger = Logger(runs)
    for run in range(runs):
        print('============================================')
        model.reset_parameters()

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(1, 1 + epochs):
            loss = train(model, train_loader, optimizer, device)
            print(f'Run: {run + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}')
            train_loader = get_train_loader(divide_list, data, gppool, y_setting, int(num_subgraphs / 4))

            if epoch > 19 and epoch % eval_steps == 0:
                train_acc, val_acc, test_acc = test(model, data, backbone, test_loader, device)
                print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
                result = train_acc, val_acc, test_acc
                logger.add_result(run, result)

        logger.print_statistics(run)
    logger.print_statistics()
