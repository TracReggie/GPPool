import numpy as np
import torch
from torch_geometric.nn import GCNConv, APPNP, SAGEConv, Linear

from graph_tools import GCN_adj



class GCN(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_layers, dropout):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hid_dim))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hid_dim, hid_dim))

        self.convs.append(GCNConv(hid_dim, out_dim))

        self.dropout = torch.nn.Dropout(dropout)
        self.act = torch.nn.ReLU()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = self.act(x)
            x = self.dropout(x)

        x = self.convs[-1](x, edge_index)

        return x

    def inference(self, x, edge_index):
        x, edge_index = x.cpu(), edge_index.cpu()
        adj_sym = GCN_adj(edge_index)

        weights = [(conv.lin.weight.t().cpu().detach().numpy(), conv.bias.cpu().detach().numpy())
                   for conv in self.convs]

        for i, (weight_W, weight_B) in enumerate(weights):
            x = adj_sym @ x @ weight_W + weight_B  # AXW + B
            x = np.clip(x, 0, None) if i < len(weights) - 1 else x  # Relu()

        x = torch.from_numpy(x)

        return x



class Net_APPNP(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_layers, dropout):
        super(Net_APPNP, self).__init__()
        self.linears = torch.nn.ModuleList()
        self.linears.append(Linear(in_dim, hid_dim))
        self.linears.append(Linear(hid_dim, hid_dim))
        self.linears.append(Linear(hid_dim, out_dim))
        self.appnp = APPNP(num_layers, 0.2)

        self.dropout = torch.nn.Dropout(dropout)
        self.act = torch.nn.ReLU()
        self.num_layers = num_layers

    def reset_parameters(self):
        for linear in self.linears:
            linear.reset_parameters()

    def forward(self, x, edge_index):
        for linear in self.linears[:-1]:
            x = linear(x)
            x = self.act(x)
            x = self.dropout(x)

        x = self.linears[-1](x)
        x = self.appnp(x, edge_index)

        return x

    def inference(self, x, edge_index):
        adj_sym = GCN_adj(edge_index)

        weights = [(linear.weight.t().cpu().detach().numpy(), linear.bias.cpu().detach().numpy())
                   for linear in self.linears]

        for i, (weight_W, weight_B) in enumerate(weights):
            x = x @ weight_W + weight_B
            x = np.clip(x, 0, None) if i < len(weights) - 1 else x  # Relu()

        x0 = x
        x0 = x0.numpy()

        for _ in range(self.num_layers):
            x = 0.8 * (adj_sym @ x) + 0.2 * x0

        x = torch.from_numpy(x)

        return x



class SAGE(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_layers, dropout):
        super(SAGE, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hid_dim))

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hid_dim, hid_dim))

        self.convs.append(SAGEConv(hid_dim, out_dim))

        self.dropout = torch.nn.Dropout(dropout)
        self.act = torch.nn.ReLU()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = self.act(x)
            x = self.dropout(x)

        x = self.convs[-1](x, edge_index)

        return x

    def inference(self, x_all, test_loader, device):
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in test_loader:
                x = x_all[batch.n_id].to(device)
                x = conv(x, batch.edge_index.to(device))
                if i != len(self.convs) - 1:
                    x = self.act(x)
                xs.append(x[:batch.batch_size].cpu())

            x_all = torch.cat(xs, dim=0)

        return x_all
