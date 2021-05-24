import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class EdgeBlock(nn.Module):
    def __init__(self, graph_feat_size, node_feat_size, edge_feat_size):
        super(EdgeBlock, self).__init__()
        self.f_e = nn.Sequential(
            nn.Linear(graph_feat_size + 2 * node_feat_size + edge_feat_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, edge_feat_size),
        )

    def forward(self, g, ns, nr, e):
        x = torch.cat([g, ns, nr, e], dim=-1)
        return self.f_e(x)


class NodeBlock(nn.Module):
    def __init__(self, graph_feat_size, node_feat_size, edge_feat_size):
        super(NodeBlock, self).__init__()
        self.f_n = nn.Sequential(
            nn.Linear(graph_feat_size + node_feat_size + edge_feat_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, node_feat_size),
        )

    def forward(self, g, n, e):
        x = torch.cat([g, n, e], dim=-1)
        return self.f_n(x)


class GraphBlock(nn.Module):
    def __init__(self, graph_feat_size, node_feat_size, edge_feat_size):
        super(GraphBlock, self).__init__()
        self.f_g = nn.Sequential(
            nn.Linear(graph_feat_size + node_feat_size + edge_feat_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, graph_feat_size),
        )

    def forward(self, g, n, e):
        x = torch.cat([g, n, e], dim=-1)
        return self.f_g(x)


class GNBlock(nn.Module):
    def __init__(self, graph_feat_size, node_feat_size, edge_feat_size, device):
        super(GNBlock, self).__init__()
        self.edge_block = EdgeBlock(graph_feat_size, node_feat_size, edge_feat_size)
        self.node_block = NodeBlock(graph_feat_size, node_feat_size, edge_feat_size)
        self.graph_block = GraphBlock(graph_feat_size, node_feat_size, edge_feat_size)

        self.graph_feat_size = graph_feat_size
        self.node_feat_size = node_feat_size
        self.edge_feat_size = edge_feat_size

        self.device = device

    def forward(self, x):
        bs = x.graph["feat"].size(0)

        # edge update
        for u, v, k in x.edges(keys=True):
            g = x.graph["feat"]
            ns = x.nodes[u]["feat"]
            nr = x.nodes[v]["feat"]
            e = x[u][v][k]["feat"]  # note the difference between Graph and MultiDiGraph
            x[u][v][k]["temp_feat"] = self.edge_block(g, ns, nr, e)

        for u, v, k in x.edges(keys=True):
            x[u][v][k]["feat"] = x[u][v][k]["temp_feat"]

        # node update
        for node in x.nodes():
            g = x.graph["feat"]
            n = x.nodes[node]["feat"]
            n_e_agg = torch.zeros(bs, self.edge_feat_size).to(self.device)
            for u, v, k in x.in_edges(keys=True):
                n_e_agg += x[u][v][k]["feat"]

            x.nodes[node]["temp_feat"] = self.node_block(g, n, n_e_agg)

        for node in x.nodes():
            x.nodes[node]["feat"] = x.nodes[node]["temp_feat"]

        # graph update
        e_agg = torch.zeros(bs, self.edge_feat_size).to(self.device)
        n_agg = torch.zeros(bs, self.node_feat_size).to(self.device)

        for u, v, k in x.edges(keys=True):
            e_agg += x[u][v][k]["feat"]
        for node in x.nodes():
            n_agg += x.nodes[node]["feat"]
        g = x.graph["feat"]
        x.graph["feat"] = self.graph_block(g, n_agg, e_agg)

        return x


class FFGN(nn.Module):
    def __init__(self, graph_feat_size, node_feat_size, edge_feat_size, device):
        super(FFGN, self).__init__()
        self.GN1 = GNBlock(graph_feat_size, node_feat_size, edge_feat_size, device)
        self.GN2 = GNBlock(
            graph_feat_size * 2, node_feat_size * 2, edge_feat_size * 2, device
        )
        self.linear = nn.Linear(node_feat_size * 2, node_feat_size)

    def forward(self, G_in):
        G = G_in.copy()
        G = self.GN1(G)
        # Graph concatenate
        G.graph["feat"] = torch.cat([G.graph["feat"], G_in.graph["feat"]], dim=-1)

        # node
        for node in G.nodes():
            G.nodes[node]["feat"] = torch.cat(
                [G.nodes[node]["feat"], G_in.nodes[node]["feat"]], dim=-1
            )

        # edge
        for u, v, k in G.edges(keys=True):
            G[u][v][k]["feat"] = torch.cat(
                [G[u][v][k]["feat"], G_in[u][v][k]["feat"]], dim=-1
            )
        G = self.GN2(G)

        for node in G.nodes():
            G.nodes[node]["feat"] = self.linear(G.nodes[node]["feat"])
        # use a linear layer to change back to original node feature size
        return G
