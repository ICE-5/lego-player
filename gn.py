import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


_node_feat_size = 128
_edge_feat_size = 128
_graph_feat_size = 128


class EdgeBlock(nn.Module):
    def __init__(self, graph_feat_size, node_feat_size, edge_feat_size):
        super(EdgeBlock, self).__init__()
        self.f_e = nn.Sequential(
            nn.Linear(graph_feat_size + 2 * node_feat_size + edge_feat_size, 256),
            nn.ReLU(inplace = True),
            nn.Linear(256,256),
            nn.ReLU(inplace = True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256,edge_feat_size),
        )
        
    def forward(self, g, ns, nr, e):
        x = torch.cat([g, ns, nr, e], dim = -1)
        return self.f_e(x)

class NodeBlock(nn.Module):
    def __init__(self, graph_feat_size, node_feat_size, edge_feat_size):
        super(NodeBlock, self).__init__()
        self.f_n = nn.Sequential(
            nn.Linear(graph_feat_size + node_feat_size + edge_feat_size, 256),
            nn.ReLU(inplace = True),
            nn.Linear(256, 256),
            nn.ReLU(inplace = True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, node_feat_size),
        )

    def forward(self, g, n, e):
        x = torch.cat([g, n, e], dim = -1)
        return self.f_n(x)

    
class GraphBlock(nn.Module):
    def __init__(self, graph_feat_size, node_feat_size, edge_feat_size):
        super(GraphBlock, self).__init__()
        self.f_g = nn.Sequential(
            nn.Linear(graph_feat_size + node_feat_size + edge_feat_size, 256),
            nn.ReLU(inplace = True),
            nn.Linear(256, 256),
            nn.ReLU(inplace = True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, graph_feat_size),
        )

    def forward(self, g, n, e):
        x = torch.cat([g, n, e], dim = -1)
        return self.f_g(x)


class GNBlock(nn.Module):
    def __init__(self, graph_feat_size, node_feat_size, edge_feat_size):
        super(GNBlock, self).__init__()
        self.edge_block = EdgeBlock(graph_feat_size, node_feat_size, edge_feat_size)
        self.node_block = NodeBlock(graph_feat_size, node_feat_size, edge_feat_size)
        self.graph_block = GraphBlock(graph_feat_size, node_feat_size, edge_feat_size)
        
        self.graph_feat_size = graph_feat_size
        self.node_feat_size = node_feat_size
        self.edge_feat_size = edge_feat_size
        
    def forward(self, x):
        bs = x.graph['feat'].size(0)
        # edge update
        for u, v in x.edges():
            g = x.graph['feat']
            ns = x.node[u]['feat']
            nr = x.node[v]['feat']
            e = x[u][v]['feat']
            x[u][v]['temp_feat'] = self.edge_block(g, ns, nr, e)
            
        for u, v in x.edges():
            x[u][v]['feat'] = x[u][v]['temp_feat']
        
        # node update
        for u in x.nodes():
            g = x.graph['feat']
            n = x.node[u]['feat']
            pred = list(x.predecessors(u))
            n_e_agg = torch.zeros(bs, self.edge_feat_size)
            if x.graph['feat'].is_cuda:
                n_e_agg = n_e_agg.cuda()
            for v in pred:
                n_e_agg += x[v][u]['feat']
            x.node[u]['temp_feat'] = self.node_block(g, n, n_e_agg)
        
        for u in x.nodes():
            x.node[u]['feat'] = x.node[u]['temp_feat']
         
        # graph update
        e_agg = torch.zeros(bs, self.edge_feat_size)
        n_agg = torch.zeros(bs, self.node_feat_size)
        if x.graph['feat'].is_cuda:
            e_agg = e_agg.cuda()
            n_agg = n_agg.cuda()

        for u, v in x.edges():
            e_agg += x[u][v]['feat']
        for u in x.nodes():
            n_agg += x.node[u]['feat']
        g = x.graph['feat']
        x.graph['feat'] = self.graph_block(g, n_agg, e_agg)
        
        return x