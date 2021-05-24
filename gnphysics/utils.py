import torch
import networkx as nx
import matplotlib.pyplot as plt
import pathlib


def calc_graph_loss(G: nx.MultiDiGraph, H: nx.MultiDiGraph) -> torch.Tensor:
    loss = 0
    for node in G.nodes:
        loss += torch.sum(
            torch.mean(
                G.nodes[node]["feat"][:, :-1].view(-1, 3) - H.nodes[node]["feat"][:, :-1].view(-1, 3)
            )
            ** 2,
            dim=-1,
        )
    return loss / len(G)

# TODO
def draw_graph(G: nx.MultiDiGraph, save_dir: pathlib.Path=None):
    pass



def detach(G):
    G.graph['feat'] = G.graph['feat'].detach()
    for node in G.nodes():
        G.nodes[node]['feat'] = G.nodes[node]['feat'].detach()
    for edge in G.edges():
        G[edge[0]][edge[1]]['feat'] = G[edge[0]][edge[1]]['feat'].detach()
    return G