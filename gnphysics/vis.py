# import torch.utils.data as data
# from torch.utils.data import DataLoader
import numpy as np
import networkx as nx
import torch
import torch.optim as optim
import os
import matplotlib.pyplot as plt

from datetime import datetime

from dataset_loader import GNPhysicsDataset

# from utils import *
from tqdm import tqdm
import argparse
import copy


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Trainer For GN Based Physics Engine")
    parser.add_argument(
        "--path-train",
        type=str,
        default="",
        help="Path of training set in .npz format",
    )
    parser.add_argument(
        "--path-eval",
        type=str,
        default="",
        help="Path of evaluation set in .npz format",
    )
    parser.add_argument(
        "--graph-feat-size", type=int, default=10, help="Graph feature size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Computing device, either 'cuda' or 'cpu'",
    )
    parser.add_argument(
        "--model", type=str, default="", help="Path for loading model in .pth format"
    )
    parser.add_argument(
        "--num-epoch", type=int, default=300, help="Number of training epochs"
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    dset_train = GNPhysicsDataset(args.path_train)

    graph_feat_size = args.graph_feat_size
    node_feat_size = dset_train.node_feat_size
    edge_feat_size = dset_train.edge_feat_size

    G = nx.MultiDiGraph()
    G.graph["feat"] = torch.zeros(1, graph_feat_size).to(device)

    save_dir = os.path.join("./logs", "runs", datetime.now().strftime("%B%d_%H:%M:%S"))

    frame = 200
    G = dset_train.get_state_graph_by_idx(10, G, device)

    node_color = []
    for node in G.nodes:
        degree = G.degree(node)

        x = 20
        d = 10

        if degree <= d:
            color = "#fcb045"
        elif degree > d  and degree <= d+x:
            color = "#fd7435"
        elif degree > d+x and degree <= d+2*x:
            color = "#fd4c2a"
        elif degree > d+2*x and degree <= d+3*x:
            color = "#fd1d1d"
        elif degree > d+3*x and degree <= d+4*x:
            color = "#c22b65"
        else:
            color = "#833ab4"

        node_color.append(copy.deepcopy(color))

    # nx.draw_kamada_kawai(
    #     G,
    #     with_labels=True,
    #     node_color=node_color,
    #     node_size=200,
    #     alpha=0.9,
    #     edge_color="#575757",
    #     linewidths=None,
    #     font_weight="bold",
    #     font_size=8,
    # )
    # if save_dir is not None:
    #     plt.savefig(save_dir)
    # else:
    #     plt.show()


    plt.figure(figsize=(7.5, 7.5))
    pos = nx.random_layout(G)
    # nx.draw_networkx_nodes(G, pos, node_color="r", node_size=100, alpha=1)
    nx.draw_networkx_nodes(
        G,
        pos,
        label=True,
        node_color=node_color,
        node_size=200,
        alpha=0.9,
    )
    ax = plt.gca()
    for e in G.edges:
        ax.annotate(
            "",
            xy=pos[e[0]],
            xycoords="data",
            xytext=pos[e[1]],
            textcoords="data",
            arrowprops=dict(
                arrowstyle="->",
                color="#575757",
                alpha=0.25,
                shrinkA=5,
                shrinkB=5,
                patchA=None,
                patchB=None,
                connectionstyle="arc3,rad=rrr".replace("rrr", str(0.01 * e[2])),
            ),
        )

    plt.axis("off")
    plt.show()

