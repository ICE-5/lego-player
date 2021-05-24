# import torch.utils.data as data
# from torch.utils.data import DataLoader
import numpy as np
import networkx as nx
import torch
import torch.optim as optim
import os

from tensorboardX import SummaryWriter
from datetime import datetime

from dataset_loader import GNPhysicsDataset
from model import FFGN
from utils import *

# from utils import *
from tqdm import tqdm
import argparse


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
    dset_eval = GNPhysicsDataset(args.path_eval)

    graph_feat_size = args.graph_feat_size
    node_feat_size = dset_train.node_feat_size
    edge_feat_size = dset_train.edge_feat_size

    G = nx.MultiDiGraph()
    G_target = nx.MultiDiGraph()
    G.graph["feat"] = torch.zeros(1, graph_feat_size).to(device)
    G_target.graph["feat"] = torch.zeros(1, graph_feat_size).to(device)

    gn = FFGN(graph_feat_size, node_feat_size, edge_feat_size, device=device)

    optimizer = optim.Adam(gn.parameters(), lr=1e-4)
    schedular = optim.lr_scheduler.StepLR(optimizer, 5e4, gamma=0.975)

    save_dir = os.path.join("./logs", "runs", datetime.now().strftime("%B%d_%H:%M:%S"))
    writer = SummaryWriter(save_dir)
    step = 0

    for epoch in tqdm(range(args.num_epoch), "Epoch"):
        idxs = np.arange(len(dset_train))
        np.random.shuffle(idxs)

        for idx in tqdm(idxs, "Frame"):
            G = dset_train.get_state_graph_by_idx(idx, G, device)
            G_out = gn(G)
            G_target = dset_train.get_delta_state_graph_by_idx(idx, G_target, device)
            loss = calc_graph_loss(G_out, G_target)
            loss.backward()

            if step % 10 == 0:
                writer.add_scalar("loss/train_loss", loss.data.item(), step)
            step += 1
            for param in gn.parameters():
                if not param.grad is None:
                    param.grad.clamp_(-3, 3)

            optimizer.step()
            schedular.step()
            if step % 5000 == 0:
                torch.save(gn.state_dict(), save_dir + "/model_{}.pth".format(step))

        eval_loss = 0.0
        for idx in range(len(dset_eval)):
            G = dset_train.get_state_graph_by_idx(idx, G, device)
            G_out = gn(G)
            G_target = dset_train.get_delta_state_graph_by_idx(idx, G_target, device)
            loss = calc_graph_loss(G_out, G_target)
            eval_loss += loss.data.item()
        writer.add_scalar("loss/eval_loss", eval_loss / len(dset_eval), step)

