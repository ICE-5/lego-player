import numpy as np
import networkx as nx
import pathlib
import torch

from typing import List, Optional


class GNPhysicsDataset:
    def __init__(self, path: pathlib.Path, graph_feat_size: int = 10) -> None:

        self.nodes = np.load(path)["nodes"]
        self.edges = np.load(path)["edges"]
        self.node_feats = np.load(path)["node_feats"]
        self.edge_feats = np.load(path)["edge_feats"]

        self.num_episode = np.max(self.nodes, axis=0)[0] + 1
        self.num_frame = np.max(self.nodes, axis=0)[1] + 1
        self.num_ingraph_node = np.max(self.nodes, axis=0)[2] + 1

        self.node_feat_size = self.node_feats.shape[1]
        self.edge_feat_size = self.edge_feats.shape[1]
        self.graph_feat_size = graph_feat_size

        print(f">>>>> # edges: {self.edges.shape[0]}")
        print(f">>>>> # nodes: {self.nodes.shape[0]}")
        print(f">>>>> # episodes: {self.num_episode}")
        print(f">>>>> # frames: {self.num_frame}")
        print(f">>>>> Avg. # edges per frame: {self.edges.shape[0] / (self.num_frame * self.num_episode)}")
        print(f">>>>> Avg. # nodes per frame: {self.nodes.shape[0] / (self.num_frame * self.num_episode)}")

    def __len__(self) -> int:
        return self.num_episode * self.num_frame

    def get_state_graph_by_idx(
        self, idx: int, G: nx.MultiDiGraph, device: torch.device,
    ) -> Optional[nx.MultiDiGraph]:
        episode = idx // self.num_frame
        frame = idx % self.num_frame

        G.clear()
        G.graph["feat"] = torch.zeros(1, self.graph_feat_size).to(device)

        if self.get_delta_node_feat(episode, frame, 0) is None:
            return None
        else:
            for i in range(self.num_ingraph_node):
                # load node feat
                G.add_node(
                    i,
                    feat=(
                        torch.Tensor(self.get_node_feat(episode, frame, i))
                        .view(1, -1)
                        .to(device)
                    ),
                )

                # query all edges in (episode, frame)
                edge_portion = self.edges[
                    np.where(
                        (self.edges[:, 0] == episode) * (self.edges[:, 1] == frame)
                    )
                ]
                edge_feat_portion = self.edge_feats[
                    np.where(
                        (self.edges[:, 0] == episode) * (self.edges[:, 1] == frame)
                    )
                ]
                num_edge = edge_portion.shape[0]
                for i in range(num_edge):
                    snode = edge_portion[i, 2]
                    rnode = edge_portion[i, 3]

                    G.add_edge(
                        snode,
                        rnode,
                        feat=torch.Tensor(edge_feat_portion[i, :])
                        .view(1, -1)
                        .to(device),
                    )
            return G

    def get_delta_state_graph_by_idx(
        self, idx: int, G: nx.MultiDiGraph, device: torch.device,
    ) -> Optional[nx.MultiDiGraph]:
        episode = idx // self.num_frame
        frame = idx % self.num_frame

        G.clear()
        G.graph["feat"] = torch.zeros(1, self.graph_feat_size).to(device)

        if self.get_delta_node_feat(episode, frame, 0) is None:
            return None
        else:
            for i in range(self.num_ingraph_node):
                # load node feat
                G.add_node(
                    i,
                    feat=(
                        torch.Tensor(self.get_delta_node_feat(episode, frame, i))
                        .view(1, -1)
                        .to(device)
                    ),
                )
            return G

    def get_node_feat(
        self, episode: int, frame: int, ingraph_node_id: int
    ) -> Optional[np.array]:
        node_feat = self.node_feats[
            np.where(
                (self.nodes[:, 0] == episode)
                * (self.nodes[:, 1] == frame)
                * (self.nodes[:, 2] == ingraph_node_id)
            )
        ]
        if node_feat.shape[0] == 0:
            return None
        else:
            return node_feat.flatten()

    def get_delta_node_feat(
        self, episode: int, frame: int, ingraph_node_id: int
    ) -> Optional[np.array]:
        curr_node_feat = self.get_node_feat(episode, frame, ingraph_node_id)
        next_node_feat = self.get_node_feat(episode, frame + 1, ingraph_node_id)
        if next_node_feat is None:
            return None
        else:
            return next_node_feat - curr_node_feat


if __name__ == "__main__":
    test_path_1 = "/Users/ice5/Desktop/lego-player/datasets/PLAIN/BRICK_GNPHYSICS_PLAIN_15_TRAIN.npz"
    test_path_2 = "/Users/ice5/Desktop/lego-player/datasets/PLAIN/BRICK_GNPHYSICS_PLAIN_15_EVAL.npz"
    dset_1 = GNPhysicsDataset(test_path_1)
    dset_2 = GNPhysicsDataset(test_path_2)

