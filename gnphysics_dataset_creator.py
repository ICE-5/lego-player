import pybullet as p
import pybullet_data
import numpy as np

import copy
import time
import pathlib
import shutil
import argparse

from typing import List, Optional
from tqdm import tqdm

from designer import *


class GNPhysicsDatasetCreator:
    def __init__(
        self,
        designer: Designer,
        dataset_prefix: str,
        num_episode: int = 300,
        num_frame: int = 50,
        min_num_brick: int = 5,
        max_num_brick: int = 25,
        train_eval_split: float = 0.8,
        save_dir: Optional[pathlib.Path] = None,
    ) -> None:
        self.designer = designer
        self.dataset_prefix = dataset_prefix
        self.num_episode = num_episode
        self.num_frame = num_frame
        self.min_num_brick = min_num_brick
        self.max_num_brick = max_num_brick
        self.train_eval_split = train_eval_split
        self.save_dir = save_dir

    def generate_dataset(self):

        for num_brick in tqdm(
            range(self.min_num_brick, self.max_num_brick + 1),
            "Number of Brick Progress",
        ):
            # separate dataset for GN-based physics engine (GN Physics)
            dataset_name = self.dataset_prefix + "_" + str(num_brick)

            nodes_train, nodes_eval = [], []
            edges_train, edges_eval = [], []
            node_feats_train, node_feats_eval = [], []
            edge_feats_train, edge_feats_eval = [], []

            split = int(self.num_episode * self.train_eval_split)
            for episode in tqdm(range(self.num_episode), "Episode Progress"):
                if episode < split:
                    nodes_dummy = nodes_train
                    edges_dummy = edges_train
                    node_feats_dummy = node_feats_train
                    edge_feats_dummy = edge_feats_train
                    episode_dummy = copy.deepcopy(episode)
                else:
                    nodes_dummy = nodes_eval
                    edges_dummy = edges_eval
                    node_feats_dummy = node_feats_eval
                    edge_feats_dummy = edge_feats_eval
                    episode_dummy = copy.deepcopy(episode) - split

                # use test_simulator.py for debugging and visualization
                p.connect(p.DIRECT)

                p.setAdditionalSearchPath(pybullet_data.getDataPath())
                p.setPhysicsEngineParameter(numSolverIterations=10)
                p.setTimeStep(1.0 / 60.0)

                # generate new design
                self.designer.clear()
                self.designer.generate_design(num_brick=num_brick)

                for b in self.designer.built:
                    pos = b.anchor
                    rot = p.getQuaternionFromEuler(np.array(b.rotation) * np.pi / 2)
                    p.loadURDF("urdf/brick.urdf", pos, rot)
                p.loadURDF("plane.urdf", useMaximalCoordinates=True)

                # start simulation
                p.setGravity(0, 0, -10)
                for frame in range(self.num_frame):
                    p.stepSimulation()
                    time.sleep(1.0 / 240.0)

                    # remember the ground!
                    for objectID in range(num_brick + 1):
                        pos, ort = p.getBasePositionAndOrientation(objectID)
                        rot = p.getEulerFromQuaternion(ort)
                        vel_linear, vel_angular = p.getBaseVelocity(objectID)
                        if objectID < num_brick:
                            mass = 1.0
                        else:
                            mass = 9999999.0

                        nodes_dummy.append([episode_dummy, frame, objectID])
                        node_feats_dummy.append(
                            pos + rot + vel_linear + vel_angular + (mass,)
                        )

                    contact_list = p.getContactPoints()
                    for contact in contact_list:
                        edges_dummy.append(
                            [episode_dummy, frame, contact[1], contact[2]]
                        )
                        edge_feats_dummy.append(
                            contact[5]  # positionA
                            + contact[6]  # positionB
                            + (
                                contact[9],  # normalForce
                                contact[10],  # friction1
                                contact[12],  # friction2
                            )
                        )

                p.disconnect()

            np.savez(
                self.save_dir / str(dataset_name + "_TRAIN"),
                nodes=np.array(nodes_train),
                node_feats=np.array(node_feats_train),
                edges=np.array(edges_train),
                edge_feats=np.array(edge_feats_train),
            )
            np.savez(
                self.save_dir / str(dataset_name + "_EVAL"),
                nodes=np.array(nodes_eval),
                node_feats=np.array(node_feats_eval),
                edges=np.array(edges_eval),
                edge_feats=np.array(edge_feats_eval),
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "GN Based Physics Engine Dataset Creator For Lego Player"
    )
    parser.add_argument(
        "--brick-type",
        type=str,
        default="plain",
        help="Type of brick, either 'plain' or 'lego'",
    )
    parser.add_argument(
        "--arena-length",
        type=float,
        default=1.0,
        help="Arena length for designer to contain width of generated design",
    )
    parser.add_argument(
        "--num-episode",
        type=int,
        default=200,
        help="Number of episode per each separate dataset",
    )
    parser.add_argument(
        "--num-frame",
        type=int,
        default=5,
        help="Number of frame to run in simulator per each episode",
    )
    parser.add_argument(
        "--min-num-brick",
        type=int,
        default=5,
        help="Minimum number of bricks for separate dataset generation",
    )
    parser.add_argument(
        "--max-num-brick",
        type=int,
        default=6,
        help="Maximum number of bricks for separate dataset generation",
    )
    parser.add_argument(
        "--brick-extents",
        nargs="+",
        type=float,
        default=[0.4, 0.2, 0.1],
        help="Brick extents for designer",
    )
    parser.add_argument(
        "--safe-margin",
        type=float,
        default=0.1,
        help="Safe margin for designer in deciding brick placement",
    )

    args = parser.parse_args()

    if args.brick_type == "plain":
        safe_margin = args.safe_margin
        is_modular = False
        mod_unit = None
        rotate_prob = [0.1, 0.1, 0.5]
        dataset_prefix = "BRICK_GNPHYSICS_PLAIN"
        save_dir = pathlib.Path(__file__).parent.absolute() / "datasets" / "PLAIN"

    elif args.brick_type == "lego":
        is_modular = True
        mod_unit = min(args.brick_extents)
        rotate_prob = [0.0, 0.0, 0.5]
        safe_margin = round(args.safe_margin / mod_unit) * mod_unit
        dataset_prefix = "BRICK_GNPHYSICS_LEGO"
        save_dir = pathlib.Path(__file__).parent.absolute() / "datasets" / "LEGO"

    desigher = Designer(
        arena_length=args.arena_length,
        brick_extents=args.brick_extents,
        safe_margin=safe_margin,
        is_modular=is_modular,
        mod_unit=mod_unit,
        rotate_prob=rotate_prob,
    )
    dc = GNPhysicsDatasetCreator(
        desigher,
        dataset_prefix,
        num_episode=args.num_episode,
        num_frame=args.num_frame,
        min_num_brick=args.min_num_brick,
        max_num_brick=args.max_num_brick,
        save_dir=save_dir,
    )
    dc.generate_dataset()

