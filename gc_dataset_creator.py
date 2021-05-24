import pybullet as p
import pybullet_data
import numpy as np

import time
import pathlib
import shutil
import argparse

from typing import List, Optional
from tqdm import tqdm

from designer import *


class GCDatasetCreator:
    def __init__(
        self,
        designer: Designer,
        dataset_prefix: str,
        num_episode: int = 300,
        num_frame: int = 50,
        min_num_brick: int = 5,
        max_num_brick: int = 25,
        movement_threshold: float = 0.15,
        save_dir: Optional[pathlib.Path] = None,
    ) -> None:
        self.designer = designer
        self.dataset_prefix = dataset_prefix
        self.movement_threshold = movement_threshold
        self.num_episode = num_episode
        self.num_frame = num_frame
        self.min_num_brick = min_num_brick
        self.max_num_brick = max_num_brick
        self.save_dir = save_dir

    def generate_dataset(self):

        # combined dataset for graph classification (GC)
        combined_dataset_name = dataset_prefix
        combined_dataset_dir = self.save_dir / combined_dataset_name
        pathlib.Path(combined_dataset_dir).mkdir(parents=True, exist_ok=True)

        combined_node_id = 0
        combined_graph_id = 0
        combined_graphnode2id = {}

        combined_a = open(
            combined_dataset_dir / str(combined_dataset_name + "_A.txt"), "w"
        )
        combined_graph_indicator = open(
            combined_dataset_dir / str(combined_dataset_name + "_graph_indicator.txt"),
            "w",
        )
        combined_graph_labels = open(
            combined_dataset_dir / str(combined_dataset_name + "_graph_labels.txt"), "w"
        )
        combined_node_attributes = open(
            combined_dataset_dir / str(combined_dataset_name + "_node_attributes.txt"),
            "w",
        )

        for num_brick in tqdm(
            range(self.min_num_brick, self.max_num_brick + 1),
            "Number of Brick Progress",
        ):
            # separate dataset for graph classification (GC)
            dataset_name = self.dataset_prefix + "_" + str(num_brick)
            dataset_dir = self.save_dir / dataset_name
            pathlib.Path(dataset_dir).mkdir(parents=True, exist_ok=True)

            node_id = 0
            graphnode2id = {}

            a = open(dataset_dir / str(dataset_name + "_A.txt"), "w")
            graph_indicator = open(
                dataset_dir / str(dataset_name + "_graph_indicator.txt"), "w"
            )
            graph_labels = open(
                dataset_dir / str(dataset_name + "_graph_labels.txt"), "w"
            )
            node_attributes = open(
                dataset_dir / str(dataset_name + "_node_attributes.txt"), "w"
            )

            for episode in tqdm(range(self.num_episode), "Episode Progress"):
                # use test_simulator.py for debugging and visualization
                p.connect(p.DIRECT)

                p.setAdditionalSearchPath(pybullet_data.getDataPath())
                p.setPhysicsEngineParameter(numSolverIterations=10)
                p.setTimeStep(1.0 / 60.0)

                # generate new design
                self.designer.clear()
                self.designer.generate_design(num_brick=num_brick)

                # update combined graph_id
                graph_id = episode + 1
                combined_graph_id += 1

                init_pos_list = []
                final_pos_list = []

                for i, b in enumerate(self.designer.built):
                    # write to separate GC dataset
                    node_id += 1
                    graphnode2id[str(graph_id) + "_" + str(i)] = node_id
                    graph_indicator.write(str(graph_id) + "\n")
                    node_attributes.write(
                        ",".join(str(i) for i in b.bounding.flatten().tolist()) + "\n"
                    )

                    # write to combined GC dataset
                    combined_node_id += 1
                    combined_graphnode2id[
                        str(combined_graph_id) + "_" + str(i)
                    ] = combined_node_id
                    combined_graph_indicator.write(str(combined_graph_id) + "\n")
                    combined_node_attributes.write(
                        ",".join(str(i) for i in b.bounding.flatten().tolist()) + "\n"
                    )

                    pos = b.anchor
                    rot = p.getQuaternionFromEuler(np.array(b.rotation) * np.pi / 2)
                    brickID = p.loadURDF("urdf/brick.urdf", pos, rot)
                    init_pos_list.append(pos)

                p.loadURDF("plane.urdf", useMaximalCoordinates=True)

                p.setGravity(0, 0, -10)
                for frame in range(self.num_frame):
                    p.stepSimulation()
                    time.sleep(1.0 / 240.0)

                    if frame == self.num_frame - 1:
                        for brickID in len(num_brick):
                            pos, _ = p.getBasePositionAndOrientation(brickID)
                            final_pos_list.append(pos)

                for edge in self.designer.G.edges:
                    # write to separate GC dataset
                    row = graphnode2id[str(graph_id) + "_" + str(edge[0])]
                    col = graphnode2id[str(graph_id) + "_" + str(edge[1])]
                    a.write(str(row) + ", " + str(col) + "\n")
                    a.write(str(col) + ", " + str(row) + "\n")

                    # write to combined GC dataset
                    combined_row = combined_graphnode2id[
                        str(combined_graph_id) + "_" + str(edge[0])
                    ]
                    combined_col = combined_graphnode2id[
                        str(combined_graph_id) + "_" + str(edge[1])
                    ]
                    combined_a.write(
                        str(combined_row) + ", " + str(combined_col) + "\n"
                    )
                    combined_a.write(
                        str(combined_col) + ", " + str(combined_row) + "\n"
                    )

                label = int(self.determine_stable(init_pos_list, final_pos_list))
                graph_labels.write(str(label) + "\n")
                combined_graph_labels.write(str(label) + "\n")

                p.disconnect()

            a.close()
            graph_indicator.close()
            graph_labels.close()
            node_attributes.close()
            shutil.make_archive(self.save_dir / dataset_name, "zip", dataset_dir)
            shutil.rmtree(dataset_dir)

        combined_a.close()
        combined_graph_indicator.close()
        combined_graph_labels.close()
        combined_node_attributes.close()
        shutil.make_archive(
            self.save_dir / combined_dataset_name, "zip", combined_dataset_dir
        )
        shutil.rmtree(combined_dataset_dir)

    def determine_stable(
        self, init_pos_list: List[List], final_pos_list: List[List]
    ) -> bool:
        num_brick = len(init_pos_list)
        dist_list = [
            np.linalg.norm(np.array(final_pos_list[i]) - np.array(init_pos_list[i]))
            for i in range(num_brick)
        ]
        total_dist = np.sum(dist_list)
        stable = False if total_dist >= self.movement_threshold else True
        return stable


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "Graph Classification Dataset Creator For Lego Player"
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
        default=500,
        help="Number of episode per each separate dataset",
    )
    parser.add_argument(
        "--num-frame",
        type=int,
        default=100,
        help="Number of frame to run in simulator per each episode",
    )
    parser.add_argument(
        "--movement-threshold",
        type=float,
        default=0.1,
        help="Threshold for determining structure stability",
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
        default=10,
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
        dataset_prefix = "BRICK_GC_PLAIN"
        save_dir = pathlib.Path(__file__).parent.absolute() / "datasets" / "PLAIN"

    elif args.brick_type == "lego":
        is_modular = True
        mod_unit = min(args.brick_extents)
        rotate_prob = [0.0, 0.0, 0.5]
        safe_margin = round(args.safe_margin / mod_unit) * mod_unit
        dataset_prefix = "BRICK_GC_LEGO"
        save_dir = pathlib.Path(__file__).parent.absolute() / "datasets" / "LEGO"

    desigher = Designer(
        arena_length=args.arena_length,
        brick_extents=args.brick_extents,
        safe_margin=safe_margin,
        is_modular=is_modular,
        mod_unit=mod_unit,
        rotate_prob=rotate_prob,
    )
    dc = GCDatasetCreator(
        desigher,
        dataset_prefix,
        num_episode=args.num_episode,
        num_frame=args.num_frame,
        min_num_brick=args.min_num_brick,
        max_num_brick=args.max_num_brick,
        movement_threshold=args.movement_threshold,
        save_dir=save_dir,
    )
    dc.generate_dataset()

