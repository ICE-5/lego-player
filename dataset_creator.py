import pybullet as p
import time, io
import pathlib
import pybullet_data
import numpy as np
from designer import *
from typing import List
from tqdm import tqdm


class DatasetCreator:
    def __init__(
        self,
        designer: Designer,
        num_episode: int = 100,
        num_frame: int = 20,
        min_num_brick: int = 5,
        max_num_brick: int = 25,
        movement_threshold: float = 0.15,
        save_dir: pathlib.Path = pathlib.Path(__file__).parent.absolute() / "datasets",
    ) -> None:
        self.designer = designer
        self.movement_threshold = movement_threshold
        self.num_episode = num_episode
        self.num_frame = num_frame
        self.min_num_brick = min_num_brick
        self.max_num_brick = max_num_brick
        self.save_dir = save_dir

    def generate_dataset(self):

        # combined dataset for graph classification (GC)
        combined_dataset_name = "BRICK_GC"
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
            dataset_name = "BRICK_GC_" + str(num_brick)
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

            # position(3), rotation(3), velocity_linear(3), velocity_angular(3)
            physics_node_obs_dim = 3 + 3 + 3 + 3
            # positionA(3), positionB(3), normalForce(1), friction1(1), friction2(1)
            # positionA(3), positionB(3), normalForce(1), friction1(1), friction1Dir(3), friction2(1), friction2Dir(3),
            physics_edge_obs_dim = 3 + 3 + 1 + 1 + 3 + 1 + 3
            physics_node_obs = np.zeros(
                [self.num_episode, self.num_frame, physics_node_obs_dim * num_brick]
            )
            physics_edge_obs = np.zeros(
                [
                    self.num_episode,
                    self.num_frame,
                    num_brick,
                    num_brick,
                    physics_edge_obs_dim,
                ]
            )

            for episode in tqdm(range(self.num_episode), "Episode Progress"):
                # use test_simulator.py for debugging and visualization
                p.connect(p.DIRECT)

                p.setAdditionalSearchPath(pybullet_data.getDataPath())
                p.setPhysicsEngineParameter(numSolverIterations=10)
                p.setTimeStep(1.0 / 60.0)
                p.loadURDF("plane.urdf", useMaximalCoordinates=True)

                # generate new design
                self.designer.clear()
                self.designer.generate_design(num_brick=num_brick)

                # update combined graph_id
                graph_id = episode + 1
                combined_graph_id += 1

                brickID_list = []
                init_pos_list = []
                final_pos_list = []

                for i, b in enumerate(self.designer.built):
                    # write to separate GC dataset
                    node_id += 1
                    graphnode2id[str(graph_id) + "_" + str(i)] = node_id
                    graph_indicator.write(str(graph_id) + "\n")
                    node_attributes.write(
                        ",".join(str(i) for i in b.bounding.flatten().tolist())  + "\n"
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
                    brickID_list.append(brickID)
                    init_pos_list.append(pos)

                p.setGravity(0, 0, -10)
                for frame in range(self.num_frame):
                    p.stepSimulation()
                    time.sleep(1./240.)

                    obs = []
                    for brickID in brickID_list:
                        pos, ort = p.getBasePositionAndOrientation(brickID)
                        rot = p.getEulerFromQuaternion(ort)
                        vel_linear, vel_angular = p.getBaseVelocity(brickID)
                        obs.extend(list(pos))
                        obs.extend(list(rot))
                        obs.extend(list(vel_linear))
                        obs.extend(list(vel_angular))

                        if frame == self.num_frame - 1:
                            final_pos_list.append(pos)

                    # TODO: physics dataset
                    # physics_node_obs[episode, frame, :] = np.array(obs)

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

        combined_a.close()
        combined_graph_indicator.close()
        combined_graph_labels.close()
        combined_node_attributes.close()

        return physics_node_obs, physics_edge_obs

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
    MIN_NUM_BRICK = 10
    MAX_NUM_BRICK = 15

    NUM_EPISODE = 50
    NUM_FRAME = 100
    MOVEMENT_THRESHOLD = 0.1

    ARENA_LENGTH = 1.0
    BRICK_EXTENTS = [0.4, 0.2, 0.1]

    desigher = Designer(arena_length=ARENA_LENGTH, brick_extents=BRICK_EXTENTS)
    dc = DatasetCreator(
        desigher,
        num_episode=NUM_EPISODE,
        num_frame=NUM_FRAME,
        min_num_brick=MIN_NUM_BRICK,
        max_num_brick=MAX_NUM_BRICK,
        movement_threshold=MOVEMENT_THRESHOLD,
    )
    dc.generate_dataset()

