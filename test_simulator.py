import pybullet as p
import time
import pathlib
import pybullet_data
import numpy as np
from designer import *
from typing import List
from tqdm import tqdm


brick_extents: List[float] = [0.4, 0.2, 0.1]
arena_length: float = 1.0
num_brick: int = 15
num_frame: int = 100
color_list = ["_r", "_g", "_b", "_y"]


p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setPhysicsEngineParameter(numSolverIterations=10)
p.setTimeStep(1.0 / 120.0)

p.loadURDF("plane.urdf", useMaximalCoordinates=True)


# d = Designer(
#         arena_length=1.0,
#         brick_extents=[0.4, 0.2, 0.1],
#         safe_margin=0.1,
#         is_modular=False,
#         rotate_prob=[0.1, 0.1, 0.5],
#     )

d = Designer(
    arena_length=1.0,
    brick_extents=[0.4, 0.2, 0.1],
    safe_margin=0.1,
    is_modular=True,
    mod_unit=0.1,
    rotate_prob=[0.0, 0.0, 0.5],
)

d.generate_design(num_brick=num_brick)
print("generated design", len(d))

brickID_list = []
for b in d.built:
    pos = b.anchor
    rot = p.getQuaternionFromEuler(np.array(b.rotation) * np.pi / 2)

    # choose color
    color = color_list[np.random.choice(len(color_list))]
    brickID = p.loadURDF("urdf/brick"+color+".urdf", pos, rot)
    brickID_list.append(brickID)

p.setGravity(0, 0, -10)

for frame in range(num_frame):
    p.stepSimulation()
    if frame == 0:
        contact_list = p.getContactPoints(bodyA=brickID_list[10])
        print("by pybullet", len(contact_list))
        # print(
        #     "by graph",
        #     d.G.in_degree(i),
        #     d.G.out_degree(i),
        #     list(d.G.predecessors(i)),
        #     list(d.G.successors(i)),
        # )
        # for contact in contact_list:
        #     # print(len(contact))
        #     print(
        #         f"brickA {contact[1]}, brickB {contact[2]}, positionA {contact[5]},  normalForce {contact[9]}, friction1 {contact[10]}, friction2 {contact[12]}"
        #     )

    time.sleep(1.0 / 240.0)

p.disconnect()

