import pybullet as p
import time
import pybullet_data
from designer import *


EXTENTS = [0.4, 0.2, 0.1]
NUM_BRICK = 90
ARENA_LENGTH = 1.


physicsClient = p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setPhysicsEngineParameter(numSolverIterations=10)
p.setTimeStep(1. / 120.)
logId = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "visualShapeBench.json")

planeID = p.loadURDF("plane.urdf", useMaximalCoordinates=True)


d = Designer(arena_length=ARENA_LENGTH, brick_extents=EXTENTS)
d.generate_design(num_brick=NUM_BRICK)

for idx, b in enumerate(d.built):
    # if idx == 0:
    pos = b.anchor
    rotation = p.getQuaternionFromEuler(np.array(b.rotation) * np.pi / 2)
    brickID = p.loadURDF("brick.urdf", pos, rotation)


p.stopStateLogging(logId)
p.setGravity(0, 0, -10)


for i in range (10000):
    p.stepSimulation()
    time.sleep(1. / 240.)

p.disconnect()