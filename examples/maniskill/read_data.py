import h5py
import numpy as np
import ipdb


POUR_WATER_PATH = "/root/openpi/examples/maniskill/pour_water/pour.h5"

with h5py.File(POUR_WATER_PATH, "r") as f:
    traj = f["traj_0"]
    env_states = traj["env_states"]
    actors = env_states["actors"]
    print(actors.keys())
    ipdb.set_trace()