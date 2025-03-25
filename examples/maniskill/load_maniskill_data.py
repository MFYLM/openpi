"""
The maniskill dataset h5 file:
Root
└── Trajectories (traj_xxx/)
    ├── Observations (obs/)
    │   ├── Agent State (agent/)
    │   │   ├── qpos: (T, 9), dtype=float32
    │   │   ├── qvel: (T, 9), dtype=float32
    │   ├── Extra Info (extra/)
    │   │   ├── is_grasp: (T,), dtype=bool
    │   │   ├── tcp_pose: (T, 7), dtype=float32
    │   │   ├── source_pose: (T, 7), dtype=float32
    │   │   ├── target_pose: (T, 3), dtype=float32
    │   │   ├── tcp_to_obj_pos: (T, 3), dtype=float32
    │   │   ├── obj_to_goal_pos: (T, 3), dtype=float32
    │   ├── Sensor Parameters (sensor_param/)
    │   │   ├── Base Camera (base_camera/)
    │   │       ├── extrinsic_cv: (T, 3, 4), dtype=float32
    │   │       ├── cam2world_gl: (T, 4, 4), dtype=float32
    │   │       ├── intrinsic_cv: (T, 3, 3), dtype=float32
    │   ├── Sensor Data (sensor_data/)
    │   │   ├── Base Camera (base_camera/)
    │   │       ├── rgb: (T, 128, 128, 3), dtype=uint8
    ├── Actions (actions): (T-1, 7), dtype=float32
    ├── Termination Flags:
    │   ├── terminated: (T-1,), dtype=bool
    │   ├── truncated: (T-1,), dtype=bool
    ├── Success Status (success): (T-1,), dtype=bool
    ├── Environment States (env_states/)
    │   ├── Actors (actors/)
    │   │   ├── table-workspace: (T, 13), dtype=float32
    │   │   ├── coke_can: (T, 13), dtype=float32
    │   │   ├── target_180: (T, 13), dtype=float32
    │   │   ├── source_goal_site: (T, 13), dtype=float32
    │   │   ├── target_goal_site: (T, 13), dtype=float32
    │   ├── Articulations (articulations/)
    │   │   ├── panda: (T, 31), dtype=float32
    ├── Rewards (rewards): (T-1,), dtype=float32

Notes:
- T represents the number of timesteps in a trajectory, which varies between different traj_xxx groups.
- Action, termination, and success datasets have a shape of (T-1, ...) because actions are applied at each step to transition between states.
- Each trajectory contains observations, actions, termination conditions, rewards, and environment state representations.
- Camera data (rgb) is stored as images of shape (T, 128, 128, 3), dtype=uint8.
- Intrinsic and extrinsic camera matrices are stored per timestep.

"""
