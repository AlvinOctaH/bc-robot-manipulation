import numpy as np
import robosuite as suite
import json, os

env = suite.make(
    env_name="Lift",
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    control_freq=20,
)

obs = env.reset()
robot = env.robots[0]
print("Robot type:", type(robot))
print("Robot attrs:", [a for a in dir(robot) if 'control' in a.lower()])

# Cek config file langsung
config_path = r"C:\Users\Alvin\anaconda3\envs\imitation_learning\lib\site-packages\robosuite\controllers\config\robots\default_panda.json"
with open(config_path) as f:
    config = json.load(f)
print("\nController config:")
print(json.dumps(config, indent=2))
env.close()