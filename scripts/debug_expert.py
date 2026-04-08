import numpy as np
import robosuite as suite

env = suite.make(
    env_name="Lift",
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    control_freq=20,
)

obs = env.reset()
print("Initial eef_pos:", obs["robot0_eef_pos"])
print("Initial cube_pos:", obs["cube_pos"])
print("Action dim:", env.action_spec[0].shape)

# Jalankan 50 step dengan action random, lihat reward
for i in range(200):
    low, high = env.action_spec
    action = np.random.uniform(low, high)
    obs, reward, done, info = env.step(action)
    if reward > 0:
        print(f"Step {i}: GOT REWARD! eef_pos={obs['robot0_eef_pos']}, cube_pos={obs['cube_pos']}")
    if i % 50 == 0:
        print(f"Step {i}: eef_pos={obs['robot0_eef_pos'].round(3)}, cube_pos={obs['cube_pos'].round(3)}, reward={reward}")

env.close()