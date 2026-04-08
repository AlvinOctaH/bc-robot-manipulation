import numpy as np
import torch
import torch.nn as nn
import robosuite as suite

class BCPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh()
        )
    def forward(self, obs):
        return self.net(obs)

device = torch.device("cpu")
checkpoint = torch.load("results/bc_policy.pth", map_location=device)
policy = BCPolicy(checkpoint["obs_dim"], checkpoint["action_dim"])
policy.load_state_dict(checkpoint["model_state"])
policy.eval()

env = suite.make(
    env_name="Lift", robots="Panda",
    has_renderer=False, has_offscreen_renderer=False,
    use_camera_obs=False, control_freq=20,
)

obs = env.reset()
print(f"Start | eef={obs['robot0_eef_pos'].round(3)} | cube={obs['cube_pos'].round(3)}")

for step in range(150):
    ob_vec = np.concatenate([
        obs["robot0_eef_pos"], obs["robot0_eef_quat"],
        obs["robot0_gripper_qpos"], obs["cube_pos"],
        obs["gripper_to_cube_pos"],
    ])
    with torch.no_grad():
        action = policy(torch.FloatTensor(ob_vec).unsqueeze(0)).squeeze(0).numpy()

    obs, reward, done, _ = env.step(action)

    if step % 20 == 0:
        eef  = obs["robot0_eef_pos"].round(3)
        cube = obs["cube_pos"].round(3)
        grip = obs["robot0_gripper_qpos"].round(3)
        dist = np.linalg.norm(obs["gripper_to_cube_pos"]).round(3)
        print(f"Step {step:3d} | eef_z={eef[2]:.3f} | cube_z={cube[2]:.3f} | dist={dist:.3f} | grip={grip} | action={action.round(2)} | r={reward:.2f}")

    if reward > 0:
        print("SUCCESS!")
        break

env.close()