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


def evaluate(model_path="results/bc_policy.pth", n_episodes=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    obs_dim    = checkpoint["obs_dim"]
    action_dim = checkpoint["action_dim"]
    obs_mean   = checkpoint["obs_mean"]
    obs_std    = checkpoint["obs_std"]

    policy = BCPolicy(obs_dim, action_dim).to(device)
    policy.load_state_dict(checkpoint["model_state"])
    policy.eval()

    env = suite.make(
        env_name="Lift", robots="Panda",
        has_renderer=False, has_offscreen_renderer=False,
        use_camera_obs=False, control_freq=20,
    )

    success_count = 0
    episode_lengths = []

    print(f"Evaluating BC policy over {n_episodes} episodes...\n")

    for ep in range(n_episodes):
        obs     = env.reset()
        success = False
        ep_len  = 0

        for step in range(400):
            ob_vec = np.concatenate([
                obs["robot0_eef_pos"],
                obs["robot0_eef_quat"],
                obs["robot0_gripper_qpos"],
                obs["cube_pos"],
                obs["gripper_to_cube_pos"],
            ])

            # Normalisasi pakai mean/std dari training data
            ob_vec = (ob_vec - obs_mean) / obs_std

            with torch.no_grad():
                action = policy(torch.FloatTensor(ob_vec).unsqueeze(0).to(device)).squeeze(0).cpu().numpy()

            obs, reward, done, _ = env.step(action)
            ep_len += 1

            if reward > 0:
                success = True
                break
            if done:
                break

        episode_lengths.append(ep_len)
        status = "✓ SUCCESS" if success else "✗ FAIL"
        print(f"  Episode {ep+1:2d}: {status} | steps={ep_len}")
        if success:
            success_count += 1

    success_rate = success_count / n_episodes * 100
    print(f"\n{'='*40}")
    print(f"Success Rate : {success_count}/{n_episodes} ({success_rate:.1f}%)")
    print(f"Avg Steps    : {np.mean(episode_lengths):.1f}")
    print(f"{'='*40}")

    env.close()
    return success_rate


if __name__ == "__main__":
    evaluate()