import numpy as np
import torch
import torch.nn as nn
import robosuite as suite
import imageio
import os

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


def record_video(model_path="results/bc_policy.pth", save_path="results/demo.mp4", n_episodes=5):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    device = torch.device("cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    policy = BCPolicy(checkpoint["obs_dim"], checkpoint["action_dim"])
    policy.load_state_dict(checkpoint["model_state"])
    policy.eval()

    obs_mean = checkpoint["obs_mean"]
    obs_std  = checkpoint["obs_std"]

    env = suite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,   # aktifkan offscreen
        use_camera_obs=False,
        camera_names="agentview",
        camera_heights=512,
        camera_widths=512,
        control_freq=20,
    )

    all_frames = []
    success_count = 0

    print(f"Recording {n_episodes} episodes...")

    for ep in range(n_episodes):
        obs = env.reset()
        ep_frames = []
        success = False

        for step in range(400):
            # Render frame
            frame = env.sim.render(camera_name="agentview", height=512, width=512, depth=False)
            frame = frame[::-1]  # flip vertical
            ep_frames.append(frame)

            ob_vec = np.concatenate([
                obs["robot0_eef_pos"],
                obs["robot0_eef_quat"],
                obs["robot0_gripper_qpos"],
                obs["cube_pos"],
                obs["gripper_to_cube_pos"],
            ])
            ob_vec = (ob_vec - obs_mean) / obs_std

            with torch.no_grad():
                action = policy(torch.FloatTensor(ob_vec).unsqueeze(0)).squeeze(0).numpy()

            obs, reward, done, _ = env.step(action)

            if reward > 0:
                success = True
                # Record beberapa frame tambahan setelah success
                for _ in range(20):
                    frame = env.sim.render(camera_name="agentview", height=512, width=512, depth=False)
                    ep_frames.append(frame[::-1])
                break
            if done:
                break

        status = "SUCCESS" if success else "FAIL"
        print(f"  Episode {ep+1}: {status} ({len(ep_frames)} frames)")

        if success:
            success_count += 1

        all_frames.extend(ep_frames)

    env.close()

    # Simpan video
    print(f"\nSaving video: {len(all_frames)} total frames...")
    writer = imageio.get_writer(save_path, fps=30)
    for frame in all_frames:
        writer.append_data(frame)
    writer.close()

    print(f"Video saved to {save_path}")
    print(f"Success: {success_count}/{n_episodes}")


if __name__ == "__main__":
    record_video()