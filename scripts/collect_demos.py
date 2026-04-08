import numpy as np
import h5py
import robosuite as suite
import os

def get_expert_action(obs, phase, phase_step):
    eef_pos = obs["robot0_eef_pos"]
    cube_pos = obs["cube_pos"]
    action = np.zeros(7)

    if phase == 0:
        target = cube_pos + np.array([0, 0, 0.1])
        delta = target - eef_pos
        action[:3] = np.clip(delta * 20.0, -1, 1)
        action[6] = -1.0
        done = np.linalg.norm(delta) < 0.015
        return action, done

    elif phase == 1:
        target = cube_pos + np.array([0, 0, 0.005])
        delta = target - eef_pos
        action[:3] = np.clip(delta * 20.0, -1, 1)
        action[6] = -1.0
        done = phase_step > 30
        return action, done

    elif phase == 2:
        action[6] = 1.0
        done = phase_step > 25
        return action, done

    elif phase == 3:
        action[2] = 1.0
        action[6] = 1.0
        done = cube_pos[2] > 0.92
        return action, done

    return action, True


def collect_demonstrations(n_demos=500, save_path="data/demos.hdf5"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    env = suite.make(
        env_name="Lift", robots="Panda",
        has_renderer=False, has_offscreen_renderer=False,
        use_camera_obs=False, control_freq=20,
    )

    all_obs, all_actions, all_rewards, all_dones = [], [], [], []
    demo_lengths = []
    success_count = 0

    print(f"Collecting {n_demos} demonstrations...")

    for demo_idx in range(n_demos):
        obs = env.reset()
        demo_obs, demo_actions, demo_rewards, demo_dones = [], [], [], []

        phase = 0
        phase_step = 0
        success = False

        for step in range(400):
            action, phase_done = get_expert_action(obs, phase, phase_step)

            # Noise injection — hanya di phase 0 dan 1 (bukan saat grasp/lift)
            if phase in [0, 1]:
                action[:3] += np.random.normal(0, 0.02, 3)
                action = np.clip(action, -1, 1)

            next_obs, reward, done, _ = env.step(action)

            ob_vec = np.concatenate([
                obs["robot0_eef_pos"],
                obs["robot0_eef_quat"],
                obs["robot0_gripper_qpos"],
                obs["cube_pos"],
                obs["gripper_to_cube_pos"],
            ])
            demo_obs.append(ob_vec)
            demo_actions.append(action.copy())
            demo_rewards.append(reward)
            demo_dones.append(done)

            obs = next_obs
            phase_step += 1

            if phase_done and phase < 3:
                phase += 1
                phase_step = 0

            if reward > 0:
                success = True
                break
            if done:
                break

        if success:
            all_obs.extend(demo_obs)
            all_actions.extend(demo_actions)
            all_rewards.extend(demo_rewards)
            all_dones.extend(demo_dones)
            demo_lengths.append(len(demo_obs))
            success_count += 1

        if (demo_idx + 1) % 50 == 0:
            print(f"  [{demo_idx+1}/{n_demos}] Success: {success_count}")

    print(f"\nTotal successful demos: {success_count}/{n_demos}")

    if success_count > 0:
        with h5py.File(save_path, "w") as f:
            f.create_dataset("obs", data=np.array(all_obs))
            f.create_dataset("actions", data=np.array(all_actions))
            f.create_dataset("rewards", data=np.array(all_rewards))
            f.create_dataset("dones", data=np.array(all_dones))
            f.create_dataset("demo_lengths", data=np.array(demo_lengths))
            f.attrs["n_demos"] = success_count
            f.attrs["obs_dim"] = np.array(all_obs).shape[1]
            f.attrs["action_dim"] = 7
        print(f"Saved {len(all_obs)} timesteps to {save_path}")
    else:
        print("No successful demos")

    env.close()


if __name__ == "__main__":
    collect_demonstrations(n_demos=500, save_path="data/demos.hdf5")