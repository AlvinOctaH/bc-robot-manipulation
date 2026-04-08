import numpy as np
import robosuite as suite

def get_expert_action(obs, phase, phase_step):
    eef_pos = obs["robot0_eef_pos"]
    cube_pos = obs["cube_pos"]
    action = np.zeros(7)

    if phase == 0:
        target = cube_pos + np.array([0, 0, 0.1])
        delta = target - eef_pos
        action[:3] = np.clip(delta * 20.0, -1, 1)
        action[6] = 1.0
        done = np.linalg.norm(delta) < 0.015
        return action, done
    elif phase == 1:
        target = cube_pos + np.array([0, 0, 0.005])
        delta = target - eef_pos
        action[:3] = np.clip(delta * 20.0, -1, 1)
        action[6] = 1.0
        done = np.linalg.norm(delta) < 0.015
        return action, done
    elif phase == 2:
        action[6] = -1.0
        done = phase_step > 20
        return action, done
    elif phase == 3:
        action[2] = 1.0
        action[6] = -1.0
        done = cube_pos[2] > 0.92
        return action, done
    return action, True

env = suite.make(
    env_name="Lift", robots="Panda",
    has_renderer=False, has_offscreen_renderer=False,
    use_camera_obs=False, control_freq=20,
)

obs = env.reset()
phase = 0
phase_step = 0

for step in range(150):
    action, phase_done = get_expert_action(obs, phase, phase_step)
    obs, reward, done, info = env.step(action)
    phase_step += 1

    eef = obs["robot0_eef_pos"].round(4)
    cube = obs["cube_pos"].round(4)
    grip = obs["robot0_gripper_qpos"].round(4)
    dist = np.linalg.norm(eef - cube)

    print(f"Step {step:3d} | Phase {phase} | eef_z={eef[2]:.4f} | cube_z={cube[2]:.4f} | dist={dist:.4f} | grip={grip} | r={reward:.2f}")

    if phase_done and phase < 3:
        print(f"  >>> Phase {phase} -> {phase+1}")
        phase += 1
        phase_step = 0

    if reward > 0:
        print("SUCCESS!")
        break

env.close()