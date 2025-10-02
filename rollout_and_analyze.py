import argparse
import os
import pickle
from pathlib import Path

import matplotlib
import numpy as np
from tqdm import tqdm

matplotlib.use("Agg")
import d3rlpy
import gymnasium as gym
import matplotlib.pyplot as plt

# Import helpers from the sibling file in the same directory
from record_replay_buffer_videos import (
    _batch_observation,
    _set_ant_yaw_init_qpos,
    ant_xml,
    render_episodes_grid_video,
)


def _quat_wxyz_to_yaw(quat_wxyz: np.ndarray) -> float:
    """Convert quaternion (w, x, y, z) to yaw around z-axis.

    Uses 2*atan2(z, w) which matches the approximation used in graph.py.
    """
    w, x, y, z = (
        float(quat_wxyz[0]),
        float(quat_wxyz[1]),
        float(quat_wxyz[2]),
        float(quat_wxyz[3]),
    )
    return float(2.0 * np.arctan2(z, w))


def _extract_quat_from_obs_or_env(obs: np.ndarray, env) -> np.ndarray:
    """Try to extract torso orientation quaternion (w, x, y, z).

    For Ant observations, the quaternion typically appears at indices [1:5].
    Fallback to MuJoCo qpos[3:7] if not available.
    """
    try:
        q = np.asarray(obs[1:5], dtype=float)
        if q.shape[0] == 4 and np.isfinite(q).all():
            return q
    except Exception:
        pass
    # Fallback to env state
    try:
        qpos = env.unwrapped.data.qpos
        return np.asarray(qpos[3:7], dtype=float)
    except Exception:
        pass
    # Last resort: identity quaternion
    return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float)


def _make_initial_yaw_dist(xs, out_dir):
    xticks = [-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2]
    # Use valid mathtext strings (single backslashes)
    xticklabels = [
        r"$-\frac{\pi}{2}$",
        r"$-\frac{\pi}{4}$",
        r"$0$",
        r"$\frac{\pi}{4}$",
        r"$\frac{\pi}{2}$",
    ]

    plt.figure()
    plt.hist(xs)
    plt.ylabel("Count")
    plt.xlabel("Initial Orientation (rad)")
    plt.xticks(xticks, xticklabels)
    plt.tight_layout()
    plt.savefig(out_dir / "initial.png", dpi=150)
    plt.close()


def _make_plots(
    xs: np.ndarray,
    ds: np.ndarray,
    fails: np.ndarray,
    turns: np.ndarray,
    overturn: np.ndarray,
    out_dir: Path,
) -> None:
    def sliding_window(x, y, window):
        smoothed = []
        for xi in x:
            mask = (x >= xi - window / 2) & (x <= xi + window / 2)
            smoothed.append(y[mask].mean() if mask.any() else 0)
        return np.array(smoothed)

    xticks = [-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2]
    # Use valid mathtext strings (single backslashes)
    xticklabels = [
        r"$-\frac{\pi}{2}$",
        r"$-\frac{\pi}{4}$",
        r"$0$",
        r"$\frac{\pi}{4}$",
        r"$\frac{\pi}{2}$",
    ]

    plt.figure()
    plt.scatter(xs, sliding_window(xs, ds, 0.1))
    plt.ylabel("Reached 100.0m (density, per rad)")
    plt.xlabel("Initial Orientation (rad)")
    plt.xticks(xticks, xticklabels)
    plt.tight_layout()
    plt.savefig(out_dir / "plot_reached.png", dpi=150)
    plt.close()

    plt.figure()
    plt.scatter(xs, sliding_window(xs, turns, 0.1))
    plt.ylabel("Turned Optimally (density, per rad)")
    plt.xlabel("Initial Orientation (rad)")
    plt.xticks(xticks, xticklabels)
    plt.tight_layout()
    plt.savefig(out_dir / "plot_turns.png", dpi=150)
    plt.close()

    plt.figure()
    plt.scatter(xs, sliding_window(xs, overturn, 0.1))
    plt.ylabel("Turned too much (density, per rad)")
    plt.xlabel("Initial Orientation (rad)")
    plt.xticks(xticks, xticklabels)
    plt.tight_layout()
    plt.savefig(out_dir / "plot_overturn.png", dpi=150)
    plt.close()

    plt.figure()
    plt.scatter(xs, sliding_window(xs, fails, 0.1))
    plt.ylabel("Failure (density, per rad)")
    plt.xlabel("Initial Orientation (rad)")
    plt.xticks(xticks, xticklabels)
    plt.tight_layout()
    plt.savefig(out_dir / "plot_fails.png", dpi=150)
    plt.close()

    plt.figure()
    should_left = np.where(xs >= 0, np.ones_like(xs), np.zeros_like(xs))
    plt.scatter(xs, sliding_window(xs, should_left, 0.1))
    plt.ylabel("Should turn left (density, per rad)")
    plt.xlabel("Initial Orientation (rad)")
    plt.xticks(xticks, xticklabels)
    plt.tight_layout()
    plt.savefig(out_dir / "plot_should_left.png", dpi=150)
    plt.close()


def _correct_yaw(yaw):
    if yaw < -np.pi / 2:
        return yaw + np.pi
    elif yaw > np.pi / 2:
        return yaw - np.pi
    else:
        return yaw


def rollout_and_analyze(
    policy_path: str,
    env_name: str = "Ant-v5",
    num_episodes: int = 100,
    max_steps: int = 1000,
    distance_threshold: float = 100.0,
    deterministic: bool = True,
    use_ant_xml: bool = True,
    make_grid_video: bool = False,
    episodes_per_row: int = 10,
):
    policy = d3rlpy.load_learnable(policy_path)

    env_kwargs = {}
    if use_ant_xml and (
        "ant" in env_name.lower() or env_name.lower().startswith("mujoco/ant")
    ):
        xml = ant_xml()
        env_kwargs["xml_file"] = xml
    # Ensure reset is deterministic when we control yaw
    env_kwargs["reset_noise_scale"] = 0.0

    env = gym.make(env_name, **env_kwargs)

    # Prepare outputs
    run_dir = Path(os.path.dirname(policy_path))
    out_dir = run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare yaw sweep
    yaw_values = np.linspace(-np.pi, np.pi, num_episodes, endpoint=True)

    # Data holders
    ds = []  # reached distance threshold
    fails = []
    turns = []  # turned optimally
    overturns = []
    initial_yaws = []
    collected_episodes = []

    from d3rlpy.dataset.components import Episode
    from d3rlpy.dataset.replay_buffer import create_infinite_replay_buffer

    for ep_idx in tqdm(range(num_episodes)):
        # Set initial yaw (Ant only)
        env_id_lower = env.unwrapped.spec.id.lower()
        if "ant" in env_id_lower:
            _set_ant_yaw_init_qpos(env, float(yaw_values[ep_idx]))

        obs, info = env.reset()

        obs_list = [obs]
        act_list = []
        rew_list = []
        terminated_flag = False

        # Track metrics across steps
        yaw_series = []

        # Initial yaw
        quat0 = _extract_quat_from_obs_or_env(obs, env)
        initial_yaw = _quat_wxyz_to_yaw(quat0)

        for t in range(max_steps):
            batched_obs = _batch_observation(obs)
            if deterministic:
                action = policy.predict(batched_obs)[0]
            else:
                action = policy.sample_action(batched_obs)[0]

            next_obs, reward, term, trunc, info = env.step(action)

            # Record yaw from observation/env
            quat_t = _extract_quat_from_obs_or_env(next_obs, env)
            yaw_series.append(_quat_wxyz_to_yaw(quat_t))

            # Metrics
            x_pos = info.get("x_position")

            act_list.append(np.asarray(action))
            rew_list.append(np.asarray([reward], dtype=np.float32))
            obs_list.append(next_obs)

            obs = next_obs
            terminated_flag = bool(term)
            if term or trunc:
                break

        # Determine turning optimality
        if len(yaw_series) == 0:
            final_mean_yaw = 0.0
        else:
            k = min(200, len(yaw_series))
            final_mean_yaw = float(np.mean(yaw_series[-k:]))

        overturned = np.abs(final_mean_yaw - initial_yaw) > np.pi / 4
        final_mean_yaw = _correct_yaw(final_mean_yaw)
        initial_yaw = _correct_yaw(initial_yaw)
        turns_optimally = (final_mean_yaw * initial_yaw) >= 0.0

        observations = np.asarray(obs_list)
        rewards = np.asarray(rew_list, dtype=np.float32)

        fail = np.any(observations[:, 0] < 0.2)
        exceeds = rewards.sum() > 2000.0

        episode = Episode(
            observations=observations,
            actions=np.asarray(act_list),
            rewards=rewards,
            terminated=terminated_flag,
        )
        collected_episodes.append(episode)

        # Append metrics
        ds.append(bool(exceeds))
        fails.append(bool(fail))
        turns.append(bool(turns_optimally))
        overturns.append(bool(overturned))
        initial_yaws.append(float(initial_yaw))

    # Save replay buffer
    buffer = create_infinite_replay_buffer(episodes=collected_episodes, env=env)
    rollouts_path = out_dir / f"rollouts_{env.unwrapped.spec.id}_{num_episodes}.pkl"
    with open(rollouts_path, "wb") as f:
        pickle.dump(buffer, f)

    # Save metrics
    xs = np.asarray(initial_yaws)
    ds_arr = np.asarray(ds, dtype=bool)
    fails_arr = np.asarray(fails, dtype=bool)
    turns_arr = np.asarray(turns, dtype=bool)
    overturns = np.asarray(overturns, dtype=bool)

    metrics = {
        "initial_yaw": xs,
        "reached_distance": ds_arr,
        "failed": fails_arr,
        "turned_optimally": turns_arr,
        "overturned": overturns,
        "distance_threshold": float(distance_threshold),
        "env": env.unwrapped.spec.id,
        "num_episodes": int(num_episodes),
    }
    with open(
        out_dir / f"metrics_{env.unwrapped.spec.id}_{num_episodes}.pkl", "wb"
    ) as f:
        pickle.dump(metrics, f)

    # Save plots
    # Sort by initial yaw for cleaner plots
    order = np.argsort(xs)
    _make_plots(
        xs[order],
        ds_arr[order],
        fails_arr[order],
        turns_arr[order],
        overturns[order],
        out_dir,
    )

    _make_initial_yaw_dist(xs, out_dir)

    # Optional grid video
    if make_grid_video:
        # Build rows for the grid
        rows = []
        for r in range(0, num_episodes, episodes_per_row):
            rows.append(collected_episodes[r : r + episodes_per_row])

        # Create a rendering env (separate, rgb_array)
        render_env = gym.make(env_name, render_mode="rgb_array", **env_kwargs)
        video_path = (
            out_dir / f"grid_rollouts_{env.unwrapped.spec.id}_{num_episodes}.mp4"
        )
        render_episodes_grid_video(
            rows,
            render_env,
            video_path,
            fps=30,
            title_return=True,
            env_kwargs=env_kwargs,
            parallel_workers=None,
        )

    env.close()

    return str(rollouts_path)


def main():
    parser = argparse.ArgumentParser(
        description="Rollout a policy, analyze, and save artifacts next to the policy."
    )
    parser.add_argument(
        "--policy-path",
        type=str,
        required=True,
        help="Path to saved d3rlpy learnable (.d3)",
    )
    parser.add_argument("--env", type=str, default="Ant-v5", help="Environment ID")
    parser.add_argument(
        "--num-episodes", type=int, default=100, help="Number of episodes to rollout"
    )
    parser.add_argument(
        "--max-steps", type=int, default=1000, help="Max steps per episode"
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=100.0,
        help="X position threshold for success",
    )
    parser.add_argument(
        "--deterministic", action="store_true", help="Use greedy actions (predict)"
    )
    parser.add_argument(
        "--use-ant-xml", action="store_true", help="Use recolored Ant XML"
    )

    args = parser.parse_args()

    rollout_and_analyze(
        policy_path=args.policy_path,
        env_name=args.env,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        distance_threshold=args.distance_threshold,
        deterministic=args.deterministic,
        use_ant_xml=args.use_ant_xml,
        make_grid_video=False,
    )


if __name__ == "__main__":
    main()