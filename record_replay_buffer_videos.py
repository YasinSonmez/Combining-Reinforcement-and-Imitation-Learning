import argparse
import os
import pickle
from itertools import zip_longest
from pathlib import Path
import xml.etree.ElementTree as ET
from importlib import resources
import tempfile
from datetime import datetime

import imageio
import numpy as np
import d3rlpy
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

os.environ["MUJOCO_GL"] = "egl"


def ant_xml():
    base_xml = resources.files("gymnasium.envs.mujoco.assets") / "ant.xml"

    tree = ET.parse(base_xml)
    root = tree.getroot()

    PURPLE = "0.60 0.30 0.85 1.0"

    def paint_leg(body_name: str, rgba: str):
        for body in root.findall(f".//body[@name='{body_name}']"):
            for geom in body.findall(".//geom"):
                geom.set("rgba", rgba)

    LEG_BODIES = ["front_left_leg", "back_leg"]

    for name in LEG_BODIES:
        paint_leg(name, PURPLE)

    with tempfile.NamedTemporaryFile(suffix="_ant_purple.xml", delete=False) as tmp:
        tree.write(tmp.name)
        custom_xml_path = tmp.name
    return custom_xml_path


def _set_ant_yaw_init_qpos(env, yaw_radians: float) -> None:
    """Set Ant's initial orientation (yaw) by modifying init_qpos quaternion.

    This updates only the torso orientation quaternion (indices 3:7) while
    keeping all other initial positions/velocities at their defaults.

    The quaternion layout in MuJoCo is (w, x, y, z). For a pure yaw about the
    z-axis by angle θ, the quaternion is [cos(θ/2), 0, 0, sin(θ/2)].
    """
    if not hasattr(env.unwrapped, "init_qpos"):
        raise ValueError("Environment does not expose init_qpos; only supported for Ant.")
    qpos0 = env.unwrapped.init_qpos.copy()
    half = 0.5 * yaw_radians
    qw = float(np.cos(half))
    qz = float(np.sin(half))
    qpos0[3:7] = np.array([qw, 0.0, 0.0, qz], dtype=qpos0.dtype)
    env.unwrapped.init_qpos[:] = qpos0


def load_replay_buffer(buffer_path):
    """Load replay buffer from pickle file."""
    with open(buffer_path, 'rb') as f:
        buffer = pickle.load(f)
    return buffer


def extract_episodes_from_buffer(buffer, max_episodes=None):
    """Extract episodes from replay buffer and rank them by returns."""
    # d3rlpy replay buffer already has episodes property
    episodes = list(buffer.episodes)
    
    # Rank episodes by their returns (best first)
    episodes.sort(key=lambda ep: ep.compute_return(), reverse=True)
    
    if max_episodes:
        episodes = episodes[:max_episodes]  # Take top episodes after ranking
    
    return episodes


def rollout_frames_generic(env, episode, missing, max_steps=1000):
    """Run one rollout for generic MuJoCo environments (e.g., Ant, HalfCheetah)."""
    frames = []
    n_qpos = env.unwrapped.model.nq
    n_qvel = env.unwrapped.model.nv
    n_steps = min(episode.actions.shape[0], max_steps)

    obs, info = env.reset()

    n_missing = len(missing)
    qpos = np.zeros(n_qpos)
    qpos[n_missing:] = episode.observations[0, : n_qpos - n_missing]
    qvel = episode.observations[
        0, n_qpos - n_missing : n_qvel - n_missing + n_qpos
    ].copy()
    env.unwrapped.set_state(qpos, qvel)

    for i in range(n_steps - 1):
        obs, reward, _, _, info = env.step(episode.actions[i])

        qpos = np.zeros(n_qpos)
        qpos[n_missing:] = episode.observations[i + 1, : n_qpos - n_missing]
        for j, k in enumerate(missing):
            qpos[j] = info.get(k, qpos[j])
        qvel = episode.observations[
            i + 1, n_qpos - n_missing : n_qpos + n_qvel - n_missing
        ]
        env.unwrapped.set_state(qpos, qvel)

        frame = env.render()
        frames.append(frame)

    return frames


def add_title_with_matplotlib(frames, title_text):
    """Render frames to a video with a matplotlib title (returns), returning new frames.

    This draws the frame as an image and uses the axes title for the text, ensuring
    consistent, readable text rendering without manual overlays per frame.
    """
    rendered_frames = []
    for frame in frames:
        # Create a figure that exactly matches the frame size in pixels
        fig, ax = plt.subplots(figsize=(frame.shape[1] / 100.0, frame.shape[0] / 100.0), dpi=100)
        ax.imshow(frame)
        ax.axis('off')
        # Remove all margins so the image fills the canvas
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        # Draw the title as a figure-level text to avoid reserving space
        fig.text(0.5, 0.985, title_text, ha='center', va='top', fontsize=18, color='black')
        fig.patch.set_alpha(0.0)
        # Render via Agg canvas and extract RGB buffer
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = np.asarray(canvas.buffer_rgba())  # shape (H, W, 4)
        img = buf[..., :3].copy()
        plt.close(fig)
        rendered_frames.append(img)
    return rendered_frames


def add_title_with_matplotlib_stream(frames_iter, title_text):
    """Yield frames with a matplotlib-rendered title, streaming one-by-one."""
    for frame in frames_iter:
        fig, ax = plt.subplots(
            figsize=(frame.shape[1] / 100.0, frame.shape[0] / 100.0), dpi=100
        )
        ax.imshow(frame)
        ax.axis('off')
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig.text(0.5, 0.985, title_text, ha='center', va='top', fontsize=18, color='black')
        fig.patch.set_alpha(0.0)
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = np.asarray(canvas.buffer_rgba())
        img = buf[..., :3].copy()
        plt.close(fig)
        yield img


def rollout_frames(env, episode, max_steps=1000):
    """Dispatch to environment-specific rollout function."""
    env_id = env.unwrapped.spec.id.lower()
    if "hopper" in env_id or "halfcheetah" in env_id or "walker2d" in env_id:
        return rollout_frames_generic(env, episode, ["x_position"], max_steps)
    elif "ant" in env_id:
        return rollout_frames_generic(
            env, episode, ["x_position", "y_position"], max_steps
        )
    else:
        raise ValueError(
            f"rollout_frames not implemented for environment: {env_id}"
        )


def rollout_frames_generic_stream(env, episode, missing, max_steps=1000):
    """Stream frames for generic MuJoCo envs by stepping the env and yielding per frame."""
    n_qpos = env.unwrapped.model.nq
    n_qvel = env.unwrapped.model.nv
    n_steps = min(episode.actions.shape[0], max_steps)

    obs, info = env.reset()

    n_missing = len(missing)
    qpos = np.zeros(n_qpos)
    qpos[n_missing:] = episode.observations[0, : n_qpos - n_missing]
    qvel = episode.observations[
        0, n_qpos - n_missing : n_qvel - n_missing + n_qpos
    ].copy()
    env.unwrapped.set_state(qpos, qvel)

    for i in range(n_steps - 1):
        obs, reward, _, _, info = env.step(episode.actions[i])

        qpos = np.zeros(n_qpos)
        qpos[n_missing:] = episode.observations[i + 1, : n_qpos - n_missing]
        for j, k in enumerate(missing):
            qpos[j] = info.get(k, qpos[j])
        qvel = episode.observations[
            i + 1, n_qpos - n_missing : n_qpos + n_qvel - n_missing
        ]
        env.unwrapped.set_state(qpos, qvel)

        frame = env.render()
        yield frame


def rollout_frames_stream(env, episode, max_steps=1000):
    """Dispatch to environment-specific streaming rollout function."""
    env_id = env.unwrapped.spec.id.lower()
    if "hopper" in env_id or "halfcheetah" in env_id or "walker2d" in env_id:
        yield from rollout_frames_generic_stream(env, episode, ["x_position"], max_steps)
    elif "ant" in env_id:
        yield from rollout_frames_generic_stream(
            env, episode, ["x_position", "y_position"], max_steps
        )
    else:
        raise ValueError(
            f"rollout_frames_stream not implemented for environment: {env_id}"
        )


def render_episode(episode, env, output_path, fps=30, max_frames=None, title_return=True):
    """Render a single episode to video using streaming writes."""
    writer = imageio.get_writer(output_path, fps=fps)

    # Stream frames directly from env
    frames_iter = rollout_frames_stream(env, episode)
    if title_return:
        ret_text = f"Return: {episode.compute_return():.2f}"
        frames_iter = add_title_with_matplotlib_stream(frames_iter, ret_text)

    for frame in frames_iter:
        writer.append_data(frame)

    writer.close()
    env.close()


def _render_episode_to_temp_video(args):
    env_id, episode, title_return, env_kwargs, fps = args
    local_env = gym.make(env_id, render_mode="rgb_array", **(env_kwargs or {}))
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp_path = tmp.name
        tmp.close()
        writer = imageio.get_writer(tmp_path, fps=fps)
        frames_iter = rollout_frames_stream(local_env, episode)
        if title_return:
            ret_text = f"Return: {episode.compute_return():.2f}"
            frames_iter = add_title_with_matplotlib_stream(frames_iter, ret_text)
        for frame in frames_iter:
            writer.append_data(frame)
        writer.close()
        return tmp_path
    finally:
        local_env.close()


def render_episodes_grid_video(
    interval_episodes,
    env,
    output_path,
    fps=30,
    max_frames=None,
    title_return=True,
    env_kwargs: dict | None = None,
    parallel_workers: int | None = None,
):
    """Render multiple episodes in a grid format with streaming composition.

    Each cell uses its own environment instance so frames can be streamed
    concurrently without buffering entire episodes in memory.
    """
    writer = imageio.get_writer(output_path, fps=fps)
    env_kwargs = env_kwargs or {}

    # Pre-roll to determine geometry and generators per cell
    num_rows = len(interval_episodes)
    num_cols = max((len(row) for row in interval_episodes), default=0)
    if num_rows == 0 or num_cols == 0:
        writer.close()
        env.close()
        return

    # Gym id for cloning envs
    env_id = env.unwrapped.spec.id

    # Heuristic: if grid is large or parallel requested, pre-render each cell to
    # a temp video (parallelizable), then compose by reading frames back.
    total_cells = sum(len(row) for row in interval_episodes)
    pre_render = (parallel_workers or 0) > 0 or total_cells > 9

    if pre_render:
        tasks = []
        for row in interval_episodes:
            for ep in row:
                tasks.append((env_id, ep, title_return, env_kwargs, fps))

        temp_paths = []
        if parallel_workers and parallel_workers > 1:
            import multiprocessing as mp
            with mp.get_context("spawn").Pool(processes=parallel_workers) as pool:
                for p in pool.imap_unordered(_render_episode_to_temp_video, tasks):
                    temp_paths.append(p)
        else:
            for t in tasks:
                temp_paths.append(_render_episode_to_temp_video(t))

        # Rebuild into grid-aligned list of paths
        it = iter(temp_paths)
        path_grid = []
        for row in interval_episodes:
            row_paths = []
            for _ in row:
                row_paths.append(next(it))
            while len(row_paths) < num_cols:
                row_paths.append(None)
            path_grid.append(row_paths)

        # Open readers and determine frame shape
        readers = []
        sample_frame = None
        for row in path_grid:
            reader_row = []
            for p in row:
                if p is None:
                    reader_row.append(None)
                    continue
                try:
                    r = imageio.get_reader(p)
                    if sample_frame is None:
                        try:
                            sample_frame = r.get_data(0)
                        except Exception:
                            pass
                    reader_row.append(r)
                except Exception:
                    reader_row.append(None)
            readers.append(reader_row)

        if sample_frame is None:
            # Cleanup
            for row in readers:
                for r in row:
                    try:
                        if r is not None:
                            r.close()
                    except Exception:
                        pass
            for row in path_grid:
                for p in row:
                    if p is not None:
                        try:
                            os.remove(p)
                        except Exception:
                            pass
            writer.close()
            env.close()
            return

        h, w, c = sample_frame.shape
        blank = np.zeros((h, w, c), dtype=np.uint8)

        t = 0
        while True:
            active = False
            row_images = []
            for row in readers:
                col_images = []
                for r in row:
                    if r is None:
                        col_images.append(blank)
                    else:
                        try:
                            frame = r.get_data(t)
                            col_images.append(frame)
                            active = True
                        except Exception:
                            col_images.append(blank)
                row_images.append(np.concatenate(col_images, axis=1))
            if not active:
                break
            grid_frame = np.concatenate(row_images, axis=0)
            writer.append_data(grid_frame)
            if max_frames is not None and (t + 1) >= max_frames:
                break
            t += 1

        # Cleanup
        for row in readers:
            for r in row:
                try:
                    if r is not None:
                        r.close()
                except Exception:
                    pass
        for row in path_grid:
            for p in row:
                if p is not None:
                    try:
                        os.remove(p)
                    except Exception:
                        pass

        writer.close()
        env.close()
        return

    # Build per-cell frame generators (streaming, with dedicated envs)
    cell_generators = []
    for row in interval_episodes:
        gens = []
        for episode in row:
            # Create a dedicated env per episode cell
            local_env = gym.make(env_id, render_mode="rgb_array", **env_kwargs)
            frames_iter = rollout_frames_stream(local_env, episode)
            if title_return:
                ret_text = f"Return: {episode.compute_return():.2f}"
                frames_iter = add_title_with_matplotlib_stream(frames_iter, ret_text)

            # Ensure local_env closes when iterator is exhausted
            def with_env_close(env_to_close, it):
                try:
                    for item in it:
                        yield item
                finally:
                    env_to_close.close()

            gens.append(iter(with_env_close(local_env, frames_iter)))
        # pad missing columns with infinite blanks
        while len(gens) < num_cols:
            gens.append(None)
        cell_generators.append(gens)

    # Determine frame shape from first available generator frame
    sample_frame = None
    for r_idx, row in enumerate(cell_generators):
        for c_idx, g in enumerate(row):
            if g is not None:
                try:
                    first = next(g)
                    sample_frame = first
                    # reinsert by chaining back the sample
                    def prepend(sample, gen):
                        yield sample
                        for x in gen:
                            yield x
                    row[c_idx] = prepend(first, g)
                except StopIteration:
                    sample_frame = None
                break
        if sample_frame is not None:
            break

    if sample_frame is None:
        writer.close()
        env.close()
        return

    h, w, c = sample_frame.shape
    blank = np.zeros((h, w, c), dtype=np.uint8)

    # Compose frames timestep by timestep until all generators are exhausted
    active = True
    while active:
        active = False
        row_images = []
        for row in cell_generators:
            col_images = []
            for g in row:
                if g is None:
                    col_images.append(blank)
                else:
                    try:
                        frame = next(g)
                        active = True
                        col_images.append(frame)
                    except StopIteration:
                        col_images.append(blank)
            row_images.append(np.concatenate(col_images, axis=1))
        if active:
            grid_frame = np.concatenate(row_images, axis=0)
            writer.append_data(grid_frame)

    writer.close()
    env.close()


def _batch_observation(observation):
    if isinstance(observation, (tuple, list)):
        return [np.expand_dims(x, axis=0) for x in observation]
    return np.expand_dims(observation, axis=0)


def rollout_policy_and_render_grid(
    policy,
    env_name: str = "Ant-v5",
    num_episodes: int = 10,
    episodes_per_interval: int = 5,
    intervals: int = 2,
    output_dir: str = "replay_buffer_videos",
    fps: int = 30,
    max_steps: int = 1000,
    use_ant_xml: bool = False,
    show_title: bool = True,
    deterministic: bool = True,
    env_kwargs: dict | None = None,
    ant_yaw_values: list[float] | None = None,
    sort_by_return: bool = True,
):
    """Roll out a d3rlpy policy, build a replay buffer, and render a grid video.

    This reuses the same utilities used by the buffer-based video renderer while
    minimizing new logic in this file.
    """
    from d3rlpy.dataset.components import Episode
    from d3rlpy.dataset.replay_buffer import create_infinite_replay_buffer

    # Create rollout env (no rendering needed here)
    env_kwargs = env_kwargs or {}
    if ant_yaw_values is not None:
        # Ensure deterministic reset so only yaw differs
        env_kwargs = {**env_kwargs, "reset_noise_scale": 0.0}
    if use_ant_xml and ("ant" in env_name.lower() or env_name.lower().startswith("mujoco/ant")):
        xml = ant_xml()
        env_kwargs = {**env_kwargs, "xml_file": xml}
        rollout_env = gym.make("Ant-v5", **env_kwargs)
        render_env = gym.make("Ant-v5", render_mode="rgb_array", **env_kwargs)
    else:
        rollout_env = gym.make(env_name, **env_kwargs)
        render_env = gym.make(env_name, render_mode="rgb_array", **env_kwargs)

    # Collect episodes
    collected_episodes = []
    for ep_idx in range(num_episodes):
        # If yaw sweep requested, set Ant's initial yaw before reset
        if ant_yaw_values is not None:
            env_id_lower = rollout_env.unwrapped.spec.id.lower()
            if "ant" not in env_id_lower:
                raise ValueError("--ant-yaw-sweep is only supported for Ant environments")
            _set_ant_yaw_init_qpos(rollout_env, float(ant_yaw_values[ep_idx]))
        obs, info = rollout_env.reset()
        obs_list = [obs]
        act_list = []
        rew_list = []
        terminated = False

        for _t in range(max_steps):
            batched_obs = _batch_observation(obs)
            if policy is None:
                action = rollout_env.action_space.sample()
            else:
                if deterministic:
                    action = policy.predict(batched_obs)[0]
                else:
                    action = policy.sample_action(batched_obs)[0]

            next_obs, reward, term, trunc, info = rollout_env.step(action)

            act_list.append(np.array(action))
            rew_list.append(np.array([reward], dtype=np.float32))
            obs_list.append(next_obs)

            obs = next_obs
            terminated = bool(term)
            if term or trunc:
                break

        episode = Episode(
            observations=np.asarray(obs_list),
            actions=np.asarray(act_list),
            rewards=np.asarray(rew_list, dtype=np.float32),
            terminated=terminated,
        )
        collected_episodes.append(episode)

    # Build a replay buffer from collected episodes
    buffer = create_infinite_replay_buffer(episodes=collected_episodes, env=rollout_env)

    # Either keep order or sort by return
    if sort_by_return:
        episodes = extract_episodes_from_buffer(buffer, max_episodes=num_episodes)
    else:
        episodes = collected_episodes[:num_episodes]

    # Prepare grid exactly like in main()
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(exist_ok=True)
    title_return = bool(show_title)

    # Construct rows
    interval_episodes = []
    for row in range(intervals):
        start_idx = row * episodes_per_interval
        end_idx = start_idx + episodes_per_interval
        row_eps = episodes[start_idx:end_idx]
        interval_episodes.append(row_eps)

    # Compose filename
    mj_env_name = render_env.unwrapped.spec.id
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    grid_video_path = output_dir_path / f"{mj_env_name}_policy_rollouts_grid_{ts}.mp4"

    # Render using existing streaming grid renderer
    render_episodes_grid_video(
        interval_episodes,
        render_env,
        grid_video_path,
        fps=fps,
        title_return=title_return,
        env_kwargs=env_kwargs,
    )

    rollout_env.close()
    # render_env is closed inside render_episodes_grid_video

    return str(grid_video_path)


def main():
    parser = argparse.ArgumentParser(
        description="Record videos from replay buffer episodes"
    )
    parser.add_argument(
        "--buffer-path",
        type=str,
        help="Path to pickle replay buffer file"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="Ant-v5",
        help="Environment name for rendering"
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=100,
        help="Maximum number of episodes to extract from buffer"
    )
    parser.add_argument(
        "--episodes-per-interval",
        type=int,
        default=5,
        help="Number of episodes per row in grid video"
    )
    parser.add_argument(
        "--intervals",
        type=int,
        default=5,
        help="Number of rows in grid video (shows top episodes by return)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="replay_buffer_videos",
        help="Output directory for videos"
    )
    parser.add_argument(
        "--single-episode",
        action="store_true",
        help="Record a single episode video instead of grid"
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="Index of episode to record when using --single-episode"
    )
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Disable matplotlib title showing total return"
    )
    parser.add_argument(
        "--use-ant-xml",
        action="store_true",
        help="Use customized Ant XML with recolored legs"
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=0,
        help="Number of worker processes for pre-rendering (0 disables parallelization)"
    )
    parser.add_argument(
        "--policy-path",
        type=str,
        help="Path to saved d3rlpy learnable (.d3) to rollout and record"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use greedy actions (predict) instead of sampling for policy rollouts"
    )
    parser.add_argument(
        "--policy-episodes",
        type=int,
        help="Number of episodes to rollout when using --policy-path (default: intervals * episodes-per-interval)"
    )
    parser.add_argument(
        "--ant-yaw-sweep",
        action="store_true",
        help="For Ant: create 25 episodes with yaw linearly spaced from -pi to pi (unsorted)"
    )

    args = parser.parse_args()

    # Validate mode
    if args.buffer_path and args.policy_path:
        raise SystemExit("Provide only one of --buffer-path or --policy-path, not both")
    if not args.buffer_path and not args.policy_path and not args.ant_yaw_sweep:
        raise SystemExit("You must provide either --buffer-path, --policy-path, or --ant-yaw-sweep")

    if args.policy_path:
        # Policy rollout mode: render grid from policy rollouts
        print(f"Loading policy from: {args.policy_path}")
        policy = d3rlpy.load_learnable(args.policy_path)
        print("Policy loaded successfully!")
        if args.ant_yaw_sweep:
            print("Ant yaw sweep enabled: generating 25 yaws from -pi to pi (unsorted)")
            yaw_values = np.linspace(-np.pi, np.pi, 25, endpoint=True).tolist()
            # Ensure grid defaults to 5x5 if user did not override
            num_eps = len(yaw_values)
            print(f"Rolling out policy for {num_eps} episode(s) with yaw sweep ...")
            video_path = rollout_policy_and_render_grid(
                policy=policy,
                env_name=args.env,
                num_episodes=num_eps,
                episodes_per_interval=args.episodes_per_interval,
                intervals=args.intervals,
                output_dir=args.output_dir,
                fps=30,
                max_steps=1000,
                use_ant_xml=args.use_ant_xml,
                show_title=not args.no_title,
                deterministic=args.deterministic,
                ant_yaw_values=yaw_values,
                sort_by_return=False,
            )
        else:
            num_eps = args.policy_episodes or (args.intervals * args.episodes_per_interval)
            print(f"Rolling out policy for {num_eps} episode(s) ...")
            video_path = rollout_policy_and_render_grid(
                policy=policy,
                env_name=args.env,
                num_episodes=num_eps,
                episodes_per_interval=args.episodes_per_interval,
                intervals=args.intervals,
                output_dir=args.output_dir,
                fps=30,
                max_steps=1000,
                use_ant_xml=args.use_ant_xml,
                show_title=not args.no_title,
                deterministic=args.deterministic,
            )
        print(f"Grid video saved to: {video_path}")
        return

    # Yaw sweep without a policy: use random actions
    if args.ant_yaw_sweep and not args.buffer_path:
        print("Ant yaw sweep enabled without a policy: using random actions")
        yaw_values = np.linspace(-np.pi, np.pi, 25, endpoint=True).tolist()
        num_eps = len(yaw_values)
        print(f"Rolling out for {num_eps} episode(s) with yaw sweep ...")
        video_path = rollout_policy_and_render_grid(
            policy=None,
            env_name=args.env,
            num_episodes=num_eps,
            episodes_per_interval=args.episodes_per_interval,
            intervals=args.intervals,
            output_dir=args.output_dir,
            fps=30,
            max_steps=1000,
            use_ant_xml=args.use_ant_xml,
            show_title=not args.no_title,
            deterministic=True,
            ant_yaw_values=yaw_values,
            sort_by_return=False,
        )
        print(f"Grid video saved to: {video_path}")
        return

    # Load replay buffer (buffer mode)
    print(f"Loading replay buffer from: {args.buffer_path}")
    buffer = load_replay_buffer(args.buffer_path)
    print(f"Buffer loaded successfully! Size: {buffer.size()} episodes, {buffer.transition_count} transitions")

    # Extract episodes
    print(f"Extracting episodes from buffer...")
    episodes = extract_episodes_from_buffer(buffer, args.max_episodes)
    print(f"Extracted {len(episodes)} episodes")
    
    # Show top episodes ranking
    if len(episodes) > 0:
        print("\nTop episodes by return:")
        for i, episode in enumerate(episodes[:min(10, len(episodes))]):  # Show top 10
            return_val = episode.compute_return()
            length = episode.size()
            print(f"  {i+1:2d}. Return: {return_val:8.2f}, Length: {length:3d} steps")
        if len(episodes) > 10:
            print(f"  ... and {len(episodes) - 10} more episodes")
        print()

    if len(episodes) == 0:
        print("No episodes found in buffer!")
        return

    # Create environment for rendering (optionally with custom Ant XML)
    if args.use_ant_xml and ("ant" in args.env.lower() or args.env.lower().startswith("mujoco/ant")):
        xml = ant_xml()
        env = gym.make("Ant-v5", xml_file=xml, render_mode="rgb_array")
        env_kwargs_local = {"xml_file": xml}
    else:
        env = gym.make(args.env, render_mode="rgb_array")
        env_kwargs_local = {}
    mj_env_name = env.unwrapped.spec.id

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Title settings
    title_return = not args.no_title
    if title_return:
        print("Matplotlib titles: total return will be shown per episode")
    else:
        print("Matplotlib titles: disabled")
    print()

    if args.single_episode:
        # Record single episode
        if args.episode_index >= len(episodes):
            print(f"Episode index {args.episode_index} out of range. Max: {len(episodes) - 1}")
            return
        
        episode = episodes[args.episode_index]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = output_dir / f"{mj_env_name}_episode_{args.episode_index}_{ts}.mp4"
        
        # Show episode info
        episode_return = episode.compute_return()
        episode_length = episode.size()
        print(f"Recording episode {args.episode_index} (Return: {episode_return:.2f}, Length: {episode_length} steps) to: {video_path}")
        render_episode(episode, env, video_path, title_return=title_return)
        print(f"Video saved to: {video_path}")
        
    else:
        # Record grid video
        # Since episodes are already ranked by returns, create a grid showing the best episodes
        print(f"Creating grid video with top episodes...")
        
        # Calculate how many episodes we can show
        total_episodes_to_show = min(args.intervals * args.episodes_per_interval, len(episodes))
        
        # Create grid layout
        interval_episodes = []
        for row in range(args.intervals):
            start_idx = row * args.episodes_per_interval
            end_idx = start_idx + args.episodes_per_interval
            row_episodes = episodes[start_idx:end_idx]
            
            if len(row_episodes) > 0:
                interval_episodes.append(row_episodes)
                print(f"  Row {row + 1}: Episodes {start_idx + 1}-{end_idx} (Returns: {row_episodes[0].compute_return():.2f} to {row_episodes[-1].compute_return():.2f})")
            else:
                interval_episodes.append([])
                print(f"  Row {row + 1}: No episodes")

        # Record grid video
        buffer_path = os.path.split(args.buffer_path)[1]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        grid_video_path = output_dir / f"{mj_env_name}_{buffer_path}_grid_{ts}.mp4"
        
        print(f"Recording grid video to: {grid_video_path}")
        render_episodes_grid_video(
            interval_episodes,
            env,
            grid_video_path,
            title_return=title_return,
            env_kwargs=env_kwargs_local,
            parallel_workers=max(0, int(args.parallel_workers)) or None,
        )
        
        print(
            f"Grid video saved ({args.intervals} rows x {args.episodes_per_interval} cols, top {total_episodes_to_show} episodes by return) to {grid_video_path}"
        )

    env.close()


if __name__ == "__main__":
    main() 