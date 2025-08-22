import argparse
import os
import pickle
from itertools import zip_longest
from pathlib import Path
import xml.etree.ElementTree as ET
from importlib import resources
import tempfile

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


def render_episode(episode, env, output_path, fps=30, max_frames=None, title_return=True):
    """Render a single episode to video."""
    writer = imageio.get_writer(output_path, fps=fps)

    frames = rollout_frames(env, episode)
    # Use matplotlib title for return text
    if title_return:
        ret = episode.compute_return()
        frames = add_title_with_matplotlib(frames, f"Return: {ret:.2f}")
    for f in frames:
        writer.append_data(f)

    writer.close()
    env.close()


def render_episodes_grid_video(
    interval_episodes, env, output_path, fps=30, max_frames=None, title_return=True
):
    """Render multiple episodes in a grid format."""
    fps = 30
    writer = imageio.get_writer(output_path, fps=fps)

    # Create frames with optional matplotlib titles (returns)
    frames_grid = []
    episode_counter = 0
    
    for row_idx, row_episodes in enumerate(interval_episodes):
        row_frames = []
        for col_idx, episode in enumerate(row_episodes):
            frames = rollout_frames(env, episode)
            if title_return:
                ret = episode.compute_return()
                frames = add_title_with_matplotlib(frames, f"Return: {ret:.2f}")
            
            row_frames.append(frames)
            episode_counter += 1
        
        frames_grid.append(row_frames)

    len(frames_grid)
    nc = max(len(row) for row in frames_grid)

    # padding
    h, w, c = frames_grid[0][0][0].shape
    blank = np.zeros((h, w, c), dtype=np.uint8)

    for row in frames_grid:
        while len(row) < nc:
            row.append([])

    row_iters = [zip_longest(*row, fillvalue=blank) for row in frames_grid]

    for row_frames in zip_longest(*row_iters, fillvalue=(blank,) * nc):
        # row_frames is an nr tuple, each elem has nc frames
        row_images = [np.concatenate(frames, axis=1) for frames in row_frames]
        grid_frame = np.concatenate(row_images, axis=0)
        writer.append_data(grid_frame)

    writer.close()
    env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Record videos from replay buffer episodes"
    )
    parser.add_argument(
        "--buffer-path",
        type=str,
        required=True,
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

    args = parser.parse_args()

    # Load replay buffer
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
    else:
        env = gym.make(args.env, render_mode="rgb_array")
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
        video_path = output_dir / f"{mj_env_name}_episode_{args.episode_index}.mp4"
        
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
        grid_video_path = output_dir / f"{mj_env_name}_replay_buffer_grid.mp4"
        
        print(f"Recording grid video to: {grid_video_path}")
        render_episodes_grid_video(interval_episodes, env, grid_video_path, title_return=title_return)
        
        print(
            f"Grid video saved ({args.intervals} rows x {args.episodes_per_interval} cols, top {total_episodes_to_show} episodes by return) to {grid_video_path}"
        )

    env.close()


if __name__ == "__main__":
    main() 