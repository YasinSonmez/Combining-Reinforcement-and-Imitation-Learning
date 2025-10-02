import argparse
import os
import pickle
import random
import torch
import d3rlpy
import gymnasium as gym
from d3rlpy.dataset.buffers import InfiniteBuffer


os.environ["WANDB_API_KEY"] = "0e47b265815b455d6221285fb4a25202bf52c47b"


def load_replay_buffer(buffer_path: str) -> d3rlpy.dataset.ReplayBufferBase:
    with open(buffer_path, "rb") as f:
        buffer = pickle.load(f)
    return buffer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BC from a recorded replay buffer")
    parser.add_argument(
        "--buffer-path",
        type=str,
        default="buffer_base_ant_1.pkl",
        help="Path to the pickled replay buffer",
    )
    parser.add_argument(
        "--buffer-path-2",
        type=str,
        default=None,
        help="Path to the second pickled replay buffer (optional)",
    )
    parser.add_argument(
        "--mix-ratio",
        type=float,
        default=0.5,
        help="Mix ratio for second buffer (0.0 = only first buffer, 1.0 = only second buffer)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="Ant-v5",
        help="Gymnasium environment ID for evaluation",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--algo",
        type=str,
        default="bc",
        choices=["bc", "td3bc", "bcsac"],
        help="Which algorithm to train",
    )
    parser.add_argument(
        "--policy-type",
        type=str,
        default="deterministic",
        choices=["deterministic", "stochastic"],
    )
    parser.add_argument("--no-compile", action="store_false", dest="compile")
    args = parser.parse_args()

    # Load replay buffer(s)
    print(f"Loading replay buffer from: {args.buffer_path}")
    buffer1 = load_replay_buffer(args.buffer_path)
    print(
        f"Loaded buffer 1 with {buffer1.size()} episodes, {buffer1.transition_count} transitions"
    )

    if args.buffer_path_2 is not None:
        print(f"Loading second replay buffer from: {args.buffer_path_2}")
        buffer2 = load_replay_buffer(args.buffer_path_2)
        print(
            f"Loaded buffer 2 with {buffer2.size()} episodes, {buffer2.transition_count} transitions"
        )

        # Combine episodes from both buffers
        episodes1 = list(buffer1.episodes)
        episodes2 = list(buffer2.episodes)

        # Calculate how many episodes to take from each buffer based on mix ratio
        total_episodes = len(episodes1) + len(episodes2)
        episodes_from_buffer1 = int(len(episodes1) * (1 - args.mix_ratio))
        episodes_from_buffer2 = int(len(episodes2) * args.mix_ratio)

        # Ensure we have at least some episodes from each buffer
        episodes_from_buffer1 = max(1, episodes_from_buffer1)
        episodes_from_buffer2 = max(1, episodes_from_buffer2)

        # Sample episodes from each buffer
        sampled_episodes1 = random.sample(episodes1, min(episodes_from_buffer1, len(episodes1)))
        sampled_episodes2 = random.sample(episodes2, min(episodes_from_buffer2, len(episodes2)))

        # Combine episodes
        combined_episodes = sampled_episodes1 + sampled_episodes2

        # Create new replay buffer with combined episodes
        dataset = d3rlpy.dataset.ReplayBuffer(
            buffer=InfiniteBuffer(),
            episodes=combined_episodes
        )

        print(f"Combined {len(sampled_episodes1)} episodes from buffer 1 and {len(sampled_episodes2)} episodes from buffer 2")
        print(f"Final combined buffer has {dataset.size()} episodes, {dataset.transition_count} transitions")
    else:
        dataset = buffer1

    # Create evaluation environment
    env = gym.make(args.env)

    # Fix seed
    d3rlpy.seed(args.seed)
    d3rlpy.envs.seed_env(env, args.seed)

    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        device = f"cuda:{torch.cuda.current_device()}"
    print(f"Using device: {device}")

    # Configure algorithm
    if args.algo == "bc":
        algo = d3rlpy.algos.BCConfig(
            learning_rate=1e-3,
            batch_size=256,
            policy_type=args.policy_type,
            observation_scaler=d3rlpy.preprocessing.StandardObservationScaler(),
            compile_graph=args.compile,
        ).create(device=device)
    elif args.algo == "td3bc":
        algo = d3rlpy.algos.TD3PlusBCConfig(
            actor_learning_rate=3e-4,
            critic_learning_rate=3e-4,
            batch_size=256,
            target_smoothing_sigma=0.2,
            target_smoothing_clip=0.5,
            alpha=2.5,
            update_actor_interval=2,
            observation_scaler=d3rlpy.preprocessing.StandardObservationScaler(),
            compile_graph=args.compile,
        ).create(device=device)
    elif args.algo == "bcsac":
        algo = d3rlpy.algos.BCSACConfig(
            actor_learning_rate=3e-4,
            critic_learning_rate=3e-4,
            temp_learning_rate=3e-4,
            batch_size=256,
            bc_lambda=2.5,
            observation_scaler=d3rlpy.preprocessing.StandardObservationScaler(),
            compile_graph=args.compile,
        ).create(device=device)
    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")

    # Train
    project = "bench-rlil"

    logger_adapter = d3rlpy.logging.CombineAdapterFactory([
        d3rlpy.logging.FileAdapterFactory(root_dir="d3rlpy_logs"),
        d3rlpy.logging.WanDBAdapterFactory(project=project),
    ])

    algo.fit(
        dataset,
        n_steps=200000,
        n_steps_per_epoch=5000,
        save_interval=10,
        evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)},
        experiment_name=f"{args.algo}_from_buffer_{args.env}_{args.seed}" + (f"_mixed_{args.mix_ratio}" if args.buffer_path_2 else ""),
        logger_adapter=logger_adapter,
    )


if __name__ == "__main__":
    main()


