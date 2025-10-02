import argparse
import os
import pickle
import torch
import d3rlpy
import gymnasium as gym


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
        "--env",
        type=str,
        default="Ant-v5",
        help="Gymnasium environment ID for evaluation",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--policy-type",
        type=str,
        default="deterministic",
        choices=["deterministic", "stochastic"],
    )
    parser.add_argument("--no-compile", action="store_false", dest="compile")
    args = parser.parse_args()

    # Load replay buffer
    print(f"Loading replay buffer from: {args.buffer_path}")
    dataset = load_replay_buffer(args.buffer_path)
    print(
        f"Loaded buffer with {dataset.size()} episodes, {dataset.transition_count} transitions"
    )

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

    # Configure BC
    bc = d3rlpy.algos.BCConfig(
        learning_rate=1e-3,
        batch_size=256,
        policy_type=args.policy_type,
        observation_scaler=d3rlpy.preprocessing.StandardObservationScaler(),
        compile_graph=args.compile,
    ).create(device=device)

    # Train
    project = "bench-rlil"
    bc.fit(
        dataset,
        n_steps=1000000,
        n_steps_per_epoch=10000,
        save_interval=10,
        logdir='d3rlpy_logs',
        evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)},
        experiment_name=f"BC_from_buffer_{args.env}_{args.seed}",
        logger_adapter=d3rlpy.logging.WanDBAdapterFactory(project=project),
    )


if __name__ == "__main__":
    main()


