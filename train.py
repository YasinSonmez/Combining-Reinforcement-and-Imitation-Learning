import argparse
import torch
import d3rlpy
import os

os.environ["WANDB_API_KEY"] = "0e47b265815b455d6221285fb4a25202bf52c47b"

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="hopper-expert-v2")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--algo", type=str, default="bcsac", choices=["bcsac", "td3bc"])
    parser.add_argument("--no-compile", action="store_false", dest="compile")
    args = parser.parse_args()

    if "mujoco" in args.dataset:
        dataset, env = d3rlpy.datasets.get_minari(args.dataset)
    else:
        dataset, env = d3rlpy.datasets.get_dataset(args.dataset)

    # Fix seed
    d3rlpy.seed(args.seed)
    d3rlpy.envs.seed_env(env, args.seed)

    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        device = f"cuda:{torch.cuda.current_device()}"
    print(f"Using device: {device}")
    args.gpu = device

    # Choose algorithm
    if args.algo == "bcsac":
        algo = d3rlpy.algos.BCSACConfig(
            actor_learning_rate=3e-4,
            critic_learning_rate=3e-4,
            temp_learning_rate=3e-4,
            batch_size=256,
            bc_lambda=2.5,
            observation_scaler=d3rlpy.preprocessing.StandardObservationScaler(),
            compile_graph=args.compile,
        ).create(device=args.gpu)
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
        ).create(device=args.gpu)
    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")

    # Train
    project = "bench-rlil"
    algo.fit(
        dataset,
        n_steps=200000,
        n_steps_per_epoch=5000,
        save_interval=10,
        evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)},
        experiment_name=f"{args.algo}_{args.dataset}_{args.seed}",
        logger_adapter=d3rlpy.logging.WanDBAdapterFactory(project=project),
    )


if __name__ == "__main__":
    main()
