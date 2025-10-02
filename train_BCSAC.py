import argparse
import torch
import d3rlpy
import os

os.environ["WANDB_API_KEY"] = "0e47b265815b455d6221285fb4a25202bf52c47b"

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="hopper-expert-v2")
    parser.add_argument("--seed", type=int, default=1)
    # parser.add_argument("--gpu", type=int)
    parser.add_argument("--no-compile", action="store_false", dest="compile")
    args = parser.parse_args()

    if "mujoco" in args.dataset:
        dataset, env = d3rlpy.datasets.get_minari(args.dataset)
    else:
        dataset, env = d3rlpy.datasets.get_dataset(args.dataset) # for D4RL datasets

    # fix seed
    d3rlpy.seed(args.seed)
    d3rlpy.envs.seed_env(env, args.seed)

    # Check if a GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Optionally, use the first available GPU (cuda:0)
    if device == "cuda":
        device = f"cuda:{torch.cuda.current_device()}"
    print(f"Using device: {device}")
    args.gpu = device

    bcsac = d3rlpy.algos.BCSACConfig(
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        temp_learning_rate=3e-4,
        batch_size=256,
        bc_lambda=10,
        observation_scaler=d3rlpy.preprocessing.StandardObservationScaler(),
        compile_graph=args.compile,
    ).create(device=args.gpu)

    project ="bench-rlil"
    bcsac.fit(
        dataset,
        n_steps=200000,
        n_steps_per_epoch=5000,
        save_interval=10,
        evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)},
        experiment_name=f"BC_obs_scaler_{args.dataset}_{args.seed}",
        logger_adapter=d3rlpy.logging.WanDBAdapterFactory(project=project),
    )


if __name__ == "__main__":
    main()
