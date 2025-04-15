import argparse
import torch
import d3rlpy
import os

os.environ["WANDB_API_KEY"] = "0e47b265815b455d62564281fb4a25202bf52"

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

    td3 = d3rlpy.algos.TD3PlusBCConfig(
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

    project ="bench-rlil"
    td3.fit(
        dataset,
        n_steps=500000,
        n_steps_per_epoch=5000,
        save_interval=10,
        evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)},
        experiment_name=f"TD3PlusBC_{args.dataset}_{args.seed}",
        logger_adapter=d3rlpy.logging.WanDBAdapterFactory(project=project),
    )


if __name__ == "__main__":
    main()
