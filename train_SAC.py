import argparse
import torch
import d3rlpy
import os
# import wandb
# wandb.login(anonymous='never')  # optional, forces offline to not ask for login
# wandb.init(mode="offline", settings=wandb.Settings(init_timeout=300))
# os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_API_KEY"] = "0e47b265815b455d6221285fb4a25202bf52c47b"

def main() -> None:
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", type=str, default="hopper-medium-v0")
    # parser.add_argument("--dataset", type=str, default="mujoco/hopper/expert-v0")
    parser.add_argument("--dataset", type=str, default="mujoco/invertedpendulum/expert-v0")
    parser.add_argument("--seed", type=int, default=1)
    # parser.add_argument("--gpu", type=int)
    parser.add_argument("--no-compile", action="store_false", dest="compile")
    args = parser.parse_args()

    # dataset, env = d3rlpy.datasets.get_dataset(args.dataset)
    dataset, env = d3rlpy.datasets.get_minari(args.dataset)

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
    project ="bench-rlil"

    encoder = d3rlpy.models.encoders.VectorEncoderFactory([256, 256, 256])

    if "medium-v0" in args.dataset:
        conservative_weight = 10.0
    else:
        conservative_weight = 5.0

    cql = d3rlpy.algos.CQLConfig(
        actor_learning_rate=1e-4,
        critic_learning_rate=3e-4,
        temp_learning_rate=1e-4,
        actor_encoder_factory=encoder,
        critic_encoder_factory=encoder,
        batch_size=256,
        n_action_samples=10,
        alpha_learning_rate=0.0,
        conservative_weight=conservative_weight,
        # compile_graph=args.compile,
        compile_graph=False
    ).create(device=args.gpu)

    cql.fit(
        dataset,
        n_steps=500000,
        n_steps_per_epoch=1000,
        save_interval=10,
        evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)},
        experiment_name=f"CQL_{args.dataset}_{args.seed}",
        logger_adapter=d3rlpy.logging.WanDBAdapterFactory(project=project),
    )


if __name__ == "__main__":
    main()
