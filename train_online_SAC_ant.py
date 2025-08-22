import argparse
import torch
import d3rlpy
import os
import pickle
from pathlib import Path

os.environ["WANDB_API_KEY"] = "0e47b265815b455d6221285fb4a25202bf52c47b"

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Ant-v4")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-compile", action="store_false", dest="compile")
    parser.add_argument("--use-base-env", action="store_true", 
                       help="Use base environment without diagonal penalties")
    parser.add_argument("--diagonal-penalty", type=float, default=-1.0, 
                       help="Penalty for using diagonal legs (negative value)")
    parser.add_argument("--diagonal-type", type=int, default=1, choices=[1, 2],
                       help="Which diagonal to penalize: 1 (legs 1&4: front left + back right) or 2 (legs 2&3: front right + back left)")
    parser.add_argument("--save-buffer", action="store_true", 
                       help="Save replay buffer at the end of training")
    parser.add_argument("--buffer-dir", type=str, default="replay_buffers",
                       help="Directory to save replay buffers")
    args = parser.parse_args()

    # Create base environment
    import gymnasium as gym
    base_env = gym.make(args.env)
    
    # Choose environment based on flag
    if args.use_base_env:
        env = base_env
        print("Using base environment without diagonal penalties")
    else:
        # Import custom environment
        from custom_ant_env import DiagonalLegPenaltyWrapper
        
        # Wrap with custom penalty
        env = DiagonalLegPenaltyWrapper(
            base_env, 
            diagonal_type=args.diagonal_type,
            penalty_weight=args.diagonal_penalty
        )
        print(f"Using modified environment with diagonal {args.diagonal_type} penalty: {args.diagonal_penalty}")
    
    # Create evaluation environment (always use base env for fair evaluation)
    eval_env = gym.make(args.env)

    # Fix seed
    d3rlpy.seed(args.seed)
    d3rlpy.envs.seed_env(env, args.seed)
    d3rlpy.envs.seed_env(eval_env, args.seed)

    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        device = f"cuda:{torch.cuda.current_device()}"
    print(f"Using device: {device}")
    
    if not args.use_base_env:
        print(f"Diagonal leg penalty: {args.diagonal_penalty}")
        print(f"Penalizing diagonal type: {args.diagonal_type}")
    
    args.gpu = device

    # Setup SAC algorithm with standard hyperparameters from the codebase
    sac = d3rlpy.algos.SACConfig(
        batch_size=256,
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        temp_learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        n_critics=2,
        initial_temperature=1.0,
        compile_graph=args.compile,
    ).create(device=args.gpu)

    # Replay buffer for experience replay
    buffer = d3rlpy.dataset.create_fifo_replay_buffer(limit=1000000, env=env)

    # Start online training
    project = "bench-rlil"
    
    # Set experiment name based on environment type
    if args.use_base_env:
        exp_name = f"online_sac_base_ant_{args.seed}"
    else:
        exp_name = f"online_sac_custom_ant_{args.diagonal_type}_{args.diagonal_penalty}_{args.seed}"
    
    print(f"Starting training with experiment name: {exp_name}")
    
    sac.fit_online(
        env,
        buffer,
        eval_env=eval_env,
        n_steps=2000000,
        n_steps_per_epoch=10000,
        update_interval=1,
        update_start_step=1000,
        save_interval=10,
        experiment_name=exp_name,
        logger_adapter=d3rlpy.logging.WanDBAdapterFactory(project=project),
    )
    
    # Save replay buffer if requested
    if args.save_buffer:
        # Create buffer directory if it doesn't exist
        buffer_dir = Path(args.buffer_dir)
        buffer_dir.mkdir(exist_ok=True)
        
        # Generate buffer filename
        if args.use_base_env:
            buffer_filename = f"buffer_base_ant_{args.seed}.pkl"
        else:
            buffer_filename = f"buffer_custom_ant_{args.diagonal_type}_{args.diagonal_penalty}_{args.seed}.pkl"
        
        buffer_path = buffer_dir / buffer_filename
        
        # Save buffer
        print(f"Saving replay buffer to: {buffer_path}")
        with open(buffer_path, 'wb') as f:
            pickle.dump(buffer, f)
        
        # Print buffer statistics
        print(f"Buffer saved successfully!")
        print(f"Buffer size: {len(buffer)} transitions")
        print(f"Buffer capacity: {buffer.capacity}")


if __name__ == "__main__":
    main() 