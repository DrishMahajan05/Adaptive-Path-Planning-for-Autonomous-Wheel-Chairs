"""
train_rl.py
===========
Training script for the wheelchair RL agent using PPO (Stable-Baselines3).

The agent learns small corrections on top of the classical pipeline to
produce smoother, collision-free paths.

Usage:
    python train_rl.py                          # Train for 200k steps
    python train_rl.py --timesteps 500000       # Train longer
    python train_rl.py --resume                 # Continue from checkpoint
    python train_rl.py --timesteps 100000 --n-envs 4   # Parallel envs
"""

import argparse
import os
import sys

import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def make_env(seed: int):
    """Factory function for creating a single env instance."""
    def _init():
        from ras.rl.rl_env import WheelchairRLEnv
        env = WheelchairRLEnv(num_obstacles=3, max_steps=2000, seed=seed)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(
        description="Train wheelchair RL agent with PPO")
    parser.add_argument("--timesteps", type=int, default=3_000_000,
                        help="Total training timesteps (default: 3000000)")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Number of parallel environments (default: 4)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--model-dir", type=str, default="rl_models",
                        help="Directory for model checkpoints")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    args = parser.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    from stable_baselines3.common.callbacks import (
        CheckpointCallback, EvalCallback, CallbackList
    )

    # ── Create output directory ──
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "wheelchair_ppo")
    best_model_path = os.path.join(args.model_dir, "best_model")

    # ── Create training environments ──
    print(f"Creating {args.n_envs} parallel training environments...")
    if args.n_envs > 1:
        train_env = SubprocVecEnv(
            [make_env(seed=i * 42) for i in range(args.n_envs)])
    else:
        train_env = DummyVecEnv(
            [make_env(seed=0)])

    # ── Create eval environment ──
    eval_env = DummyVecEnv([make_env(seed=999)])

    # ── Load or create model ──
    if args.resume and os.path.exists(model_path + ".zip"):
        print(f"Resuming training from {model_path}.zip")
        model = PPO.load(model_path, env=train_env)
        model.learning_rate = args.lr
    else:
        print("Creating new PPO model...")
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=args.lr,
            n_steps=1024,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            policy_kwargs=dict(
                net_arch=dict(pi=[128, 128], vf=[128, 128]),
            ),
        )

    # ── Callbacks ──
    checkpoint_cb = CheckpointCallback(
        save_freq=max(10_000 // args.n_envs, 1000),
        save_path=args.model_dir,
        name_prefix="wheelchair_ppo_ckpt",
        verbose=1,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=args.model_dir,
        log_path=args.model_dir,
        eval_freq=max(20_000 // args.n_envs, 2000),
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )
    callbacks = CallbackList([checkpoint_cb, eval_cb])

    # ── Train ──
    print(f"\n{'=' * 56}")
    print(f"  PPO Training — {args.timesteps:,} timesteps")
    print(f"  Environments: {args.n_envs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Model dir: {args.model_dir}/")
    print(f"{'=' * 56}\n")

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    # ── Save final model ──
    model.save(model_path)
    print(f"\nModel saved to {model_path}.zip")

    # ── Quick evaluation ──
    print("\nRunning final evaluation (10 episodes)...")
    from stable_baselines3.common.evaluation import evaluate_policy
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True)
    print(f"Mean reward: {mean_reward:.1f} +/- {std_reward:.1f}")

    train_env.close()
    eval_env.close()
    print("Done!")


if __name__ == "__main__":
    main()
