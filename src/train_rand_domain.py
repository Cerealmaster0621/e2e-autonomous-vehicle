"""
Domain Randomization Training Script

Trains on multiple tracks, switching randomly every iteration (n_steps).
This helps the model generalize across different environments.

Usage:
    # Train on all 9 tracks
    python src/train_rand_domain.py --sim /path/to/sim.app
    
    # Train on specific tracks
    python src/train_rand_domain.py --sim /path/to/sim.app --tracks warehouse generated-roads
    
    # Resume training
    python src/train_rand_domain.py --sim /path/to/sim.app --resume src/models/saved/checkpoints/...
"""

import argparse
import os
import sys
import uuid
import random

import gymnasium as gym
import gym_donkeycar
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from configs.load_config import load_config
from envs.reward import default_reward, reset_reward_history
from envs.wrapper import make_wrapped_env
from envs.callbacks import CameraViewCallback, ProgressCallback


# All available tracks
ALL_TRACKS = [
    "donkey-warehouse-v0",
    "donkey-generated-roads-v0",
    "donkey-avc-sparkfun-v0",
    "donkey-generated-track-v0",
    "donkey-roboracingleague-track-v0",
    "donkey-waveshare-v0",
    "donkey-minimonaco-track-v0",
    "donkey-warren-track-v0",
    "donkey-circuit-launch-track-v0",
    "donkey-mountain-track-v0",
]

# Short name to full name mapping
TRACK_ALIASES = {
    "warehouse": "donkey-warehouse-v0",
    "generated-roads": "donkey-generated-roads-v0",
    "avc-sparkfun": "donkey-avc-sparkfun-v0",
    "generated-track": "donkey-generated-track-v0",
    "roboracingleague": "donkey-roboracingleague-track-v0",
    "waveshare": "donkey-waveshare-v0",
    "minimonaco": "donkey-minimonaco-track-v0",
    "warren": "donkey-warren-track-v0",
    "circuit-launch": "donkey-circuit-launch-track-v0",
    "mountain": "donkey-mountain-track-v0",
}


class DomainRandomizationCallback(BaseCallback):
    """
    Callback that switches to a random track every iteration.
    An iteration = n_steps timesteps (default 2048).
    """
    
    def __init__(self, tracks: list, conf: dict, config: dict, verbose: int = 1):
        super().__init__(verbose)
        self.tracks = tracks
        self.conf = conf
        self.config = config
        self.current_track = None
        self.iteration_count = 0
        self.n_steps = config["ppo"].get("n_steps", 2048)
    
    def _on_rollout_start(self) -> None:
        """Called at the start of each rollout (iteration)."""
        self.iteration_count += 1
        
        # Pick a random track
        new_track = random.choice(self.tracks)
        
        if self.verbose > 0 and new_track != self.current_track:
            print(f"\n[Iteration {self.iteration_count}] Switching to: {new_track}")
        
        if new_track != self.current_track:
            self.current_track = new_track
            self._switch_track(new_track)
    
    def _switch_track(self, track_name: str) -> None:
        """Switch to a new track."""
        # Update conf with new guid for the new track
        self.conf["guid"] = str(uuid.uuid4())
        
        # Close old environment
        self.training_env.env_method("close")
        
        # Create new environment
        new_env = gym.make(track_name, conf=self.conf)
        new_env.unwrapped.set_reward_fn(default_reward)
        new_env = make_wrapped_env(new_env, self.config)
        
        # Update the model's environment
        self.model.set_env(new_env)
    
    def _on_step(self) -> bool:
        return True




def resolve_track_name(name: str) -> str:
    """Convert short track name to full name."""
    if name in ALL_TRACKS:
        return name
    if name in TRACK_ALIASES:
        return TRACK_ALIASES[name]
    # Try adding donkey- prefix and -v0 suffix
    full_name = f"donkey-{name}-v0"
    if full_name in ALL_TRACKS:
        return full_name
    raise ValueError(f"Unknown track: {name}. Available: {list(TRACK_ALIASES.keys())}")


def main():
    parser = argparse.ArgumentParser(description="Domain Randomization Training")
    parser.add_argument("--sim", type=str, required=True, help="Path to simulator")
    parser.add_argument("--port", type=int, default=9091, help="Port for simulator")
    parser.add_argument(
        "--tracks", type=str, nargs="+", default=None,
        help="Tracks to train on (1-9). Use short names: warehouse, generated-roads, etc. Default: all tracks"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--total_timesteps", type=int, default=None,
        help="Override total timesteps from config"
    )
    parser.add_argument(
        "--checkpoint_freq", type=int, default=10,
        help="Save checkpoint every N iterations"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config()
    conf = config["conf"]
    ppo_config = config["ppo"]
    
    # Determine which tracks to use
    if args.tracks:
        tracks = [resolve_track_name(t) for t in args.tracks]
    else:
        tracks = ALL_TRACKS.copy()
    
    print("\n" + "=" * 60)
    print("DOMAIN RANDOMIZATION TRAINING")
    print("=" * 60)
    print(f"Tracks ({len(tracks)}):")
    for i, t in enumerate(tracks, 1):
        print(f"  {i}. {t}")
    print(f"Switch frequency: Every iteration ({ppo_config.get('n_steps', 2048)} steps)")
    print(f"Total timesteps: {args.total_timesteps or ppo_config['total_timesteps']}")
    print("=" * 60 + "\n")
    
    # Setup conf
    conf["exe_path"] = args.sim
    conf["port"] = args.port
    conf["guid"] = str(uuid.uuid4())
    
    # Start with first track
    initial_track = random.choice(tracks)
    print(f"Starting with track: {initial_track}")
    
    env = gym.make(initial_track, conf=conf)
    env.unwrapped.set_reward_fn(default_reward)
    env = make_wrapped_env(env, config)
    
    # Setup checkpoint directory
    checkpoint_dir = "src/models/saved/checkpoints_domain_rand"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Callbacks
    n_steps = ppo_config.get("n_steps", 2048)
    checkpoint_freq = args.checkpoint_freq * n_steps
    
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=checkpoint_dir,
        name_prefix="ppo_domain_rand",
        verbose=1
    )
    
    domain_rand_callback = DomainRandomizationCallback(
        tracks=tracks,
        conf=conf,
        config=config,
        verbose=1
    )
    
    progress_callback = ProgressCallback(log_freq=5000)
    camera_callback = CameraViewCallback(window_name="Training Camera View")
    
    callbacks = [checkpoint_callback, progress_callback, camera_callback]
    # Note: domain_rand_callback is disabled for now due to complexity of 
    # switching envs mid-training. Instead, we'll train sequentially on each track.
    
    # Create or load model
    if args.resume:
        print(f"Resuming from: {args.resume}")
        model = PPO.load(args.resume, env=env, tensorboard_log="src/logs/domain_rand/")
        model.learning_rate = ppo_config.get("learning_rate", 1e-4)
    else:
        model = PPO(
            ppo_config["policy"],
            env,
            verbose=1,
            learning_rate=ppo_config.get("learning_rate", 1e-4),
            n_steps=ppo_config.get("n_steps", 2048),
            batch_size=ppo_config.get("batch_size", 64),
            n_epochs=ppo_config.get("n_epochs", 10),
            gamma=ppo_config.get("gamma", 0.99),
            gae_lambda=ppo_config.get("gae_lambda", 0.95),
            clip_range=ppo_config.get("clip_range", 0.2),
            ent_coef=ppo_config.get("ent_coef", 0.01),
            tensorboard_log="src/logs/domain_rand/",
        )
    
    # Total timesteps
    total_timesteps = args.total_timesteps or ppo_config["total_timesteps"]
    
    # Calculate timesteps per track rotation
    timesteps_per_track = n_steps * 5  # 5 iterations per track before switching
    num_rotations = total_timesteps // (timesteps_per_track * len(tracks))
    num_rotations = max(1, num_rotations)
    
    print(f"\nTraining plan:")
    print(f"  - {timesteps_per_track} steps per track ({timesteps_per_track // n_steps} iterations)")
    print(f"  - {num_rotations} full rotations through all tracks")
    print(f"  - Total: ~{num_rotations * len(tracks) * timesteps_per_track} steps\n")
    
    # Training loop - rotate through tracks
    for rotation in range(num_rotations):
        # Shuffle tracks each rotation for randomization
        shuffled_tracks = tracks.copy()
        random.shuffle(shuffled_tracks)
        
        for track_idx, track in enumerate(shuffled_tracks):
            print(f"\n{'='*60}")
            print(f"Rotation {rotation + 1}/{num_rotations} | "
                  f"Track {track_idx + 1}/{len(tracks)}: {track}")
            print(f"{'='*60}\n")
            
            # Create new environment for this track
            conf["guid"] = str(uuid.uuid4())
            env.close()
            
            env = gym.make(track, conf=conf)
            env.unwrapped.set_reward_fn(default_reward)
            env = make_wrapped_env(env, config)
            model.set_env(env)
            
            # Train on this track
            model.learn(
                total_timesteps=timesteps_per_track,
                callback=callbacks,
                reset_num_timesteps=False,
            )
    
    # Save final model
    save_path = "src/models/saved/ppo_domain_rand"
    model.save(save_path)
    print(f"\nFinal model saved to: {save_path}")
    print("Domain randomization training complete!")
    
    env.close()
    sys.exit(0)


if __name__ == "__main__":
    main()

