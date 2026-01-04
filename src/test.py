import argparse
import os
import sys
import uuid

import gymnasium as gym
import gym_donkeycar
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from configs.load_config import load_config
from envs.reward import default_reward, reset_reward_history
from envs.wrapper import make_wrapped_env

if __name__ == "__main__":
    # load configs
    config = load_config()
    conf = config["conf"]
    ppo_config = config["ppo"]
    
    # Initialize the donkey environment
    # where env_name one of:
    env_list = [
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

    parser = argparse.ArgumentParser(description="test")
    parser.add_argument(
        "--sim",
        type=str,
        default="sim_path",
        help="path to unity simulator. maybe be left at manual if you would like to start the sim on your own.",
    )
    parser.add_argument("--port", type=int, default=9091, help="port to use for tcp")
    parser.add_argument("--test", action="store_true", help="load the trained model and play")
    parser.add_argument("--multi", action="store_true", help="start multiple sims at once")
    parser.add_argument(
        "--env_name", type=str, default="donkey-warehouse-v0", help="name of donkey sim environment", choices=env_list
    )
    parser.add_argument(
        "--resume", type=str, default=None, 
        help="path to checkpoint to resume training from (e.g., src/models/saved/checkpoints/ppo_donkey_40960_steps.zip)"
    )
    parser.add_argument(
        "--checkpoint_freq", type=int, default=10, 
        help="save checkpoint every N iterations (default: 10, means every 10*2048=20480 steps)"
    )

    args = parser.parse_args()

    if args.sim == "sim_path" and args.multi:
        print("you must supply the sim path with --sim when running multiple environments")
        exit(1)

    env_id = args.env_name
    
    # Fill in CLI arguments
    conf["exe_path"] = args.sim
    conf["port"] = args.port
    conf["guid"] = str(uuid.uuid4())

    if args.test:
        # Make an environment test our trained policy
        env = gym.make(args.env_name, conf=conf)
        env.unwrapped.set_reward_fn(default_reward)
        
        # Apply image processing wrappers
        env = make_wrapped_env(env, config)

        # Load model WITH environment to ensure observation space is properly handled
        model = PPO.load(ppo_config["save_path"], env=env)
        print(f"Timesteps trained: {model.num_timesteps}")

        obs, info = env.reset()
        reset_reward_history(env.unwrapped.viewer.handler)
        
        # Debug: Print observation and action info
        print(f"Observation shape from env: {obs.shape}")
        print(f"Model observation space: {model.observation_space.shape}")
        
        step_count = 0
        for _ in range(10000):
            action, _states = model.predict(obs, deterministic=True)
            
            # Debug: Print first 5 actions to verify model is outputting meaningful values
            if step_count < 5:
                print(f"Step {step_count}: steering={action[0]:.4f}, throttle={action[1]:.4f}")
            step_count += 1
            
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            
            if terminated or truncated:
                obs, info = env.reset()
                reset_reward_history(env.unwrapped.viewer.handler)

        print("done testing")

    else:
        # make gym env
        env = gym.make(args.env_name, conf=conf)
        env.unwrapped.set_reward_fn(default_reward)
        
        # Apply image processing wrappers
        env = make_wrapped_env(env, config)
        
        # Setup checkpoint directory
        checkpoint_dir = "src/models/saved/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Checkpoint callback: saves every N iterations (N * n_steps timesteps)
        n_steps = ppo_config.get("n_steps", 2048)
        checkpoint_freq = args.checkpoint_freq * n_steps  # Convert iterations to timesteps
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=checkpoint_dir,
            name_prefix="ppo_donkey",
            save_replay_buffer=False,
            save_vecnormalize=True,
            verbose=1
        )
        
        if args.resume:
            # Resume training from checkpoint
            print(f"Resuming training from: {args.resume}")
            model = PPO.load(
                args.resume, 
                env=env,
                tensorboard_log="src/logs/",
            )
            # Update learning rate if different from checkpoint
            model.learning_rate = ppo_config.get("learning_rate", 1e-4)
        else:
            # Create new model
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
                tensorboard_log="src/logs/",
            )

        print(f"Training for {ppo_config['total_timesteps']} timesteps...")
        print(f"Checkpoints will be saved every {args.checkpoint_freq} iterations ({checkpoint_freq} steps)")
        print(f"Checkpoint directory: {checkpoint_dir}")
        
        # Train the model with checkpoint callback
        model.learn(
            total_timesteps=ppo_config["total_timesteps"],
            callback=checkpoint_callback,
            reset_num_timesteps=(args.resume is None),  # Continue timestep count if resuming
        )

        # Save the final model
        model.save(ppo_config["save_path"])
        print(f"Final model saved to: {ppo_config['save_path']}")
        print("Training complete!")

    env.close()
    sys.exit(0)
