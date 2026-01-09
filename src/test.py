import argparse
import os
import sys
import uuid

import gymnasium as gym
import gym_donkeycar
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import get_schedule_fn

import cv2
import numpy as np

from configs.load_config import load_config
from envs.reward import default_reward, reset_reward_history
from envs.wrapper import make_wrapped_env
from envs.callbacks import CameraViewCallback, display_observation

# Post-Mortem Analysis imports
from analysis.visual_backprop import VisualBackProp
from utils.recorder import BlackBoxRecorder

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
        # We need access to the raw environment for raw images
        raw_env = gym.make(args.env_name, conf=conf)
        raw_env.unwrapped.set_reward_fn(default_reward)
        
        # Apply image processing wrappers
        env = make_wrapped_env(raw_env, config)

        # Load model WITH environment to ensure observation space is properly handled
        model = PPO.load(ppo_config["save_path"], env=env)
        print(f"Timesteps trained: {model.num_timesteps}")
        
        # ========== POST-MORTEM ANALYSIS SETUP ==========
        pm_config = config.get("post_mortem", {})
        pm_enabled = pm_config.get("enabled", False)
        
        recorder = None
        vbp = None
        
        if pm_enabled:
            # Get ROI crop from wrapper config for proper attention map alignment
            wrapper_cfg = config.get("wrapper", {})
            roi_crop = wrapper_cfg.get("roi_crop", None)
            if roi_crop:
                roi_crop = tuple(roi_crop)  # Convert list to tuple
            
            # Initialize BlackBoxRecorder (ring buffer)
            recorder = BlackBoxRecorder(
                buffer_seconds=pm_config.get("buffer_seconds", 4.0),
                fps=pm_config.get("fps", 20),
                roi_crop=roi_crop  # Pass crop params for visualization alignment
            )
            
            # Initialize VisualBackProp (will register hooks on CNN)
            vbp = VisualBackProp(model)
            
            print(f"\n[Post-Mortem] Enabled - Recording last {pm_config.get('buffer_seconds', 4.0)}s before crashes")
            print(f"[Post-Mortem] Output directory: {pm_config.get('output_dir', 'src/logs/crashes')}\n")
        # ================================================

        obs, info = env.reset()
        reset_reward_history(env.unwrapped.viewer.handler)
        
        # Start recording for this episode
        if recorder is not None:
            recorder.on_episode_start()
        
        # Debug: Print observation and action info
        print(f"Observation shape from env: {obs.shape}")
        print(f"Model observation space: {model.observation_space.shape}")
        
        step_count = 0
        print("\nPress 'q' in the camera window to quit testing.\n")
        
        for _ in range(10000):
            action, _states = model.predict(obs, deterministic=True)
            
            # Debug: Print first 5 actions to verify model is outputting meaningful values
            if step_count < 5:
                print(f"Step {step_count}: steering={action[0]:.4f}, throttle={action[1]:.4f}")
            step_count += 1
            
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            
            # ========== RECORD FRAME FOR POST-MORTEM ==========
            # Capture raw image AFTER step (when new frame has arrived)
            if recorder is not None:
                try:
                    # Access the raw image from the simulator handler
                    # In gym-donkeycar, the image is stored in handler.image_array
                    handler = env.unwrapped.viewer.handler
                    raw_image = handler.image_array.copy()
                    
                    # Build telemetry info
                    telemetry = {
                        'cte': getattr(handler, 'cte', 0.0),
                        'speed': getattr(handler, 'forward_vel', 0.0),
                        'hit': getattr(handler, 'hit', 'none'),
                    }
                    
                    recorder.record(
                        raw_image=raw_image,
                        processed_obs=obs,
                        action=action,
                        reward=reward,
                        info=telemetry
                    )
                except Exception as e:
                    if step_count <= 5:  # Only print first few errors
                        print(f"[Post-Mortem] Warning: Could not capture frame: {e}")
            # ==================================================
            
            # Display what the model sees
            display_observation(obs, window_name="Model Camera View")
            
            # Check for 'q' key to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quit requested by user")
                break
            
            if terminated or truncated:
                # ========== TRIGGER POST-MORTEM ON CRASH ==========
                if recorder is not None and pm_config.get("auto_save_on_crash", True):
                    print(f"\n[Post-Mortem] Crash detected! Processing {len(recorder)} buffered frames...")
                    
                    # Save post-mortem video with attention maps
                    video_path = recorder.save_post_mortem(
                        vbp=vbp if pm_config.get("include_attention", True) else None,
                        output_dir=pm_config.get("output_dir", "src/analysis/crashes"),
                        include_attention=pm_config.get("include_attention", True),
                        include_telemetry=pm_config.get("include_telemetry", True),
                        codec=pm_config.get("video_codec", "mp4v")
                    )
                    
                    # Optionally display replay
                    if pm_config.get("display_replay", False) and video_path:
                        recorder.display_buffer_live(vbp, window_name="Crash Replay")
                # ==================================================
                
                obs, info = env.reset()
                reset_reward_history(env.unwrapped.viewer.handler)
                
                # Start fresh recording for new episode
                if recorder is not None:
                    recorder.on_episode_start()

        cv2.destroyAllWindows()
        
        # Cleanup
        if vbp is not None:
            vbp.close()
        
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
            # Apply ALL hyperparameters from config (overwrite checkpoint values)
            new_lr = ppo_config.get("learning_rate", 1e-4)
            new_clip = ppo_config.get("clip_range", 0.2)
            
            model.learning_rate = get_schedule_fn(new_lr)
            model.lr_schedule = get_schedule_fn(new_lr)
            model.n_epochs = ppo_config.get("n_epochs", 10)
            model.batch_size = ppo_config.get("batch_size", 64)
            model.gamma = ppo_config.get("gamma", 0.99)
            model.gae_lambda = ppo_config.get("gae_lambda", 0.95)
            model.clip_range = get_schedule_fn(new_clip)
            model.ent_coef = ppo_config.get("ent_coef", 0.01)
            
            print(f"[Config Override] Applied new hyperparameters from default.yaml:")
            print(f"  learning_rate={new_lr}, n_epochs={model.n_epochs}, "
                  f"batch_size={model.batch_size}, clip_range={new_clip}, ent_coef={model.ent_coef}")
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
        
        # Setup camera view callback for visualization
        camera_callback = CameraViewCallback(window_name="Training Camera View", display_freq=10)
        callbacks = [checkpoint_callback]#, camera_callback]
        
        # Train the model with callbacks
        model.learn(
            total_timesteps=ppo_config["total_timesteps"],
            callback=callbacks,
            reset_num_timesteps=(args.resume is None),  # Continue timestep count if resuming
        )
        
        cv2.destroyAllWindows()

        # Save the final model
        model.save(ppo_config["save_path"])
        print(f"Final model saved to: {ppo_config['save_path']}")
        print("Training complete!")

    env.close()
    sys.exit(0)
