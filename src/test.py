import argparse
import uuid

import gymnasium as gym
import gym_donkeycar
from stable_baselines3 import PPO

from configs.load_config import load_config
from envs.test_reward import custom_reward

if __name__ == "__main__":
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
        "donkey-thunderhill-track-v0",
        "donkey-circuit-launch-track-v0",
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

    args = parser.parse_args()

    if args.sim == "sim_path" and args.multi:
        print("you must supply the sim path with --sim when running multiple environments")
        exit(1)

    env_id = args.env_name
    
    # load configs
    config = load_config()
    conf = config["conf"]
    ppo_config = config["ppo"]
    
    conf["exe_path"] = args.sim
    conf["port"] = args.port
    conf["guid"] = str(uuid.uuid4())
    

    if args.test:
        # Make an environment test our trained policy
        env = gym.make(args.env_name, conf=conf)
        env.unwrapped.set_reward_fn(custom_reward)

        model = PPO.load(ppo_config["save_path"])

        obs, info = env.reset()
        for _ in range(10):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            if terminated or truncated:
                obs, info = env.reset()

        print("done testing")

    else:
        # make gym env
        env = gym.make(args.env_name, conf=conf)
        env.unwrapped.set_reward_fn(custom_reward)
        # create cnn policy(mlp won't work if the environment is not a image)
        model = PPO(ppo_config["policy"], env, verbose=1)

        # set up model in learning mode with goal number of timesteps to complete
        model.learn(total_timesteps=ppo_config["total_timesteps"])

        obs, info = env.reset()

        for i in range(10):
            action, _states = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)

            try:
                env.render()
            except Exception as e:
                print(e)
                print("failure in render, continuing...")

            if terminated or truncated:
                obs, info = env.reset()

            if i % 100 == 0:
                print("saving...")
                model.save(ppo_config["save_path"])

        # Save the agent
        model.save(ppo_config["save_path"])
        print("done training")

    env.close()
