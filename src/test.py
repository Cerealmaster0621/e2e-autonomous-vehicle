import argparse
import uuid

import gym
import gym_donkeycar
from stable_baselines3 import PPO

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

    conf = {
        "exe_path": args.sim,
        "host": "127.0.0.1",
        "port": args.port,
        "body_style": "donkey",
        "body_rgb": (128, 128, 128),
        "car_name": "me",
        "font_size": 100,
        "racer_name": "TEST",
        "country": "KOREA",
        "guid": str(uuid.uuid4()),
        "max_cte": 10,
    }

    if args.test:
        # Make an environment test our trained policy
        env = gym.make(args.env_name, conf=conf)
        env.unwrapped.set_reward_fn(custom_reward)

        model = PPO.load("ppo_donkey")

        obs = env.reset()
        for _ in range(10):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()
            if done:
                obs = env.reset()

        print("done testing")

    else:
        # make gym env
        env = gym.make(args.env_name, conf=conf)
        env.unwrapped.set_reward_fn(custom_reward)
        # create cnn policy(mlp won't work if the environment is not a image)
        model = PPO("CnnPolicy", env, verbose=1)

        # set up model in learning mode with goal number of timesteps to complete
        model.learn(total_timesteps=10)

        obs = env.reset()

        for i in range(10):
            action, _states = model.predict(obs, deterministic=True)

            obs, reward, done, info = env.step(action)

            try:
                env.render()
            except Exception as e:
                print(e)
                print("failure in render, continuing...")

            if done:
                obs = env.reset()

            if i % 100 == 0:
                print("saving...")
                model.save("ppo_donkey")

        # Save the agent
        model.save("ppo_donkey")
        print("done training")

    env.close()
