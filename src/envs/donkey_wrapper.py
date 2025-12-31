import gym_donkeycar.envs  # Must import .envs to register environments
import gymnasium as gym
import numpy as np
import os
from gymnasium import spaces

class DonkeyGymnasiumWrapper(gym.Env):
    """
    gym-donkeycar(Old Gym) -> Gymnasium(New Gym) adaptor
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, env_name, conf):
        # Set environment variables for donkey simulator configuration
        os.environ['DONKEY_SIM_PATH'] = conf.get('exe_path', '')
        os.environ['DONKEY_SIM_PORT'] = str(conf.get('port', 9091))
        os.environ['DONKEY_SIM_HEADLESS'] = '0'  # Show UI by default
        
        # 1. generate old gym env
        import gym as old_gym
        self.env = old_gym.make(env_name)

        # 2. Space definition - Gymnasium style mapping
        # Action: [Steer, Throttle]
        self.action_space = spaces.Box(
            low=self.env.action_space.low,
            high=self.env.action_space.high,
            dtype=np.float32
        )

        # Observation: (Height, Width, 3) RGB Image
        self.observation_space = spaces.Box(
            low=self.env.observation_space.low,
            high=self.env.observation_space.high,
            dtype=np.uint8
        )

    def reset(self, seed=None, options=None):
        # old: obs
        obs = self.env.reset()

        # new: (obs, info)
        info = {}
        return obs, info

    def step(self, action):
        # old: obs, reward, done, info
        obs, reward, done, info = self.env.step(action)

        # new: obs, reward, terminated, truncated, info
        terminated = done
        truncated = False

        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()