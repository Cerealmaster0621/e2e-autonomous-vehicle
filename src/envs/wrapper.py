"""
Wrapper for the DonkeyCar environment

Image Processing Order:
1. ROI Crop: Remove unnecessary sky/hood
2. Resize: Downsample to reduce computation
3. Random Lighting: Augment brightness (Sim-to-Real)
4. Grayscale: Reduce dimensionality
5. Gaussian Noise: Augment sensor noise
6. Canny Edge: Structure extraction (replaces Binary Threshold)
7. Cutout: Augment occlusion robustness
8. Normalization: Scale to [0, 1]
9. Frame Stacking: Temporal information
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2
from collections import deque


class ROICropWrapper(gym.ObservationWrapper):
    """Crop region of interest - removes sky and irrelevant areas."""
    
    def __init__(self, env, crop=(40, 0, 0, 0)):
        super().__init__(env)
        self.crop = crop  # (top, bottom, left, right)
        old_shape = self.observation_space.shape
        
        new_h = old_shape[0] - crop[0] - crop[1]
        new_w = old_shape[1] - crop[2] - crop[3]
        
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(new_h, new_w, old_shape[2]),
            dtype=np.uint8
        )
    
    def observation(self, obs):
        top, bottom, left, right = self.crop
        h, w = obs.shape[:2]
        
        bottom_idx = h - bottom if bottom > 0 else h
        right_idx = w - right if right > 0 else w
        
        return obs[top:bottom_idx, left:right_idx]


class ResizeWrapper(gym.ObservationWrapper):
    """Resize observations to target size."""
    
    def __init__(self, env, size=(80, 80)):
        super().__init__(env)
        self.size = size  # (width, height)
        old_shape = self.observation_space.shape
        self.n_channels = old_shape[2]
        
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(size[1], size[0], old_shape[2]),
            dtype=np.uint8
        )
    
    def observation(self, obs):
        resized = cv2.resize(obs, self.size, interpolation=cv2.INTER_AREA)
        # cv2.resize drops channel dim for single-channel images, restore it
        if len(resized.shape) == 2:
            resized = np.expand_dims(resized, axis=-1)
        return resized


class GrayscaleWrapper(gym.ObservationWrapper):
    """Convert RGB observations to grayscale."""
    
    def __init__(self, env):
        super().__init__(env)
        old_shape = self.observation_space.shape
        # (H, W, C) -> (H, W, 1)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(old_shape[0], old_shape[1], 1),
            dtype=np.uint8
        )
    
    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return np.expand_dims(gray, axis=-1)


class BinaryThresholdWrapper(gym.ObservationWrapper):
    """Apply binary thresholding to observations."""
    
    def __init__(self, env, threshold=128):
        super().__init__(env)
        self.threshold = threshold
    
    def observation(self, obs):
        # Convert to grayscale if needed
        if len(obs.shape) == 3 and obs.shape[2] == 3:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            gray = obs.squeeze() if len(obs.shape) == 3 else obs
        
        _, binary = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
        return np.expand_dims(binary, axis=-1)


class NormalizeWrapper(gym.ObservationWrapper):
    """Normalize observations to [0, 1] range."""
    
    def __init__(self, env):
        super().__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=old_shape,
            dtype=np.float32
        )
    
    def observation(self, obs):
        return obs.astype(np.float32) / 255.0


class FrameStackWrapper(gym.ObservationWrapper):
    """Stack multiple frames for temporal information."""
    
    def __init__(self, env, n_frames=4):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)
        
        old_shape = self.observation_space.shape
        # Stack along channel dimension, keep uint8 for SB3 CnnPolicy compatibility
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(old_shape[0], old_shape[1], old_shape[2] * n_frames),
            dtype=np.uint8
        )
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_obs(), info
    
    def observation(self, obs):
        self.frames.append(obs)
        return self._get_obs()
    
    def _get_obs(self):
        return np.concatenate(list(self.frames), axis=-1).astype(np.uint8)


def make_wrapped_env(env, config):
    wrapper_cfg = config.get("wrapper", {})
    
    # 1. ROI Crop (remove sky/irrelevant areas)
    if wrapper_cfg.get("roi_crop"):
        crop = tuple(wrapper_cfg["roi_crop"])
        env = ROICropWrapper(env, crop=crop)
    
    # 2. Grayscale conversion
    if wrapper_cfg.get("grayscale", False):
        env = GrayscaleWrapper(env)
    
    # 3. Binary thresholding (optional, after grayscale)
    if wrapper_cfg.get("binary_threshold") is not None:
        env = BinaryThresholdWrapper(env, threshold=wrapper_cfg["binary_threshold"])
    
    # 4. Resize
    if wrapper_cfg.get("resize"):
        size = tuple(wrapper_cfg["resize"])
        env = ResizeWrapper(env, size=size)
    
    # 5. Frame stacking (keep uint8 for SB3 compatibility)
    if wrapper_cfg.get("frame_stack", 1) > 1:
        env = FrameStackWrapper(env, n_frames=wrapper_cfg["frame_stack"])
    
    return env
