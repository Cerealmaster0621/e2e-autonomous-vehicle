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


class RandomLightingWrapper(gym.ObservationWrapper):
    """
    Random brightness/contrast augmentation for Sim-to-Real transfer.
    Only applied during training (stochastic), not during testing.
    """
    
    def __init__(self, env, brightness_range=(-50, 50), contrast_range=(0.7, 1.3), probability=0.5):
        super().__init__(env)
        self.brightness_range = brightness_range  # (min, max) additive brightness
        self.contrast_range = contrast_range      # (min, max) multiplicative contrast
        self.probability = probability            # Probability of applying augmentation
    
    def observation(self, obs):
        if np.random.random() > self.probability:
            return obs
        
        # Random brightness adjustment
        brightness = np.random.uniform(*self.brightness_range)
        
        # Random contrast adjustment
        contrast = np.random.uniform(*self.contrast_range)
        
        # Apply: output = contrast * (input - mean) + mean + brightness
        obs_float = obs.astype(np.float32)
        mean = np.mean(obs_float)
        augmented = contrast * (obs_float - mean) + mean + brightness
        
        # Clip to valid range
        augmented = np.clip(augmented, 0, 255).astype(np.uint8)
        
        return augmented


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


class GaussianNoiseWrapper(gym.ObservationWrapper):
    """
    Add Gaussian noise to simulate sensor noise.
    Helps with Sim-to-Real transfer.
    """
    
    def __init__(self, env, std=10.0, probability=0.5):
        super().__init__(env)
        self.std = std                  # Standard deviation of noise
        self.probability = probability  # Probability of applying noise
    
    def observation(self, obs):
        if np.random.random() > self.probability:
            return obs
        
        noise = np.random.normal(0, self.std, obs.shape).astype(np.float32)
        noisy = obs.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        
        return noisy


class CannyEdgeWrapper(gym.ObservationWrapper):
    """
    Apply Canny edge detection for structure extraction.
    Extracts lane lines and road boundaries while removing texture details.
    """
    
    def __init__(self, env, low_threshold=50, high_threshold=150):
        super().__init__(env)
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        
        old_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(old_shape[0], old_shape[1], 1),
            dtype=np.uint8
        )
    
    def observation(self, obs):
        # If already grayscale (H, W, 1), squeeze to (H, W)
        if len(obs.shape) == 3 and obs.shape[2] == 1:
            gray = obs.squeeze()
        elif len(obs.shape) == 3 and obs.shape[2] == 3:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            gray = obs
        
        # Apply Gaussian blur to reduce noise before edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, self.low_threshold, self.high_threshold)
        
        return np.expand_dims(edges, axis=-1)


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


class CutoutWrapper(gym.ObservationWrapper):
    """
    Random cutout augmentation for occlusion robustness.
    Randomly masks rectangular regions of the image.
    """
    
    def __init__(self, env, num_cutouts=1, cutout_size=(10, 20), probability=0.5):
        super().__init__(env)
        self.num_cutouts = num_cutouts      # Number of cutout regions
        self.cutout_size = cutout_size      # (min_size, max_size) for width/height
        self.probability = probability      # Probability of applying cutout
    
    def observation(self, obs):
        if np.random.random() > self.probability:
            return obs
        
        obs = obs.copy()
        h, w = obs.shape[:2]
        
        for _ in range(self.num_cutouts):
            # Random cutout size
            cut_h = np.random.randint(self.cutout_size[0], self.cutout_size[1] + 1)
            cut_w = np.random.randint(self.cutout_size[0], self.cutout_size[1] + 1)
            
            # Random position
            y = np.random.randint(0, max(1, h - cut_h))
            x = np.random.randint(0, max(1, w - cut_w))
            
            # Apply cutout (set to 0 or mean value)
            obs[y:y+cut_h, x:x+cut_w] = 0
        
        return obs


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
    """
    Apply wrappers in the correct order based on config.
    
    Order:
    1. ROI Crop
    2. Resize
    3. Random Lighting (augmentation)
    4. Grayscale
    5. Gaussian Noise (augmentation)
    6. Canny Edge OR Binary Threshold
    7. Cutout (augmentation)
    8. Normalization (disabled by default for SB3 CnnPolicy)
    9. Frame Stacking
    """
    wrapper_cfg = config.get("wrapper", {})
    
    # 1. ROI Crop (remove sky/irrelevant areas)
    if wrapper_cfg.get("roi_crop"):
        crop = tuple(wrapper_cfg["roi_crop"])
        env = ROICropWrapper(env, crop=crop)
    
    # 2. Resize
    if wrapper_cfg.get("resize"):
        size = tuple(wrapper_cfg["resize"])
        env = ResizeWrapper(env, size=size)
    
    # 3. Random Lighting augmentation (before grayscale)
    if wrapper_cfg.get("random_lighting", {}).get("enabled", False):
        lighting_cfg = wrapper_cfg["random_lighting"]
        env = RandomLightingWrapper(
            env,
            brightness_range=tuple(lighting_cfg.get("brightness_range", [-50, 50])),
            contrast_range=tuple(lighting_cfg.get("contrast_range", [0.7, 1.3])),
            probability=lighting_cfg.get("probability", 0.5)
        )
    
    # 4. Grayscale conversion
    if wrapper_cfg.get("grayscale", False):
        env = GrayscaleWrapper(env)
    
    # 5. Gaussian Noise augmentation
    if wrapper_cfg.get("gaussian_noise", {}).get("enabled", False):
        noise_cfg = wrapper_cfg["gaussian_noise"]
        env = GaussianNoiseWrapper(
            env,
            std=noise_cfg.get("std", 10.0),
            probability=noise_cfg.get("probability", 0.5)
        )
    
    # 6. Edge detection OR Binary thresholding (mutually exclusive)
    if wrapper_cfg.get("canny_edge", {}).get("enabled", False):
        edge_cfg = wrapper_cfg["canny_edge"]
        env = CannyEdgeWrapper(
            env,
            low_threshold=edge_cfg.get("low_threshold", 50),
            high_threshold=edge_cfg.get("high_threshold", 150)
        )
    elif wrapper_cfg.get("binary_threshold") is not None:
        env = BinaryThresholdWrapper(env, threshold=wrapper_cfg["binary_threshold"])
    
    # 7. Cutout augmentation
    if wrapper_cfg.get("cutout", {}).get("enabled", False):
        cutout_cfg = wrapper_cfg["cutout"]
        env = CutoutWrapper(
            env,
            num_cutouts=cutout_cfg.get("num_cutouts", 1),
            cutout_size=tuple(cutout_cfg.get("cutout_size", [10, 20])),
            probability=cutout_cfg.get("probability", 0.5)
        )
    
    # 8. Normalization (disabled by default - SB3 CnnPolicy handles it)
    if wrapper_cfg.get("normalize", False):
        env = NormalizeWrapper(env)
    
    # 9. Frame stacking (keep uint8 for SB3 compatibility)
    if wrapper_cfg.get("frame_stack", 1) > 1:
        env = FrameStackWrapper(env, n_frames=wrapper_cfg["frame_stack"])
    
    return env
