"""
Custom Callbacks for Training

Includes:
- CameraViewCallback: Display what the model sees during training/testing
- ProgressCallback: Log training progress
"""

import cv2
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class CameraViewCallback(BaseCallback):
    
    def __init__(self, window_name: str = "Donkey Car Camera View", 
                 display_freq: int = 1,
                 verbose: int = 0):
        super().__init__(verbose)
        self.window_name = window_name
        self.display_freq = display_freq
    
    def _on_step(self) -> bool:
        # Only update display at specified frequency
        if self.n_calls % self.display_freq != 0:
            return True
        
        # Get the current observation from the model
        # SB3 observations are in shape (batch, channels, height, width)
        obs = self.locals.get('new_obs')
        if obs is None:
            return True
        
        # Get first environment's image: (Channels, Height, Width)
        img = obs[0]
        
        # Transpose from (C, H, W) to (H, W, C) for OpenCV
        img = np.transpose(img, (1, 2, 0))
        
        # Convert normalized [0,1] to [0,255] if needed
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        
        # Handle different channel configurations
        if img.shape[2] == 1:
            # Single channel grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            # Frame stacked grayscale - show only the latest frame
            img = img[:, :, -1]
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 3:
            # RGB to BGR for OpenCV
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif img.shape[2] > 4:
            # Multiple stacked frames - show latest
            # Assume last N channels are the most recent frame
            img = img[:, :, -1]
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Resize for better visibility
        img = cv2.resize(img, (320, 320), interpolation=cv2.INTER_NEAREST)
        
        # Add step info overlay
        cv2.putText(img, f"Step: {self.num_timesteps}", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Display
        cv2.imshow(self.window_name, img)
        cv2.waitKey(1)
        
        return True
    
    def _on_training_end(self) -> None:
        cv2.destroyAllWindows()


class ProgressCallback(BaseCallback):
    """Log training progress with episode statistics."""
    
    def __init__(self, log_freq: int = 5000, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
    
    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0 and len(self.model.ep_info_buffer) > 0:
            ep_rewards = [ep["r"] for ep in self.model.ep_info_buffer]
            ep_lengths = [ep["l"] for ep in self.model.ep_info_buffer]
            print(f"[Step {self.num_timesteps}] "
                  f"Mean reward: {np.mean(ep_rewards):.2f}, "
                  f"Mean length: {np.mean(ep_lengths):.1f}")
        return True


def display_observation(obs, window_name: str = "Observation"):
    img = obs.copy()
    
    # If channels first, transpose to channels last
    if len(img.shape) == 3 and img.shape[0] in [1, 3, 4]:
        img = np.transpose(img, (1, 2, 0))
    
    # Convert to uint8 if normalized
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    
    # Handle channel configurations
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img.squeeze(), cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img[:, :, -1], cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Resize for visibility
    img = cv2.resize(img, (320, 320), interpolation=cv2.INTER_NEAREST)
    
    cv2.imshow(window_name, img)
    cv2.waitKey(1)

