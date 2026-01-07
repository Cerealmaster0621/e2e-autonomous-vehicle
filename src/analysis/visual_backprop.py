"""
VisualBackProp Implementation for Post-Mortem Analysis

Algorithm:
1. Capture feature maps from all convolutional layers using forward hooks
2. Average feature maps in each layer to get a single activation map
3. Starting from the deepest layer, upscale to match previous layer's spatial size
4. Point-wise multiply with averaged feature maps of the previous layer
5. Repeat until reaching input image size
6. The result is a saliency mask highlighting important regions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Union
from collections import OrderedDict


class VisualBackProp:
    
    def __init__(self, model, device: str = "auto"):
        self.model = model
        self.policy = model.policy
        
        # Determine device
        if device == "auto":
            self.device = next(self.policy.parameters()).device
        else:
            self.device = torch.device(device)
        
        # Storage for feature maps captured by hooks
        self.feature_maps: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        
        # Track conv layer info for proper upscaling
        self.conv_layers_info: List[Dict] = []
        
        # Register hooks on conv layers
        self._register_hooks()
    
    def _register_hooks(self):
        self._remove_hooks()  # Clean up any existing hooks
        self.feature_maps.clear()
        self.conv_layers_info.clear()
        
        # Get the features extractor (CNN part of the policy)
        features_extractor = self.policy.features_extractor
        
        # Find all Conv2d layers recursively
        conv_idx = 0
        for name, module in features_extractor.named_modules():
            if isinstance(module, nn.Conv2d):
                layer_name = f"conv_{conv_idx}"
                hook = module.register_forward_hook(
                    self._create_hook(layer_name)
                )
                self.hooks.append(hook)
                
                # Store layer info for upscaling calculations
                self.conv_layers_info.append({
                    'name': layer_name,
                    'module': module,
                    'kernel_size': module.kernel_size,
                    'stride': module.stride,
                    'padding': module.padding
                })
                conv_idx += 1
        
        if conv_idx == 0:
            raise ValueError("No Conv2d layers found in features_extractor. "
                           "Ensure the model uses CnnPolicy or similar.")
        
        print(f"[VisualBackProp] Registered hooks on {conv_idx} convolutional layers")
    
    def _create_hook(self, layer_name: str):
        def hook(module, input, output):
            # Store the output feature maps
            # output shape: (batch, channels, height, width)
            self.feature_maps[layer_name] = output.detach()
        return hook
    
    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def _forward_pass(self, obs: Union[np.ndarray, torch.Tensor]) -> None:
        self.feature_maps.clear()
        
        # Ensure tensor format
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs)
        
        # Add batch dimension if needed
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)
        
        # Ensure correct dtype and device
        obs = obs.float().to(self.device)
        
        # Run forward pass through features extractor only
        # This triggers all the hooks
        with torch.no_grad():
            _ = self.policy.features_extractor(obs)
    
    def compute_attention_mask(
        self, 
        obs: Union[np.ndarray, torch.Tensor],
        target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        # Handle input format
        if isinstance(obs, np.ndarray):
            if len(obs.shape) == 3 and obs.shape[-1] in [1, 3, 4]:
                # Channels-last format (H, W, C) -> (C, H, W)
                obs = np.transpose(obs, (2, 0, 1))
            input_shape = obs.shape[-2:]  # (H, W)
        else:
            if len(obs.shape) == 3 and obs.shape[-1] in [1, 3, 4]:
                obs = obs.permute(2, 0, 1)
            input_shape = obs.shape[-2:]
        
        if target_size is None:
            target_size = input_shape
        
        # Run forward pass to capture feature maps
        self._forward_pass(obs)
        
        if len(self.feature_maps) == 0:
            raise RuntimeError("No feature maps captured. Ensure hooks are registered.")
        
        # Get feature maps in order (from shallow to deep)
        layer_names = list(self.feature_maps.keys())
        
        # Start from the deepest layer
        # Average across channels: (batch, C, H, W) -> (batch, 1, H, W)
        mask = self.feature_maps[layer_names[-1]].mean(dim=1, keepdim=True)
        
        # Apply ReLU to keep only positive activations
        mask = F.relu(mask)
        
        # Propagate backwards through layers (deep to shallow)
        for i in range(len(layer_names) - 2, -1, -1):
            layer_name = layer_names[i]
            feature_map = self.feature_maps[layer_name]
            
            # Average across channels
            avg_feature = feature_map.mean(dim=1, keepdim=True)
            avg_feature = F.relu(avg_feature)
            
            # Upscale mask to match this layer's spatial dimensions
            target_h, target_w = avg_feature.shape[2], avg_feature.shape[3]
            mask = F.interpolate(
                mask, 
                size=(target_h, target_w), 
                mode='bilinear', 
                align_corners=False
            )
            
            # Point-wise multiplication
            mask = mask * avg_feature
        
        # Final upscale to target size
        mask = F.interpolate(
            mask, 
            size=target_size, 
            mode='bilinear', 
            align_corners=False
        )
        
        # Convert to numpy and normalize to [0, 1]
        mask = mask.squeeze().cpu().numpy()
        
        # Normalize
        mask_min = mask.min()
        mask_max = mask.max()
        if mask_max - mask_min > 1e-8:
            mask = (mask - mask_min) / (mask_max - mask_min)
        else:
            mask = np.zeros_like(mask)
        
        return mask
    
    def compute_batch_attention_masks(
        self, 
        observations: Union[np.ndarray, torch.Tensor],
        target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        # Handle input format
        if isinstance(observations, np.ndarray):
            # Check if channels-last format
            if len(observations.shape) == 4 and observations.shape[-1] in [1, 3, 4]:
                # (B, H, W, C) -> (B, C, H, W)
                observations = np.transpose(observations, (0, 3, 1, 2))
            input_shape = observations.shape[-2:]
            observations = torch.from_numpy(observations)
        else:
            if len(observations.shape) == 4 and observations.shape[-1] in [1, 3, 4]:
                observations = observations.permute(0, 3, 1, 2)
            input_shape = observations.shape[-2:]
        
        if target_size is None:
            target_size = input_shape
        
        batch_size = observations.shape[0]
        
        # Clear and run forward pass
        self.feature_maps.clear()
        observations = observations.float().to(self.device)
        
        with torch.no_grad():
            _ = self.policy.features_extractor(observations)
        
        # Get layer names
        layer_names = list(self.feature_maps.keys())
        
        # Start from deepest layer
        mask = self.feature_maps[layer_names[-1]].mean(dim=1, keepdim=True)
        mask = F.relu(mask)
        
        # Propagate backwards
        for i in range(len(layer_names) - 2, -1, -1):
            layer_name = layer_names[i]
            feature_map = self.feature_maps[layer_name]
            
            avg_feature = feature_map.mean(dim=1, keepdim=True)
            avg_feature = F.relu(avg_feature)
            
            target_h, target_w = avg_feature.shape[2], avg_feature.shape[3]
            mask = F.interpolate(
                mask, 
                size=(target_h, target_w), 
                mode='bilinear', 
                align_corners=False
            )
            
            mask = mask * avg_feature
        
        # Final upscale
        mask = F.interpolate(
            mask, 
            size=target_size, 
            mode='bilinear', 
            align_corners=False
        )
        
        # Convert to numpy (batch, H, W)
        masks = mask.squeeze(1).cpu().numpy()
        
        # Normalize each mask individually
        for i in range(batch_size):
            m = masks[i]
            m_min, m_max = m.min(), m.max()
            if m_max - m_min > 1e-8:
                masks[i] = (m - m_min) / (m_max - m_min)
            else:
                masks[i] = np.zeros_like(m)
        
        return masks
    
    def overlay_heatmap(
        self, 
        image: np.ndarray, 
        mask: np.ndarray,
        colormap: int = cv2.COLORMAP_JET,
        alpha: float = 0.5
    ) -> np.ndarray:
        # Ensure image is RGB and uint8
        if image.dtype != np.uint8:
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
        
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image.squeeze(), cv2.COLOR_GRAY2RGB)
        
        # Resize mask to match image if needed
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        
        # Convert mask to heatmap
        heatmap = (mask * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Blend
        blended = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        
        return blended
    
    def create_visualization_frame(
        self,
        raw_image: np.ndarray,
        processed_obs: np.ndarray,
        mask: np.ndarray,
        frame_idx: int = 0,
        total_frames: int = 0,
        extra_info: Optional[Dict] = None
    ) -> np.ndarray:
        h, w = raw_image.shape[:2]
        
        # Ensure raw image is RGB uint8
        if raw_image.dtype != np.uint8:
            raw_image = np.clip(raw_image * 255, 0, 255).astype(np.uint8)
        
        # Create heatmap overlay on raw image
        overlay = self.overlay_heatmap(raw_image.copy(), mask, alpha=0.6)
        
        # Prepare processed observation for display
        proc_display = processed_obs.copy()
        
        # Handle frame stacking - extract latest frame
        if len(proc_display.shape) == 3 and proc_display.shape[-1] > 3:
            # Frame stacked: (H, W, N*C) -> take last channel
            proc_display = proc_display[:, :, -1]
        elif len(proc_display.shape) == 3 and proc_display.shape[-1] == 1:
            proc_display = proc_display.squeeze()
        
        # Convert to proper display format [0, 255]
        # Handle various input formats:
        # - float32 [0, 1]: multiply by 255
        # - uint8 [0, 1]: from normalize->frame_stack bug, multiply by 255
        # - uint8 [0, 255]: already correct
        proc_display = proc_display.astype(np.float32)
        max_val = proc_display.max()
        
        if max_val <= 1.0 + 1e-6:
            # Values in [0, 1] range (normalized or edge detection with normalize)
            proc_display = proc_display * 255.0
        elif max_val < 255.0:
            # Values in some intermediate range, normalize to full range
            if max_val > 0:
                proc_display = (proc_display / max_val) * 255.0
        # else: already in [0, 255] range
        
        proc_display = np.clip(proc_display, 0, 255).astype(np.uint8)
        
        # Convert grayscale to RGB for display
        if len(proc_display.shape) == 2:
            proc_display = cv2.cvtColor(proc_display, cv2.COLOR_GRAY2RGB)
        elif len(proc_display.shape) == 3 and proc_display.shape[-1] == 1:
            proc_display = cv2.cvtColor(proc_display.squeeze(), cv2.COLOR_GRAY2RGB)
        
        # Resize for consistent display
        display_size = (w, h)
        overlay = cv2.resize(overlay, display_size)
        proc_display = cv2.resize(proc_display, display_size)
        
        # Horizontal concatenation: [Raw | Overlay | Processed]
        combined = np.hstack([raw_image, overlay, proc_display])
        
        # Add text annotations (small font - 20% of original)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35  # Reduced from 0.5
        color = (255, 255, 255)
        thickness = 1
        
        # Labels at top of each panel
        cv2.putText(combined, "Raw", (5, 12), font, font_scale, color, thickness)
        cv2.putText(combined, "Attention", (w + 5, 12), font, font_scale, color, thickness)
        cv2.putText(combined, "Model Input", (2 * w + 5, 12), font, font_scale, color, thickness)
        
        # Frame counter (bottom left)
        if total_frames > 0:
            time_text = f"{frame_idx + 1}/{total_frames}"
            cv2.putText(combined, time_text, (5, h - 5), font, font_scale, color, thickness)
        
        # Telemetry info on left edge (below "Raw" label)
        if extra_info:
            y_offset = 24
            for key, value in extra_info.items():
                if isinstance(value, float):
                    text = f"{key}: {value:.2f}"
                else:
                    text = f"{key}: {value}"
                cv2.putText(combined, text, (5, y_offset), font, font_scale * 0.9, color, thickness)
                y_offset += 10
        
        return combined
    
    def __del__(self):
        self._remove_hooks()
    
    def close(self):
        self._remove_hooks()
        self.feature_maps.clear()
