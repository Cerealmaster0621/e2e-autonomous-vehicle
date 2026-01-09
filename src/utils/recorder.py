"""
This module provides circular buffer to store the last N seconds of driving data
(raw images + processed observations) and converts them to video when a crash occurs
"""

import os
import cv2
import numpy as np
from collections import deque
from datetime import datetime
from typing import Deque, Tuple, Optional, Dict, List, Any, Union
from dataclasses import dataclass, field


@dataclass
class FrameData:
    """Container for a single frame's data."""
    raw_image: np.ndarray          # Original RGB image before any processing
    processed_obs: np.ndarray      # Processed observation (what model sees)
    action: Optional[np.ndarray] = None  # Action taken
    reward: Optional[float] = None       # Reward received
    info: Optional[Dict] = None          # Additional info (steering, throttle, etc.)
    timestamp: float = 0.0               # Simulation timestamp


class BlackBoxRecorder:
    """
    Usage:
        recorder = BlackBoxRecorder(buffer_seconds=4.0, fps=20)
        
        # During driving loop:
        recorder.record(raw_image, processed_obs, action, reward, info)
        
        # On crash:
        if done:
            recorder.save_post_mortem(vbp, output_dir="crashes/")
    """
    
    def __init__(
        self, 
        buffer_seconds: float = 4.0,
        fps: int = 20,
        raw_image_size: Optional[Tuple[int, int]] = None,
        roi_crop: Optional[Tuple[int, int, int, int]] = None
    ):
        self.buffer_seconds = buffer_seconds
        self.fps = fps
        self.raw_image_size = raw_image_size
        self.roi_crop = roi_crop  # Store crop parameters for visualization alignment
        
        # Calculate buffer size
        self.buffer_size = int(buffer_seconds * fps)
        
        # Ring buffer using deque with maxlen
        self.buffer: Deque[FrameData] = deque(maxlen=self.buffer_size)
        
        # Tracking
        self.frame_count = 0
        self.episode_count = 0
        self.crash_count = 0
        
        print(f"[BlackBoxRecorder] Initialized: {buffer_seconds}s buffer @ {fps} FPS = {self.buffer_size} frames")
        if roi_crop:
            print(f"[BlackBoxRecorder] ROI crop for visualization: top={roi_crop[0]}, bottom={roi_crop[1]}, left={roi_crop[2]}, right={roi_crop[3]}")
    
    def record(
        self,
        raw_image: np.ndarray,
        processed_obs: np.ndarray,
        action: Optional[np.ndarray] = None,
        reward: Optional[float] = None,
        info: Optional[Dict] = None,
        timestamp: Optional[float] = None
    ) -> None:
        # Auto-detect raw image size from first frame
        if self.raw_image_size is None:
            self.raw_image_size = (raw_image.shape[0], raw_image.shape[1])
        
        frame_data = FrameData(
            raw_image=raw_image.copy(),  # Copy to prevent reference issues
            processed_obs=processed_obs.copy(),
            action=action.copy() if action is not None else None,
            reward=reward,
            info=info.copy() if info is not None else None,
            timestamp=timestamp if timestamp is not None else self.frame_count / self.fps
        )
        
        self.buffer.append(frame_data)
        self.frame_count += 1
    
    def get_buffer_data(self) -> Tuple[np.ndarray, np.ndarray, List[FrameData]]:
        if len(self.buffer) == 0:
            return np.array([]), np.array([]), []
        
        frames = list(self.buffer)
        raw_images = np.stack([f.raw_image for f in frames])
        processed_obs = np.stack([f.processed_obs for f in frames])
        
        return raw_images, processed_obs, frames
    
    def save_post_mortem(
        self,
        vbp: Optional[Any] = None,  # VisualBackProp instance
        output_dir: str = "crashes",
        video_filename: Optional[str] = None,
        include_attention: bool = True,
        include_telemetry: bool = True,
        codec: str = "mp4v"
    ) -> Optional[str]:
        if len(self.buffer) == 0:
            print("[BlackBoxRecorder] Buffer is empty, nothing to save")
            return None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename
        if video_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = f"crash_{self.crash_count:04d}_{timestamp}.mp4"
        
        output_path = os.path.join(output_dir, video_filename)
        
        # Get buffered data
        raw_images, processed_obs, frames = self.get_buffer_data()
        num_frames = len(frames)
        
        print(f"[BlackBoxRecorder] Processing {num_frames} frames for post-mortem analysis...")
        
        # Compute attention maps in batch if VBP is provided
        attention_masks = None
        if include_attention and vbp is not None:
            print("[BlackBoxRecorder] Computing VisualBackProp attention maps (batch)...")
            try:
                # Target size matches CROPPED raw image dimensions
                raw_h, raw_w = raw_images.shape[1], raw_images.shape[2]
                if self.roi_crop:
                    top, bottom, left, right = self.roi_crop
                    cropped_h = raw_h - top - bottom
                    cropped_w = raw_w - left - right
                    target_size = (cropped_h, cropped_w)
                else:
                    target_size = (raw_h, raw_w)
                
                attention_masks = vbp.compute_batch_attention_masks(
                    processed_obs, 
                    target_size=target_size
                )
                print(f"[BlackBoxRecorder] Computed {len(attention_masks)} attention masks")
            except Exception as e:
                print(f"[BlackBoxRecorder] Warning: Failed to compute attention maps: {e}")
                attention_masks = None
        
        # Determine video frame size AFTER ROI crop
        # Layout: [Raw Image | Attention Overlay | Processed Input]
        raw_h, raw_w = raw_images.shape[1], raw_images.shape[2]
        if self.roi_crop:
            top, bottom, left, right = self.roi_crop
            h = raw_h - top - bottom
            w = raw_w - left - right
        else:
            h, w = raw_h, raw_w
        
        if include_attention and attention_masks is not None:
            frame_width = w * 3  # Three panels
        else:
            frame_width = w * 2  # Two panels (raw + processed)
        
        frame_height = h
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, self.fps, (frame_width, frame_height))
        
        if not writer.isOpened():
            print(f"[BlackBoxRecorder] Error: Could not open video writer for {output_path}")
            return None
        
        # Process each frame
        for i, frame_data in enumerate(frames):
            raw_img = frame_data.raw_image.copy()
            proc_obs = frame_data.processed_obs.copy()
            
            # Build telemetry info dict
            extra_info = {}
            if include_telemetry:
                if frame_data.action is not None:
                    extra_info["Steering"] = frame_data.action[0]
                    extra_info["Throttle"] = frame_data.action[1]
                if frame_data.reward is not None:
                    extra_info["Reward"] = frame_data.reward
                if frame_data.info:
                    for key in ["cte", "speed", "forward_vel"]:
                        if key in frame_data.info:
                            extra_info[key] = frame_data.info[key]
            
            if attention_masks is not None:
                # Use VisualBackProp visualization
                viz_frame = vbp.create_visualization_frame(
                    raw_image=raw_img,
                    processed_obs=proc_obs,
                    mask=attention_masks[i],
                    frame_idx=i,
                    total_frames=num_frames,
                    extra_info=extra_info,
                    roi_crop=self.roi_crop  # Pass crop params for proper alignment
                )
            else:
                # Simple two-panel visualization without attention
                viz_frame = self._create_simple_frame(
                    raw_img, proc_obs, i, num_frames, extra_info
                )
            
            # Convert RGB to BGR for OpenCV
            viz_frame_bgr = cv2.cvtColor(viz_frame, cv2.COLOR_RGB2BGR)
            writer.write(viz_frame_bgr)
        
        writer.release()
        self.crash_count += 1
        
        print(f"[BlackBoxRecorder] Saved post-mortem video: {output_path}")
        return output_path
    
    def _create_simple_frame(
        self,
        raw_image: np.ndarray,
        processed_obs: np.ndarray,
        frame_idx: int,
        total_frames: int,
        extra_info: Dict
    ) -> np.ndarray:
        """Create a simple visualization frame without attention overlay."""
        # Apply ROI crop to match what model sees
        if self.roi_crop is not None:
            top, bottom, left, right = self.roi_crop
            img_h, img_w = raw_image.shape[:2]
            bottom_idx = img_h - bottom if bottom > 0 else img_h
            right_idx = img_w - right if right > 0 else img_w
            raw_image = raw_image[top:bottom_idx, left:right_idx].copy()
        
        h, w = raw_image.shape[:2]
        
        # Ensure uint8
        if raw_image.dtype != np.uint8:
            raw_image = np.clip(raw_image * 255, 0, 255).astype(np.uint8)
        
        # Process the observation for display
        proc_display = processed_obs.copy()
        
        # Handle frame stacking - extract latest frame
        if len(proc_display.shape) == 3 and proc_display.shape[-1] > 3:
            proc_display = proc_display[:, :, -1]
        elif len(proc_display.shape) == 3 and proc_display.shape[-1] == 1:
            proc_display = proc_display.squeeze()
        
        # Convert to proper display format [0, 255]
        proc_display = proc_display.astype(np.float32)
        max_val = proc_display.max()
        
        if max_val <= 1.0 + 1e-6:
            proc_display = proc_display * 255.0
        elif max_val < 255.0 and max_val > 0:
            proc_display = (proc_display / max_val) * 255.0
        
        proc_display = np.clip(proc_display, 0, 255).astype(np.uint8)
        
        # Convert grayscale to RGB
        if len(proc_display.shape) == 2:
            proc_display = cv2.cvtColor(proc_display, cv2.COLOR_GRAY2RGB)
        
        proc_display = cv2.resize(proc_display, (w, h))
        
        # Concatenate
        combined = np.hstack([raw_image, proc_display])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.1
        color = (255, 255, 255)
        
        cv2.putText(combined, "Raw", (5, 12), font, font_scale, color, 1)
        cv2.putText(combined, "Model Input", (w + 5, 12), font, font_scale, color, 1)
        cv2.putText(combined, f"{frame_idx + 1}/{total_frames}", (5, h - 5), font, font_scale, color, 1)
        
        # Telemetry info on left edge
        if extra_info:
            y_offset = 24
            for key, value in extra_info.items():
                if isinstance(value, float):
                    text = f"{key}: {value:.2f}"
                else:
                    text = f"{key}: {value}"
                cv2.putText(combined, text, (5, y_offset), font, font_scale * 0.9, color, 1)
                y_offset += 10
        
        return combined
    
    def display_buffer_live(
        self,
        vbp: Optional[Any] = None,
        window_name: str = "Post-Mortem Replay"
    ) -> None:
        if len(self.buffer) == 0:
            print("[BlackBoxRecorder] Buffer is empty")
            return
        
        raw_images, processed_obs, frames = self.get_buffer_data()
        num_frames = len(frames)
        
        # Compute attention maps
        attention_masks = None
        if vbp is not None:
            print("[BlackBoxRecorder] Computing attention maps for replay...")
            target_size = (raw_images.shape[1], raw_images.shape[2])
            attention_masks = vbp.compute_batch_attention_masks(processed_obs, target_size)
        
        # Playback
        idx = 0
        paused = False
        delay = int(1000 / self.fps)  # ms between frames
        
        print(f"[BlackBoxRecorder] Replaying {num_frames} frames. Q=quit, Space=pause, Arrows=step")
        
        while True:
            frame_data = frames[idx]
            
            if attention_masks is not None:
                viz_frame = vbp.create_visualization_frame(
                    raw_image=frame_data.raw_image,
                    processed_obs=frame_data.processed_obs,
                    mask=attention_masks[idx],
                    frame_idx=idx,
                    total_frames=num_frames,
                    roi_crop=self.roi_crop  # Pass crop params for proper alignment
                )
            else:
                viz_frame = self._create_simple_frame(
                    frame_data.raw_image,
                    frame_data.processed_obs,
                    idx, num_frames, {}
                )
            
            viz_frame_bgr = cv2.cvtColor(viz_frame, cv2.COLOR_RGB2BGR)
            cv2.imshow(window_name, viz_frame_bgr)
            
            key = cv2.waitKey(delay if not paused else 0) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
            elif key == 83 or key == ord('d'):  # Right arrow
                idx = min(idx + 1, num_frames - 1)
            elif key == 81 or key == ord('a'):  # Left arrow
                idx = max(idx - 1, 0)
            elif not paused:
                idx = (idx + 1) % num_frames
        
        cv2.destroyWindow(window_name)
    
    def clear(self) -> None:
        """Clear the buffer"""
        self.buffer.clear()
        self.frame_count = 0
    
    def on_episode_start(self) -> None:
        """Called at the start of a new episode."""
        self.clear()
        self.episode_count += 1
    
    def __len__(self) -> int:
        """Return current number of frames in buffer."""
        return len(self.buffer)
    
    @property
    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        return len(self.buffer) >= self.buffer_size
    
    def get_statistics(self) -> Dict:
        """Get recorder statistics."""
        return {
            "buffer_size": self.buffer_size,
            "current_frames": len(self.buffer),
            "buffer_seconds": self.buffer_seconds,
            "fps": self.fps,
            "total_frames_recorded": self.frame_count,
            "episodes": self.episode_count,
            "crashes_saved": self.crash_count
        }
