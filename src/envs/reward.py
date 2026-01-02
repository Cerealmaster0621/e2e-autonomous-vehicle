"""
Reward Function:
    Safety Elements:
        `+` Speed reward (keeping target velocity).
        `+` progress reward (distance).
        `-` Collision penalty.
        `-` Lane Centering & Out-of-Lane penalty.
    Comfort Elements:
        `-` Steering Smoothness penalty(restrict High-frequency oscillation)
        `-` Acceleration Smoothness penalty
        `-` Lateral Stability(Restrict Lateral Acceleration when high speed)
"""

from configs.load_config import load_config

# Load config once at module level
_config = load_config()
_reward_cfg = _config["reward"]
_conf_cfg = _config["conf"]


def default_reward(self, done: bool) -> float:
    if not hasattr(self, '_prev_forward_vel'):
        self._prev_forward_vel = 0.0
    if not hasattr(self, '_prev_yaw'):
        self._prev_yaw = 0.0
    
    # Get config values
    target_speed = _reward_cfg["target_velocity"]
    max_cte = _conf_cfg["max_cte"]
    
    reward = 0.0
    
    # ========== TERMINAL CONDITIONS ==========
    # Collision: Large penalty and episode ends
    if self.hit != "none":
        return _reward_cfg["collision_penalty"]
    
    # Out of lane: Large penalty (done flag usually set by sim)
    if done:
        return _reward_cfg["collision_penalty"]
    
    # ========== SAFETY REWARDS ==========
    
    # 1. Speed Reward: Encourage maintaining target speed
    # R_speed = w * min(v / v_target, 1.0)
    # Cap at 1.0 to not reward going too fast
    if target_speed > 0:
        speed_ratio = min(self.forward_vel / target_speed, 1.0)
        speed_reward = _reward_cfg["speed_reward"] * max(speed_ratio, 0.0)
        reward += speed_reward
    
    # 2. Progress Reward: Reward for moving forward
    # Simple: proportional to forward velocity (encourages moving, not stopping)
    if self.forward_vel > 0:
        progress_reward = _reward_cfg["progress_reward"] * (self.forward_vel / target_speed)
        reward += progress_reward
    
    # 3. Lane Centering Penalty: Penalize being off-center
    # R_lane = -w * (|cte| / max_cte)
    # cte = cross-track error (0 = perfectly centered)
    if max_cte > 0:
        cte_normalized = min(abs(self.cte) / max_cte, 1.0)
        # Use squared to penalize large deviations more heavily
        lane_penalty = _reward_cfg["lane_centering_penalty"] * (cte_normalized ** 2)
        reward += lane_penalty  # Already negative in config
    
    # ========== COMFORT REWARDS ==========
    
    # 4. Steering Smoothness: Penalize rapid yaw changes
    # Uses yaw angle changes as proxy for steering oscillation
    yaw_delta = abs(self.yaw - self._prev_yaw)
    # Normalize: yaw is in degrees, large changes are bad
    # Wrap around handling for yaw (-180 to 180)
    if yaw_delta > 180:
        yaw_delta = 360 - yaw_delta
    yaw_delta_normalized = min(yaw_delta / 30.0, 1.0)  # 30 degrees = max penalty
    steering_penalty = _reward_cfg["steering_smoothness_penalty"] * (yaw_delta_normalized ** 2)
    reward += steering_penalty
    
    # 5. Acceleration Smoothness: Penalize rapid velocity changes
    vel_delta = abs(self.forward_vel - self._prev_forward_vel)
    # Normalize: large acceleration changes are bad
    vel_delta_normalized = min(vel_delta / 5.0, 1.0)  # 5 m/s change = max penalty
    accel_penalty = _reward_cfg["acceleration_smoothness_penalty"] * (vel_delta_normalized ** 2)
    reward += accel_penalty
    
    # 6. Lateral Stability: Penalize high-speed sharp turns
    # High speed + high yaw rate = unstable
    # Use gyro_z (yaw rate) if available, otherwise use yaw_delta
    yaw_rate = abs(getattr(self, 'gyro_z', yaw_delta))
    if target_speed > 0:
        speed_factor = self.forward_vel / target_speed
        lateral_factor = speed_factor * (yaw_rate / 50.0)  # Normalize yaw rate
        lateral_penalty = _reward_cfg["lateral_stability_penalty"] * min(lateral_factor, 1.0)
        reward += lateral_penalty
    
    # ========== UPDATE HISTORY ==========
    self._prev_forward_vel = self.forward_vel
    self._prev_yaw = self.yaw
    
    return reward


def reset_reward_history(handler) -> None:
    if hasattr(handler, '_prev_forward_vel'):
        handler._prev_forward_vel = 0.0
    if hasattr(handler, '_prev_yaw'):
        handler._prev_yaw = 0.0
