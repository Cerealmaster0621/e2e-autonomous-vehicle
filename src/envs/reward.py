"""
Reward Function:
    Safety Elements:
        `+` Speed reward (keeping target velocity).
        `+` Lane Centering reward (staying centered on track).
        `-` Collision penalty.
        `-` Idle penalty (prevents "do nothing" strategy).
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
    
    # Get normalization constants
    max_yaw_delta = _reward_cfg.get("max_yaw_delta", 30.0)
    max_vel_delta = _reward_cfg.get("max_vel_delta", 5.0)
    max_yaw_rate = _reward_cfg.get("max_yaw_rate", 50.0)
    
    reward = 0.0
    
    # ========== TERMINAL CONDITIONS ==========
    # These return immediately with large penalties
    
    # Collision: Strong penalty and episode ends
    if self.hit != "none":
        return _reward_cfg["collision_penalty"]
    
    # Out of lane / other terminal conditions
    if done:
        return _reward_cfg["collision_penalty"]
    
    # ========== COMPUTE RAW DELTAS (before normalization) ==========
    # Store these before any modifications for later use
    
    raw_yaw_delta = abs(self.yaw - self._prev_yaw)
    # Handle yaw wrap-around (-180 to 180)
    if raw_yaw_delta > 180:
        raw_yaw_delta = 360 - raw_yaw_delta
    
    raw_vel_delta = abs(self.forward_vel - self._prev_forward_vel)
    
    # ========== SAFETY REWARDS ==========
    
    # 1. Speed Reward: Encourage maintaining target speed
    # R_speed = w * min(v / v_target, 1.0)
    # Capped at 1.0 to not reward exceeding target speed
    if target_speed > 0:
        speed_ratio = min(self.forward_vel / target_speed, 1.0)
        speed_ratio = max(speed_ratio, 0.0)  # Ensure non-negative
        speed_reward = _reward_cfg["speed_reward"] * speed_ratio
        reward += speed_reward
    
    # 2. Idle Penalty: Penalize standing still / not moving
    # This prevents the agent from learning to "do nothing" to avoid crashes
    min_velocity = _reward_cfg.get("min_velocity", 2.0)
    if abs(self.forward_vel) < min_velocity:
        idle_penalty = _reward_cfg.get("idle_penalty", -1.0)
        reward += idle_penalty
    
    # 3. Lane Centering Reward: Reward staying centered
    # R_center = w * (1.0 - |cte| / max_cte)
    # Returns 1.0 when perfectly centered, 0.0 at lane edge
    if max_cte > 0:
        centering_factor = max(1.0 - (abs(self.cte) / max_cte), 0.0)
        lane_reward = _reward_cfg["lane_centering_reward"] * centering_factor
        reward += lane_reward
    
    # ========== COMFORT PENALTIES ==========
    # These use the raw deltas computed earlier
    
    # 4. Steering Smoothness: Penalize rapid yaw changes
    # Uses squared penalty for smooth gradient near zero
    yaw_delta_normalized = min(raw_yaw_delta / max_yaw_delta, 1.0)
    steering_penalty = _reward_cfg["steering_smoothness_penalty"] * (yaw_delta_normalized ** 2)
    reward += steering_penalty  # Note: penalty is already negative in config
    
    # 5. Acceleration Smoothness: Penalize rapid velocity changes
    vel_delta_normalized = min(raw_vel_delta / max_vel_delta, 1.0)
    accel_penalty = _reward_cfg["acceleration_smoothness_penalty"] * (vel_delta_normalized ** 2)
    reward += accel_penalty
    
    # 6. Lateral Stability: Penalize high-speed sharp turns
    # Uses gyro_z (yaw rate) if available, otherwise falls back to raw yaw delta
    # Penalty scales with speed: high speed + high yaw rate = more penalty
    yaw_rate = abs(getattr(self, 'gyro_z', raw_yaw_delta))
    if target_speed > 0 and self.forward_vel > 0:
        speed_factor = min(self.forward_vel / target_speed, 1.0)
        yaw_rate_normalized = min(yaw_rate / max_yaw_rate, 1.0)
        lateral_factor = speed_factor * yaw_rate_normalized
        lateral_penalty = _reward_cfg["lateral_stability_penalty"] * lateral_factor
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


def get_reward_info(self) -> dict:
    target_speed = _reward_cfg["target_velocity"]
    max_cte = _conf_cfg["max_cte"]
    
    # Speed component
    speed_ratio = min(self.forward_vel / target_speed, 1.0) if target_speed > 0 else 0
    speed_reward = _reward_cfg["speed_reward"] * max(speed_ratio, 0.0)
    
    # Centering component
    centering_factor = max(1.0 - (abs(self.cte) / max_cte), 0.0) if max_cte > 0 else 0
    lane_reward = _reward_cfg["lane_centering_reward"] * centering_factor
    
    return {
        "speed_reward": speed_reward,
        "lane_reward": lane_reward,
        "speed_ratio": speed_ratio,
        "centering_factor": centering_factor,
        "cte": self.cte,
        "forward_vel": self.forward_vel,
        "yaw": self.yaw,
    }
