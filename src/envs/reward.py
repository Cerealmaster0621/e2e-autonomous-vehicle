"""
**Reward Function:**
  **Safety Elements**
    `+` Speed reward (keeping target velocity).
    `+` progress reward (distance).
    `-` Collision penalty.
    `-` Lane Centering & Out-of-Lane penalty.
  **Comfort Elements**
    `-` Steering Smoothness penalty(restrict High-frequency oscillation)
    `-` Acceleration Smoothness penalty
    `-` Lateral Stability(Restrict Lateral Acceleration when high speed)
"""