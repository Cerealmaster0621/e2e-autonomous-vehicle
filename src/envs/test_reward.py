# src/envs/test_reward.py

def custom_reward(self, done):
    # Example: penalize hitting, encourage speed, etc.
    if done:
        return -5.0
    if self.hit != "none":
        return -10.0
    return self.forward_vel  # Or any logic you prefer!