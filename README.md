# End-to-End Reinforcement Learning for Autonomous Driving

> Visual-based Autonomous Driving Agent using PPO & SAC with Different Environments and Interpretability Analysis

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![CARLA](https://img.shields.io/badge/Donkey_Simulator-25.10.06-orange.svg)](https://docs.donkeycar.com/guide/deep_learning/simulator/)
[![Framework](https://img.shields.io/badge/Framework-Gymnasium%20%7C%20SB3-green.svg)](https://stable-baselines3.readthedocs.io/)
[![License](https://img.shields.io/badge/license-Apache%20License%202.0-lightgrey.svg)](LICENSE)

## Project Overview

This project implements an **End-to-End Reinforcement Learning (RL)** pipeline for autonomous driving using the **Donkey Simulator(v25.10.06)**.

The primary goal is to benchmark on-policy (PPO) vs. off-policy (SAC) algorithms and analyze their robustness against environmental variations (weather, dynamic obstacles) and interpret their failure modes using Visual Attention Maps.

## System Architecture

### 1. Perception Module (Input)

### 2. Reinforcement Learning

- **Action Space:** Continuous control `[Steer, Acceleration]`(Throttle when Acceleration > 0, Brake when less than 0).
- **Reward Function:**
  - **Safety Elements**
    - `+` Speed reward (keeping target velocity).
    - `+` progress reward (distance).
    - `-` Collision penalty.
    - `-` Lane Centering & Out-of-Lane penalty.
  - **Comfort Elements**
    - `-` Steering Smoothness penalty(restrict High-frequency oscillation)
    - `-` Acceleration Smoothness penalty
    - `-` Lateral Stability(Restrict Lateral Acceleration when high speed)

### 3. Interpretability (Post-Mortem Analysis)

- Triggers upon collision events.
- Generates **Attention Maps** overlaying the last few(undecided) seconds of input frames to visualize what the agent was focusing on (e.g., did it see the car or was it looking at the clouds?).

---

## Getting Started

1. download `DonkeySim` simulator following the instruction inside the project folder
2. clone `gym_donkeycar` following the instruction inside the project folder

- if you are using mac os, add following code inside `gym_donkeycar/envs/donkey_proc.py` file
  ```python3
          # line 25 ~
          # On macOS, .app is a directory.
          if sim_path.endswith(".app"):
          # Try to find the executable inside the bundle
          app_name = os.path.splitext(os.path.basename(sim_path))[0]
          mac_exe = os.path.join(sim_path, "Contents", "MacOS", app_name)
          if os.path.exists(mac_exe):
              sim_path = mac_exe
  ```

3. clone this repository

```bash
  git clone https://github.com/Cerealmaster0621/e2e-autonomous-vehicle.git
  cd e2e-autonomous-vehicle
```

4. make conda environment and activate

```bash
  conda create -n e2e-autonomous-vehicle python=3.11 -y
  conda activate e2e-autonomous-vehicle
```

5. (optional)run the test

```bash
  sudo python src/test.py --sim {path to donkey_sim.app file}
```

---
