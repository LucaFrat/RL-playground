# RL Playground 🤖

This is my personal sandbox for learning Deep Reinforcement Learning. The goal was simple: get my hands dirty with simulators, write algorithms from scratch, and eventually graduate to using industrial-grade libraries.

## What's Inside?

### 1. From Scratch (PyTorch) 🧠
I built PPO (Proximal Policy Optimization) from the ground up to understand the math behind the magic.
- **`ppo_discrete.py`**: A clean implementation for discrete action spaces (CartPole, etc.).
- **`ppo_continuous.py`**: Extended to continuous control for MuJoCo robots.
- **`humanoid_v5.py` & `hopper_v5.py`**: Training standard MuJoCo agents with my custom PPO implementation.
- **`humanoid_v5_parallel.py`**: Scaled up training using vectorized environments to speed things up.

### 2. Using Libraries (Stable Baselines 3) 🚀
Once I understood the core mechanics, I switched to **Stable Baselines 3** to see how the pros do it.
- **`hopper_sb3.py`**: Solving the Hopper task with SB3's optimized PPO.
- **`humanoid_sb3.py`**: Tackling the complex Humanoid-v5 environment using SB3 and vectorized environments (`SubprocVecEnv`).

### 3. Simulators 🌍
- **Gymnasium / MuJoCo**: The primary physics engine used for all locomotion tasks.

## Why this exists
This repo documents my journey from "How does Policy Gradient work?" to training a Humanoid to walk using vectorized parallel environments. It's rough, experimental, and exactly what I needed to learn the ropes.

## Usage
Most scripts are standalone. Just run them directly:
```bash
python ppo_continuous.py
# or
python humanoid_sb3.py