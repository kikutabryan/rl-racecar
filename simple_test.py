import beamnggym
import gymnasium as gym
from stable_baselines3 import SAC
import torch
import os

# Constants
VEHICLE_INDEX = 0  # Change this to use different vehicles (0-6)
TIMESTEPS = 10000

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_path = os.path.join("models", "sac_stage_6_etk_800_100pct.zip")

# Create BeamNG gym with same settings as training
env = gym.make(
    "BNG-WCA-Race-Geometry-v0",
    start_lap_percent=1.0,  # Test on full track
    final_lap_percent=1.0,
    lap_percent_increment=0.0,
    learn_starts=0,
    real_time=True,
    vehicle_index=VEHICLE_INDEX,
    randomize_start=False,
    enable_logging=False,
)

model = SAC.load(model_path, device=device)

# Test the model
obs, info = env.reset()
for _ in range(TIMESTEPS):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
