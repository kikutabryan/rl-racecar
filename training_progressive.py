import beamnggym
import gymnasium as gym
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
import os
import time

# Learning starts above this value
learn_starts = 100

# Training stages with their respective lap percentages, timesteps, and vehicle indices
training_stages = [
    (0.05, 50_000, 0),  # 5% of track, ETK 800
    (0.1, 50_000, 0),  # 10% of track, ETK 800
    (0.3, 50_000, 0),  # 30% of track, ETK 800
    (0.5, 50_000, 0),  # 50% of track, ETK 800
    (0.7, 50_000, 0),  # 70% of track, ETK 800
    (0.9, 50_000, 0),  # 90% of track, ETK 800
    (1.0, 50_000, 0),  # 100% of track, ETK 800
    (1.0, 50_000, 0),  # Additional training on full track
    (1.0, 50_000, 0),  # Additional training on full track
    (1.0, 50_000, 0),  # Additional training on full track
]

# Vehicle names for reference
vehicle_names = [
    "ETK 800",  # 0
    "Bruckell LeGran",  # 1
    "Bruckell Bastion",  # 2
    "Gavril H-Series",  # 3
    "Gavril T-Series",  # 4
    "Wentward DT40L",  # 5
    "ETK K-Series",  # 6
]

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model as None
model = None

# Train through each stage
for stage_idx, (lap_percent, timesteps, vehicle_index) in enumerate(training_stages):
    # Generate model name from stage information
    vehicle_name = vehicle_names[vehicle_index].lower().replace(" ", "_")
    model_name = f"sac_stage_{stage_idx}_{vehicle_name}_{int(lap_percent*100)}pct"

    # Check if model already exists
    save_path = os.path.join("models", model_name)
    if os.path.exists(save_path + ".zip"):
        print(f"Model {model_name} already exists, skipping training")
    else:
        # Set up logger for this stage
        log_path = os.path.join("logs", model_name)
        logger = configure(log_path, ["stdout", "csv", "tensorboard"])

        # Create BeamNG gym with current lap percentage and vehicle
        env = gym.make(
            "BNG-WCA-Race-Geometry-v0",
            start_lap_percent=lap_percent,
            final_lap_percent=lap_percent,
            lap_percent_increment=0.0,
            learn_starts=learn_starts,
            real_time=False,
            vehicle_index=vehicle_index,
            randomize_start=True,
            enable_logging=False,
        )

        if stage_idx == 0:
            # First stage - create new model
            model = SAC(
                "MlpPolicy",
                env,
                verbose=1,
                device=device,
                learning_starts=learn_starts,
                seed=42,
            )
        else:
            # Load previous model for next stage
            prev_vehicle_name = (
                vehicle_names[training_stages[stage_idx - 1][2]]
                .lower()
                .replace(" ", "_")
            )
            prev_lap_percent = training_stages[stage_idx - 1][0]
            prev_model_name = f"sac_stage_{stage_idx-1}_{prev_vehicle_name}_{int(prev_lap_percent*100)}pct"

            load_path = os.path.join("models", prev_model_name)
            print(f"Loading previous model from {load_path}")
            model = SAC.load(load_path, env=env, verbose=1, device=device)

        # Set logger and train
        model.set_logger(logger)
        model.learn(total_timesteps=timesteps)

        # Save model
        model.save(save_path)
        print(f"Saved model to {save_path}")

        # Clea up both environment and model
        env.close()
        del model
        del env
