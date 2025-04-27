# RL Racecar Project

This project demonstrates reinforcement learning for autonomous racecar driving using BeamNG and Stable Baselines3.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/kikutabryan/rl-racecar.git
cd rl-racecar
```

### 2. Clone and Install the BeamNG.gym

You need to clone the [BeamNG.gym](https://github.com/kikutabryan/BeamNG.gym.git) repository fork and install it:

```bash
git clone https://github.com/kikutabryan/BeamNG.gym.git
cd BeamNG.gym
pip install -e .
cd ..
```

### 3. Install Python Dependencies

It is recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Make sure you have [BeamNG.tech](https://beamng.tech/) installed and properly configured. You must set the `BNG_HOME` environment variable to the path of your BeamNG.tech installation (the folder containing `EULA.pdf`).

## Running the Test Script

To test a trained model, run:

```bash
python simple_test.py
```

This will load a pre-trained model and run it in the BeamNG environment.

## Demo Video

You can view a demonstration video of the trained agent below:

---

**Demo Video:**

[![Demo Video](https://img.youtube.com/vi/-dIdvcbgOhM/0.jpg)](https://youtu.be/-dIdvcbgOhM)

---

## Training the Agent

Training is performed using the `training_progressive.py` script. This script implements progressive curriculum training, incrementally increasing the track length and saving models at each stage.

To start training, run:

```bash
python training_progressive.py
```

This will save trained models in the `models/` directory and logs in the `logs/` directory.

## Notes

- Adjust the `simple_test.py` script as needed for your experiments.
- For more information, see the documentation in the [BeamNG.gym repository](https://github.com/kikutabryan/BeamNG.gym.git).
