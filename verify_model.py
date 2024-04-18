import torch
import gym
import numpy as np
from algorithms.dqn_agent import DQN
from envs.escape_room_env import EscapeRoomEnv
import pygame


def load_model(model_path):
    try:
        model = torch.load(model_path)
    except FileNotFoundError:
        print("Checkpoint file not found.")
        return None
    else:
        return {key: DQN.custom_load(data) for key, data in model.items()}


def simulate_model(env, model, device):
    state, _ = env.reset()
    done = False
    steps = 0
    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action = model(state_tensor).argmax().item()
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()
        steps += 1
        print(f"Steps :: {steps}")


def main():
    env = EscapeRoomEnv(max_steps_per_episode=500)
    models = load_model("model_basic.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if models:
        for key, model in models.items():
            print(f"Simulating with model: {key}")
            simulate_model(env, model, device)
            env.close()  # Ensure the environment is closed properly


if __name__ == "__main__":
    main()