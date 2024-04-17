import torch
import gym
import numpy as np
from algorithms.dqn_agent import DQN
from envs.escape_room_env import EscapeRoomEnv


def load_model(model_path):
    try:
        model = torch.load(model_path)
    except FileNotFoundError:
        print("Checkpoint file not found.")
        return None
    else:
        model_dict = {key: DQN.custom_load(data) for key, data in model.items()}
        return model_dict


def simulate_model(env, model):
    state, _ = env.reset()
    done = False
    while not done:
        state_tensor = torch.tensor([state], dtype=torch.float32)
        with torch.no_grad():
            action = model(state_tensor).argmax().item()
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()


def main():
    env = EscapeRoomEnv()
    models = load_model(f"model_basic.pt")

    if models:
        for key, model in models.items():
            print(f"Simulating with model: {key}")
            simulate_model(env, model)
            env.close()  # Close the environment properly


if __name__ == "__main__":
    main()
