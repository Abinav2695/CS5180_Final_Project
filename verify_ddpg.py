import gym
import numpy as np
from tqdm import tqdm
import pygame
from envs.escape_room_continuous_space_env import EscapeRoomEnv
from ddpg_torch.ddpg_agent import Agent

def simulate_trained_agent(n_episodes=5, render=True):
    # Initialize the environment
    env = EscapeRoomEnv(max_steps_per_episode=500)

    # Initialize the agent (make sure to initialize with the same parameters as during training)
    agent = Agent(
        alpha=0.0001,  # Alpha is not used in simulation but required for initialization
        beta=0.001,    # Beta is not used in simulation but required for initialization
        input_dims=env.observation_space.shape,
        tau=0.001,     # Tau is not used in simulation but required for initialization
        batch_size=64, # Batch size is not used in simulation but required for initialization
        fc1_dims=400,
        fc2_dims=300,
        n_actions=env.action_space.shape[0],
    )

    # Load the trained model
    agent.load_models()

    # Simulation loop
    for episode in range(n_episodes):
        state, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            if render:
                env.render()

            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state

        print(f"Episode {episode + 1}/{n_episodes} - Total Reward: {total_reward}")

    env.close()  # Don't forget to close the environment


def simulate_trained_agent_ll(n_episodes=5, render=True):
    # Initialize the environment
    env = gym.make('LunarLanderContinuous-v2', render_mode="human")

    # Initialize the agent (make sure to initialize with the same parameters as during training)
    agent = Agent(
        alpha=0.0001,  # Alpha is not used in simulation but required for initialization
        beta=0.001,    # Beta is not used in simulation but required for initialization
        input_dims=env.observation_space.shape,
        tau=0.001,     # Tau is not used in simulation but required for initialization
        batch_size=64, # Batch size is not used in simulation but required for initialization
        fc1_dims=400,
        fc2_dims=300,
        n_actions=env.action_space.shape[0],
    )

    # Load the trained model
    agent.load_models()

    # Simulation loop
    for episode in range(n_episodes):
        state, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            if render:
                env.render()

            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state

        print(f"Episode {episode + 1}/{n_episodes} - Total Reward: {total_reward}")

    env.close()  # Don't forget to close the environment

if __name__ == "__main__":
    # simulate_trained_agent_ll(n_episodes=5, render=True)
    simulate_trained_agent( n_episodes=5, render=True )
