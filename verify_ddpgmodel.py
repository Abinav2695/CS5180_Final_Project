# import gym
# import numpy as np
# from ddpg_torch.ddpg_agent import Agent
# from envs.escape_room_continuous_space_env import EscapeRoomEnv

# def simulate_trained_model(n_episodes=5, render=True):
#     # Initialize the environment
#     env = EscapeRoomEnv()

#     # Initialize the agent - ensure this matches the training configuration
#     agent = Agent(
#         alpha=0.0001,  # Learning rate for the actor
#         beta=0.001,    # Learning rate for the critic
#         input_dims=env.observation_space.shape,
#         tau=0.001,
#         batch_size=64,
#         fc1_dims=400,
#         fc2_dims=300,
#         n_actions=env.action_space.shape[0]
#     )

#     # Load the models from the saved files
#     agent.load_models()

#     for episode in range(n_episodes):
#         state, info = env.reset()
#         done = False
#         score = 0

#         while not done:
#             action = agent.choose_action(state)
#             state, reward, terminated, truncated, info = env.step(action)
#             done = terminated or truncated
#             score += reward
            
#             if render:
#                 env.render()

#         print(f"Episode {episode + 1}: Score = {score}")

#     env.close()

# if __name__ == "__main__":
#     simulate_trained_model()
    
import gym
import numpy as np
from ddpg_torch.ddpg_agent import Agent
from envs.escape_room_continuous_space_env import EscapeRoomEnv

def simulate_trained_model(selected_episodes=[0, 25, 50, 75, 100], render=True):
    # Initialize the environment
    env = EscapeRoomEnv()

    # Initialize the agent with matching configuration from training
    agent = Agent(
        alpha=0.0001,  # Learning rate for the actor
        beta=0.001,    # Learning rate for the critic
        input_dims=env.observation_space.shape,
        tau=0.001,
        batch_size=64,
        fc1_dims=400,
        fc2_dims=300,
        n_actions=env.action_space.shape[0]
    )

    # Load the models from the saved files
    agent.load_models()
    # Simulate for each selected episode
    for episode in selected_episodes:
        state, info = env.reset()
        done = False
        score = 0
        step_count = 0  # Initialize step counter

        while not done:
            action = agent.choose_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score += reward
            step_count += 1  # Increment step count

            # Render the environment if requested
            if render:
                env.render()

            # Check if the step count has reached 500
            if step_count >= 500:
                print(f"Reached 500 steps on Episode {episode}.")
                break  # Stop the episode after 500 steps

        # Print the score for each episode
        print(f"Episode {episode}: Score = {score}")

    env.close()

    # # Simulate for each selected episode
    # for episode in selected_episodes:
    #     state, info = env.reset()
    #     done = False
    #     score = 0

    #     while not done:
    #         action = agent.choose_action(state)
    #         state, reward, terminated, truncated, info = env.step(action)
    #         done = terminated or truncated
    #         score += reward

    #         if render:
    #             env.render()

    #     # Print the score for each episode
    #     print(f"Episode {episode}: Score = {score}")

    # env.close()

if __name__ == "__main__":
    simulate_trained_model(selected_episodes=[0, 25, 50, 75, 100])
