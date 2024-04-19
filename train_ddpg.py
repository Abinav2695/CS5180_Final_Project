import os
import gym
from matplotlib import pyplot as plt
import numpy as np
from ddpg_torch.ddpg_agent import Agent
from utils.plot import plot_learning_curve
from envs.escape_room_continuous_space_env import EscapeRoomEnv
from tqdm import trange

def plot_learning_curve(x, scores, figure_file):
    plt.figure()
    plt.plot(x, scores, label='Score per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)

    # Check if the directory exists before saving; create it if it doesn't
    os.makedirs(os.path.dirname(figure_file), exist_ok=True)
    
    plt.savefig(figure_file)
    plt.close()  # Close the plot to avoid memory issues

def train_diff_robot_custom_env(alpha=0.0001, beta=0.001, tau=0.001, n_games=5000):
    env = EscapeRoomEnv()  # Assuming EscapeRoomEnv is imported

    agent = Agent(
        alpha=alpha,
        beta=beta,
        input_dims=env.observation_space.shape,
        tau=tau,
        batch_size=64,
        fc1_dims=400,
        fc2_dims=300,
        n_actions=env.action_space.shape[0],
    )

    filename = f"EscapeRoom_alpha_{agent.alpha}_beta_{agent.beta}_{n_games}_games"
    figure_file = f"plots/{filename}.png"

    # Set milestones for saving models at 0%, 25%, 50%, 75%, and 100% of the training
    milestones = [int(n_games * percentage / 100) for percentage in (0, 25, 50, 75, 100)]

    score_history = []

    pbar = trange(n_games, desc='Training progress')

    for i in pbar:
        state, info = env.reset()
        done = False
        score = 0
        agent.noise.reset()

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.remember(state, action, reward, next_state, done)
            agent.learn()
            score += reward
            state = next_state

        score_history.append(score)
        avg_score = np.mean(score_history[-100:]) if len(score_history) > 100 else np.mean(score_history)

        pbar.set_description(f"Episode {i}: Score {score:.1f}, Average Score {avg_score:.1f}")

        # Save the model at specified milestones
        if i in milestones:
            agent.save_models()
            print(f"Saved models at {i/n_games*100:.0f}% completion")

    x = [i + 1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)

if __name__ == "__main__":
    train_diff_robot_custom_env()



# import os
# import gym
# from matplotlib import pyplot as plt
# import numpy as np
# from ddpg_torch.ddpg_agent import Agent
# from utils.plot import plot_learning_curve
# from envs.escape_room_continuous_space_env import EscapeRoomEnv
# from tqdm import tqdm, trange


# def plot_learning_curve(x, scores, figure_file):
#     plt.figure()
#     plt.plot(x, scores, label='Score per Episode')
#     plt.xlabel('Episode')
#     plt.ylabel('Score')
#     plt.title('Learning Curve')
#     plt.legend()
#     plt.grid(True)

#     # Check if the directory exists before saving; create it if it doesn't
#     os.makedirs(os.path.dirname(figure_file), exist_ok=True)
    
#     plt.savefig(figure_file)
#     plt.show()

# def train_diff_robot_custom_env(alpha=0.0001, beta=0.001, tau=0.001, n_games=50):
#     env = EscapeRoomEnv()  # Assuming EscapeRoomEnv is imported

#     agent = Agent(
#         alpha=alpha,
#         beta=beta,
#         input_dims=env.observation_space.shape,
#         tau=tau,
#         batch_size=64,
#         fc1_dims=400,
#         fc2_dims=300,
#         n_actions=env.action_space.shape[0],
#     )

#     filename = f"EscapeRoom_alpha_{agent.alpha}_beta_{agent.beta}_{n_games}_games"
#     figure_file = f"plots/{filename}.png"

#     best_score = env.reward_range[0]
#     score_history = []

#     pbar = trange(n_games)

#     for i in pbar:
#         state, info = env.reset()  # Correctly unpack the tuple returned by env.reset()
#         done = False
#         score = 0
#         agent.noise.reset()

#         while not done:
#             action = agent.choose_action(state)
#             # print(f"Action :: {action}")
#             next_state, reward, terminated, truncated, info = env.step(action)
#             done = terminated or truncated
#             agent.remember(state, action, reward, next_state, done)
#             agent.learn()
#             score += reward
#             state = next_state

#         score_history.append(score)
#         avg_score = np.mean(score_history[-100:])

#         if avg_score > best_score:
#             best_score = avg_score
#             agent.save_models()

#         # print(f"Episode {i}: Score {score:.1f}, Average Score {avg_score:.1f}")
#         pbar.set_description(
#             f"Episode {i}: Score {score:.1f}, Average Score {avg_score:.1f}"
#         )



#     x = [i + 1 for i in range(n_games)]
#     plot_learning_curve(x, score_history, figure_file)


# if __name__ == "__main__":
#     train_diff_robot_custom_env()
#     # train_lunarLander()
