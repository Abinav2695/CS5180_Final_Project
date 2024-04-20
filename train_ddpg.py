import gym
import numpy as np
from ddpg_torch.ddpg_agent import Agent
from utils.plot import plot_learning_curve
from envs.escape_room_continuous_space_env import EscapeRoomEnv
from tqdm import tqdm, trange

def train_diff_robot_custom_env(alpha=0.0001, beta=0.001, tau=0.001, n_games=100, max_steps_per_episode=1000):
    env = EscapeRoomEnv(max_steps_per_episode=max_steps_per_episode)  # Assuming EscapeRoomEnv is imported

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

    best_score = env.reward_range[0]
    score_history = []

    pbar = trange(n_games)

    for i in pbar:
        state, info = env.reset()
        done = False
        score = 0
        step_count = 0  # Initialize step count for the episode
        agent.noise.reset()

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.remember(state, action, reward, next_state, done)
            agent.learn()
            score += reward
            state = next_state
            step_count += 1  # Increment step count

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        pbar.set_description(
            f"Episode {i}: Score {score:.1f}, Average Score {avg_score:.1f}, Steps {step_count}"
        )

    x = [i + 1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)

def train_lunar_lander():
    env = gym.make('LunarLanderContinuous-v2')
    agent = Agent(alpha=0.0001, beta=0.001, 
                    input_dims=env.observation_space.shape, tau=0.001,
                    batch_size=64, fc1_dims=400, fc2_dims=300, 
                    n_actions=env.action_space.shape[0])
    n_games = 1000
    filename = 'LunarLander_alpha_' + str(agent.alpha) + '_beta_' + \
                str(agent.beta) + '_' + str(n_games) + '_games'
    figure_file = 'plots/' + filename + '.png'

    best_score = env.reward_range[0]
    score_history = []
    pbar = trange(n_games)

    for i in pbar:
        observation, info = env.reset()
        done = False
        score = 0
        agent.noise.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            score += reward
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        # print('episode ', i, 'score %.1f' % score,
        #         'average score %.1f' % avg_score)
        pbar.set_description(
            f"Episode {i}: Score {score:.1f}, Average Score {avg_score:.1f}"
        )
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)

if __name__ == "__main__":
    train_diff_robot_custom_env(n_games=2000, max_steps_per_episode=500)
    # train_lunar_lander()
