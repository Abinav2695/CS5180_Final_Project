import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)


# Define the Actor model using PyTorch
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(
            state_dim, 64
        )  # Adjusted to take 'state_dim' as input size
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
        self.action_bound = action_bound

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x)) * self.action_bound
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.s1 = nn.Linear(state_dim, 32)
        self.s2 = nn.Linear(32, 32)
        self.a1 = nn.Linear(action_dim, 32)
        self.a2 = nn.Linear(32, 32)
        self.fc1 = nn.Linear(64, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, state, action):
        s = F.relu(self.s1(state))
        s = self.s2(s)
        a = F.relu(self.a1(action))
        a = self.a2(a)
        x = torch.cat([s, a], dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ReplayBuffer modified for device handling
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.stack(state).to(device),
            torch.stack(action).to(device),
            torch.stack(reward).to(device),
            torch.stack(next_state).to(device),
            torch.stack(done).to(device).float(),
        )

    def __len__(self):
        return len(self.buffer)


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, size, mu=0, sigma=0.2, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mu = mu if isinstance(mu, np.ndarray) else np.array([mu] * size)
        self.sigma = sigma
        self.dt = dt
        self.x_initial = x_initial if x_initial is not None else np.zeros_like(self.mu)
        self.reset()

    def reset(self):
        self.x_prev = self.x_initial

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mu - self.x_prev) * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        )
        self.x_prev = x
        return x


class DDPGAgent:
    def __init__(
        self,
        actor,
        critic,
        actor_target,
        critic_target,
        actor_lr=0.001,
        critic_lr=0.002,
    ):
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.actor_target = actor_target.to(device)
        self.critic_target = critic_target.to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def update_target_network(self, target_network, network, tau=0.005):
        for target_param, param in zip(
            target_network.parameters(), network.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def train_ddpg(agent, env, episodes, noise_process, buffer_size=100000, batch_size=64):
    replay_buffer = ReplayBuffer(capacity=buffer_size)
    response = []  # List to store the total rewards per episode

    for episode in range(episodes):
        state = env.reset()
        noise_process.reset()
        episode_reward = 0
        done = False
        while not done:
            state_tensor = (
                torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)
            )
            action = agent.actor(state_tensor).detach().cpu().numpy()[0]
            noise = noise_process()  # Adding exploration noise
            action = np.clip(
                action + noise, -agent.actor.action_bound, agent.actor.action_bound
            )

            next_state, reward, done, _ = env.step(action)
            replay_buffer.add(
                torch.tensor(state, dtype=torch.float32).to(device),
                torch.tensor(action, dtype=torch.float32).to(device),
                torch.tensor([reward], dtype=torch.float32).to(device),
                torch.tensor(next_state, dtype=torch.float32).to(device),
                torch.tensor([done], dtype=torch.float32).to(device),
            )

            state = next_state
            episode_reward += reward

            if len(replay_buffer) >= batch_size:
                s, a, r, ns, d = replay_buffer.sample(batch_size)
                # Update critic
                agent.critic_optimizer.zero_grad()
                target_actions = agent.actor_target(ns)
                y = r + 0.99 * (1 - d) * agent.critic_target(ns, target_actions)
                critic_value = agent.critic(s, a)
                critic_loss = F.mse_loss(y, critic_value)
                critic_loss.backward()
                agent.critic_optimizer.step()

                # Update actor
                agent.actor_optimizer.zero_grad()
                actions = agent.actor(s)
                critic_value = -agent.critic(s, actions).mean()
                critic_value.backward()
                agent.actor_optimizer.step()

                # Update target networks
                agent.update_target_network(agent.actor_target, agent.actor)
                agent.update_target_network(agent.critic_target, agent.critic)

        print(f"Episode: {episode}, Reward: {episode_reward}")
        response.append(episode_reward)  # Store the total reward for the episode

    return response


def test_actor_critic_model():
    state_dim = (
        10  # Assuming the state space dimension is 10, as indicated by your error
    )
    action_dim = 2  # Two actions for the wheels
    action_bound = 1  # Action values range from -1 to 1

    actor = Actor(state_dim, action_dim, action_bound).to(device)
    critic = Critic(state_dim, action_dim).to(device)

    num_dummy_data = 10

    states = torch.randn(num_dummy_data, state_dim).to(device)
    actions = torch.randn(num_dummy_data, action_dim).to(device)
    rewards = torch.randn(num_dummy_data, 1).to(device)
    next_states = torch.randn(num_dummy_data, state_dim).to(device)
    dones = torch.zeros(num_dummy_data, 1).to(device)  # All false for simplicity

    # Initialize the replay buffer
    buffer_capacity = 100  # Larger than num_dummy_data for this test
    replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    # Add dummy data to the buffer
    for i in range(num_dummy_data):
        replay_buffer.add(states[i], actions[i], rewards[i], next_states[i], dones[i])

    # Sample a batch
    batch_size = 5  # Sample smaller than the total data added for clarity
    (
        sampled_states,
        sampled_actions,
        sampled_rewards,
        sampled_next_states,
        sampled_dones,
    ) = replay_buffer.sample(batch_size)

    # Run the sampled batch through the actor and critic
    predicted_actions = actor(sampled_states)
    critic_values = critic(sampled_states, predicted_actions)

    print("Sampled actions from the actor:")
    print(predicted_actions)
    print("\nCritic values for the sampled actions:")
    print(critic_values)


if __name__ == "__main__":
    test_actor_critic_model()
