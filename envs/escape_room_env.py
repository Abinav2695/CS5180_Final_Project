import gym
import numpy as np
import pygame
from gym import spaces

from constants import CHECKPOINT_RADIUS, ENV_HEIGHT, ENV_WIDTH, MAX_WHEEL_VELOCITY
from robots.checkpoint import Checkpoint
from robots.robot import Robot
from robots.walls import Wall, walls_mapping
from utils.drawing_utils import draw_robot


class EscapeRoomEnv(gym.Env):
    def __init__(self, max_steps_per_episode=2000, continuous=False):
        super().__init__()
        ### Observation Space
        ### Observation Shape (8,) [ X, Y, Theta, Vx, Vy, Omega, ] #Have to add robot arm

        ### Observation High [1.5 1.5 3.14 5.  5.  5.  ]
        ### Observation Low [-1.5 -1.5 -3.14 -5. -5. -5. ]  These are percentages 0-1 = 0%- 100% so 1.5 = 150%
        # these are bounds for position. Realistically the environment should have ended
        # long before we reach more than 50% outside
        self.spawn_x = 70
        self.spawn_y = 70
        self.goal_position = np.array([950, 750])

        self.walls = [Wall(**wall_data) for wall_data in walls_mapping]

        self.intermediate_goals = [
            Checkpoint((300, 50), CHECKPOINT_RADIUS, (255, 0, 0), "A"),
            Checkpoint((900, 50), CHECKPOINT_RADIUS, (0, 255, 0), "B"),
            Checkpoint((50, 750), CHECKPOINT_RADIUS, (0, 0, 255), "C"),
        ]
        self.goal = Checkpoint(
            self.goal_position, CHECKPOINT_RADIUS, (255, 0, 0), "G"
        )  # Goal

        low = np.array(
            [
                -1.5 * ENV_WIDTH,
                -1.5 * ENV_HEIGHT,
                -np.pi,
                -5.0 * MAX_WHEEL_VELOCITY,
                -5.0 * MAX_WHEEL_VELOCITY,
                -5.0 * MAX_WHEEL_VELOCITY,
            ]
        )
        high = np.array(
            [
                1.5 * ENV_WIDTH,
                1.5 * ENV_HEIGHT,
                np.pi,
                5.0 * MAX_WHEEL_VELOCITY,
                5.0 * MAX_WHEEL_VELOCITY,
                5.0 * MAX_WHEEL_VELOCITY,
            ]
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.continuous = continuous
        self.max_vel = MAX_WHEEL_VELOCITY

        # Define action space
        if self.continuous:
            # continuous action space
            self.action_space = spaces.Box(
                low=np.array([-1, -1]),
                high=np.array([1, 1]),
                dtype=np.float32,
            )
        else:
            # Discrete action space: 0 - Forward, 1 - Backward, 2 - Stop, 3 - Rotate Left, 4 - Rotate Right
            self.action_space = spaces.Discrete(4)

        # Define action to velocity mapping
        self.action_to_velocity = {
            0: (self.max_vel, self.max_vel),  # Forward
            1: (-self.max_vel, -self.max_vel),  # Backward
            # 2: (0, 0),  # Stop
            2: (-self.max_vel, self.max_vel),  # Rotate Left
            3: (self.max_vel, -self.max_vel),  # Rotate Right
        }

        self.robot = Robot((self.spawn_x, self.spawn_y))
        self.max_steps_per_episode = max_steps_per_episode
        self.t = 0  ##Time step counter

        # pygame display initialization
        self.screen = None
        self.clock = None

    # def step(self, action):
    #     assert self.action_space.contains(action), "Action is out of bounds!"
    #     # Penalties and rewards
    #     step_penalty = 0  # Penalty for each step taken
    #     distance_reward_factor = 0.1  # Scale factor for distance reward

    #     # Current state before taking the action
    #     current_pos = np.array([self.robot.x, self.robot.y])
    #     current_distance = np.linalg.norm(current_pos - self.goal_position)

    #     # Update state by taking an action
    #     if self.continuous:
    #         # Scale actions from [-1, 1] to actual velocity values [-max_vel, max_vel]
    #         left_vel = action[0] * self.max_vel
    #         right_vel = action[1] * self.max_vel
    #     else:
    #         # Map discrete actions to velocities
    #         left_vel, right_vel = self.action_to_velocity.get(action, (0, 0))

    #     penalty, out_of_bounds = self.robot.update_and_check_collisions(
    #         left_vel, right_vel, self.walls, dt=1
    #     )
    #     # New position after taking the action
    #     new_pos = np.array([self.robot.x, self.robot.y])
    #     new_distance = np.linalg.norm(new_pos - self.goal_position)

    #     # Distance reward shaping
    #     distance_reward = (current_distance - new_distance) * distance_reward_factor
    #     reward = penalty + distance_reward + step_penalty

    #     state = np.array(
    #         [
    #             self.robot.x,
    #             self.robot.y,
    #             self.robot.theta,
    #             self.robot.vx,
    #             self.robot.vy,
    #             self.robot.omega,
    #         ]
    #     )
    #     assert len(state) == 6
    #     self.t += 1

    #     terminated = False
    #     truncated = False
    #     info = {}

    #     if self.goal.check_goal_reached((self.robot.x, self.robot.y)):
    #         reward += 100  # goal reward
    #         terminated = True
    #         info = {"reason": "goal_reached"}
    #     elif out_of_bounds:
    #         terminated = True
    #         info = {"reason": "out_of_bounds"}
    #     elif self.t >= self.max_steps_per_episode:
    #         truncated = True
    #         info = {"reason": "max_steps_reached"}

    #     return state, reward, terminated, truncated, info

    # def step_bk(self, action):
    #     assert self.action_space.contains(action), "Action is out of bounds!"
    #     # Penalties and rewards
    #     step_penalty = 0  # Penalty for each step taken
    #     distance_reward_factor = 0.1  # Scale factor for distance reward

    #     # Current state before taking the action
    #     current_pos = np.array([self.robot.x, self.robot.y])
    #     current_distance = np.linalg.norm(current_pos - self.goal_position)

    #     # Update state by taking an action
    #     if self.continuous:
    #         # Scale actions from [-1, 1] to actual velocity values [-max_vel, max_vel]
    #         left_vel = action[0] * self.max_vel
    #         right_vel = action[1] * self.max_vel
    #     else:
    #         # Map discrete actions to velocities
    #         left_vel, right_vel = self.action_to_velocity.get(action, (0, 0))

    #     penalty, out_of_bounds = self.robot.update_and_check_collisions(
    #         left_vel, right_vel, self.walls, dt=1
    #     )

    #     # New position after taking the action
    #     new_pos = np.array([self.robot.x, self.robot.y])
    #     new_distance = np.linalg.norm(new_pos - self.goal_position)

    #     # Enhanced Distance reward shaping using exponential scaling
    #     distance_improvement = current_distance - new_distance
    #     distance_reward = (np.exp(distance_improvement) - 1) * distance_reward_factor
    #     reward = penalty + distance_reward + step_penalty

    #     state = np.array(
    #         [
    #             self.robot.x,
    #             self.robot.y,
    #             self.robot.theta,
    #             self.robot.vx,
    #             self.robot.vy,
    #             self.robot.omega,
    #         ]
    #     )
    #     self.t += 1

    #     terminated = False
    #     truncated = False
    #     info = {}

    #     if self.goal.check_goal_reached((self.robot.x, self.robot.y)):
    #         reward += 100  # goal reward
    #         terminated = True
    #         info = {"reason": "goal_reached"}
    #     elif out_of_bounds:
    #         # print("Robot Out of bounds")
    #         terminated = True
    #         info = {"reason": "out_of_bounds"}
    #     elif self.t >= self.max_steps_per_episode:
    #         truncated = True
    #         info = {"reason": "max_steps_reached"}

    #     return state, reward, terminated, truncated, info

    def step(self, action):
        assert self.action_space.contains(action), "Action is out of bounds!"

        # Current position before taking the action
        current_pos = np.array([self.robot.x, self.robot.y])
        step_penalty = -0.01

        # Update state by taking an action
        if self.continuous:
            # Scale actions from [-1, 1] to actual velocity values [-max_vel, max_vel]
            left_vel = action[0] * self.max_vel
            right_vel = action[1] * self.max_vel
        else:
            # Map discrete actions to velocities based on the selected action
            left_vel, right_vel = self.action_to_velocity.get(action, (0, 0))

        # Update the robot's position and check for collisions
        penalty, out_of_bounds = self.robot.update_and_check_collisions(
            left_vel, right_vel, self.walls, dt=1
        )

        # New position after taking the action
        new_pos = np.array([self.robot.x, self.robot.y])
        new_distance = np.linalg.norm(new_pos - self.goal_position)

        # Calculate the reward as a function of the error in distance
        error = new_distance
        reward = (
            1 / (1 + error)
        ) + penalty  # Immediate reward based on the distance and penalty for collisions

        # reward = penalty
        state = np.array(
            [
                self.robot.x,
                self.robot.y,
                self.robot.theta,
                self.robot.vx,
                self.robot.vy,
                self.robot.omega,
            ]
        )
        self.t += 1

        terminated = False
        truncated = False
        info = {}

        if self.goal.check_goal_reached((self.robot.x, self.robot.y)):
            reward += 100  # Additional reward for reaching the goal
            terminated = True
            info = {"reason": "goal_reached"}
            print("Goal Reached")
        elif out_of_bounds:
            terminated = True
            info = {"reason": "out_of_bounds"}
        elif self.t >= self.max_steps_per_episode:
            truncated = True
            info = {"reason": "max_steps_reached"}

        return state, reward, terminated, truncated, info

    def reset(self):
        self.robot = Robot([self.spawn_x, self.spawn_y], init_angle=0)
        self.t = 0  # Reset the timestep counter
        info = {
            "message": "Environment reset."
        }  # Optionally provide more info about the reset
        self.screen = None
        self.clock = None
        # Return the complete state including velocities
        return (
            np.array(
                [
                    self.robot.x,
                    self.robot.y,
                    self.robot.theta,
                    self.robot.vx,
                    self.robot.vy,
                    self.robot.omega,
                ]
            ),
            info,
        )

    def render(self, mode="human"):
        if mode == "human":
            if self.screen is None:
                pygame.init()
                pygame.display.init()
                self.screen = pygame.display.set_mode((ENV_WIDTH, ENV_HEIGHT))
            if self.clock is None:
                self.clock = pygame.time.Clock()

            # Clear the screen with white background at the start of each render cycle
            self.screen.fill((255, 255, 255))

            # Draw all walls
            for wall in self.walls:
                wall.draw(self.screen)

            # Draw all checkpoints including the goal if they have not been reached
            for checkpoint in self.intermediate_goals:
                checkpoint.draw(self.screen)

            # Draw the final goal
            self.goal.draw(self.screen)

            # Draw the robot on the screen
            draw_robot(self.screen, self.robot)

            # Update the full display Surface to the screen
            pygame.display.flip()

            # Limit the frame rate to maintain a consistent rendering speed
            self.clock.tick(30)

    def close(self):
        pygame.quit()


# Main executable block
if __name__ == "__main__":
    env = EscapeRoomEnv()
    assert (
        isinstance(env.observation_space, gym.spaces.Box)
        and len(env.observation_space.shape) == 1
    )
    try:
        for _ in range(1000):
            action = env.action_space.sample()
            env.step(action)
            env.render()
    except KeyboardInterrupt:
        print("Simulation stopped manually.")
    finally:
        env.close()
