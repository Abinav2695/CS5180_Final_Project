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
    def __init__(self, max_steps_per_episode=2500):
        super().__init__()

        self.spawn_x = 70
        self.spawn_y = 70
        self.goal_position = np.array([460, 460])

        self.walls = [Wall(**wall_data) for wall_data in walls_mapping]

        self.intermediate_goals = [
            # Checkpoint((300, 50), CHECKPOINT_RADIUS, (255, 0, 0), "A"),
            # Checkpoint((900, 50), CHECKPOINT_RADIUS, (0, 255, 0), "B"),
            # Checkpoint((50, 750), CHECKPOINT_RADIUS, (0, 0, 255), "C"),
        ]
        self.goal = Checkpoint(self.goal_position, CHECKPOINT_RADIUS, (255, 0, 0), "G")

        low = np.array([-1.5 * ENV_WIDTH, -1.5 * ENV_HEIGHT, -np.pi, -5.0, -5.0, -5.0])
        high = np.array([1.5 * ENV_WIDTH, 1.5 * ENV_HEIGHT, np.pi, 5.0, 5.0, 5.0])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Continuous action space for left and right wheel velocities
        # self.action_space = spaces.Box(
        #     low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32
        # )
        self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)

        self.robot = Robot((self.spawn_x, self.spawn_y))
        self.max_steps_per_episode = max_steps_per_episode
        self.t = 0  # Time step counter

        self.screen = None
        self.clock = None

    # def step(self, action):
    #     # Ensure action values are within the allowable range
    #     action = np.clip(action, -1, +1).astype(np.float32)

    #     # Current position before the move
    #     current_pos = np.array([self.robot.x, self.robot.y])
    #     current_distance = np.linalg.norm(current_pos - self.goal_position)

    #     # Convert actions to actual wheel velocities
    #     left_vel = action[0] * MAX_WHEEL_VELOCITY
    #     right_vel = action[1] * MAX_WHEEL_VELOCITY

    #     # Update robot position and check for collisions
    #     penalty, out_of_bounds = self.robot.update_and_check_collisions(
    #         left_vel, right_vel, self.walls, dt=1
    #     )

    #     # Compute new position and new distance to the goal after the move
    #     new_pos = np.array([self.robot.x, self.robot.y])
    #     new_distance = np.linalg.norm(new_pos - self.goal_position)

    #     # Calculate reward based on distance reduction
    #     step_penalty = 0
    #     reward = step_penalty  # Small step penalty to encourage efficiency
    #     if new_distance < current_distance:
    #         reward += (1/ (1 + new_distance))  # Reward for moving closer

    #     reward += penalty  # Add collision penalty

    #     # Initialize termination flags and info dictionary
    #     terminated = False
    #     truncated = False
    #     info = {}

    #     # Check termination conditions
    #     if self.goal.check_goal_reached((self.robot.x, self.robot.y)):
    #         terminated = True
    #         reward += 100  # Bonus for reaching the goal
    #         info = {"reason": "goal_reached"}
    #         print("goal reached")
    #     elif out_of_bounds:
    #         terminated = True
    #         info = {"reason": "out_of_bounds"}
    #     elif self.t >= self.max_steps_per_episode:
    #         truncated = True
    #         info = {"reason": "max_steps_reached"}

    #     # Prepare the state vector for the agent
    #     state = np.array([
    #         self.robot.x,
    #         self.robot.y,
    #         self.robot.theta,
    #         self.robot.vx,
    #         self.robot.vy,
    #         self.robot.omega,
    #     ])

    #     # Increment the timestep counter
    #     self.t += 1

    #     return state, reward, terminated, truncated, info

    def step(self, action):
        # Ensure action values are within the allowable range
        action = np.clip(action, -1, +1).astype(np.float32)

        # Calculate the current position and distance to the goal
        current_pos = np.array([self.robot.x, self.robot.y])
        current_distance = np.linalg.norm(current_pos - self.goal_position)

        # Convert actions to actual wheel velocities
        left_vel = action[0] * MAX_WHEEL_VELOCITY
        right_vel = action[1] * MAX_WHEEL_VELOCITY

        # Update the robot's position and check for collisions
        penalty, out_of_bounds = self.robot.update_and_check_collisions(
            left_vel, right_vel, self.walls, dt=1
        )

        # Calculate the new position and distance to the goal
        new_pos = np.array([self.robot.x, self.robot.y])
        new_distance = np.linalg.norm(new_pos - self.goal_position)

        # Calculate reward based on distance to the goal
        reward = -0.1  # Small step penalty to encourage efficiency
        if new_distance < current_distance:
            reward += 0.1 * (current_distance - new_distance)  # Scale reward with distance reduction
            # Add an efficiency bonus if the velocities are within a controlled range
            if np.abs(left_vel) + np.abs(right_vel) < MAX_WHEEL_VELOCITY:
                reward += 0.5  # Adjust this value based on desired efficiency
        reward += penalty  # Include collision penalty

        terminated = False
        truncated = False
        info = {}

        # Check for termination conditions
        if self.goal.check_goal_reached((self.robot.x, self.robot.y)):
            terminated = True
            reward += 100  # Large bonus for reaching the goal
            info = {"reason": "goal_reached"}
            print("Goal Reached ................")
        elif out_of_bounds:
            terminated = True
            reward -= 100  # Large penalty for going out of bounds
            info = {"reason": "out_of_bounds"}
        elif self.t >= self.max_steps_per_episode:
            truncated = True
            info = {"reason": "max_steps_reached"}

        # Update the state vector
        state = np.array([
            self.robot.x, self.robot.y, self.robot.theta,
            self.robot.vx, self.robot.vy, self.robot.omega
        ])

        # Increment the timestep counter
        self.t += 1

        return state, reward, terminated, truncated, info

    def reset(self):
        self.robot = Robot([self.spawn_x, self.spawn_y], init_angle=0)
        self.t = 0  # Reset timestep counter
        self.screen = None
        self.clock = None
        info = {"message": "Environment reset."}
        self.goal.reset()
        for chk_pts in self.intermediate_goals:
            chk_pts.reset()

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
