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
    def __init__(self, max_steps_per_episode=3000):
        super().__init__()

        self.spawn_x = 70
        self.spawn_y = 70
        self.goal_position = np.array([200, 200])
        # self.goal_position = np.array([900, 50])

        self.walls = [Wall(**wall_data) for wall_data in walls_mapping]

        self.intermediate_goals = [
            Checkpoint((300, 50), CHECKPOINT_RADIUS, (255, 0, 0), "A"),
            Checkpoint((900, 50), CHECKPOINT_RADIUS, (0, 255, 0), "B"),
            Checkpoint((50, 750), CHECKPOINT_RADIUS, (0, 0, 255), "C"),
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

    def step(self, action):
        action = np.clip(action, -1, +1).astype(np.float32)

        left_vel = action[0] * MAX_WHEEL_VELOCITY
        right_vel = action[1] * MAX_WHEEL_VELOCITY

        penalty, out_of_bounds = self.robot.update_and_check_collisions(
            left_vel, right_vel, self.walls, dt=1
        )
        
        new_pos = np.array([self.robot.x, self.robot.y])
        old_distance = self.old_distance  # this needs to be stored after each step
        new_distance = np.linalg.norm(new_pos - self.goal_position)
        self.old_distance = new_distance  # update the old distance

        alpha = 0.1
        # # Simplified and less penalizing distance reward
        if new_distance > self.old_distance:
            reward_distance = -np.log1p(new_distance) * alpha # Logarithmic penalty for smoother gradient
        else:
            reward_distance = +np.log1p(self.old_distance) * alpha # Logarithmic penalty for smoother gradient

        reward_efficiency = max(old_distance - new_distance, 0) * (1-alpha) # Only reward forward movement #max

        reward = reward_distance + reward_efficiency + (penalty * (1-alpha))
        # Aggregate reward components
        # reward = reward_distance + penalty

        # new_pos = np.array([self.robot.x, self.robot.y])
        # new_distance = np.linalg.norm(new_pos - self.goal_position)
        # error = new_distance 
        # reward = (
        #     1 / (1 + error) + penalty
        # )  # Reward based on the distance and collision penalty

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
            reward += alpha * 5000  # goal reward
            terminated = True 
            info = {"reason": "goal_reached"}
        elif out_of_bounds:
            terminated = True
            info = {"reason": "out_of_bounds"}
        elif self.t >= self.max_steps_per_episode:
            truncated = True
            reward += (alpha - 1)
            info = {"reason": "max_steps_reached"}

        return state, reward, terminated, truncated, info

    def reset(self):
        self.robot = Robot([self.spawn_x, self.spawn_y], init_angle=0)
        self.t = 0  # Reset timestep counter
        self.old_distance = np.linalg.norm(
            np.array([self.robot.x, self.robot.y]) - self.goal_position
        )
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
