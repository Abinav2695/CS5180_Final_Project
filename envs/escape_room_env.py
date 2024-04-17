import pygame
import gym
from gym import spaces
import numpy as np

from robots.robot import Robot
from utils.drawing_utils import draw_robot
from constants import (
    ENV_WIDTH,
    ENV_HEIGHT,
    MAX_WHEEL_VELOCITY,
    GOAL_X_MAX,
    GOAL_X_MIN,
    GOAL_Y_MAX,
    GOAL_Y_MIN,
)


class EscapeRoomEnv(gym.Env):
    def __init__(self, continuous=False):
        super().__init__()
        ### Observation Space
        ### Observation Shape (8,) [ X, Y, Theta, Vx, Vy, Omega, ] #Have to add robot arm

        ### Observation High [1.5 1.5 3.14 5.  5.  5.  ]
        ### Observation Low [-1.5 -1.5 -3.14 -5. -5. -5. ]  These are percentages 0-1 = 0%- 100% so 1.5 = 150%
        # these are bounds for position. Realistically the environment should have ended
        # long before we reach more than 50% outside

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
        self.goal_pos = np.array([700, 700])

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
            self.action_space = spaces.Discrete(5)

        self.robot = Robot((400, 300))
        self.max_steps_per_episode = 2000
        self.t = 0  ##Time step counter

        pygame.init()
        self.screen = pygame.display.set_mode((ENV_WIDTH, ENV_HEIGHT))
        self.clock = pygame.time.Clock()

    def step(self, action):
        assert self.action_space.contains(action), "Action is out of bounds!"

        if self.continuous:
            # Scale actions from [-1, 1] to actual velocity values [-max_vel, max_vel]
            left_vel = action[0] * self.max_vel
            right_vel = action[1] * self.max_vel
        else:
            # Map discrete actions to velocities
            if action == 0:  # Forward
                left_vel, right_vel = self.max_vel, self.max_vel
            elif action == 1:  # Backward
                left_vel, right_vel = -self.max_vel, -self.max_vel
            elif action == 2:  # Stop
                left_vel, right_vel = 0, 0
            elif action == 3:  # Rotate Left
                left_vel, right_vel = -self.max_vel, self.max_vel
            elif action == 4:  # Rotate Right
                left_vel, right_vel = self.max_vel, -self.max_vel
            else:
                left_vel, right_vel = 0, 0  # Default to stop if invalid action

        penalty, out_of_bounds = self.robot.update_position(left_vel, right_vel, dt=1)
        reward = penalty
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

        if self.has_reached_goal([self.robot.x, self.robot.y]):
            reward += 100
            terminated = True
            info = {"reason": "goal_reached"}
        elif out_of_bounds:
            terminated = True
            info = {"reason": "out_of_bounds"}
        elif self.t >= self.max_steps_per_episode:
            truncated = True
            info = {"reason": "max_steps_reached"}

        return state, reward, terminated, truncated, info

    def reset(self):
        self.robot = Robot([400, 300], init_angle=0)
        self.t = 0  # Reset the timestep counter
        info = {
            "message": "Environment reset."
        }  # Optionally provide more info about the reset
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

    def has_reached_goal(self, position):
        x, y = position
        # Calculate euclidean distance to goal position
        dist_to_goal = np.sqrt(
            (x - self.goal_pos[0]) ** 2 + (y - self.goal_pos[1]) ** 2
        )
        if dist_to_goal <= 0.1:
            return True
        return False

    def render(self, mode="human"):
        if mode == "human":
            self.screen.fill((255, 255, 255))  # Clear the screen
            draw_robot(self.screen, self.robot)
            pygame.display.flip()
            self.clock.tick(30)

    def close(self):
        pygame.quit()


# Main executable block
if __name__ == "__main__":
    env = EscapeRoomEnv()
    try:
        for _ in range(500):
            action = env.action_space.sample()
            env.step(action)
            env.render()
    except KeyboardInterrupt:
        print("Simulation stopped manually.")
    finally:
        env.close()
