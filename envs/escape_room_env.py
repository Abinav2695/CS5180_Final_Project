import pygame
import gym
from gym import spaces
import numpy as np
from robots.robot import Robot
from utils.drawing_utils import draw_robot
from constants import ENV_WIDTH, ENV_HEIGHT

class EscapeRoomEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(9)
        self.robot = Robot([400, 300])
        pygame.init()
        self.screen = pygame.display.set_mode((ENV_WIDTH, ENV_HEIGHT))
        self.clock = pygame.time.Clock()

    def step(self, action):
        actions = {
            0: lambda: self.robot.move_forward(10),
            1: lambda: self.robot.move_backward(10),
            2: lambda: self.robot.rotate(True, 10),
            3: lambda: self.robot.rotate(False, 10),
            4: lambda: self.robot.rotate_servo(10),
            5: lambda: self.robot.rotate_servo(-10),
            6: lambda: self.robot.extend_link(True),
            7: lambda: self.robot.extend_link(False),
            8: self.robot.toggle_gripper
        }
        actions[action]()
        return np.array(self.robot.position), 0, False, {}

    def reset(self):
        self.robot = Robot([400, 300])
        return np.array(self.robot.position)

    def render(self, mode='human'):
        self.screen.fill((255, 255, 255))
        draw_robot(self.screen, self.robot)
        pygame.display.flip()
        self.clock.tick(60)

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
