import math
import time

import gym
import numpy as np
import pygame
from gym import spaces

# Constants for environment and robot dimensions
ENV_WIDTH, ENV_HEIGHT = 1000, 800
ROBOT_RADIUS = 30
WHEEL_WIDTH, WHEEL_HEIGHT = 10, 20
LINK_LENGTH_MIN, LINK_LENGTH_MAX = 50, 120  # Min and max lengths of the link


class Robot:
    def __init__(self, init_position):
        self.position = init_position
        self.angle = 0  # Robot's angle in degrees
        self.servo_angle = 0  # Servo (link) angle in degrees
        self.link_length = LINK_LENGTH_MIN  # Initial length of the link
        self.gripper_closed = False

    def move_forward(self, distance):
        # Convert the angle to radians
        rad_angle = math.radians(self.angle)

        self.position[0] += distance * math.sin(rad_angle)
        self.position[1] -= distance * math.cos(rad_angle)

    def move_backward(self, distance):
        # To move backward, we simply call move_forward with a negative distance
        self.move_forward(-distance)

    def rotate(self, clockwise, degrees):
        if clockwise:
            self.angle -= degrees
        else:
            self.angle += degrees
        self.angle %= 360

    def toggle_gripper(self):
        # Toggle the state of the gripper between open and closed
        self.gripper_closed = not self.gripper_closed

    def rotate_servo(self, angle):
        # Update the servo angle with the provided angle value
        self.servo_angle += angle

        # Normalize the servo angle to stay within the range 0-360 degrees
        self.servo_angle %= 360

    def extend_link(self, extend):
        if extend:
            self.link_length = min(self.link_length + 10, LINK_LENGTH_MAX)
        else:
            self.link_length = max(self.link_length - 10, LINK_LENGTH_MIN)

    def draw(self, screen):
        # Draw the robot body as a circle
        pygame.draw.circle(screen, (128, 128, 128), self.position, ROBOT_RADIUS)

        # Calculate the angle in radians
        rad_angle = math.radians(self.angle)

        # The offset from the robot's center to where the wheel's center should be
        wheel_offset = ROBOT_RADIUS + WHEEL_HEIGHT * 0.5 / 2

        # Calculate the left wheel center, taking the offset into account
        left_wheel_center = (
            self.position[0] - wheel_offset * math.cos(rad_angle),
            self.position[1] - wheel_offset * math.sin(rad_angle),
        )

        # Calculate the right wheel center, taking the offset into account
        right_wheel_center = (
            self.position[0] + wheel_offset * math.cos(rad_angle),
            self.position[1] + wheel_offset * math.sin(rad_angle),
        )

        # Create surfaces for wheels to allow rotation
        left_wheel_surf = pygame.Surface((WHEEL_WIDTH, WHEEL_HEIGHT), pygame.SRCALPHA)
        right_wheel_surf = pygame.Surface((WHEEL_WIDTH, WHEEL_HEIGHT), pygame.SRCALPHA)

        # Draw the wheels on their respective surfaces
        pygame.draw.rect(left_wheel_surf, (0, 0, 0), [0, 0, WHEEL_WIDTH, WHEEL_HEIGHT])
        pygame.draw.rect(right_wheel_surf, (0, 0, 0), [0, 0, WHEEL_WIDTH, WHEEL_HEIGHT])

        # Rotate the wheel surfaces according to the robot's angle
        left_wheel_rotated = pygame.transform.rotate(
            left_wheel_surf, -math.degrees(rad_angle)
        )
        right_wheel_rotated = pygame.transform.rotate(
            right_wheel_surf, -math.degrees(rad_angle)
        )

        # Blit the rotated wheel images to the screen, positioned such that the wheels appear tangent to the robot's body
        screen.blit(
            left_wheel_rotated, left_wheel_rotated.get_rect(center=left_wheel_center)
        )
        screen.blit(
            right_wheel_rotated, right_wheel_rotated.get_rect(center=right_wheel_center)
        )

        link_angle = rad_angle + math.radians(
            self.servo_angle
        )  # Servo angle relative to body
        link_end = (
            self.position[0] + self.link_length * math.cos(link_angle),
            self.position[1] + self.link_length * math.sin(link_angle),
        )

        # Draw the gripper with a visual indication if it's gripping an item
        gripper_color = (255, 0, 0) if self.gripper_closed else (0, 0, 255)

        # Draw the link rotating around its axis relative to the body
        pygame.draw.line(screen, (0, 255, 0), self.position, link_end, 5)
        pygame.draw.circle(
            screen, gripper_color, link_end, 8
        )  # Use gripper_color which changes based on state


class EscapeRoomEnv(gym.Env):
    def __init__(self):
        super(EscapeRoomEnv, self).__init__()
        self.action_space = spaces.Discrete(
            9
        )  # Forward, backward, rotate left, rotate right, servo rotate, extend link, retract link
        self.robot = Robot([400, 300])
        pygame.init()
        self.screen = pygame.display.set_mode((ENV_WIDTH, ENV_HEIGHT))
        self.clock = pygame.time.Clock()

    def step(self, action):
        if action == 0:
            self.robot.move_forward(10)
        elif action == 1:
            self.robot.move_backward(10)
        elif action == 2:
            self.robot.rotate(True, 10)  # Clockwise
        elif action == 3:
            self.robot.rotate(False, 10)  # Counterclockwise
        elif action == 4:
            self.robot.rotate_servo(10)
        elif action == 5:
            self.robot.rotate_servo(-10)
        elif action == 6:
            self.robot.extend_link(True)
        elif action == 7:
            self.robot.extend_link(False)
        elif action == 8:
            # This is the new action for toggling the gripper
            self.robot.toggle_gripper()

        return np.array(self.robot.position), 0, False, {}

    def reset(self):
        self.robot = Robot([400, 300])
        return np.array(self.robot.position)

    def render(self, mode="human"):
        if mode == "human":
            self.screen.fill((255, 255, 255))  # Clear screen
            self.robot.draw(self.screen)
            pygame.display.flip()
            self.clock.tick(60)

    def close(self):
        pygame.quit()


# Initialize the environment
env = EscapeRoomEnv()
env.reset()

# Run the environment with a random policy
try:
    for _ in range(500):  # Limit the number of steps for demonstration
        action = env.action_space.sample()  # Randomly sample an action
        env.step(action)  # Take the action
        env.render()  # Render the current state of the environment
        # time.sleep(0.1)  # Pause for 0.05 seconds
except KeyboardInterrupt:
    print("Simulation stopped manually.")

env.close()  # Clean up the environment
