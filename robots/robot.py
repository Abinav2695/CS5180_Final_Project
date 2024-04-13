import math
import pygame
from constants import (
    ENV_WIDTH,
    ENV_HEIGHT,
    ROBOT_RADIUS,
    WHEEL_WIDTH,
    WHEEL_HEIGHT,
    LINK_LENGTH_MIN,
    LINK_LENGTH_MAX,
    AXEL_LENGTH,
)


class Robot:
    def __init__(self, init_position):
        self.position = init_position
        self.angle = 0  # in degrees
        self.servo_angle = 0  # in degrees
        self.link_length = LINK_LENGTH_MIN
        self.gripper_closed = False

    def move_forward(self, distance):
        rad_angle = math.radians(self.angle)
        self.position[0] += distance * math.sin(rad_angle)
        self.position[1] -= distance * math.cos(rad_angle)

    def move_backward(self, distance):
        self.move_forward(-distance)

    def rotate(self, clockwise, degrees):
        self.angle = (
            (self.angle - degrees) % 360 if clockwise else (self.angle + degrees) % 360
        )

    def toggle_gripper(self):
        self.gripper_closed = not self.gripper_closed

    def rotate_servo(self, angle):
        self.servo_angle = (self.servo_angle + angle) % 360

    def extend_link(self, extend):
        self.link_length = (
            min(self.link_length + 10, LINK_LENGTH_MAX)
            if extend
            else max(self.link_length - 10, LINK_LENGTH_MIN)
        )
