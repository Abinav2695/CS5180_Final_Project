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
from walls import Wall,all_walls





class Robot:
    def __init__(self, init_position: tuple, init_angle: float = 0):
        self.x, self.y = init_position
        self.theta = init_angle  # Orientation in radians
        self.vx, self.vy, self.omega = 0, 0, 0  # Initial velocities
        self.servo_angle = 0  # Servo angle in degrees
        self.link_length = LINK_LENGTH_MIN
        self.gripper_closed = False

    def normalize_angle(self, angle):
        """Normalize an angle to the range [-pi, pi]."""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def update_position(self, left_vel, right_vel, walls, dt=1):
        v = (left_vel + right_vel) / 2
        omega = (right_vel - left_vel) / AXEL_LENGTH
        self.theta += omega * dt
        self.theta = self.normalize_angle(
            self.theta
        )  # Normalize theta to be within [-pi, pi]

        self.vx = v * math.cos(self.theta)  # Compute current velocity components
        self.vy = v * math.sin(self.theta)
        self.omega = omega  # Angular velocity
        self.x += self.vx * dt  # Update position based on velocity
        self.y += self.vy * dt

        penalty, out_of_bounds = self.check_boundaries(walls)
        return penalty, out_of_bounds

    def check_boundaries(self, walls):
        robot_rect = self.get_collision_rect()
        if self.x < 0 or self.x > ENV_WIDTH or self.y < 0 or self.y > ENV_HEIGHT:
            return -100, True  # Collision penalty and signal that it's out of bounds
        for wall in walls:
            if wall.rect.colliderect(robot_rect):
                return -100, True  # Penalty for colliding with a wall
        return 0, False  # No penalty if no collision with walls or boundaries
    

    def get_collision_rect(self):
        """ Get the collision rectangle encompassing the robot and its wheels. """
        total_robot_width = ROBOT_RADIUS * 2 + WHEEL_WIDTH
        total_robot_height = ROBOT_RADIUS * 2
        return pygame.Rect(
            self.x - total_robot_width / 2,
            self.y - total_robot_height / 2,
            total_robot_width,
            total_robot_height
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
