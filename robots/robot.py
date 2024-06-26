import math

import pygame

from constants import (
    AXEL_LENGTH,
    ENV_HEIGHT,
    ENV_WIDTH,
    LINK_LENGTH_MAX,
    LINK_LENGTH_MIN,
    ROBOT_RADIUS,
    WHEEL_HEIGHT,
    WHEEL_WIDTH,
)


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

    def get_collision_circle(self):
        """Returns the position and radius for the robot's collision circle."""
        return (self.x, self.y, ROBOT_RADIUS + WHEEL_WIDTH)

    def circle_rect_collision(self, circle, rect: pygame.Rect):
        """Check collision between a circle and a rectangle.

        Args:
            circle (tuple): A tuple (cx, cy, radius) representing the circle.
            rect (pygame.Rect): A rectangle object.

        Returns:
            bool: True if there is a collision, False otherwise.
        """
        cx, cy, radius = circle
        # Find the closest point on the rectangle to the circle
        closest_x = max(rect.left, min(cx, rect.right))
        closest_y = max(rect.top, min(cy, rect.bottom))

        # Calculate the distance between the circle's center and this closest point
        distance_x = cx - closest_x
        distance_y = cy - closest_y

        # If the distance is less than the circle's radius, there's a collision
        return (distance_x**2 + distance_y**2) <= (radius**2)

    def check_boundary_collision(self, circle):
        """Check if the robot collides with the environment boundaries.

        Args:
            circle (tuple): A tuple (cx, cy, radius) representing the robot's collision circle.

        Returns:
            bool: True if there is a boundary collision, False otherwise.
        """
        cx, cy, radius = circle
        if cx - radius < 0 or cx + radius > ENV_WIDTH:
            return True
        if cy - radius < 0 or cy + radius > ENV_HEIGHT:
            return True
        return False

    def update_and_check_collisions(self, left_vel, right_vel, walls, dt=1):
        """Update the robot's position and check for collisions with walls.

        Args:
            left_vel (float): Velocity of the left wheel.
            right_vel (float): Velocity of the right wheel.
            dt (int): Time delta.
            walls (list): A list of pygame.Rect objects representing the walls.

        Returns:
            tuple: (penalty, collision_flag) where penalty is a numerical value indicating
                the collision penalty, and collision_flag is a boolean indicating if a collision occurred.
        """
        # Update position
        v = (left_vel + right_vel) / 2
        omega = (right_vel - left_vel) / AXEL_LENGTH
        self.theta += omega * dt
        self.theta = self.normalize_angle(self.theta)

        self.vx = v * math.cos(self.theta)
        self.vy = v * math.sin(self.theta)
        self.x += self.vx * dt
        self.y += self.vy * dt

        # Check for collisions
        collision_flag = False
        robot_circle = self.get_collision_circle()

        if self.check_boundary_collision(robot_circle):
            collision_flag = True

        for wall in walls:
            if self.circle_rect_collision(robot_circle, wall.rect):
                collision_flag = True
                break

        penalty = -100 if collision_flag else 0
        return (penalty, collision_flag)

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
