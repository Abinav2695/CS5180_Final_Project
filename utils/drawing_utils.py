import pygame
import math
from constants import ROBOT_RADIUS, WHEEL_WIDTH, WHEEL_HEIGHT

def draw_robot(screen, robot):
    pygame.draw.circle(screen, (128, 128, 128), robot.position, ROBOT_RADIUS)
    draw_wheels(screen, robot)
    draw_link(screen, robot)

def draw_wheels(screen, robot):
    rad_angle = math.radians(robot.angle)
    wheel_offset = ROBOT_RADIUS + WHEEL_HEIGHT*0.5 / 2
    left_wheel_center = [robot.position[0] - wheel_offset * math.cos(rad_angle), robot.position[1] - wheel_offset * math.sin(rad_angle)]
    right_wheel_center = [robot.position[0] + wheel_offset * math.cos(rad_angle), robot.position[1] + wheel_offset * math.sin(rad_angle)]
    draw_wheel(screen, left_wheel_center, rad_angle)
    draw_wheel(screen, right_wheel_center, rad_angle)

def draw_wheel(screen, center, angle):
    wheel_surf = pygame.Surface((WHEEL_WIDTH, WHEEL_HEIGHT), pygame.SRCALPHA)
    pygame.draw.rect(wheel_surf, (0, 0, 0), [0, 0, WHEEL_WIDTH, WHEEL_HEIGHT])
    rotated_surf = pygame.transform.rotate(wheel_surf, -math.degrees(angle))
    screen.blit(rotated_surf, rotated_surf.get_rect(center=center))

def draw_link(screen, robot):
    rad_angle = math.radians(robot.angle + robot.servo_angle)
    link_end = [robot.position[0] + robot.link_length * math.cos(rad_angle), robot.position[1] + robot.link_length * math.sin(rad_angle)]
    gripper_color = (255, 0, 0) if robot.gripper_closed else (0, 0, 255)
    pygame.draw.line(screen, (0, 255, 0), robot.position, link_end, 5)
    pygame.draw.circle(screen, gripper_color, link_end, 8)
