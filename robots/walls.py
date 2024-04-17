import pygame
class Wall:
    def __init__(self, start_pos, end_pos, thickness):
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.thickness = thickness
        # Create a pygame.Rect object for the wall
        # This will help with drawing and collision detection
        # Inflate the rectangle by the robot's diameter to simulate the robot's area for collision detection
        if start_pos[0] == end_pos[0]:  # Vertical wall
            self.rect = pygame.Rect(start_pos[0] - thickness // 2, min(start_pos[1], end_pos[1]),
                                    thickness, abs(end_pos[1] - start_pos[1]))
        else:  # Horizontal wall
            self.rect = pygame.Rect(min(start_pos[0], end_pos[0]), start_pos[1] - thickness // 2,
                                    abs(end_pos[0] - start_pos[0]), thickness)

    def draw(self, screen):
        # Draw a black rectangle (wall) on the screen
        pygame.draw.rect(screen, (0, 0, 0), self.rect)




wall_thickness = 10

# Vertical walls (x stays constant, y changes)
vertical_walls = [
    ((200, 0), (200,120)),  # Vertical wall on the left side
    ((200, 240), (200, 420)),  # Vertical wall on the right before the bottom corridor
    ((400, 540), (400, 660)),
    ((400, 760), (400, 800)),  # Short vertical wall in the middle
    ((650, 390), (650, 610)),
    ((730, 0), (730, 250)),  # Vertical wall in the middle left
]

# Horizontal walls (y stays constant, x changes)
horizontal_walls = [
    ((130, 420), (200, 420)),  # Bottom long horizontal wall
    ((0, 540), (400, 540)),  # Short horizontal wall before the right vertical wall
    ((400, 610), (650, 610)),  # Top horizontal wall after the middle left vertical wall
    ((650, 390), (880, 390)),  # Short horizontal wall connecting to the short vertical wall
    ((880, 250), (1000, 250))
]

all_walls = vertical_walls+horizontal_walls
