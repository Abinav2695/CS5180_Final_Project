import pygame

from constants import ENV_HEIGHT, ENV_WIDTH

walls_mapping = [
    {"start_pos": (275, 0), "width": 10, "height": 180},
    {"start_pos": (0, 200), "width": 160, "height": 10},
    {"start_pos": (200, 350), "width": ENV_WIDTH, "height": 10},
    # {"start_pos": (400, 760), "width": 10, "height": 40},
    # {"start_pos": (650, 390), "width": 10, "height": 220},
    # {"start_pos": (730, 0), "width": 10, "height": 250},
    # {"start_pos": (0, 420), "width": 210, "height": 10},
    # {"start_pos": (0, 540), "width": 400, "height": 10},
    # {"start_pos": (400, 610), "width": 260, "height": 10},
    # {"start_pos": (650, 390), "width": 230, "height": 10},
    # {"start_pos": (880, 250), "width": 120, "height": 10},
]


class Wall:
    def __init__(self, start_pos, width=10, height=10, color=(0, 0, 0)):
        self.start_pos = start_pos
        self.width = width
        self.height = height
        self.color = color
        self.rect = self.create_rect()

    def create_rect(self):
        """Create a pygame.Rect object for the wall based on its orientation."""
        return pygame.Rect(
            self.start_pos[0], self.start_pos[1], self.width, self.height
        )

    def draw(self, screen):
        """Draw the wall on the screen."""
        pygame.draw.rect(screen, self.color, self.rect)


# Example usage:
if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((ENV_WIDTH, ENV_HEIGHT))

    # Define walls
    wall_objs = [Wall(**wall_data) for wall_data in walls_mapping]
    screen.fill((255, 255, 255))  # Clear screen with white at the start of each frame

    # Draw all walls
    for wall in wall_objs:
        wall.draw(screen)

    running = True
    while running:
        pygame.display.flip()  # Update the full display Surface to the screen

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()
