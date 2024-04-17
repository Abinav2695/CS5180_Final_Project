import pygame
class Checkpoint:
    def __init__(self, center_pos, size, color, label):
        self.center_pos = center_pos
        self.size = size
        self.color = color
        self.label = label
        self.reached = False
        self.rect = pygame.Rect(center_pos[0] - size // 2, center_pos[1] - size // 2, size, size)

    def draw(self, screen):
        if not self.reached:  # Only draw if the checkpoint hasn't been reached
            pygame.draw.rect(screen, self.color, self.rect)
            font = pygame.font.Font(None, 36)
            text = font.render(self.label, True, (255, 255, 255))
            text_rect = text.get_rect(center=self.center_pos)
            screen.blit(text, text_rect)

    def check_collision(self, robot_rect):
        return self.rect.colliderect(robot_rect)