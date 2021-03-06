__author__ = "Travis Manderson"
__copyright__ = "Copyright 2018, Travis Manderson"

import pygame, math, sys
import platform
pygame.font.init()
# font = pygame.font.SysFont("Verdana", 12)
font = pygame.font.Font('resources/COMIC.TTF', 12)


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 50, 50)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 50)
BLUE = (50, 50, 255)
GREY = (150, 150, 150)
ORANGE = (200, 100, 50)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
TRANS = (1, 1, 1)

#hints from https://www.dreamincode.net/forums/topic/401541-buttons-and-sliders-in-pygame/
#https://kivy.org/doc/stable/api-kivy.uix.slider.html
class VSlider():
    def __init__(self, screen, name, val, mini, maxi, position, height):
        self.screen = screen
        self.val = val  # start value
        self.mini = mini  # minimum at slider position left
        self.maxi = maxi  # maximum at slider position right
        self.xpos = position[0]  # x-location on screen
        self.ypos = position[1]
        self.height = height
        self.top_border = 44
        self.bottom_border = 30
        self.slide_height = self.height - self.top_border - self.bottom_border

        self.surf = pygame.surface.Surface((60, self.height))
        self.hit = False  # the hit attribute indicates slider movement due to mouse interaction


        if platform.system() == 'Darwin':
            self.ntxt_surf = font.render(name, 0, WHITE)
        else:
            self.ntxt_surf = font.render(name, 1, WHITE)
        self.ntxt_rect = self.ntxt_surf.get_rect(center=(30, self.height-10))
        # Static graphics - slider background #
        # self.surf.fill((100, 100, 100))
        self.surf.fill(TRANS)
        #pygame.draw.rect(self.surf, GREY, [0, 0, 60, 30], 3)
        #pygame.draw.rect(self.surf, WHITE, [0, 0, 100, 30], 0)
        # pygame.draw.rect(self.surf, ORANGE, [10, self.top_border, 10, self.slide_height], 0)
        pygame.draw.rect(self.surf, [200, 200, 200], [22, self.top_border, 6, self.slide_height], 0)

        self.surf.blit(self.ntxt_surf, self.ntxt_rect)  # this surface never changes

        # dynamic graphics - button surface #
        self.button_surf = pygame.surface.Surface((50, 50))
        self.button_surf.fill(TRANS)
        self.button_surf.set_colorkey(TRANS)
        pygame.draw.circle(self.button_surf, [49, 72, 86], (20, 20), 16, 0) #dark blue
        pygame.draw.circle(self.button_surf, [57, 174, 227], (20, 20), 8, 0) #light blue
        pygame.draw.circle(self.button_surf, [57, 174, 227], (20, 20), 16, 2) #light blue

    def draw(self):
        # static
        surf = self.surf.copy()
        # dynamic
        pos = (30, self.top_border + int((1.0 - float(self.val-self.mini) / (self.maxi-self.mini)) * self.slide_height))
        self.button_rect = self.button_surf.get_rect(center=pos)
        surf.blit(self.button_surf, self.button_rect)
        self.button_rect.move_ip(self.xpos, self.ypos)  # move of button box to correct screen position

        val_txt = '{:6.2f}'.format(self.val)
        self.txt_surf = font.render(val_txt, 1, WHITE)
        self.txt_rect = self.txt_surf.get_rect(center=(30, 15))
        surf.blit(self.txt_surf, self.txt_rect)  # this surface never changes

        # screen
        self.screen.blit(surf, (self.xpos, self.ypos))

    def move(self):
        """
    The dynamic part; reacts to movement of the slider button.
    """
        # print(pygame.mouse.get_pos()[0])
        self.val = self.mini + (self.maxi - self.mini) * (1.0 - float(pygame.mouse.get_pos()[1] - self.ypos - self.top_border)/float(self.slide_height))
        #+ (1.0 - ((float(pygame.mouse.get_pos()[1]) - self.ypos - self.top_border) / float(self.slide_height)) * (self.maxi - self.mini)
        # self.val = (self.maxi - self.mini) - (float(pygame.mouse.get_pos()[1]) - self.ypos - self.top_border) / float(self.slide_height) * (self.maxi - self.mini) + self.mini
        if self.val < self.mini:
            self.val = self.mini
        if self.val > self.maxi:
            self.val = self.maxi


if __name__ == '__main__':
    pygame.init()
    X = 900  # screen width
    Y = 600  # screen height
    font = pygame.font.SysFont("Verdana", 12)
    screen = pygame.display.set_mode((X, Y))
    clock = pygame.time.Clock()

    COLORS = [MAGENTA, RED, YELLOW, GREEN, CYAN, BLUE]

    test1 = VSlider(screen, "Test", 100, 1, 1000, (50, 50), 240)
    test2 = VSlider(screen, "Test2", 0.5, 0.0, 1.0, (150, 50), 240)
    test3 = VSlider(screen, "Test3", -2, -10.0, 10.0, (250, 50), 240)
    slides = [test1, test2, test3]

    num = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                for s in slides:
                    if s.button_rect.collidepoint(pos):
                        s.hit = True
            elif event.type == pygame.MOUSEBUTTONUP:
                for s in slides:
                    s.hit = False

        # Move slides
        for s in slides:
            if s.hit:
                s.move()

        # Update screen
        screen.fill(BLACK)
        num += 2

        for s in slides:
            s.draw()

        pygame.display.flip()
        #clock.tick(speed.val)
    clock.tick(50)