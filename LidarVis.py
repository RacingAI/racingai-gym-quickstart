import numpy as np
import math
import pygame, sys

def calc_end_pos(start_pos, distance, num):
    theta_sign = -1.0
    if num < 540:
        theta_sign = 1.0
        theta = math.radians(((num / 540) * 90))
        delta_x = (distance * math.cos(theta)) * theta_sign
    else: 
        num -= 540
        theta = math.radians(90 - ((num / 540) * 90))
        delta_x = (distance * math.cos(theta)) * theta_sign

    
    delta_y = (distance * math.sin(theta)) * -1

    delta_y = int(delta_y * 10)
    delta_x = int(delta_x * 10)

    end_pos = (start_pos[0] + delta_x, start_pos[1] + delta_y)
    return end_pos

class Visualiser:
    CAR_WIDTH, CAR_HEIGHT = 14, 20
    def __init__(self, swidth=600, sheight=600):
        pygame.init()
        self.clock = pygame.time.Clock()
        WINDOW_SIZE = (swidth, sheight)
        self.dis = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption('Lidar Visualisation')
        self.START_POS = (self.dis.get_width() / 2, (self.dis.get_height() / 2) - (self.CAR_HEIGHT /2)+150)

    def display(self, proc_ranges):
        pygame.init()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            self.dis.fill((0, 0, 0))
            car_width, car_height = 14, 20
            start_pos = (self.dis.get_width() / 2, (self.dis.get_height() / 2) - (car_height /2)+150)

            for num, distance in enumerate(proc_ranges):
                end_pos = calc_end_pos(start_pos, distance, num)
                if num < 135 or num > 945:
                    pygame.draw.line(self.dis, (155, 155, 155), start_pos, end_pos, 1)
                else:
                    pygame.draw.line(self.dis, (255, 255, 255), start_pos, end_pos, 1)
            if len(proc_ranges) > 0:
                #pygame.draw.line(dis, (0, 0, 255), start_pos, calc_end_pos(start_pos, best_speed, 135 + best_point), 5)
                pygame.draw.rect(self.dis, (255, 0, 0), pygame.Rect((self.dis.get_width() / 2) - (car_width / 2), (dis.get_height() / 2) - (car_height / 2) + 150, car_width, car_height))
                pygame.draw.circle(self.dis, (100, 100, 100), start_pos, 30, 2)
                pygame.draw.circle(self.dis, (150, 150, 150), start_pos, 50, 2)
                pygame.display.update()

    def step(self, proc_ranges):
        self.dis.fill((0, 0, 0))
        for num, distance in enumerate(proc_ranges):
            end_pos = calc_end_pos(self.START_POS, distance, num)
            if num < 135 or num > 945:
                pygame.draw.line(self.dis, (155, 155, 155), self.START_POS, end_pos, 1)
            else:
                pygame.draw.line(self.dis, (255, 255, 255), self.START_POS, end_pos, 1)
        if len(proc_ranges) > 0:
            #pygame.draw.line(self.dis, (0, 0, 255), start_pos, calc_end_pos(start_pos, best_speed, 135 + best_point), 5)
            pygame.draw.rect(self.dis, (255, 0, 0), pygame.Rect((self.dis.get_width() / 2) - (self.CAR_WIDTH / 2), \
                (self.dis.get_height() / 2) - (self.CAR_HEIGHT / 2) + 150, self.CAR_WIDTH, self.CAR_HEIGHT))
            pygame.draw.circle(self.dis, (100, 100, 100), self.START_POS, 30, 2)
            pygame.draw.circle(self.dis, (150, 150, 150), self.START_POS, 50, 2)
            pygame.display.update()
