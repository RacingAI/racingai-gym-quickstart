import time
import yaml
import gym
import numpy as np
from argparse import Namespace
import concurrent.futures

# import your drivers here
from follow_the_gap import GapFollower
from starting_point import SimpleDriver
from drivers.ctq import CTQmk4

#Visualisation
import pygame,sys
from LidarVis import *

# choose your drivers here (1-4)
drivers = [GapFollower()]

# choose your racetrack here (TRACK_1, TRACK_2, TRACK_3, OBSTACLES)
RACETRACK = 'TRACK_1'

# visualiser
visualise_lidar = True

if __name__ == '__main__':
    with open('maps/{}.yaml'.format(RACETRACK)) as map_conf_file:
        map_conf = yaml.load(map_conf_file, Loader=yaml.FullLoader)
    scale = map_conf['resolution'] / map_conf['default_resolution']
    starting_angle = map_conf['starting_angle']
    env = gym.make('f110_gym:f110-v0', map="maps/{}".format(RACETRACK),
            map_ext=".png", num_agents=len(drivers))
    # specify starting positions of each agent
    poses = np.array([[-1.25*scale + (i * 0.75*scale), 0., starting_angle] for i in range(len(drivers))])
    if visualise_lidar:
        pygame.init()
        clock = pygame.time.Clock()
        WINDOW_SIZE = (600, 600)
        dis = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption('Lidar Visualisation')
        car_width, car_height = 14, 20
        start_pos = (dis.get_width() / 2, (dis.get_height() / 2) - (car_height /2)+150)
    obs, step_reward, done, info = env.reset(poses=poses)
    env.render()

    laptime = 0.0
    start = time.time()

    while not done:
        actions = []
        futures = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for i, driver in enumerate(drivers):
                output = executor.submit(
                    driver.process_lidar,
                    obs['scans'][i])
                futures.append(output)
        for future in futures:
            speed, steer = future.result()
            actions.append([steer, speed])
        actions = np.array(actions)
        obs, step_reward, done, info = env.step(actions)
        if len(drivers) >= 1 and visualise_lidar:
            proc_ranges = obs['scans'][0]
            dis.fill((0, 0, 0))
            for num, distance in enumerate(proc_ranges):
                end_pos = calc_end_pos(start_pos, distance, num)
                if num < 135 or num > 945:
                    pygame.draw.line(dis, (155, 155, 155), start_pos, end_pos, 1)
                else:
                    pygame.draw.line(dis, (255, 255, 255), start_pos, end_pos, 1)
            if len(proc_ranges) > 0:
                #pygame.draw.line(dis, (0, 0, 255), start_pos, calc_end_pos(start_pos, best_speed, 135 + best_point), 5)
                pygame.draw.rect(dis, (255, 0, 0), pygame.Rect((dis.get_width() / 2) - (car_width / 2), (dis.get_height() / 2) - (car_height / 2) + 150, car_width, car_height))
                pygame.draw.circle(dis, (100, 100, 100), start_pos, 30, 2)
                pygame.draw.circle(dis, (150, 150, 150), start_pos, 50, 2)
                pygame.display.update()


        laptime += step_reward
        env.render(mode='human')
        if obs['collisions'].any() == 1.0:
            break
    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)
