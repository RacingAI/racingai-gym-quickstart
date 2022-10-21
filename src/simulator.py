import time
import yaml
import gym
import numpy as np
from argparse import Namespace
import concurrent.futures

# import your drivers here
from drivers.follow_the_gap import GapFollower
from drivers.starting_point import SimpleDriver
from drivers.ctq import CTQmk4
from drivers.disparity import DisparityExtender

#Visualisation
from LidarVis import Visualiser, calc_end_pos

def race(drivers=[DisparityExtender()],
         racetrack='TRACK_1',
         vis_driver_idx=0,
         visualise_lidar=True):
    """
    :param racetrack: (TRACK_1, TRACK_2, TRACK_3, OBSTACLES)
    :param drivers: A list of classes with a process_lidar
    function.
    """
    with open('maps/{}.yaml'.format(racetrack)) as map_conf_file:
        map_conf = yaml.load(map_conf_file, Loader=yaml.FullLoader)
    scale = map_conf['resolution'] / map_conf['default_resolution']
    starting_angle = map_conf['starting_angle']
    env = gym.make('f110_gym:f110-v0', map="maps/{}".format(racetrack),
            map_ext=".png", num_agents=len(drivers), disable_env_checker = True)
    # specify starting positions of each agent
    poses = np.array([[-1.25*scale + (i * 0.75*scale), 0., starting_angle] for i in range(len(drivers))])
    if visualise_lidar:
        vis = Visualiser()
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
        if visualise_lidar and vis_driver_idx >= 0 and vis_driver_idx < len(drivers):
            proc_ranges = obs['scans'][vis_driver_idx]
            vis.step(proc_ranges)
        laptime += step_reward
        env.render(mode='human')
        if obs['collisions'].any() == 1.0:
            break
    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)

if __name__ == "__main__":
    race()
