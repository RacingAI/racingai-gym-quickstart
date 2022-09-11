import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib as mpl
from drivers.disparity import DisparityExtender
from LidarVis import *

mpl.rcParams['font.family'] = 'serif'
driver = DisparityExtender()

pkl_file = open('C:/Users/aaron/Desktop/Coding/RidersAI/Gym-Quickstart-Master/src/scans.pkl', 'rb')
scans_list = pickle.load(pkl_file)
scan = scans_list[0]
# scans = np.random.randn(1080)

def Slide1(scans):
    length = len(scans)
    rads = ((np.pi) / length) * np.arange(length) - np.pi / 2
    plt.style.use("dark_background")  # Initialize style BEFORE you create fig object
    #plt.rcParams['font.size'] = '10'
    fig = plt.figure()
    fig2 = plt.figure()
    fig3 = plt.figure()
    ax = fig.add_subplot(111, polar=True) # First two numbers specify the subplot size eg (1 row 3 cols). Third number is which subplot you choose
    bx = fig2.add_subplot(111, polar=True)
    cx = fig3.add_subplot(111, polar=True)

    ax.bar(rads, scans, width=0.01, bottom=0.0)
    ax.title.set_text("LiDAR Scan Data (Filled)")
    ax.title.set_fontsize(16)
    ax.grid = True
    ax.set_theta_zero_location('N')
    ax.set_theta_direction('anticlockwise')
    ax.set_xticks(np.array([-180, -135, -90, -45, 0, 45, 90, 135])/180*np.pi) # negative angle indicate turn right and positive angle indicate turn left
    ax.set_thetalim(-np.pi, np.pi)

    bx.scatter(rads, scans, s=5, c='#fa8174')
    bx.title.set_text("LiDAR Scan Data (Point Cloud)")
    bx.title.set_fontsize(16)
    bx.grid = True
    bx.set_theta_zero_location('N')
    bx.set_theta_direction('anticlockwise')
    bx.set_xticks(np.array([-180, -135, -90, -45, 0, 45, 90, 135])/180*np.pi)
    bx.set_thetalim(-np.pi, np.pi)

    cx.plot(rads, scans, lw=5, c='#feffb3')
    cx.title.set_text("LiDAR Scan Data (Boundary)")
    cx.title.set_fontsize(16)
    cx.grid = True
    cx.set_theta_zero_location('N')
    cx.set_theta_direction('anticlockwise')
    cx.set_xticks(np.array([-180, -135, -90, -45, 0, 45, 90, 135])/180*np.pi)
    cx.set_thetalim(-np.pi, np.pi)
    plt.show()

def Preprocess_plot(scans, processed_scans):
    length_scan = len(scans)
    rads_preprocessed = (np.pi / length_scan) * np.arange(135, 945) - np.pi / 2
    plt.style.use("dark_background")  # Initialize style BEFORE you create fig object
    # plt.rcParams['font.size'] = '10'
    fig = plt.figure()
    bx = fig.add_subplot(111, polar=True)

    bx.bar(rads_preprocessed, processed_scans, width=0.1, bottom=0.0, color='#fa8174')
    bx.title.set_text("LiDAR Scan Data (Pre Processed)")
    bx.title.set_fontsize(16)
    bx.grid = True
    bx.set_theta_zero_location('N')
    bx.set_theta_direction('anticlockwise')
    bx.set_xticks(np.array([-180, -135, -90, -45, 0, 45, 90, 135])/180*np.pi)
    bx.set_thetalim(-np.pi, np.pi)
    plt.show()


def disparities(ranges):
    plt.style.use("dark_background")
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    differences = driver.get_differences(ranges)
    disparities_index = driver.get_disparities(differences, driver.DIFFERENCE_THRESHOLD)
    rads = np.array(disparities_index)*(np.pi/1080) - np.pi / 2
    lengths = [ranges[x] for x in disparities_index]

    rads_preprocessed = (np.pi / 1080) * np.arange(135, 945) - np.pi / 2

    ax.bar(rads_preprocessed, ranges, width=0.01,  bottom=0.0, zorder=0)
    ax.bar(rads, lengths, width=0.06,  bottom=0.0, color='#fdb462', zorder=1)
    ax.title.set_text("Disparities Found")
    ax.title.set_fontsize(16)
    ax.grid = True
    ax.set_theta_zero_location('N')
    ax.set_theta_direction('anticlockwise')
    ax.set_xticks(np.array([-180, -135, -90, -45, 0, 45, 90, 135])/180*np.pi) # negative angle indicate turn right and positive angle indicate turn left
    ax.set_thetalim(-np.pi, np.pi)
    plt.show()



proc_ranges = driver.preprocess_lidar(scan)
disparities(proc_ranges)
pkl_file.close()