# Ghiotto Andrea   2118418

import matplotlib.pyplot as plt
import math
import numpy as np

def start_plot(train_data, sonar_radius=50, figsize = (15,10)):
    plt.figure(figsize=figsize)

    ax = plt.gca()
    ax.set_xlim([0, train_data.poses[:, 0].max()+sonar_radius])
    ax.set_ylim([0, train_data.poses[:, 1].max()+sonar_radius])

def plot_synth_poses_train(td, color="blue"):
    plt.scatter(td.poses[:, 0], td.poses[:, 1], c=color, marker='o', linestyle='None', s=1)

def plot_synth_poses_val(vd, color="red"):
    plt.scatter(vd.poses[:, 0], vd.poses[:, 1], c=color, marker='o', linestyle='None', s = 1)

def scatter_point(x, y, color, label=None):
    if label is None:
        plt.scatter(x, y, c=color, s = 20.51)
    else:
        plt.scatter(x, y, c=color, s = 20.51, label=label)
        
def scatter_orientation(x, y, Y_r, color, rad=50):
    dy, dx = rad*math.cos(Y_r), rad*math.sin(Y_r)
    plt.arrow(x, y, dx, dy, color=color)
       
def scatter_real_orientation(x, y, Y, color, rad=50):
    dx, dy = rad*math.cos(Y), rad*math.sin(Y)
    plt.arrow(x, y, dy, dx, color=color)

def plot_real_poses(rd, color="pink"):
    plt.scatter(rd.poses[:, 0], rd.poses[:, 1], c=color, marker='o', linestyle='None', s =1)
    for i in range(0, rd.poses.shape[0], 5):
        q_x, q_y, q_Y_deg = rd.poses[i, :]
        q_Y = (q_Y_deg+90)*np.pi/180
        q_Y %= 2*np.pi
        scatter_real_orientation(q_x, q_y, q_Y, "mediumturquoise")

def plot_train_data(data):
    plt.scatter(data.poses[:, 0], data.poses[:, 1], c="pink", marker='o', linestyle='None', s =1)
    for i in range(0, data.poses.shape[0], 20):
        q_x, q_y, q_Y_deg = data.poses[i, :]
        q_Y = (q_Y_deg+90)*np.pi/180
        q_Y %= 2*np.pi
        scatter_real_orientation(q_x, q_y, q_Y, "mediumturquoise", rad=10)

def plot_data(data):
    plt.scatter(data.poses[:, 0], data.poses[:, 1], c="pink", marker='o', linestyle='None', s =1)
    for i in range(0, data.poses.shape[0], 5):
        q_x, q_y, q_Y_deg = data.poses[i, :]
        q_Y = (q_Y_deg+90)*np.pi/180
        q_Y %= 2*np.pi
        scatter_real_orientation(q_x, q_y, q_Y, "mediumturquoise", rad=10)