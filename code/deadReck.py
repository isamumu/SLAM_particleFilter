import math
import random
from matplotlib import pyplot as plt
import scipy
from scipy.signal import butter, lfilter, freqz
import numpy as np
# from pr2_utils import bresenham2D

def gauss_2d(mu, sigmav, sigmaw):
    x = random.gauss(mu, sigmav)
    y = random.gauss(mu, sigmaw)
    return [x, y]

def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# obtains the pose matrix of the robot
def dt_pose_kin(theta, p_t, tau, w_t, v_t):
    R = np.array([[np.cos(theta), -1*np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    r0 = np.concatenate((R[0], p_t[0]))
    r1 = np.concatenate((R[1], p_t[1]))
    r2 = np.array([0, 0, 1])
    R_t = np.array([r0, r1, r2])
    arg = tau * np.array([[0, -1*w_t, v_t], [w_t, 0, 0], [0, 0, 0]])
    R_next = np.matmul(R_t, scipy.linalg.expm(arg))

    return R_next

def predict(x, u, tau):
    state = find_next_state(x, u[0], tau, u[1])

    return state

# yaw angle state after tau seconds (with w and v at time t0)
def find_next_state(state, v, tau, omega):
    theta = state[2]
    return state + tau*np.array([v*np.cos(theta), v*np.sin(theta), omega])

def find_nearest_time(a, b, i, prev_idx):
    idx = -1
    ref = a[i] 

    prev = float('inf')
    for j in range(prev_idx+1, len(b)):
        diff = abs(ref - b[j])

        if diff < prev:
            idx = j
        prev = diff

    # if -1 is returned, a unique, matching index was not found
    return idx

def corr(x_coords, y_coords, grid_m, thresh):
    score = 0
    
    x_obj = int(x_coords[len(x_coords)-1])
    y_obj = int(y_coords[len(x_coords)-1])
    if findGamma(grid_m[y_obj][x_obj]) > thresh:
       score += 1

    return score

def find_v(FR, FL, RR, RL, tau):
    d1 = ((FR+RR)/2)*0.0022
    d2 = ((FL+RL)/2)*0.0022
    vL = d1/tau
    vR = d2/tau

    return (vR+vL) / 2

def find_avg_w(start, end, yaw_rates):
    sum = 0
    for i in range(start+1, end+1):
        sum += yaw_rates[i]

    return sum / (end - start)

def toMapCoord(x, y, height, width):
    return width - x, height - y

def lidarToCartesian(alpha, r):
    return r*np.cos(alpha), r*np.sin(alpha)

def bresenham2D(sx, sy, ex, ey):
  '''
  Bresenham's ray tracing algorithm in 2D.
  Inputs:
	  (sx, sy)	start point of ray
	  (ex, ey)	end point of ray
  '''
  sx = int(np.round(sx))
  sy = int(np.round(sy))
  ex = int(np.round(ex))
  ey = int(np.round(ey))
  dx = abs(ex-sx)
  dy = abs(ey-sy)
  steep = abs(dy)>abs(dx)
  if steep:
    dx,dy = dy,dx # swap 

  if dy == 0:
    q = np.zeros((dx+1,1))
  else:
    q = np.append(0,np.greater_equal(np.diff(np.mod(np.arange( np.floor(dx/2), -dy*dx+np.floor(dx/2)-1,-dy),dx)),0))
  if steep:
    if sy <= ey:
      y = np.arange(sy,ey+1)
    else:
      y = np.arange(sy,ey-1,-1)
    if sx <= ex:
      x = sx + np.cumsum(q)
    else:
      x = sx - np.cumsum(q)
  else:
    if sx <= ex:
      x = np.arange(sx,ex+1)
    else:
      x = np.arange(sx,ex-1,-1)
    if sy <= ey:
      y = sy + np.cumsum(q)
    else:
      y = sy - np.cumsum(q)
  return np.vstack((x,y))

def convertToPixelHeight(y, height):
   return height - y - 1

# map pmf recovered from log odds
def findGamma(lamb):
    val = -1
    if lamb < -300:
        val = -300
    elif lamb > 300:
        val = 300
    else:
        val = lamb
    
    return np.exp(val) / (1+np.exp(val))

def plt_map(occ, height, width):
    map = np.array([np.array([-1 for x in range(width)]) for y in range(height)]) # later will be used to draw the boundaries with occupancy map

    for i in range(height):
        for j in range(width):
            y_conv = int(convertToPixelHeight(i, height))
            if occ[i][j] != 0:
                map[y_conv][j] = 1 # color encoded -1

    plt.imshow(map, cmap="gray")
    plt.show()

def main():
    dataset = 20
    
    # count the rotations of the four wheels at 40Hz
    # wheel diameter = 0.254m
    # 360 ticks/rev -> 0.0022m/tic
    # format: [FR, FL, RR, RL] ... left travel = (FL + RL)/2*0.0022m ... right travel = (FR + RR)/2*0.0022m
    with np.load("../data/Encoders%d.npz"%dataset) as data:
        encoder_counts = data["counts"] # 4 x n encoder counts
        encoder_stamps = data["time_stamps"] # encoder time stamps

    # lidar measurements
    with np.load("../data/Hokuyo%d.npz"%dataset) as data:
        lidar_angle_min = data["angle_min"] # start angle of the scan [rad]
        lidar_angle_max = data["angle_max"] # end angle of the scan [rad]
        lidar_angle_increment = data["angle_increment"] # angular distance between measurements [rad]
        lidar_range_min = data["range_min"] # minimum range value [m]
        lidar_range_max = data["range_max"] # maximum range value [m]
        lidar_ranges = data["ranges"]       # range data [m] (Note: values < range_min or > range_max should be discarded)
        lidar_stamps = data["time_stamps"]  # acquisition times of the lidar scans
    
    # linear and angular velocity data (consider applying LPF w/ BW 10Hz). Only use the yaw rate of the IMU for ang vel
    with np.load("../data/Imu%d.npz"%dataset) as data:
        imu_angular_velocity = data["angular_velocity"] # angular velocity in rad/sec
        imu_linear_acceleration = data["linear_acceleration"] # Accelerations in gs (gravity acceleration scaling)
        imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements
  
    with np.load("../data/Kinect%d.npz"%dataset) as data:
        disp_stamps = data["disparity_time_stamps"] # acquisition times of the disparity images
        rgb_stamps = data["rgb_time_stamps"] # acquisition times of the rgb images

    '''
    Use the first LiDAR scan to initialize the map:
    1. convert the scan to Cartesian coordinates
    2. transform the scan from the lidar frame to the body frame and then to the
    world frame
    3. convert the scan to cells (via bresenham2D or cv2.drawContours) and
    update the map log-odds
    '''

    # =================================== IMU filtering =================================== #
    # obtain the yaw rate from the IMU and filter it at BW 10Hz) 
    order = 6
    fs = 1000.0       # sample rate, Hz
    cutoff = 10  # desired cutoff frequency of the filter, Hz
    filtered_yaw_rate = butter_lowpass_filter(imu_angular_velocity[2], cutoff, fs, order)

    # =================================== dead reconning =================================== #
    state = np.array([float(0), float(0), float(0)])

    FR = encoder_counts[0]
    FL = encoder_counts[1]
    RR = encoder_counts[2]
    RL = encoder_counts[3]

    start_pt = find_nearest_time(imu_stamps, encoder_stamps, 0, -1)

    # adjust the timesteps such that imu and encoder steps start overlapping
    encoder_new_stamps = encoder_stamps[start_pt:]
    imu_new_stamps = imu_stamps[start_pt:]
    new_yaws = filtered_yaw_rate[start_pt:]
    FR_new = FR[start_pt:]
    FL_new = FL[start_pt:]
    RR_new = RR[start_pt:]
    RL_new = RL[start_pt:]

    idx = find_nearest_time(encoder_new_stamps, imu_new_stamps, 0, -1)
    
    t_prev_enc = encoder_new_stamps[0]
    prev_imu_idx = idx
    t_prev_imu = imu_new_stamps[idx]

    # create function which computes the average angular velocity between time steps
    x_start = 300
    y_start = 300

    width = 800
    height = 800

    occ_map = np.zeros((height, width)) # storing the probabilities here
    angle = -2.356194490192345
    

    idx_lid = find_nearest_time(encoder_new_stamps, lidar_stamps, 0, -1) # match first encoder time stamp
    new_lidar_ranges = lidar_ranges[idx_lid:]
    new_lidar_stamps = lidar_stamps[idx_lid:]

    tfm = np.array([[1, 0, 0], [0, 1, -0.33276], [0, 0, 1]]) # no change in rotation, but shift in y
    scale_factor = 0.15

    for i in range(len(new_lidar_ranges)):
        if new_lidar_ranges[i][0] < lidar_range_min or new_lidar_ranges[i][0] > lidar_range_max:
            angle += lidar_angle_increment[0][0]
            continue

        x, y = lidarToCartesian(angle, new_lidar_ranges[i][0])
        coord = np.matmul(tfm, np.array([x, y, 1]))
        xp = coord[0] # scale the meters into box values
        yp = coord[1]

        xb,yb = bresenham2D(0, 0, int(xp / scale_factor), int(yp /scale_factor))
        numpts = len(xb)

        for j in range(numpts): # last point is an object
            y_grid = int(yb[j] + y_start)
            x_grid = int(xb[j] + x_start)
            occ_map[y_grid][x_grid] += np.log(4)

        angle += lidar_angle_increment[0][0]

   
    prev_lid_idx = 0
    for t in range(1, len(encoder_new_stamps)):
        tau_enc = encoder_new_stamps[t] - t_prev_enc
        idx = find_nearest_time(encoder_new_stamps, imu_new_stamps, t, prev_imu_idx)
        tau_imu = imu_new_stamps[idx] - t_prev_imu
        idx_lid = find_nearest_time(encoder_new_stamps, new_lidar_stamps, t, prev_lid_idx) # match first encoder time stamp

        if idx == -1:
            break
        
        avg_w = find_avg_w(prev_imu_idx, idx, new_yaws)
        prev_state = state

        v = find_v(FR_new[t], FL_new[t], RR_new[t], RL_new[t], tau_enc)
        state = find_next_state(prev_state, v, tau_imu, avg_w)
        w_T_b = np.array([[np.cos(state[2]), -1*np.sin(state[2]), state[0]], [np.sin(state[2]), np.cos(state[2]), state[1]], [0, 0, 1]])
        
        angle = -2.356194490192345
        for i in range(len(new_lidar_ranges)):
            if new_lidar_ranges[i][idx_lid] < lidar_range_min or new_lidar_ranges[i][idx_lid] > lidar_range_max:
                angle += lidar_angle_increment[0][0]
                continue

            x, y = lidarToCartesian(angle, new_lidar_ranges[i][idx_lid])
            coord = np.matmul(tfm, np.array([x, y, 1]))
            w_coord = np.matmul(w_T_b, coord)
            xp = w_coord[0] # scale the meters into box values
            yp = w_coord[1]

            xb,yb = bresenham2D(int(state[0]/scale_factor), int(state[1]/scale_factor), int((state[0]+xp)/scale_factor), int((state[1]+yp)/scale_factor))
            numpts = len(xb)

            for j in range(numpts): # last point is an object
                y_grid = int(yb[j] + y_start)
                x_grid = int(xb[j] + x_start)
                if j == numpts - 1:
                    occ_map[y_grid][x_grid] += np.log(4)
                else:
                    occ_map[y_grid][x_grid] -= np.log(4)

            angle += lidar_angle_increment[0][0]

        prev_imu_idx = idx
        prev_lid_idx = idx_lid
        t_prev_enc = encoder_new_stamps[t]
        t_prev_imu = imu_new_stamps[idx]

    # plot the robot trajectory "dead reckoning"
    plt_map(occ_map, height, width)

if __name__ == "__main__":
    main()