import math
import random
from matplotlib import pyplot as plt
import scipy
from scipy.signal import butter, lfilter, freqz
import numpy as np
# from pr2_utils import bresenham2D

def mapCorrelation(im, x_im, y_im, vp, xs, ys):
    nx = im.shape[0]
    ny = im.shape[1]
    xmin = x_im[0]
    xmax = x_im[-1]
    xresolution = (xmax-xmin)/(nx-1)
    ymin = y_im[0]
    ymax = y_im[-1]
    yresolution = (ymax-ymin)/(ny-1)
    nxs = xs.size
    nys = ys.size
    cpr = np.zeros((nxs, nys))
    for jy in range(0,nys):
        y1 = vp[1,:] + ys[jy] # 1 x 1076
        iy = np.int16(np.round((y1-ymin)/yresolution))
        for jx in range(0,nxs):
            x1 = vp[0,:] + xs[jx] # 1 x 1076
            ix = np.int16(np.round((x1-xmin)/xresolution))
            valid = np.logical_and( np.logical_and((iy >=0), (iy < ny)), \
                                            np.logical_and((ix >=0), (ix < nx)))
            cpr[jx,jy] = np.sum(im[ix[valid],iy[valid]])
    return cpr

def gauss_2d(mu, sigmav, sigmaw):
    x = np.random.normal(mu, sigmav)
    y = np.random.normal(mu, sigmaw)
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

def corr(x_coords, y_coords, grid_m):
    score = 1
    
    # scan for empties
    # for i in range(len(x_coords)-1):
    #     x = int(x_coords[i])
    #     y = int(y_coords[i])
    #     if grid_m[y][x] == -1:
    #         score += 1

    x_obj = int(x_coords[len(x_coords)-1])
    y_obj = int(y_coords[len(x_coords)-1])
    # scan the endpoint (supposed object)
    if findGamma(grid_m[y_obj][x_obj]) == 1:
       score += 1
    if (y_obj - 1 >= 0 and findGamma(grid_m[y_obj - 1][x_obj]) == 1):
        score += 1
    if (x_obj - 1 >= 0 and findGamma(grid_m[y_obj][x_obj-1]) == 1):
        score += 1
    if y_obj+1 < len(grid_m) and findGamma(grid_m[y_obj + 1][x_obj]) == 1:
        score += 1
    if x_obj+1 < len(grid_m[0]) and findGamma(grid_m[y_obj][x_obj+1]) == 1:
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
    if lamb < -100:
        val = -100
    elif lamb > 100:
        val = 100
    else:
        val = lamb
    
    return np.exp(val) / (1+np.exp(val))

def plt_map(occ, occ2, height, width):
    map = np.array([np.array([-1 for x in range(width)]) for y in range(height)]) # later will be used to draw the boundaries with occupancy map

    for i in range(height):
        for j in range(width):
            y_conv = int(convertToPixelHeight(i, height))
            prob = findGamma(occ[i][j])
            
            if prob > 0.9:
                map[y_conv][j] = 100 # color encoded 1
            elif occ[i][j] != 0 and prob < 0.10:
                map[y_conv][j] = 200 # color encoded -1

    plt.imshow(map, cmap="magma")
    plt.show()

def plt_map2(MAP):
    map = np.array([np.array([-1 for x in range(MAP['sizex'])]) for y in range(MAP['sizey'])]) # later will be used to draw the boundaries with occupancy map
    height = MAP['sizey']
    width = MAP['sizex']

    for i in range(height):
        for j in range(width):
            y_conv = int(convertToPixelHeight(i, height))
            
            if MAP['map'][j, i] == 1:
                map[y_conv][j] = 100 # color encoded 1
            elif MAP['map'][j,i] == -1:
                map[y_conv][j] = 200 # color encoded -1

    plt.imshow(map, cmap="magma")
    plt.show()

def Neff(alphas):
    sum = 0
    N = len(alphas)
    for i in range(N):
        sum += (alphas[i])*(alphas[i])

    if sum == 0: 
        print(alphas)
    
    return 1/float(sum)

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

    # =================================== initial states =================================== #
    N = 10
    # inital state 
    mews = [None]*N # (x, y, theta)
    alphas = [None]*N

    for i in range(N):
        mews[i] = np.array([float(0), float(0), float(0)])
        alphas[i] = 1/N

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
    x_max = float(-1)
    x_min = float('inf')
    y_max = float(-1)
    y_min = float('inf')
    x_start = 100
    y_start = 100

    theta = []
    xs = []
    ys = []
    for t in range(1, len(encoder_new_stamps)):
        tau_enc = encoder_new_stamps[t] - t_prev_enc
        idx = find_nearest_time(encoder_new_stamps, imu_new_stamps, t, prev_imu_idx)
        tau_imu = imu_new_stamps[idx] - t_prev_imu

        if idx == -1:
            break
        
        avg_w = find_avg_w(prev_imu_idx, idx, new_yaws)
        prev_state = state

        v = find_v(FR_new[t], FL_new[t], RR_new[t], RL_new[t], tau_enc)
        
        state = find_next_state(prev_state, v, tau_imu, avg_w)
        
        x_min = min(x_min, state[0])
        y_min = min(y_min, state[1])
        x_max = max(x_max, state[0])
        y_max = max(y_max, state[1])

        prev_imu_idx = idx
        t_prev_enc = encoder_new_stamps[t]
        t_prev_imu = imu_new_stamps[idx]
        theta.append(state[2])
        xs.append(state[0])
        ys.append(state[1])

    # plot the robot trajectory "dead reckoning"
    # plt.plot(theta)
    # plt.show()

    scale_factor = 0.0375
    scale_factor2 = 0.15

    min_width = (x_max - x_min)/scale_factor
    min_height = (y_max - y_min)/scale_factor
    width = int(min_width)
    height = int(min_height)

    # first lidar scan
    idx_lid = find_nearest_time(encoder_new_stamps, lidar_stamps, 0, -1) # match first encoder time stamp
    new_lidar_ranges = lidar_ranges[idx_lid:]
    new_lidar_stamps = lidar_stamps[idx_lid:]
    
    tfm = np.array([[1, 0, 0], [0, 1, -0.33276], [0, 0, 1]]) # no change in rotation, but shift in y
    x_start = int(width/4)
    y_start = int(height/2)
    # w_var = 0.005
    # v_var = 0.05

    occ_map = np.zeros((height, width)) # storing the probabilities here
    occ_map2 = np.full(shape=(height, width), fill_value=-1) # indicators of there or not there
    
    MAP = {}
    MAP['res']   = scale_factor #meters
    MAP['xmin']  = -1*(x_start)*scale_factor #meters
    MAP['ymin']  = -1*(y_start)*scale_factor
    MAP['xmax']  = (width - x_start)*scale_factor
    MAP['ymax']  =  (height - y_start)*scale_factor
    MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
    MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
    MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8
        
    
    angle = -2.356194490192345
    for i in range(len(new_lidar_ranges)):
        if new_lidar_ranges[i][0] < lidar_range_min or new_lidar_ranges[i][0] > lidar_range_max:
            angle += lidar_angle_increment[0][0]
            continue

        x, y = lidarToCartesian(angle, new_lidar_ranges[i][0])
        coord = np.matmul(tfm, np.array([x, y, 1]))
        xp = int(coord[0]/scale_factor2) # scale the meters into box values
        yp = int(coord[1]/scale_factor2)

        if findGamma(occ_map[y_start+yp][x_start+xp]) > 0.9:
            # occ_map2[y_start+yp][x_start+xp] = 1
            MAP['map'][x_start+xp, y_start+yp] = 1
        
        if findGamma(occ_map[y_start+yp][x_start+xp]) < 0.4:
            MAP['map'][x_start+xp, y_start+yp] = -1

        xb,yb = bresenham2D(0, 0, xp, yp)
        numpts = len(xb)

        for j in range(numpts): # last point is an object
            y_grid = int(yb[j]) + y_start
            x_grid = int(xb[j]) + x_start
            if j == numpts - 1:
                occ_map[y_grid][x_grid] += np.log(4)
            else:
                occ_map[y_grid][x_grid] -= np.log(4)

        angle += lidar_angle_increment[0][0]

    # iterate predict/update over remaining time steps
    idx = find_nearest_time(encoder_new_stamps, imu_new_stamps, 0, -1)
    t_prev_enc = encoder_new_stamps[0]
    prev_imu_idx = idx
    t_prev_imu = imu_new_stamps[idx]
    prev_lid_idx = idx_lid
    # len(encoder_new_stamps)
    for t in range(1, len(encoder_new_stamps), 10):
        tau_enc = encoder_new_stamps[t] - t_prev_enc
        idx_imu = find_nearest_time(encoder_new_stamps, imu_new_stamps, t, prev_imu_idx)
        tau_imu = imu_new_stamps[idx_imu] - t_prev_imu
        idx_lid = find_nearest_time(encoder_new_stamps, new_lidar_stamps, t, prev_lid_idx) # match first encoder time stamp
        if idx_imu == -1:
            break # nearest timestamp was not found

        # ================= predict step =================
        w = find_avg_w(prev_imu_idx, idx_imu, new_yaws)
        vel = find_v(FR_new[t], FL_new[t], RR_new[t], RL_new[t], tau_enc)
        u = np.array([vel, w])

        w_var = abs(w * 0.01)
        v_var = abs(vel * 0.01)
        for i in range(N):
            prev_state = mews[i]
            e = gauss_2d(0, v_var, w_var)
            
            mews[i] = predict(prev_state, u+e, tau_imu)

        # ================ store the lidar coordinates into a matrix (4xNscans) (homogeneous) ================
        angle = -2.356194490192345
        lidar_coords = []
        for i in range(len(new_lidar_ranges)):
            if new_lidar_ranges[i][idx_lid] < lidar_range_min or new_lidar_ranges[i][idx_lid] > lidar_range_max:
                angle += lidar_angle_increment[0][0]
                continue
                
            # find the lidar scan: convert to cartesian
            x, y = lidarToCartesian(angle, new_lidar_ranges[i][idx_lid])

            # convert cartesian to body
            coord = np.matmul(tfm, np.array([x, y, 1]))
            col = [coord[0], coord[1], 1]
            
            lidar_coords.append(col)
            angle += lidar_angle_increment[0][0]
        
        lidar_coords = np.array(lidar_coords).T

        # ================ store the mew transformations into a matrix per hypothesis n (Nx3x3) ================
        particle_poses = []
        for n in range(N):
            orient = mews[n][2]
            w_T_b = np.array([[np.cos(orient), -1*np.sin(orient), mews[n][0]], [np.sin(orient), np.cos(orient), mews[n][1]], [0, 0, 1]])
            particle_poses.append(w_T_b)
        particle_poses = np.array(particle_poses)

        # result should be of shape Nx3xNscan
        
        # ================= update step =================
        # for every mew, test out the lidar correlations

        particle_scans = np.dot(particle_poses, lidar_coords)
        for n in range(N):
            score = 0
            # convert body to world (for lidar coords)
            num_rows, num_cols = particle_scans[n].shape
            for i in range(num_cols):
                w_coord = particle_scans[n][:,i]
                xp = w_coord[0] # scale the meters into box values
                yp = w_coord[1]

                # (x_start, y_start) is the origin for our occupancy grid
                xb, yb = bresenham2D((mews[n][0]/scale_factor2), (mews[n][1]/scale_factor2), int((mews[n][0]+xp)/scale_factor2), int((mews[n][1]+yp)/scale_factor2))
                
                # accumulate the score for each lidar measurement in this time step
                x_im = np.arange(MAP['xmin'],MAP['xmax']+MAP['res'],MAP['res']) #x-positions of each pixel of the map
                y_im = np.arange(MAP['ymin'],MAP['ymax']+MAP['res'],MAP['res']) #y-positions of each pixel of the map

                Y = np.stack((xb,yb))
                x_range = np.arange(MAP['xmin']/100,MAP['xmax']/100+scale_factor,scale_factor)
                y_range = np.arange(MAP['ymin']/100,MAP['xmax']/100+scale_factor,scale_factor)

                corrmtx = mapCorrelation(MAP['map'], x_im, y_im, Y, x_range, y_range)
                ind = np.unravel_index(np.argmax(corrmtx, axis=None), corrmtx.shape) 
                
                if ind[1] != 0 and ind[0] != 0:
                    mews[n][0] += (4 - ind[1])*0.0001 
                    mews[n][1] += (ind[0]-4)*0.0001
             
                score += corrmtx.max()
                angle += lidar_angle_increment[0][0]

            if score != 0:
                alphas[n] = alphas[n]*score

        # normalize the alphas
        den = np.sum(alphas)
        if den == 0:
            print(alphas)
        alphas = alphas / (den)
        
        # find the index with the highest correlation
        index = np.argmax(alphas)
        mew_opt = mews[index]
        ang = -2.356194490192345

        # resample if necessary
        neff = Neff(alphas)
        if neff < int(0.5*N): 
            alphaResample = [1/N]*N
            newMews = [None]*N
            # resample N times
            for i in range(N):
                sampleIdx = random.randint(0, N-1)
                newMews[i] = mews[sampleIdx]

            alphas = alphaResample
            mews = newMews

        # for each lidar measurement, find the projection
        for i in range(len(lidar_coords[0,:])): # ~1081
        
            if new_lidar_ranges[i][idx_lid] < lidar_range_min or new_lidar_ranges[i][idx_lid] > lidar_range_max:
                ang += lidar_angle_increment[0][0]
                continue

            coord = lidar_coords[:,i] # eacg col

            orient = mew_opt[2]
            w_T_b = np.array([[np.cos(orient), -1*np.sin(orient), mew_opt[0]], [np.sin(orient), np.cos(orient), mew_opt[1]], [0, 0, 1]])
            w_coord = np.matmul(w_T_b, np.array([coord[0], coord[1], 1]))
            xp = w_coord[0] # scale the meters into box values
            yp = w_coord[1] 

            # x_start/y_start is meant for grid offset'
            xb,yb = bresenham2D(int(mew_opt[0]/scale_factor2), int(mew_opt[1]/scale_factor2), int((mew_opt[0]+xp)/scale_factor2), int((mew_opt[1]+yp)/scale_factor2))
            numpts = len(xb)

            # inrease log odds
            y_grid = int(yb[-1]) + y_start
            x_grid = int(xb[-1]) + x_start
            occ_map[y_grid][x_grid] += np.log(4)

            # decrease log odds
            y_grid = yb[:-1] + y_start
            x_grid = xb[:-1] + x_start
            x_grid = x_grid.astype(int)
            y_grid = y_grid.astype(int)
            occ_map[y_grid, x_grid] -= np.log(4)

            for j in range(numpts): # last point is an object
                y_grid = int(yb[j]) + y_start
                x_grid = int(xb[j]) + x_start
                if findGamma(occ_map[y_grid][x_grid]) > 0.9:
                    MAP['map'][x_grid, y_grid] = 1
                
                if findGamma(occ_map[y_grid][x_grid]) < 0.4:
                    MAP['map'][x_grid, y_grid] = -1
            
            # MAP['map'] = np.where(MAP['map'] > np.log(10), 1, MAP['map'])
            ang += lidar_angle_increment[0][0]
            

        # update the indices and times
        prev_imu_idx = idx_imu
        prev_lid_idx = idx_lid
        t_prev_enc = encoder_new_stamps[t]
        t_prev_imu = imu_new_stamps[idx_imu]

    # last step of generating the occupancy map
    plt_map2(MAP)
    
if __name__ == "__main__":
    main()