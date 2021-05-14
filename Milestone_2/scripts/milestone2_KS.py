#!/usr/bin/env python
'''
Outputs: outputs functions that initialize the robot 
'''
import csv
import modern_robotics as mr
import numpy as np
import math

def TrajectoryGenerator(Xstart, Xend, Tf, N,gripper_state,write):
    """Computes a trajectory as a list of N SE(3) matrices corresponding to
      the straight line motion. 
    :param Xstart: The initial end-effector configuration
    :param Xend: The final end-effector configuration
    :param Tf: Total time of the motion in seconds from rest to rest
    :param N: The number of points N > 1 (Start and stop) in the discrete
              representation of the trajectory
    :param gripper_state: 0- open, 1-close
    :write: a csv_write object
    :return: The discretized trajectory as a list of N matrices in SE(3)
             separated in time by Tf/(N-1). The first in the list is Xstart
             and the Nth is Xend. R is the rotation matrix in X, and p is the linear position part of X. 
             13-array: [9 R variables (from first row to last row), 3 P variables (from x to z), gripper_state ]
    Example Input:
        Xstart = np.array([[1, 0, 0, 1],
                           [0, 1, 0, 0],
                           [0, 0, 1, 1],
                           [0, 0, 0, 1]])
        Xend = np.array([[0, 0, 1, 0.1],
                         [1, 0, 0,   0],
                         [0, 1, 0, 4.1],
                         [0, 0, 0,   1]])
        Tf = 5
        N = 4
        gripper_state = 0
        write = csv.writer(csv_file,delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    Output:
    [1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.1992,0.0,0.7535,0.0]
    """
    N = int(N)
    timegap = Tf / (N - 1.0)
    traj = [[None]] * N
    for i in range(N):
        s = mr.QuinticTimeScaling(Tf, timegap * i)
        Rstart, pstart = mr.TransToRp(Xstart)
        Rend, pend = mr.TransToRp(Xend)
        traj[i] = np.r_[np.c_[np.dot(Rstart, \
        mr.MatrixExp3(mr.MatrixLog3(np.dot(np.array(Rstart).T,Rend)) * s)), \
                   s * np.array(pend) + (1 - s) * np.array(pstart)], \
                   [[0, 0, 0, 1]]]
    #    traj[i]  = np.dot(Xstart, mr.MatrixExp6(mr.MatrixLog6(np.dot(mr.TransInv(Xstart), Xend)) * s))
        output = traj[i][:-1,:-1].flatten()
        output = np.append( output, traj[i][:,-1][:-1].flatten())
        output = np.append(output, gripper_state)
        write.writerow( output)
        

def trajectory_generator_main( X_sc_init = [], X_se_init = []):
    '''
    there are 8 segments in the trajectory. 
    0. open the gripper
    1. initial pose to standoff (a few cm above ground) (3rd or 5th order polynomial) (3s)
    2. standoff down to cube (up and down motion)(1s)
    3.(grasp) (1s)
    4. cube back up to stand off (1s)
    5. first standoff to 2nd standoff(3s)
    6. 2nd stand off desired location (1s)
    7.(open)(1s)
    8. back to the 2nd stand off(1s)
    '''
    T0 = 4
    T1 = 8
    T2 = 4
    T3 = 4.5
    T4 = T2
    T5 = 15
    T6 = T2
    T7 = T3
    T8 = T4
    Tn = T2
    delta_t = 0.01

    #initial and desired end positions of the cube. Cube is 5cm tall.
    if len( X_sc_init) == 0:
        X_sc_init = np.array([[1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.025],
        [0.0, 0.0, 0.0, 1.0]])

    X_sc_goal = np.array([[0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, -1.0],
        [0.0, 0.0, 1.0, 0.025],
        [0.0, 0.0, 0.0, 1.0]])

    #Desired initial position of the mobile base
    X_sb_init = np.array([[1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.1845],
        [0.0, 0.0, 0.0, 1.0]])

    #Fixed offset bw the mobile base and the arm
    X_b0 = np.array([[1.0, 0.0, 0.0, -0.00024],
        [0.0, 1.0, 0.0, 0.142],
        [0.0, 0.0, 1.0, 0.3203],
        [0.0, 0.0, 0.0, 1.0]])

    #Initial end effector configuration
    M_0e = np.array([[1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]])
    # 180 deg Rotation about x 
    Rotation_x_180 = np.array([[1.0, 0.0, 0.0, 0.0],
                           [0.0, math.cos(179 * math.pi/180), math.sin(179 * math.pi/180), 0.0],
                           [0.0,-math.sin(179 * math.pi/180), math.cos(179 * math.pi/180), 0.0],
                           [0.0, 0.0, 0.0, 1.0]])
    # 180 deg Rotation about y 
    Rotation_y_180 = np.array([[math.cos(179 * math.pi/180), 0.0, -math.sin(179 * math.pi/180), 0.0],
                           [0.0, 1.0, 0.0, 0.0],
                           [math.sin(179 * math.pi/180), 0.0, math.cos(179 * math.pi/180), 0.0],
                           [0.0, 0.0, 0.0, 1.0]])
    # 180 deg Rotation about z 
    Rotation_z_180 = np.array([[math.cos(179 * math.pi/180), math.sin(179 * math.pi/180), 0.0, 0.0],
                           [-math.sin(179 * math.pi/180), math.cos(179 * math.pi/180), 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]])
    # 90 deg Rotation about x 
    Rotation_x_90 = np.array([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, -1.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]])                              
    Rotation_x_90_ccw = np.array([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, -1.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]])                              
    # 90 deg Rotation about y 
    Rotation_y_90 = np.array([[0.0, 0.0, -1.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0],
                           [1.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]])                              
    Rotation_y_90_ccw = np.array([[0.0, 0.0, 1.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0],
                           [-1.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]])                              
    # 90 deg Rotation about z 
    Rotation_z_90 =  np.array([[math.cos(90 * math.pi/180), math.sin(90 * math.pi/180), 0.0, 0.0],
                           [-math.sin(90 * math.pi/180), math.cos(90 * math.pi/180), 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]])                             
    Rotation_z_90_ccw =  np.array([[math.cos(90 * math.pi/180), -math.sin(90 * math.pi/180), 0.0, 0.0],
                           [math.sin(90 * math.pi/180), math.cos(90 * math.pi/180), 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]])                             

    #From above, we can get the end effector at each segment X_se
    if len(X_se_init) == 0:
        X_se_init = X_sb_init.dot( X_b0).dot(M_0e)

    #open gripper at the origin
    X_se_0 = np.copy(X_se_init)
    N0 = T0/delta_t + 1
    gripper_state_0 = 0

    # X_se_x = np.copy(X_se_init)
    # X_se_x[1][3] -= 0.2576
    # Nn = Tn/delta_t+1
    # gripper_state_1 = 0

    X_se_x = np.copy(X_sc_init)
    X_se_x = X_se_x.dot( Rotation_z_90 )
    X_se_x[0][3] -= 0.7728
    # X_se_x[1][3] -= 0.2576
    Nn = Tn/delta_t+1
    gripper_state_2 = 0         

    X_se_y = np.copy(X_se_x)
    X_se_y = X_se_y.dot( Rotation_x_90 )
    X_se_y[2][3] += 0.73
    Nn = Tn/delta_t+1
    gripper_state_1 = 0     

    #stand off 1
    X_se_1 = np.copy(X_se_y)
    #rotate about x axis by 180 deg.    
    X_se_1[2][3] -= 0.475
    X_se_1[0][3] += 0.375
    N1 = T1/delta_t+1
    gripper_state_1 = 0
 
    
    #2 Come down to cube initial position
    # X_se_2 = np.copy(X_se_1)
    # X_se_2[2][3] -= 0.33
    # N2 = T2/delta_t+1
    # gripper_state_2 = 0

    #3 Close Gripper
    X_se_3 = np.copy(X_se_1)
    N3 = T3/delta_t+1
    gripper_state_3 = 1

    #4 Coming back up to stand_off 1
    X_se_4 = np.copy( X_se_1)
    X_se_4[2][3] += 0.73
    N4 = T4/delta_t+1
    gripper_state_4 = 1

    #5 Going to stand_off 2
    X_se_5 = np.copy(X_sc_goal)
    #X_se_5 = Rotation_y.dot(X_se_5)
    X_se_5 = X_se_5.dot( Rotation_z_90 )
    X_se_5 = X_se_5.dot( Rotation_x_90 )
    X_se_5[2][3] +=0.73
    X_se_5[1][3] +=0.4
    N5 = T5/delta_t+1
    gripper_state_5 = 1

    #6 going to goal location
    X_se_6 = np.copy(X_se_5)
    X_se_6[2][3] -=0.47
    N6 = T6/delta_t+1
    gripper_state_6 = 1

    #7 Open Gripper
    X_se_7 = np.copy(X_se_6)
    N7 = T7/delta_t+1
    gripper_state_7 = 0

    #8 Come back up to stand off 2
    X_se_8 = np.copy( X_se_5)
    N8 = T8/delta_t+1
    gripper_state_8 = 0
    

    with open('trajectory.csv',mode='w') as csv_file:
        write = csv.writer(csv_file,delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # TrajectoryGenerator(X_se_init, X_se_init, T0, N0, gripper_state_0, write)
        TrajectoryGenerator(X_se_init, X_se_x, T1, N1, gripper_state_1, write)
        TrajectoryGenerator(X_se_x , X_se_y, Tn, Nn, gripper_state_2, write)    
        TrajectoryGenerator(X_se_y , X_se_1, Tn, Nn, gripper_state_2, write)
        # TrajectoryGenerator(X_se_1 , X_se_2, T2, N2, gripper_state_2, write)
        # TrajectoryGenerator(X_se_x , X_se_2, Tn, Nn, gripper_state_2, write)
        TrajectoryGenerator(X_se_1 , X_se_3, T3, N3, gripper_state_3, write)
        TrajectoryGenerator(X_se_3 , X_se_4, T4, N4, gripper_state_4, write)
        TrajectoryGenerator(X_se_4 , X_se_5, T5, N5, gripper_state_5, write)
        TrajectoryGenerator(X_se_5 , X_se_6, T6, N6, gripper_state_6, write)
        TrajectoryGenerator(X_se_6 , X_se_7, T7, N7, gripper_state_7, write)
        TrajectoryGenerator(X_se_7 , X_se_8, T8, N8, gripper_state_8, write)

def get_initial_cube_poses():
    #initial and desired end positions of the cube. Cube is 5cm tall. 
    X_sc_init = np.array([[1,0,0,1],
            [0,1,0,0],
            [0,0,1,0.025],
            [0,0,0,1]])   

    X_sc_goal = np.array([[0,1,0,0],
        [-1,0,0,-1],
        [0,0,1,0.025],
        [0,0,0,1]])
    
    return X_sc_init, X_sc_goal

def get_initial_desired_robot_poses():
    #Desired initial position of the mobile base
    X_sb_init = np.array([[1,0,0,0],
        [0,1,0,0],
        [0,0,1,0.1845],
        [0,0,0,1]])

    #Fixed offset bw the mobile base and the arm
    X_b0 = np.array([[1,0,0,-0.00024],
        [0,1,0,0.142],
        [0,0,1,0.3203],
        [0,0,0,1]])

    #Initial end effector configuration
    M_0e = np.array([[1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1]])

    #From above, we can get the end effector at each segment X_se    
    X_se_init = X_sb_init.dot( X_b0).dot(M_0e)

    return X_sb_init, X_se_init



if __name__ == "__main__":
    trajectory_generator_main()