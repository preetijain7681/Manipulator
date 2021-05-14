#!/usr/bin/env python
'''
This file calculates the configuration of YouBot after time step delta_t, using first order euler integration
    new arm joint angles = (old arm joint angles) + (joint speeds) * delta_t
    new wheel angles = (old wheel angles) + (wheel speeds) * delta_t 
    
Inputs: 
    1. Current configuration
    2. Joint speed
    3. Simulation time step delta_t
    4. Joint speed limits
    
Output:
    1. Configuration of the robot after time step delta_t
'''
import csv
import numpy as np

def Rot_body(phi):
    '''
    Generates a rotational matrix about the z axis by phi for the car. The  chassis configuration should be [phi, x, y] 
    Input: angle of rotation about z axis phi
    Output: 3x3 rotational matrix
    '''
    c, s = np.cos(phi), np.sin(phi)
    R = np.array([[1,0,0],\
                  [0, c, s],\
                  [0, -s, c]])
    return R

def NextState(X_current, qdot, delta_t, qdot_lim, gripper_state, write = None):
    '''
    Description is the same as the file description
    
    Inputs: 
        1. X_curent: 12-array representing the current configuration of the robot (3 variables for the chassis configuration [phi, x,y], 5 variables for the arm configuration, and 4 variables for the wheel angles, the wheel sequence start from the left front wheel of Youbot and go clock wise).
        2. qdot: 9-array of controls indicating the arm joint speeds (5 variables) and the wheel speeds u (4 variables).
        3. delta_t: a timestep delta_t.
        4. qdot_lim: 2-array indicating the maximum angular speed of the arm joints and the wheels, [-vmax, +vmax]. 
        5. an CSV write object for writing the result into the CSV file
    Outputs: 
        1. A 12-array representing the configuration of the robot time delta_t later.
        2. CSV file containing the above 12-array and gripper state 
    '''
    
    #configuration variables
    r = 0.075
    # l = 0.47/2.0
    # w = 0.3/2.0
    d = 0.2375
    q_dot = np.copy( qdot )
    #Speed limiting - arm
    for i in range(5):
        if q_dot[i] > qdot_lim[0]: 
            q_dot[i] = qdot_lim[0]
        if q_dot[i] < -1.0*qdot_lim[0]:
            q_dot[i] = -1.0*qdot_lim[0]

    #Speed limiting - wheels
    for i in range(4):
        if q_dot[5+i] > qdot_lim[1]:
            q_dot[5+i] = qdot_lim[1]
        if q_dot[5+i] < -1.0*qdot_lim[1]:
            q_dot[5+i] = -1.0*qdot_lim[1]
   
    #Joint config updating
    X = np.array( X_current )
    X_new = np.copy(X)
    X_new[3:] += q_dot*delta_t

    #Car config updating: X_new[8:] = H0*(R(phi)*X_new[0:3]) --> x_new[0:3] =  X[0:3]+R(phi)^T * H0^+ * X[8:]
    H0 = 1/r * np.array([[d, -1, 0],\
                         [-d, 0, 1],\
                         [-d, -1, 0],\
                         [d, 0, 1]]) #H in body frame 
    u_increment = (q_dot*delta_t)[5:]
    xb = np.linalg.pinv(H0).dot(u_increment)  #car configuration X in body frame
    phi = X[0]
    RT = Rot_body(phi).T
    X_new[0:3] = X_new[0:3] + RT.dot(xb) # RT is pre-multiplied to change the co-ordinates and if post multiplied we would have changed the frame
    write_output = np.append(X_new,gripper_state)
    write.writerow( write_output)
    return X_new
    

def main():
    '''
    This is a test function for NextState. It tests the robot with sample wheel and joint speed commands. 
    '''
    T = 1.0
    delta_t = 0.01
    gripper_state = 1
    X_current = np.zeros(12)
    qdot = np.zeros(9)
    N = T/delta_t
    u_total_1 = np.array([-10,0,-10,0])
    u_total_2 = np.array([0,30,0,30])
    u_total_3 = np.array([-10,10,10,-10])
    qdot_lim = np.array([10,np.pi])/N


    with open('config_update.csv',mode='w') as csv_file:
        write = csv.writer(csv_file,delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in np.arange(N): 
            wheels_speed1 = u_total_1/T
            qdot[5:] = wheels_speed1
            X_current = NextState(X_current, qdot, delta_t, qdot_lim, gripper_state, write)

        for i in np.arange(N):
            wheels_speed2 = u_total_2/T
            qdot[5:] = wheels_speed2
            X_current = NextState(X_current, qdot, delta_t, qdot_lim, gripper_state, write)
 
 
        for i in np.arange(N):
            wheels_speed3 = u_total_3/T
            qdot[5:] = wheels_speed3
            X_current = NextState(X_current, qdot, delta_t, qdot_lim, gripper_state, write)
 
if __name__ == "__main__":
    main()