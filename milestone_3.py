#!/usr/bin/env python

'''
Milestone 3. 
- Purpose:
    Given a reference trajectory that is stored in "trajectory.csv", the code generates a series of control inputs [5 joint velocities, 4 wheel velocities] to Youbot, and stores the simulated positions of the robot in a file called "configurations_milestone3.csv". 
- Logic Flow
    Part A: gnerate Vd, for pure feedforward control. 
            1. Generate desired trajectory. 
            2. Get twist, vd = Adj(x^-1, Xd)*(1/delta_t)*log(Xd^-1Xd,next)
                get desired initial twists
            2.7 update current pose from milestone 1
            
            4 then use Pseudo - inverse to get joint speeds (u theta_dot) = Je^T * V
            5. update milestone 1 so the arm configuration can be taken into account as well.  
            6.  Test with sample input outputs. 
            7. while i in (N-1)
                a. send controls to milestone 1, calculate new configuration. 
                b.store the kth configuration in milestone 1
                c. store kth Xerr - plot over time
 
    Part B: add PI controller
            1. test with identity Kp matrix. o
            2. Add 0 tolerance for pseudo-inverse. 
            3. try in scene 6, physics engine. 
   
    Part C: add joint limits to avoid self collision
            0. create a new function
            1. constrain joints 3 and 4 to always be less than − 0.2 radians (or so).
            2. Use scene 3 to get collision avoidance params
            3. write function that returns a list of joint limits given the robot arm's configuration
- How to use this code: 
        0. Input: initial_pose 13-array of the robot: 
            [phi, x,y ,theta1, theta2, theta3, theta4, theta5, u1, u2, u3, u4]
        
        1. Output: 
            a. csv file called configurations_milestone3.csv, which is ready for Youbot Scene6
            b. plot that shows 6-array error twist, from actual end-effector configruation to reference end-effector configuration.
        
        2. Example:
            initial_pose = [0.1, 0.2, 0.3, 1.0, 2.0, 3.0, 4.0, 5.9, 1.0, 1.2, 1.3, 1.4]
            #[phi, x,y ,theta1, theta2, theta3, theta4, theta5, u1, u2, u3, u4]
            control = Controller(initial_pose)
'''
import csv
import numpy as np
from milestone_2 import get_initial_cube_poses, get_initial_desired_robot_poses
from milestone_1 import NextState
import modern_robotics as mr
import matplotlib.pyplot as plt
import logging

#Test
import time 

DELTA_T = 0.01
JOINT_LIMITS = np.array([[-2.297, 2.297],
                         [-1.671, 1.853],
                         [-1.6, -0.2],
                         [-1.717, -0.2],
                         [-2.89, 2.89] ])
#
#JOINT_LIMITS = np.array([[-2.297, 2.297],
#                         [-1.071, 1.553],
#                         [-1.505, -0.2],
#                         [-1.117, -0.2],
#                         [-2.89, 2.89] ])
#configuration variables of the mobile platform
R = 0.0475
L = 0.47/2.0
W = 0.3/2.0
QDOT_LIM = [30.0,60.0]    #joint speed limits



def to_X(trajectory_output):
    # Input: a 12-array or 13-array that represents the pose in SE(3). 13-array: [9 R matrix entries, 3 P entries, 1 gripper state], 12_array: [9 R matrix entries, 3 P entries]
    # Output: Returns 4 x 4 SE(3) representation of transformation in an np array object 

    T = np.zeros((4,4))
    trajectory_output = np.array(trajectory_output)
    # print(trajectory_output)    
    R = trajectory_output[:9].reshape(3,3)
    T[0:3,0:3] = R

    P = trajectory_output[9:12]
    T[0:3,-1] = P
    T[-1,:] = [0.0,0.0,0.0,1.0]
    return T



def to_X12array(X):
    '''
    Converts a SE(3) matrix into a 12-array, [9 R matrix entries, 3 P entries]
    '''
    X = np.array(X)
    array_12 = X[:-1,:-1].flatten()
    array_12 = np.append( array_12, X[:,-1][:-1].flatten())
    return array_12



def read_entry(str):
    '''
    convert string from file reader
    '''
    try:
        s = float(str)
        return s
    except ValueError:
        pass



def get_poses(pose_8_list):
    '''
    Calculates 1. the end effector pose in fixed frame {s}, 2. chassis pose in end effector frame{e}
    Inputs: 8-array robot configuration [phi, x,y ,theta1, theta2, theta3, theta4, theta5]
    Outputs: Tse, Tbe, both in 4x4 SE(3)
    '''
    phi = pose_8_list[0]
    c = np.cos(phi)
    s = np.sin(phi)
    x = pose_8_list[1]
    y = pose_8_list[2]
    Tsb = np.array([[c, -s, 0.0, x],\
                    [s, c, 0.0, y],\
                    [0.0, 0.0, 1.0, 0.0963],\
                    [0.0, 0.0, 0.0, 1.0]])
    
    Tb0 = np.array([[1,0,0,0.1662],
        [0,1,0,0],
        [0,0,1,0.0026],
        [0,0,0,1]])

    M0e = np.array([[1,0,0,0.033],
        [0,1,0,0],
        [0,0,1,0.6546],
        [0,0,0,1]])

    Blist = np.array([[0.0, 0.0, 1.0, 0.0, 0.033, 0.0],
                      [0.0, -1.0, 0.0, -0.5076, 0.0, 0.0],
                      [0.0, -1.0, 0.0, -0.3526, 0.0, 0.0],
                      [0.0, -1.0, 0.0, -0.2176, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]).T

    thetalist = np.array( pose_8_list[3:] )
    T0e = mr.FKinBody(M0e, Blist, thetalist)
    Tse = (Tsb.dot(Tb0)).dot(T0e)
    
    #Angle error
    #Tse = np.array([[0.0, 0.0, 1.0, 0.0],[0.0, 1.0, 0.0, 0.0],[-1.0, 0.0, 0.0, 0.5],[0.0, 0.0, 0.0, 1.0]])
    Teb = (np.linalg.inv(Tb0.dot(T0e)))
    return Tse, Teb


LOG_FILE_NAME = 'Log_Result.log'
#Debugging log
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
logging.basicConfig(filename= LOG_FILE_NAME, level=logging.DEBUG)


class Controller:

    def __init__(self, IF_JOINT_LIMIT = False, initial_pose = [],mode = 'best'):
        '''
        Key params: 
            1. U_thetadot - 9-array of the commanded velocity [4 wheel velocities U, 5 joint velocities theta] 
            2. current_X - 4x4 SE(3) of the current robot pose 
        '''
 
        global DELTA_T, R,L, W, QDOT_LIM
        self.trajectory_list = self.read_trajectory_file()
        self.pose_12_list = np.array([30.0/180.0*np.pi, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])            #[phi, x,y ,theta1, theta2, theta3, theta4, theta5, u1, u2, u3, u4]
        self.if_joint_limit = IF_JOINT_LIMIT
        self.mode = mode

        #logging msgs to file
        error_log_msg = "Initial configuration of the robot - [phi, x,y ,theta1, theta2, theta3, theta4, theta5, u1, u2, u3, u4]: "
        for i in range(len( self.pose_12_list)):
            error_log_msg+= str(self.pose_12_list[i])
        logging.debug(error_log_msg) 


        # if len(initial_pose) != None:
        #     self.pose_12_list = np.array(initial_pose)
        # print(self.pose_12_list)
        #[phi, x,y ,theta1, theta2, theta3, theta4, theta5]
        self.pose_8_list = self.pose_12_list[:8]
        # print(self.pose_8_list)
        self.current_X, self.Teb = get_poses(self.pose_8_list)
        self.X_sb_init, self.X_se_init = get_initial_desired_robot_poses()      # X_sb_init is the mobile base initial pose
        self.V = None
        
        self.qdot = None
        self.gripper_state = 0                                                  #1 for close, 0 for open
        self.X_err_vec_integral = 0.0
        self.thetadot_u = [None]*9

        self.total_config_list = []                                             #list for all configurations 13-array
        self.total_error_vec_list = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])                                          #list for all errors
        
        self.main_loop()
        self.write_error_config_files()
    
        self.plot_errors()
    
    def plot_errors(self):
        T0 = 0.7
        T1 = 1.5
        T2 = 0.7
        T3 = 0.75
        T4 = T2
        T5 = 2.7
        T6 = T2
        T7 = T3
        T8 = T4

        theta_x_vec = self.total_error_vec_list[:, 0]
        theta_y_vec = self.total_error_vec_list[:, 1]
        theta_z_vec = self.total_error_vec_list[:, 2]
        x_vec = self.total_error_vec_list[:, 3]
        y_vec = self.total_error_vec_list[:, 4]
        z_vec = self.total_error_vec_list[:, 5]
        t_vec = np.linspace(0.0, T1+T2+T3+T4+T5+T6+T7+T8, num=len(theta_x_vec), endpoint=False)
        
        plt.plot(t_vec, theta_x_vec, label = 'theta_x vec')
        plt.plot(t_vec, theta_y_vec, label = 'theta_y vec')
        plt.plot(t_vec, theta_z_vec, label = 'theta_z vec')
        plt.plot(t_vec, x_vec, label = 'x vec')
        plt.plot(t_vec, y_vec, label = 'y vec')
        plt.plot(t_vec, z_vec, label = 'z vec')
        plt.legend()
        if self.mode == 'best':
            plt.title("Error Plot for Best Case")
        elif self.mode == 'new':
            plt.title("Error Plot for New Case")
        elif self.mode == 'overshoot':
            plt.title("Error Plot for Overshoot Case")
        plt.show()


    def write_error_config_files(self):
        error_file_name = 'X_errors.csv'
        config_file_name = 'configurations.csv'
        
        with open( error_file_name, mode = 'w') as error_file:
            write = csv.writer(error_file,delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for row in self.total_error_vec_list:
                write.writerow( row )

        with open( config_file_name, mode = 'w') as config_file:
            write = csv.writer(config_file,delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for row in self.total_config_list:
                write.writerow( row )
                
        logging.debug("Error is written to file")

    def read_trajectory_file(self): #good
        # Read from a trajectory file that is generated by milestone 2
        # Input trajectory: [9 R matrix entries, 3 P entries, 1 gripper state]
        # Output: A list of trajectory points represented in 13-array

        trajectory_list = []
        with open('trajectory.csv') as trajectory_file:
            reader = csv.reader( trajectory_file, delimiter = ',')
            for line in reader:
                line = [read_entry(i) for i in line if read_entry(i) is not None]
                trajectory_list.append(line)
        return trajectory_list
    

    def main_loop(self):
        N = len( self.trajectory_list )
        print(N)
        logging.debug("enters main loop of milestone 3")
        
        for i in range(N-1):
            print(i)
            self.current_X, self.Teb = get_poses(self. pose_8_list)
            # if 6*i > N:
            # if self.trajectory_list[6*i] == [] :
            #     break
            self.V = self.get_V_command(i)

            self.thetadot_u = self.update_velocity_commands() 
            
            self.pose_12_list = NextState( self.pose_12_list, self.thetadot_u, DELTA_T, QDOT_LIM, self.gripper_state)
            self.pose_8_list = self.pose_12_list[:8]

            config = np.append(self.pose_12_list , [self.gripper_state])
            self.total_config_list.append( config ) 

        logging.debug("Loop done. A csv file for the configuration is generated.")

    def get_V_command(self, i):
        '''
        Returns the feedforward + feedback body twist V as the control command for the robot. 
        Input: index of the current trajectory point in the trajectory list
        Output: Body Feedforwad Twist V that takes the robot from the current pose to the next desired pose. 
        '''
         #feedforward
        self.current_X, self.Teb = get_poses(self. pose_8_list) 
        
        Xd = to_X( self.trajectory_list[i] )
        Xd_next = to_X( self.trajectory_list[i+1])

        self.gripper_state = self.trajectory_list[i][-1]
        X_d2dnext = np.linalg.inv(Xd).dot(Xd_next)
        X_err = np.linalg.inv( self.current_X).dot(Xd) #current_x is the end effector pose in the world frame
        Adj = mr.Adjoint(X_err)
        V_feedforward = 1.0/DELTA_T*Adj.dot(( mr.se3ToVec(mr.MatrixLog6(X_d2dnext))))

        #feedforward + feedback
        X_err_vec = mr.se3ToVec(mr.MatrixLog6(X_err))
        self.total_error_vec_list = np.vstack(( self.total_error_vec_list, X_err_vec))                        #add to total error list

        if self.mode == 'overshoot':
            Kp = 1.0*np.array([[30.0, 0.0, 0.0, 0.0, 0.0, 0.0],\
                         [0.0, 20.0, 0.0, 0.0, 0.0, 0.0],\
                         [0.0, 0.0, 30.0, 0.0, 0.0, 0.0],\
                         [0.0, 0.0, 0.0, 200.0, 0.0, 0.0],\
                         [0.0, 0.0, 0.0, 0.0, 200.0, 0.0],\
                         [0.0, 0.0, 0.0, 0.0, 0.0, 30.0]])
        else:
            Kp = 1.0*np.array([[30.0, 0.0, 0.0, 0.0, 0.0, 0.0],\
                          [0.0, 20.0, 0.0, 0.0, 0.0, 0.0],\
                          [0.0, 0.0, 30.0, 0.0, 0.0, 0.0],\
                          [0.0, 0.0, 0.0, 20.0, 0.0, 0.0],\
                          [0.0, 0.0, 0.0, 0.0, 30.0, 0.0],\
                          [0.0, 0.0, 0.0, 0.0, 0.0, 30.0]])

        Ki = 1.0*np.array([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0],\
                      [0.0, 0.2, 0.0, 0.0, 0.0, 0.0],\
                      [0.0, 0.0, 0.2, 0.0, 0.0, 0.0],\
                      [0.0, 0.0, 0.0, 0.1, 0.0, 0.0],\
                      [0.0, 0.0, 0.0, 0.0, 0.1, 0.0],\
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.2]])

        self.X_err_vec_integral += DELTA_T*X_err_vec
        
        V = V_feedforward+Kp.dot( X_err_vec )+ Ki.dot( self.X_err_vec_integral )
        #V = Kp.dot( X_err_vec )

        return V
    
    def update_velocity_commands(self):
        '''
        Input: 
            1. joint angles of the onboard arm
            2. Twist from controller: V command - 6x1 array
            3. Transformation from e to b
        Output:
            [5 joint velocities, 4 wheel velocities]'''
 
        Blist = np.array([[0.0, 0.0, 1.0, 0.0, 0.033, 0.0],
                          [0.0, -1.0, 0.0, -0.5076, 0.0, 0.0],
                          [0.0, -1.0, 0.0, -0.3526, 0.0, 0.0],
                          [0.0, -1.0, 0.0, -0.2176, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]).T
       
        thetalist = np.array( self.pose_8_list[3:] )

        J_arm = mr.JacobianBody(Blist, thetalist)
        
        Adj_Teb = mr.Adjoint( self.Teb )
        
        #test
        R_modified = 2.0*R
        F = R_modified/4.0*np.array([[-1.0/(L+W), 1.0/(L+W), 1.0/(L+W), -1.0/(L+W)],
                            [1.0, 1.0, 1.0, 1.0],
                            [-1.0, 1.0, -1.0, 1.0]])
        F6_temp = np.vstack(( np.zeros((2,4)), F))
        F6 = np.vstack((F6_temp, np.zeros((1,4))))
        J_base = Adj_Teb.dot(F6)
        J= np.hstack((J_base, J_arm))
        Je_pinv = np.linalg.pinv(J,rcond=1e-3)

        u_thetadot = Je_pinv.dot( self.V )
        thetadot_u = np.hstack([u_thetadot[4:], u_thetadot[:4]])

        #test_joint_limits
        if self.if_joint_limit:
            new_thetalist = thetalist + thetadot_u[:5]*DELTA_T
            for i, theta in enumerate(new_thetalist):
                if theta<JOINT_LIMITS[i][0] or theta>JOINT_LIMITS[i][1]:
                    J_arm[:,i] = np.zeros(6)

            J = np.hstack((J_base, J_arm))
            Je_pinv = np.linalg.pinv(J,rcond=1e-3)

            u_thetadot = Je_pinv.dot( self.V )
            thetadot_u = np.hstack([u_thetadot[4:], u_thetadot[:4]])


        return thetadot_u


def final_configuration_main(IF_JOINT_LIMIT = False, initial_pose = [], mode = 'best'):
    control = Controller(IF_JOINT_LIMIT, initial_pose, mode)


    
if __name__ == "__main__":
    final_configuration_main()