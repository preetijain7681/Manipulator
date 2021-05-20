'''
This file is the main file that generates 1. desired trajectory 2. configuration file 3. A result log file 4. an configuration error file (X_error.csv)
How to use this code:
    1. modify at the bottom to run the code you intend to run
    2. put configurations.csv in vrep and visualize
'''
#!/usr/bin/env python
import numpy as np
from milestone2 import trajectory_generator_main
from milestone3 import final_configuration_main

INPUTS = {

    #cube configurations
    "new_case_cube_init_configuration": np.array([[1.0, 0.0, 0.0, 1.5],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.025],
        [0.0, 0.0, 0.0, 1.0]]),
    "regular_cube_init_configuration":np.array([[1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.025],
        [0.0, 0.0, 0.0, 1.0]]),

     #Robot configurations [phi, x,y ,theta1, theta2, theta3, theta4, theta5, u1, u2, u3, u4]
    "chassis_configuration_with_vertical_gripper": np.array([0.0/180.0*np.pi, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "chassis_configuration_with_horizontal_gripper": np.array([30.0/180.0*np.pi, -1.0, -1.0, 0.0, 0.0, 0.0, np.pi/2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),

    #Gripper configuration. Vertical is not given as it will be automatically generated if nothing is specified
    "horizontal_gripper_configuration":  np.array([[0.0, 0.0, 1.0, 0.0],
                                                   [0.0, 1.0, 0.0, 0.0],
                                                   [-1.0, 0.0, 0.0, 0.5],
                                                   [0.0, 0.0, 0.0, 1.0]])   #Controller does not work well with this configuration
}

#new case, with horizonal gripper configuration
def new_case_horizontal_gripper():
    trajectory_generator_main( X_sc_init = INPUTS["new_case_cube_init_configuration"],  X_se_init = INPUTS["horizontal_gripper_configuration"])   #generated trajectory.csv in the current directory, for horizontal gripper configuration
    final_configuration_main(IF_JOINT_LIMIT = True, initial_pose=INPUTS["chassis_configuration_with_horizontal_gripper"], mode = "new")    #generated configurations.csv

def new_case_vertical_gripper():
    trajectory_generator_main( X_sc_init = INPUTS["new_case_cube_init_configuration"])   #generated trajectory.csv in the current directory, for vertical gripper configuration
    final_configuration_main(IF_JOINT_LIMIT = True, initial_pose=INPUTS["chassis_configuration_with_vertical_gripper"], mode = "new")    #generated configurations.csv

def best_case_vertical_gripper():
    trajectory_generator_main( X_sc_init = INPUTS["regular_cube_init_configuration"])   #generated trajectory.csv in the current directory, for vertical gripper configuration
    final_configuration_main(IF_JOINT_LIMIT= True, initial_pose=INPUTS["chassis_configuration_with_vertical_gripper"], mode = "best")

def best_case_vertical_gripper_no_jointlimits():
    trajectory_generator_main( X_sc_init = INPUTS["regular_cube_init_configuration"])   #generated trajectory.csv in the current directory, for vertical gripper configuration
    final_configuration_main(IF_JOINT_LIMIT= False, initial_pose=INPUTS["chassis_configuration_with_vertical_gripper"], mode = "best")

def overshoot_vertical_gripper():
    trajectory_generator_main(X_sc_init = INPUTS["regular_cube_init_configuration"])
    final_configuration_main(IF_JOINT_LIMIT = True, initial_pose=INPUTS["chassis_configuration_with_vertical_gripper"], mode = "overshoot")    #generated configurations.csv


#Call the function of your choice here
best_case_vertical_gripper()