import sim as vrep
import sys
from yolo_detection import detection
import numpy as np
import cv2
flag = 0
vrep.simxFinish(-1)

clientID = vrep.simxStart('127.0.0.1',19999,True,True,5000,5)

if clientID!= -1:
    print("Connected to Remote API Server")
else:
    print("Connection failed")
    sys.exit('Could not reconnect')

errorcode,left_motor_handle = vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx_leftMotor',vrep.simx_opmode_oneshot_wait)
errorcode,right_motor_handle = vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx_rightMotor',vrep.simx_opmode_oneshot_wait)
errorcode,cam_handle = vrep.simxGetObjectHandle(clientID,'Cam',vrep.simx_opmode_oneshot_wait)

while True:
    returncode = vrep.simxGetFloatSignal(clientID, 'Sim_State',vrep.simx_opmode_streaming)
    print(returncode)  
    if returncode == (0,0.0):
        flag = 1  
    if flag == 1 and returncode == (1,0.0):
        print('exiting')
        cv2.destroyAllWindows()
        break
    vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,1,vrep.simx_opmode_streaming)
    vrep.simxSetJointTargetVelocity(clientID,right_motor_handle,1,vrep.simx_opmode_streaming)

    errorCode,resolution,image=vrep.simxGetVisionSensorImage(clientID,cam_handle,0,vrep.simx_opmode_streaming)
    if len(image)>0:
        # print(resolution)
        image = np.array(image,dtype=np.dtype('uint8'))
        image = np.reshape(image,(resolution[1],resolution[0],3))
        # image = cv2.imread('./yolo_detection/edge-detection.png')
        # print(image.shape)
        detection.image_detect(image,"")
        cv2.waitKey(1)
    
vrep.simxFinish(clientID)