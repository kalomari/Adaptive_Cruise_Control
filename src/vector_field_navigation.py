# Navigating using Virtual Force Field Approach

#!/usr/bin/env python2

import rospy
import math
import tf
import rospkg
import sys

import numpy as np

from nav_msgs.msg import Odometry
from std_msgs.msg import UInt8, Float64, Int16, Float64MultiArray, Bool, Float32
from autominy_msgs.msg import NormalizedSteeringCommand, NormalizedSpeedCommand


class field_Controller:
    def __init__(self):
        
        self.lane=1
        self.speed_value= 0.14
        self.old_speed=0
        rospy.on_shutdown(self.shutdown)
        self.publishing=False
        rospack = rospkg.RosPack()
        #self.position=[]
        
        
        #load forcefield
        self.file_path=rospack.get_path('fub_navigation')+'/src/'
        if (self.lane==0):
            self.matrix = np.load(self.file_path+'matrix50cm_lane1.npy')
        else:
            self.matrix = np.load(self.file_path+'matrix50cm_lane2.npy')
        
        #dynamic mapsize
        self.resolution = 1 # cm
        self.yaw= 0.0
        self.map_size_x, self.map_size_y=self.matrix.shape[:2]
        self.map_size_y *= self.resolution
        self.map_size_x *= self.resolution
        
        
        # PID controller intern variables
        self.aq_error= 0.0
        self.last_error= 0.0
        self.last_time = rospy.Time.now()
        
        #Publisher & Subscribers
        self.vel_pub = rospy.Publisher("/actuators/speed_normalized",NormalizedSpeedCommand, queue_size=100)
        self.str_pub = rospy.Publisher("/actuators/steering_normalized",NormalizedSteeringCommand, queue_size=1)

        self.dummya_1 = rospy.Subscriber("/sensors/localization/filtered_map", Odometry, self.call_back, queue_size=1)


    def call_back(self,raw_msgs):
        
        #retrieving values from message
        x = raw_msgs.pose.pose.position.x
        y = raw_msgs.pose.pose.position.y
        #self.position.append((x,y))

        orientation= raw_msgs.pose.pose.orientation
        orientation_array = [orientation.x, orientation.y, orientation.z, orientation.w]
        
        
        #change the read-out data from Euler to Rad
        orientation_in_Rad= tf.transformations.euler_from_quaternion(orientation_array)
        
        self.yaw =orientation_in_Rad[2]
        x_ind = np.int(x*(100.0/self.resolution))
        y_ind = np.int(y*(100.0/self.resolution))
        
        #if abroad, car virtually set on the map
        if x_ind < 0:
            x_ind = 0
        if x_ind > self.map_size_x/self.resolution -1:
            x_ind = self.map_size_x/self.resolution -1
        
        if y_ind < 0:
            y_ind = 0
        if y_ind > self.map_size_y/self.resolution -1:
            y_ind = self.map_size_y/self.resolution -1
        
        x_map, y_map = self.matrix[x_ind, y_ind]
        #print(x_ind,y_ind)
        #if we imagine the odometry as polar coordinates 
        #we have some radius r and a angle phi
        #so now we combine the steering angle given from the force field alpha and phi
        #with x = r * (cos(alpha-phi)) and y = (sin(alpha-phi))
        #maybe it was phi - alpha in both i wrote it on a paper
        #formula was given on the assignment sheet
    
        x_car = np.cos(self.yaw)*x_map + np.sin(self.yaw)*y_map
        y_car = -np.sin(self.yaw)*x_map + np.cos(self.yaw)*y_map
        
        #hyperparameters for PID
        Kp= 4.2
        Ki= 0.001
        Kd= 0.0
        
        #so the error is the steepness of the correction anlge
        error= np.arctan2(y_car, x_car)
        #print(x_ind, y_ind, x_car, y_car)
        
        
        #pseudo integral memory in borders
        self.aq_error = self.aq_error + error
        if self.aq_error > 10:
            self.aq_error=10.0
        elif self.aq_error < -10:
            self.aq_error = -10.0
        
        #time between measurements for delta t
        current_time = rospy.Time.now()
        dif_time = (current_time - self.last_time).to_sec()

        if dif_time == 0.0:
            return
        
        #error manipulation with PI
        PID= Kp * error  + Ki * self.aq_error * dif_time + Kd * (error - self.last_error) / dif_time
        #print(PID)
        
        self.last_time = current_time
        self.last_error = error
        
        
        #detect min dist direction
        vel=NormalizedSpeedCommand()
        if (x_car<0):
            vel.value = -self.speed_value
        else:
            vel.value = self.speed_value
        #if x_car > 0:
        
            #reduce speed based on steering
            #vel.value = max(self.speed_value, vel.value * ((np.pi/3.0)/(abs(PID)+1.0)))
        
            #valid steering angle -100<=alpha<=80 
        
        if PID > 1:
            PID= 1.0
        elif PID < -1:
            PID = -1.0
        
        #offset 100 == straight ahead
        #PID = 180 - (90 + PID)
        
        #print("yaw: {0}, vff_angle: {1}".format(np.rad2deg(self.yaw), np.rad2deg(np.arctan(y_map / x_map))))
        #print(np.rad2deg(error),UInt8(PID))
        str_val = NormalizedSteeringCommand()
        str_val.value = PID
        
        #dont start acceleration after shutdown
        if not self.publishing:
            self.str_pub.publish(str_val)
            self.vel_pub.publish(vel)

    def shutdown(self):
        #set speed to 0
        print("shutdown!")
        self.publishing=True
        msgs=NormalizedSpeedCommand()
        self.vel_pub.publish(msgs)
        rospy.sleep(1)
        #np.savetxt('Lane_PID.txt', self.position)

def main(args):
    rospy.init_node("field_Controller") # have to define node for ROS
    control = field_Controller() # call the class
    try:
        rospy.spin()
    except:
        print("Shutting down")

if __name__ == '__main__':
    main(sys.argv)