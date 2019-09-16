# Final Version of the adaptive cruise control using Fuzzy logic approach
# this code it to be run on the ego car
# target car has to run vector_field_navigation.py
# Both cars navigates using Vector force field approach

#!/usr/bin/env python
import numpy as np
import sys
import rospy
import rospkg
import math
import tf

from std_msgs.msg import Int32, UInt8, Float64, Int16, Float64MultiArray, Bool, Float32
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge, CvBridgeError
import sensor_msgs.point_cloud2 as pc2

from nav_msgs.msg import Odometry
from autominy_msgs.msg import NormalizedSteeringCommand, NormalizedSpeedCommand
from collections import deque

import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl


class obst_driver:

    def __init__(self):
        
        self.lane=1
        self.speed_value= 0.19
        self.publishing=False
        self.acc=0
        rospy.set_param('/desired_distance', 80)
        self.test_data=[]
        self.first_time=rospy.Time.now().to_sec()

        # imageimport
        self.bridge = CvBridge()
        
        # load forcefield
        rospack = rospkg.RosPack()
        self.file_path=rospack.get_path('ACC')+'/src/'
        if (self.lane==0):
            self.matrix = np.load(self.file_path+'matrix50cm_lane1.npy')
        else:
            self.matrix = np.load(self.file_path+'matrix50cm_lane2.npy')

        rospy.on_shutdown(self.shutdown)
        self.shutdown_ = False

        # dynamic mapsize
        self.resolution = 1  # cm
        self.map_size_x, self.map_size_y = self.matrix.shape[:2]
        self.map_size_y *= self.resolution
        self.map_size_x *= self.resolution
        self.distance = 1500
        
        self.Target_speeds= deque(maxlen = 100)
        self.ego_speeds= deque(maxlen = 100)
        self.ACC_speeds= deque(maxlen = 40)

        #publisher
        self.obstacle_pub = rospy.Publisher("/obstacle/warning", Int32, queue_size=1)
        self.vel_pub = rospy.Publisher("/actuators/speed_normalized",NormalizedSpeedCommand, queue_size=100)
        self.str_pub = rospy.Publisher("/actuators/steering_normalized",NormalizedSteeringCommand, queue_size=1)
        self.lane_change = rospy.Publisher("/Lane_change", Bool, queue_size=1)
        
        self.delt_speed = rospy.Publisher("/Delta_Speed" , Float64, queue_size=1)
        self.delt_distance = rospy.Publisher("/Delta_Distance" , Float64, queue_size=1)
        self.t_speed = rospy.Publisher ("/Desired_speed", Float32, queue_size=1)
        self.t_distance = rospy.Publisher ("/Desired_Distance", Float32, queue_size=1)
        self.act_distance = rospy.Publisher ("/Act_Distance", Float32, queue_size=1)
        self.act_speed = rospy.Publisher("/Act_Speed" , Float32, queue_size=1)
        self.acceleration = rospy.Publisher("/Acceleration" , Float32, queue_size=1)


        #subscribers
        self.image_sub = rospy.Subscriber("/voxel_grid/output", PointCloud2,self.obstacle_callback, queue_size=1)
        self.obstacle_sub = rospy.Subscriber("/obstacle/warning", Int32, self.callback, queue_size=1)
        self.odom_sub = rospy.Subscriber("/sensors/localization/filtered_map", Odometry, self.PID_call_back)
        self.lane_change_sub = rospy.Subscriber("/Lane_change", Bool, self.Lane_cahange)
        ego = rospy.Subscriber("/sensors/localization/filtered_map", Odometry, self.call_back_ego) #subscribe to topic
        target = rospy.Subscriber("/communication/132/localization", Odometry, self.call_back_tar)  #subscribe to topic
        
        #PID controller intern variables
        self.aq_error = 0.0
        self.last_error = 0.0
        self.last_time = rospy.Time.now()
        #self.str_val = 0.0
        
        #object avoiding
        self.time_acc = rospy.Time.now()


        # import lane points and save them in lane_ponits array
        self.desired_dis = rospy.get_param("/desired_distance")        
        self.lane_ponits = []
        with open("new_map_loop2.txt") as infile:
            for x in infile:
                self.lane_ponits.append(x.split())



    def fuzzy_controller(self):
        x_vel = np.arange(-0.95, 0.95, 0.01)
        x_dis = np.arange(-300, 101, 1)
        x_acc  = np.arange(-0.5, 0.6, 0.1)

        vel_fast = fuzz.trimf(x_vel, [-0.95, -0.95, 0])
        vel_ok = fuzz.trimf(x_vel, [-0.95, 0, 0.95])
        vel_slow = fuzz.trimf(x_vel, [0, 0.95, 0.96])
        dis_far = fuzz.trimf(x_dis, [-300, -300, 0])
        dis_ok = fuzz.trimf(x_dis, [-300, 0, 100])
        dis_close = fuzz.trimf(x_dis, [0, 100, 100])
        decelerate = fuzz.trimf(x_acc, [-1, -0.5, 0])
        const = fuzz.trimf(x_acc, [-0.5, 0, 0.5])
        accelerate = fuzz.trimf(x_acc, [0, 0.5, 0.5])

        dis_level_close = fuzz.interp_membership(x_dis, dis_close, self.delta_dis)
        dis_level_ok= fuzz.interp_membership(x_dis, dis_ok, self.delta_dis)
        dis_level_far = fuzz.interp_membership(x_dis, dis_far, self.delta_dis)

        vel_level_slow = fuzz.interp_membership(x_vel, vel_slow, self.delta_speed)
        vel_level_ok= fuzz.interp_membership(x_vel, vel_ok, self.delta_speed)
        vel_level_fast = fuzz.interp_membership(x_vel, vel_fast, self.delta_speed)
        
        #print('vel level is:', vel_level_slow, vel_level_ok, vel_level_fast)
        #print('dis level is:', dis_level_close, dis_level_ok, dis_level_far)

        rule1= np.fmin(dis_level_close, vel_level_slow)
        rule2= np.fmin(dis_level_close, vel_level_ok)
        rule3= np.fmin(dis_level_close, vel_level_fast)
        rule4= np.fmin(dis_level_ok, vel_level_slow)
        rule5= np.fmin(dis_level_ok, vel_level_ok)
        rule6= np.fmin(dis_level_ok, vel_level_fast)
        rule7= np.fmin(dis_level_far, vel_level_slow)
        rule8= np.fmin(dis_level_far, vel_level_ok)
        rule9= np.fmin(dis_level_far, vel_level_fast)

        #rule5=rule5/0.65
        #rule7=rule7/2.5
        
        #print(rule1, rule2,rule3, rule4, rule5, rule6, rule7, rule8, rule9)

        decelerate_tot= max(rule1, rule2, rule3, rule6, rule9)
        const_tot= max(rule4, rule5, rule8)
        accelerate_tot = rule7

        decelerate_activation= np.fmin(decelerate_tot, decelerate)
        const_activation= np.fmin(const_tot, const)
        accelerate_activation= np.fmin(accelerate_tot, accelerate)
        
        ####################################################################

        # Aggregate all three output membership functions together
        aggregated = np.fmax(decelerate_activation, np.fmax(const_activation, accelerate_activation))
        #print (aggregated)
        
        # Calculate defuzzified result
        self.acc = fuzz.defuzz(x_acc, aggregated, 'centroid')
        
        acc_activation = fuzz.interp_membership(x_acc, aggregated, self.acc)  # for plot

    def calc_Euclidean_dis(self, point1, point2): #[x,y]
        dis_x = point1[0] - float(point2[0])
        dis_y = point1[1] - float(point2[1])
        dis = math.sqrt((dis_x**2) + (dis_y**2))
        return dis

    def calc_dis_on_curve(self, index1, index2):
        diff = (int(index1)-int(index2)) + 25

#        if int(index1) + diff > len(self.lane_ponits):
 #           return (int(index1) + diff) % len(self.lane_ponits)
  #      else:
   #         return diff

        if diff < 0:
            return (diff) + len(self.lane_ponits)
        else:
            return diff

    # find closest point to the path
    def get_closest_point(self, input_point): #[x,y]
        res = float("inf")
        index = None
        for point in self.lane_ponits:
            dis = self.calc_Euclidean_dis(input_point,point[1:3])
            if dis < res :
                res = dis
                index = point[0]

        return index.split(".")[2]

    # Data from target car
    def call_back_tar (self,raw_msgs):
        tar_position_msg= raw_msgs.pose.pose.position
        tar_position_array = [tar_position_msg.x, tar_position_msg.y, tar_position_msg.z]
        tar_position = [tar_position_array[0], tar_position_array[1]]
        
        self.tar_closest_point= self.get_closest_point(tar_position)
        #print("tar :" +self.tar_closest_point)
        
        tar_speed_msg= raw_msgs.twist.twist.linear
        tar_speed_array = [tar_speed_msg.x, tar_speed_msg.y, tar_speed_msg.z]
        self.tar_speed_x = tar_speed_array[0]
        #self.tar_speed_y = tar_speed_array[1]
        #self.tar_speed = self.tar_speed_x
        
        self.Target_speeds.append(self.tar_speed_x)
        self.tar_speed = 0.0
        for speed in self.Target_speeds:
            self.tar_speed += speed / float(len(self.Target_speeds))

        #print("tar speed is:" , self.tar_speed)
        
        #self.desired_dis = rospy.get_param("/desired_distance")

        
        self.distance = self.calc_dis_on_curve(self.tar_closest_point, self.ego_closest_point)
        self.delta_dis = self.desired_dis - self.distance
        self.delta_speed= self.tar_speed - self.ego_speed
        
        meas_1= Float64(self.delta_dis)
        self.delt_distance.publish(meas_1)
        
        meas_2 = Float64(self.delta_speed)
        self.delt_speed.publish(meas_2)

        meas_3 = Float32(self.tar_speed)
        self.t_speed.publish(meas_3)

        meas_4 = Float32(self. desired_dis) #desiered_dis
        self.t_distance.publish(meas_4)

        meas_5 = Float32(self.distance)
        self.act_distance.publish(meas_5)

        meas_6 = Float32(self.ego_speed)
        self.act_speed.publish(meas_6)

        meas_7 = Float32(self.acc)
        self.acceleration.publish(meas_7)

        self.test_data.append(((rospy.Time.now().to_sec()-self.first_time),self.desired_dis, self.distance, self.tar_speed, self.ego_speed, self.acc, self.speed_value))
        

        
    def call_back_ego (self,raw_msgs):
        
        ego_position_msg= raw_msgs.pose.pose.position
        ego_position_array = [ego_position_msg.x, ego_position_msg.y, ego_position_msg.z]
        ego_position = [ego_position_array[0], ego_position_array[1]]
        self.ego_closest_point= self.get_closest_point(ego_position)
        #print("ego :" + self.ego_closest_point)

        ego_speed_msg= raw_msgs.twist.twist.linear
        ego_speed_array = [ego_speed_msg.x, ego_speed_msg.y, ego_speed_msg.z]
        self.ego_speed_x = ego_speed_array[0]
        #self.ego_speed_y = ego_speed_array[1]
        #self.ego_speed = self.ego_speed_x
        #print("ego speed :" , self.ego_speed)

        self.ego_speeds.append(self.ego_speed_x)
        self.ego_speed = 0.0
        for speed in self.ego_speeds:
            self.ego_speed += speed / float(len(self.ego_speeds))
        

    def obstacle_callback(self, data):
        gen = pc2.read_points(data, skip_nans=True, field_names=("x", "y", "z"))
        for P in gen:
            #print (P)
            x = P[0]
            #print (x)
            switch=0.95;
            if x > switch:
                if len (data.data) > 10:
                    self.obstacle_pub.publish(Int32(switch))
                    #print (x, switch)

        #print("switch Lane")

    def Lane_cahange(self, raw_msgs):
        # swap out loaded force field
        self.lane = (self.lane + 1) % 2
        if (self.lane == 0):
            self.matrix = np.load(self.file_path + 'matrix50cm_lane1.npy')
            #print("Lane_1")
        else:
            self.matrix = np.load(self.file_path + 'matrix50cm_lane2.npy')
            #print("Lane_2")

    def PID_call_back(self, raw_msgs):
        #retrieving values from message
        x = raw_msgs.pose.pose.position.x
        y = raw_msgs.pose.pose.position.y
        orientation= raw_msgs.pose.pose.orientation
        orientation_array = [orientation.x, orientation.y, orientation.z, orientation.w]
        
        #change the read-out data from Euler to Rad
        #(roll, pitch, yaw) = (orientation_array)
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
    
        x_car = np.cos(self.yaw)*x_map + np.sin(self.yaw)*y_map
        y_car = -np.sin(self.yaw)*x_map + np.cos(self.yaw)*y_map
        
        #hyperparameters for PID
        Kp= 3.0
        Ki= 0.00001
        Kd= 0.0003

        #so the error is the steepness of the correction anlge
        error= np.arctan2(y_car, x_car)
        #print(x_ind, y_ind, x_car, y_car)
        
        #pseudo integral memory in borders
        self.aq_error = self.aq_error + error
        if self.aq_error > 10:
            self.aq_error=10
        elif self.aq_error < -10:
            self.aq_error = -10
        
        #time between measurements for delta t
        current_time = rospy.Time.now()
        dif_time = (current_time - self.last_time).to_sec()
        #if dif_time == 0.0:
        #   return

        #error manipulation with PID
        PID= Kp * error  + Ki * self.aq_error * dif_time + Kd * (error - self.last_error) / dif_time
        
        self.last_time = current_time
        self.last_error = error

        
        #detect min dist direction
        vel=NormalizedSpeedCommand()

        excep = False
        if self.distance > 200 or self.distance < 0:        #dis= 400
            vel.value = self.speed_value

        else:
            try:
                self.fuzzy_controller()
                print ("ACC is active")
                print ("delta speed is:" , self.delta_speed)
                print("delta_distance is :" , self.delta_dis)
                print("acceleration is :", self.acc)
                #vel.value = (self.ego_speed / 1.95  + self.acc *0 )

                # low level control 
                self.ACC_speed_x = (self.ego_speed / 6.75 + self.acc / 7 + 0.058)
                self.ACC_speeds.append(self.ACC_speed_x)
                self.ACC_speed = 0.0
                for speed in self.ACC_speeds:
                    self.ACC_speed += speed / float(len(self.ACC_speeds))
        

                vel.value = self.ACC_speed
                                
                #print("tar speed is:" , self.tar_speed)
                #print("ego speed :" , self.ego_speed)
                print("new speed is :" ,vel.value)
            except:
                print ("Exception")
                excep = True

        if vel.value > 0.19:
            vel.value = 0.19

        #if vel.value < 0.09:
         #   vel.value = 0.1

        #if x_car > 0:        
            #reduce speed based on steering
            #vel.value = max(self.speed_value, vel.value * ((np.pi/3.0)/(abs(PID)+1.0)))
        
            #valid steering angle -100<=alpha<=80 
        
        if PID > 1:
            PID= 1
        elif PID < -1:
            PID = -1

        str_val = NormalizedSteeringCommand()
        str_val.value = PID
        
        #dont start acceleration after shutdown
        if not self.publishing:
            self.str_pub.publish(str_val)
            if not excep:
                self.vel_pub.publish(vel)

    def callback(self,raw_msg):
        #flag something detected
        if raw_msg.data != 0.95:
            if (rospy.Time.now() - self.time_acc).to_sec() > 3 :
                self.lane_change.publish(Bool(True))
                self.time_acc = rospy.Time.now()

    def shutdown(self):
        #set speed to 0
        print("shutdown!")
        self.publishing=True
        msgs=NormalizedSpeedCommand()
        self.vel_pub.publish(msgs)
        rospy.sleep(1)
        np.savetxt('test_new_31.txt', self.test_data)

def main(args):
    rospy.init_node('obst_driver', anonymous=True)
    driv = obst_driver()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)