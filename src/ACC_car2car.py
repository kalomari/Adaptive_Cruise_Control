# Fuzzy logic approach for testing adptive cruise controller
# The output of the contoller is not connected to the car
# and is only testing the Functionallity of the controller
# this code it to be run on the ego car
# target car has to run vector_field_navigation.py
# Both cars navigates using Vector force field approach

#!/usr/bin/env python

import math
import rospy
import sys
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from nav_msgs.msg import Odometry
from std_msgs.msg import Int16 
from skfuzzy import control as ctrl



class adaptive_cruise_control:
	def __init__(self):
		
		# import lane points and save them in lane_ponits array
		self.lane_ponits = []
		with open("new_map_loop2.txt") as infile:
			for x in infile:
				self.lane_ponits.append(x.split())
		
		ego = rospy.Subscriber("/communication/131/odometry", Odometry, self.call_back)	#subscribe to topic
		target = rospy.Subscriber("/communication/132/odometry", Odometry, self.call_back_tar)	#subscribe to topic
		
	def fuzzy_controller(self):
		x_vel = np.arange(-3.5, 2.1, 0.1)
		x_dis = np.arange(-80, 301, 1)
		x_acc  = np.arange(-0.5, 0.6, 0.1)


		vel_slow = fuzz.trimf(x_vel, [-3.5, -3.5, 0])
		vel_ok = fuzz.trimf(x_vel, [-3.5, 0, 2.0])
		vel_fast = fuzz.trimf(x_vel, [0, 2.0, 2.1])
		dis_close = fuzz.trimf(x_dis, [-80, -80, 0])
		dis_ok = fuzz.trimf(x_dis, [-80, 0, 300])
		dis_far = fuzz.trimf(x_dis, [0, 300, 300])
		decelerate = fuzz.trimf(x_acc, [-1, -0.5, 0])
		const = fuzz.trimf(x_acc, [-0.5, 0, 0.5])
		accelerate = fuzz.trimf(x_acc, [0, 0.5, 0.5])

		dis_level_close = fuzz.interp_membership(x_dis, dis_close, self.distance)
		dis_level_ok= fuzz.interp_membership(x_dis, dis_ok, self.distance)
		dis_level_far = fuzz.interp_membership(x_dis, dis_far, self.distance)

		vel_level_slow = fuzz.interp_membership(x_vel, vel_slow, self.delta_speed)
		vel_level_ok= fuzz.interp_membership(x_vel, vel_ok, self.delta_speed)
		vel_level_fast = fuzz.interp_membership(x_vel, vel_fast, self.delta_speed)

		rule1= np.fmin(dis_level_close, vel_level_slow)
		rule2= np.fmin(dis_level_close, vel_level_ok)
		rule3= np.fmin(dis_level_close, vel_level_fast)
		rule4= np.fmin(dis_level_ok, vel_level_slow)
		rule5= np.fmin(dis_level_ok, vel_level_ok)
		rule6= np.fmin(dis_level_ok, vel_level_fast)
		rule7= np.fmin(dis_level_far, vel_level_slow)
		rule8= np.fmin(dis_level_far, vel_level_ok)
		rule9= np.fmin(dis_level_far, vel_level_fast)

		decelerate_tot= max(rule1, rule2, rule3, rule6, rule9)
		const_tot= max(rule4, rule5, rule8)
		accelerate_tot = rule7

		decelerate_activation= np.fmin(decelerate_tot, decelerate)
		const_activation= np.fmin(const_tot, const)
		accelerate_activation= np.fmin(accelerate_tot, accelerate)
		
		tip0 = np.zeros_like(x_acc)



		####################################################################

		# Aggregate all three output membership functions together
		aggregated = np.fmax(decelerate_activation, np.fmax(const_activation, accelerate_activation))
		#print (aggregated)
		#print("Fuzzy")
		# Calculate defuzzified result
		acc = fuzz.defuzz(x_acc, aggregated, 'centroid')
		
		self.acc_activation = fuzz.interp_membership(x_acc, aggregated, acc)  # for plot		
	
	def calc_Euclidean_dis(self, point1, point2): #[x,y]
		dis_x = point1[0] - float(point2[0])
		dis_y = point1[1] - float(point2[1])
		dis = math.sqrt((dis_x**2) + (dis_y**2))
		return dis

	def calc_dis_on_curve(self, index1, index2):
		return abs(int(index1)-int(index2))

	def get_closest_point(self, input_point): #[x,y]
		res = float("inf")
		index = None
		for point in self.lane_ponits:
			dis = self.calc_Euclidean_dis(input_point,point[1:3])
			if dis < res :
				res = dis
				index = point[0]

		return index.split(".")[2]

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
		self.tar_speed = self.tar_speed_x 
		
		self.distance = self.calc_dis_on_curve(self.tar_closest_point,self.ego_closest_point)
		self.delta_speed= self.tar_speed - self.ego_speed

		print("Distance" ,self.distance)
		print("Delta speed:" , self.delta_speed)
		
		if self.distance < 900:
			self.fuzzy_controller()
			#sprint ("Fuzzy")
			print(self.acc_activation)
		
	def call_back (self,raw_msgs):
		
		ego_position_msg= raw_msgs.pose.pose.position
		ego_position_array = [ego_position_msg.x, ego_position_msg.y, ego_position_msg.z]
		ego_position = [ego_position_array[0], ego_position_array[1]]
		self.ego_closest_point= self.get_closest_point(ego_position)
		#print("ego :" + self.ego_closest_point)

		ego_speed_msg= raw_msgs.twist.twist.linear
		ego_speed_array = [ego_speed_msg.x, ego_speed_msg.y, ego_speed_msg.z]
		self.ego_speed_x = ego_speed_array[0]
		#self.ego_speed_y = ego_speed_array[1]
		self.ego_speed = self.ego_speed_x
		
		
	
def main(args):

	rospy.init_node("ACC") # have to define node for ROS
	
	control = adaptive_cruise_control() # call the class
	try:
		rospy.spin()
	except:
		print("Shutting down")


if __name__ == '__main__':
	main(sys.argv)