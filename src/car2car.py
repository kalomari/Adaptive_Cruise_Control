# Testing car2car communication to calculate the actual distance between the cars on the path

#!/usr/bin/env python

import math
import rospy
import sys
import numpy as np
from nav_msgs.msg import Odometry
from std_msgs.msg import Int16 

class adaptive_cruise_control:
	def __init__(self):
		self.last_time = rospy.Time.now()
		self.yaw= 0. 
		self.Stop= 1
		
		# import lane points as save them in lane_ponits array
		self.lane_ponits = []
		with open("new_map_loop2.txt") as infile:
			for x in infile:
				self.lane_ponits.append(x.split())
		
		ego = rospy.Subscriber("/communication/131/odometry", Odometry, self.call_back)	#subscribe to topic
		target = rospy.Subscriber("/communication/132/odometry", Odometry, self.call_back_tar)	#subscribe to topic
		
		
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
		self.tar_speed_y = tar_speed_array[1]
		self.tar_speed = math.sqrt((self.tar_speed_x**2) + (self.tar_speed_y**2))
		
		distance = self.calc_dis_on_curve(self.tar_closest_point,self.ego_closest_point)
		delta_speed= self.ego_speed - self.tar_speed
		print(distance)
		print(delta_speed)

		
	def call_back (self,raw_msgs):
		
		ego_position_msg= raw_msgs.pose.pose.position
		ego_position_array = [ego_position_msg.x, ego_position_msg.y, ego_position_msg.z]
		#tar_position_in_Red= tf.transformations.euler_from_quaternion(orientation_array) #change the read-out data from Euler to Rad
		ego_position = [ego_position_array[0], ego_position_array[1]]
		self.ego_closest_point= self.get_closest_point(ego_position)
		#print("ego :" + self.ego_closest_point)
		#print(self.ego_position_x,self.ego_position_y)
		
		#calculate the distance and read_out the target speed
		# dis_x = self.ego_position_x - self.tar_speed_x
		# dis_y = self.ego_position_y - self.tar_speed_y
		# dis = math.sqrt((dis_x**2) + (dis_y**2))
		# print (dis)

		ego_speed_msg= raw_msgs.twist.twist.linear
		ego_speed_array = [ego_speed_msg.x, ego_speed_msg.y, ego_speed_msg.z]
		self.ego_speed_x = ego_speed_array[0]
		self.ego_speed_y = ego_speed_array[1]
		self.ego_speed = math.sqrt((self.ego_speed_x**2) + (self.ego_speed_y**2))

		
	
def main(args):

	rospy.init_node("Local_GPS_data") # have to define node for ROS
	
	control = adaptive_cruise_control() # call the class
	try:
		rospy.spin()
	except:
		print("Shutting down")


if __name__ == '__main__':
	main(sys.argv)