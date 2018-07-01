cmake_minimum_required(VERSION 1.8.3)
project(ssd_keras_ros)

find_package(catkin REQUIRED COMPONENTS
  mdr_perception_libs
  roslint
  rospy
)

catkin_python_setup()

catkin_package()

#############
## Install ##
#############

#############
## Testing ##
#############
roslint_python()
