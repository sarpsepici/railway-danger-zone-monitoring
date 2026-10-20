cmake_minimum_required(VERSION 3.0.2)
project(sensor_fusion_package)

find_package(catkin REQUIRED COMPONENTS)

catkin_package()
catkin_python_setup()

catkin_install_python(PROGRAMS 
  scripts/sensor_fusion_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
