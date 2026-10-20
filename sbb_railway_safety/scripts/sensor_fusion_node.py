#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image, CompressedImage, PointCloud2
from sensor_msgs.msg import NavSatFix
from novatel_oem7_msgs.msg import HEADING2
# from novatel_oem7_msgs.msg import BESTGNSSPOS
from pyproj import Proj
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from ultralytics import YOLO
from geometry_msgs.msg import Twist, TwistStamped
import time

from scripts.helper_functions_detection import (preprocess_image, transform_lidar_points, project_to_image_with_depth, detect_humans, calculate_human_distances)
from scripts.helper_functions_UTM import (calculate_local_worker_positions, transform_safe_zone_to_robot_frame)
from scripts.helper_functions_positioning import compute_optimal_positioning
from scripts.helper_functions_visualization import (visualize_positioning, visualize_detections, visualize_path)
from scripts.helper_functions_navigation import (plan_path, create_waypoints, trajectory_following, coord_to_index)


model = YOLO('yolov8s.pt')

K = np.array([
    [700.1733239760443, 0, 755.8118827756588],
    [0, 700.0133240981131, 554.4259117189192],
    [0, 0, 1]])

transformation_matrix = np.array([
    [0.9999353156286391, 0.006113157552193651, -0.00959134314868424, -0.0031391209194647106],
    [0.010503421799860262, -0.17276975498184988, 0.9849062340619058, 0.06680829559166526],
    [0.0043637929772131255, -0.9849432679440241, -0.1728227885511584, -0.0779811866869374],
    [0.0, 0.0, 0.0, 1.0]])

safe_zone_utm = [(358337.22723645007, 5200322.708462884), 
                 (358324.21524241485, 5200339.173590461), 
                 (358482.8547377299, 5200442.777622888), 
                 (358495.13754519203, 5200427.698816285)]

bridge = CvBridge()

optimal_robot_position = (0.0, 0.0)


# def sensor_callback(camera_msg, lidar_msg, gps_msg, heading_msg):
def sensor_callback(camera_msg, lidar_msg):


    rospy.loginfo("Received synchronized data.")

    global optimal_robot_position


    # Detection

    start_time = time.perf_counter()

    camera_img = preprocess_image(camera_msg)                                                           # 0.1378 seconds
    points_transformed = transform_lidar_points(lidar_msg)                                              # 0.0908 seconds
    image_points, depths = project_to_image_with_depth(points_transformed, K, camera_img.shape)         # 0.0025 seconds
    bounding_boxes = detect_humans(camera_img)                                                          # 0.3854 seconds
    human_distances = calculate_human_distances(image_points, depths, bounding_boxes)                   # 0.0003 seconds
    local_worker_positions = calculate_local_worker_positions(bounding_boxes, human_distances, K)       # 0.0002 seconds
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Detection Elapsed Time: {elapsed_time:.4f} seconds")


    # GPS Coordinates

    # robot_heading = np.radians(heading_msg.heading) + 1.05 * np.pi
    # utm_proj = Proj(proj='utm', zone=32, ellps='WGS84', datum='WGS84')
    # robot_lon, robot_lat = gps_msg.longitude, gps_msg.latitude
    # robot_x, robot_y = utm_proj(robot_lon, robot_lat)
    
    
    # Positioning

    # local_safe_zone = transform_safe_zone_to_robot_frame(robot_x, robot_y, robot_heading, safe_zone_utm)
    # cost_map, best_orientation_map, best_global_pose, x_vals, y_vals = compute_optimal_positioning(local_safe_zone, local_worker_positions, optimal_robot_position)
    cost_map, best_orientation_map, best_global_pose, x_vals, y_vals = compute_optimal_positioning(local_worker_positions, optimal_robot_position)
    if best_global_pose == (None, None, None):
        return
    optimal_robot_position = (best_global_pose[0], best_global_pose[1])


    # Navigation 

    # current_position = [0.0, 0.0, 0.0]
    current_position = [0.2631578947368416, 0.1315789473684208, 0.0]
    goal_coord = (best_global_pose[0], best_global_pose[1])
    path_coords, path_indices = plan_path(cost_map, goal_coord, x_vals, y_vals)
    
    # path_coords[1] = (3.0, 0.0)
    # path_indices[1] = coord_to_index(3.0, 0.0, x_vals, y_vals)
    # best_orientation_map[coord_to_index(3.0, 0.0, x_vals, y_vals)] = 0.0
    
    waypoints = create_waypoints(path_coords, path_indices, best_orientation_map)
    linear_velocity, angular_velocity = trajectory_following(waypoints, current_position)


    # Visualize

    camera_img = visualize_detections(camera_img, bounding_boxes, human_distances)
    positioning_img = visualize_positioning(cost_map, best_orientation_map, best_global_pose, x_vals, y_vals, local_worker_positions)
    path_img = visualize_path(cost_map, x_vals, y_vals, path_coords, goal_coord, local_worker_positions, best_orientation_map)
    image_pub.publish(bridge.cv2_to_imgmsg(camera_img, encoding="bgr8"))
    image_pub_positioning.publish(bridge.cv2_to_imgmsg(positioning_img, encoding="bgr8"))
    image_pub_path.publish(bridge.cv2_to_imgmsg(path_img, encoding="bgr8"))


    # Velocity Commands

    # vel_msg = Twist()
    # vel_msg.header.stamp = rospy.Time.now()
    # vel_msg.linear.x, vel_msg.linear.y, vel_msg.linear.z = linear_velocity[0], linear_velocity[1], 0.0
    # vel_msg.angular.x, vel_msg.angular.y, vel_msg.angular.z = 0.0, 0.0, angular_velocity
    vel_msg = TwistStamped()
    vel_msg.header.frame_id = 'base'
    vel_msg.twist.linear.x, vel_msg.twist.linear.y, vel_msg.twist.linear.z = linear_velocity[0], linear_velocity[1], 0.0
    vel_msg.twist.angular.x, vel_msg.twist.angular.y, vel_msg.twist.angular.z = 0.0, 0.0, angular_velocity
    velocity_publisher.publish(vel_msg)


rospy.init_node('sensor_fusion_node')

camera_sub = Subscriber('/gt_box/alphasense_driver_node/cam1/color/image/compressed', CompressedImage)
lidar_sub = Subscriber('/gt_box/hesai/points', PointCloud2)
# gps_sub = Subscriber('/gt_box/cpt7/gps/fix', NavSatFix)
# heading_sub = Subscriber('/gt_box/cpt7/heading2', HEADING2)

# ats = ApproximateTimeSynchronizer([camera_sub, lidar_sub, gps_sub, heading_sub], queue_size=10, slop=0.1)
ats = ApproximateTimeSynchronizer([camera_sub, lidar_sub], queue_size=100, slop=0.1)
ats.registerCallback(sensor_callback)

image_pub = rospy.Publisher('/processed_camera_image', Image, queue_size=1)
image_pub_positioning = rospy.Publisher('/positioning_plot', Image, queue_size=1)
image_pub_path = rospy.Publisher('/path_plot', Image, queue_size=1)
velocity_publisher = rospy.Publisher('/computed_velocity', TwistStamped, queue_size=1)

rospy.loginfo("Sensor Fusion Node started.")
rospy.spin()






