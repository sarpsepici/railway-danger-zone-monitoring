import numpy as np


def calculate_local_worker_positions(bounding_boxes, depths, K):

    local_positions = []
    K_inv = np.linalg.inv(K)

    for bbox, depth in zip(bounding_boxes, depths):
        if depth == None:
            continue
        x_min, y_min, x_max, y_max = bbox
        u = (x_min + x_max) / 2
        v = (y_min + y_max) / 2
        pixel_coords = np.array([u, v, 1])
        camera_coords = depth * np.dot(K_inv, pixel_coords)
        x, _, z = camera_coords
        local_positions.append((x, z))

    return np.array(local_positions)


def transform_safe_zone_to_robot_frame(robot_utm_x, robot_utm_y, robot_heading_rad, safe_zone_corners):
    
    rotation_matrix = np.array([
        [np.cos(robot_heading_rad), -np.sin(robot_heading_rad)],
        [np.sin(robot_heading_rad),  np.cos(robot_heading_rad)]])

    transformed_corners = []
    for (px, py) in safe_zone_corners:
        dx = px - robot_utm_x
        dy = py - robot_utm_y
        local_xy = rotation_matrix @ np.array([dx, dy])
        transformed_corners.append(local_xy)

    return np.array(transformed_corners)








