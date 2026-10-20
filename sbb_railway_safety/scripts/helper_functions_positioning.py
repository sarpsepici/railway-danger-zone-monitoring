'''
import numpy as np
from matplotlib.path import Path


# ------------------------------------
# Helper Functions :
# ------------------------------------


def is_visible(x_robot, y_robot, orientation, x_worker, y_worker, fov):
    
    dx = x_worker - x_robot
    dy = y_worker - y_robot
    angle_to_worker = np.arctan2(dy, dx)
    rel_angle = angle_to_worker - orientation
    rel_angle = np.arctan2(np.sin(rel_angle), np.cos(rel_angle))
    return abs(rel_angle) <= fov / 2

def compute_angle_spread(angles):
    
    if len(angles) <= 1:
        return 0.0
    angles_sorted = np.sort(angles)
    diffs = np.diff(angles_sorted)
    wrap_diff = 2 * np.pi - (angles_sorted[-1] - angles_sorted[0])
    return 2 * np.pi - max(np.max(diffs), wrap_diff)


# ------------------------------------
# Sample Points Within the Safe-Zone
# ------------------------------------


def generate_sampling_points(map_size, n_samples):
    
    half = map_size / 2.0
    return np.linspace(-half, half, n_samples)

def points_inside_polygon(xs, ys, polygon):
    
    poly_path = Path(polygon)
    return poly_path.contains_points(np.column_stack((xs, ys)))




def cost_function(x, y, orientation, workers, prev_opt_pos):
    
    w1=10000000.0  # Visibility Cost
    w2=10.0     # Angular Separation Cost
    w3=1.1      # Worker Proximity Cost
    w4=1.0     # Motion Penalty
    fov = np.deg2rad(90)

    visible_angles = []
    visible_dists = []
    
    for wx, wy in workers:
        if is_visible(x, y, orientation, wx, wy, fov):
            visible_angles.append(np.arctan2(wy - y, wx - x))
            visible_dists.append(np.sqrt((wx - x)**2 + (wy - y)**2))
    
    num_visible = len(visible_angles)
    cost_visible = -w1 * (4 - num_visible)
    cost_angle_spread = w2 * compute_angle_spread(visible_angles) if num_visible > 1 else 0.0
    # If no workers are visible, use the average distance from all workers.
    avg_dist_visible = (np.mean(visible_dists) if num_visible > 0 
                        else np.mean([np.sqrt((wx - x)**2 + (wy - y)**2) for wx, wy in workers]))
    cost_dist_workers = w3 * avg_dist_visible
    cost_dist_origin = w4 * np.sqrt((x-prev_opt_pos[0])**2 + (y-prev_opt_pos[1])**2)
    
    return cost_visible + cost_angle_spread + cost_dist_workers + cost_dist_origin


# ------------------------------------
# Main :
# ------------------------------------


def compute_optimal_positioning(safe_zone_polygon, workers, prev_opt_pos):

    map_size=10.0
    n_samples=20
    ori_res_deg=36
    worker_threshold=1.0
    ori_res = np.deg2rad(ori_res_deg)
    
    x_vals = generate_sampling_points(map_size, n_samples)
    y_vals = generate_sampling_points(map_size, n_samples)
    orientations = np.arange(0, 2*np.pi, ori_res)
    
    cost_map = np.zeros((n_samples, n_samples))
    best_orientation_map = np.zeros((n_samples, n_samples))
    min_global_cost = float('inf')
    best_global_pose = (None, None, None)

    # Mask for points inside the safe zone.
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals, indexing='ij')
    X_flat, Y_flat = X_grid.ravel(), Y_grid.ravel()
    inside_mask = points_inside_polygon(X_flat, Y_flat, safe_zone_polygon)

    for i in range(n_samples):
        for j in range(n_samples):
            idx = i * n_samples + j
            if not inside_mask[idx]:
                cost_map[i, j] = np.inf
                best_orientation_map[i, j] = 0.0
                continue

            xv, yv = x_vals[i], y_vals[j]
            
            # Skip cells too close to any worker.
            distances = np.sqrt((np.array([wx for wx, _ in workers]) - xv)**2 + (np.array([wy for _, wy in workers]) - yv)**2)
            if np.any(distances < worker_threshold):
                cost_map[i, j] = 1000000000.0
                best_orientation_map[i, j] = 0.0
                continue

            min_cost_here = float('inf')
            best_ori_here = 0.0

            for ori in orientations:
                c = cost_function(xv, yv, ori, workers, prev_opt_pos)
                if c < min_cost_here:
                    min_cost_here = c
                    best_ori_here = ori
            
            cost_map[i, j] = min_cost_here
            best_orientation_map[i, j] = best_ori_here

            if min_cost_here < min_global_cost:
                min_global_cost = min_cost_here
                best_global_pose = (xv, yv, best_ori_here)

    return cost_map, best_orientation_map, best_global_pose, x_vals, y_vals



'''

import numpy as np
from matplotlib.path import Path


# ------------------------------------
# Helper Functions :
# ------------------------------------


def is_visible(x_robot, y_robot, orientation, x_worker, y_worker, fov):
    dx = x_worker - x_robot
    dy = y_worker - y_robot
    angle_to_worker = np.arctan2(dy, dx)
    rel_angle = angle_to_worker - orientation
    rel_angle = np.arctan2(np.sin(rel_angle), np.cos(rel_angle)) # Normalize to [-pi, pi]
    return abs(rel_angle) <= fov / 2

def compute_angle_spread(angles):
    angles_sorted = np.sort(angles)
    diffs = np.diff(angles_sorted)
    wrap_diff = 2*np.pi - (angles_sorted[-1] - angles_sorted[0])
    return 2*np.pi - max(np.max(diffs), wrap_diff)

def average_angle(angles):
    if len(angles) == 0:
        return None
    return np.arctan2(np.sum(np.sin(angles)), np.sum(np.cos(angles)))

def angle_diff(a, b):
    d = a - b
    return np.arctan2(np.sin(d), np.cos(d))

def cost_function(x, y, orientation, workers, prev_opt_pos):

    w1=1000.0         # Visibility Cost
    w2=5.0              # Angular Separation Cost
    w3=1.5              # Worker Proximity Cost
    w4=0.5              # Motion Penalty From Previous Optimum
    w5=0.4              # Motion Penalty From Origin
    w_center=0.5        # Centering Cost

    fov = np.deg2rad(90)

    visible_angles = []
    visible_dists = []
    
    for wx, wy in workers:
        if is_visible(x, y, orientation, wx, wy, fov):
            visible_angles.append(np.arctan2(wy - y, wx - x))
            visible_dists.append(np.sqrt((wx - x)**2 + (wy - y)**2))
    
    num_visible = len(visible_angles)
    cost_num_visible = w1 * (6 - num_visible)
    cost_angle_spread = w2 * compute_angle_spread(visible_angles) if num_visible > 1 else 5.0
    avg_dist_visible = np.mean(visible_dists) if num_visible > 0 else np.mean([np.sqrt((wx - x)**2 + (wy - y)**2) for wx, wy in workers])
    cost_dist_workers = w3 * avg_dist_visible
    cost_dist_optimum = w4 * np.sqrt((x-prev_opt_pos[0])**2 + (y-prev_opt_pos[1])**2)
    cost_dist_origin = w5 * np.sqrt((x)**2 + (y)**2)
    cost_centering = w_center * abs(angle_diff(average_angle(visible_angles), orientation)) if num_visible > 0 else 0.0

    return cost_num_visible + cost_angle_spread + cost_dist_workers + cost_dist_optimum + cost_dist_origin + cost_centering


def generate_sampling_points(map_size, n_samples):
    half = map_size / 2.0
    return np.linspace(-half, half, n_samples)

def points_inside_polygon(xs, ys, polygon):
    poly_path = Path(polygon)
    return poly_path.contains_points(np.column_stack((xs, ys)))


# ------------------------------------
# Main Function :
# ------------------------------------


# def compute_optimal_positioning(safe_zone_polygon, workers, prev_opt_pos):
def compute_optimal_positioning(workers, prev_opt_pos):


    map_size=10.0
    n_samples=20
    ori_res_deg=36
    ori_res = np.deg2rad(ori_res_deg)
    worker_threshold=1.0
    
    x_vals = generate_sampling_points(map_size, n_samples)
    y_vals = generate_sampling_points(map_size, n_samples) + 2.5
    orientations = np.arange(0, 2*np.pi, ori_res)
    
    cost_map = np.zeros((n_samples, n_samples))
    best_orientation_map = np.zeros((n_samples, n_samples))
    min_global_cost = float('inf')
    best_global_pose = (None, None, None)

    X_grid, Y_grid = np.meshgrid(x_vals, y_vals, indexing='ij')
    X_flat, Y_flat = X_grid.ravel(), Y_grid.ravel()
    # inside_mask = points_inside_polygon(X_flat, Y_flat, safe_zone_polygon)

    for i in range(n_samples):
        for j in range(n_samples):
            idx = i * n_samples + j
            # if not inside_mask[idx]:
            #     cost_map[i, j] = np.inf
            #     best_orientation_map[i, j] = 0.0
            #     continue

            xv, yv = x_vals[i], y_vals[j]

            # Skip cells too close to any worker.
            distances = np.sqrt((np.array([wx for wx, _ in workers]) - xv)**2 + (np.array([wy for _, wy in workers]) - yv)**2)
            if np.any(distances < worker_threshold):
                cost_map[i, j] = 10000.0
                best_orientation_map[i, j] = 0.0
                continue

            min_cost_here = float('inf')
            best_ori_here = 0.0

            for ori in orientations:
                c = cost_function(xv, yv, ori, workers, prev_opt_pos)
                if c < min_cost_here:
                    min_cost_here = c
                    best_ori_here = ori
            
            cost_map[i, j] = min_cost_here
            best_orientation_map[i, j] = best_ori_here

            if min_cost_here < min_global_cost:
                min_global_cost = min_cost_here
                best_global_pose = (xv, yv, best_ori_here)

    return cost_map, best_orientation_map, best_global_pose, x_vals, y_vals

