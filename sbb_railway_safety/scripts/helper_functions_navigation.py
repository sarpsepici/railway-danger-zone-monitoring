import numpy as np
import heapq

# ------------------------------------
# Dijkstra Path Planning :
# ------------------------------------

def coord_to_index(x, y, x_vals, y_vals):
    i = np.argmin(np.abs(x_vals - x))
    j = np.argmin(np.abs(y_vals - y))
    return (i, j)

def index_to_coord(i, j, x_vals, y_vals):
    return x_vals[i], y_vals[j]

def dijkstra(cost_map, start_idx, goal_idx, x_vals, y_vals):

    n_rows, n_cols = cost_map.shape
    open_set = []
    heapq.heappush(open_set, (0, start_idx))
    came_from = {}
    cost_so_far = {start_idx: 0}
    closed_set = set()

    while open_set:

        current_cost, current = heapq.heappop(open_set)
        if current == goal_idx:
            # Reconstruct the path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start_idx)
            path.reverse()
            return path
        closed_set.add(current)
        
        # 8-connected neighbors.
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                neighbor = (current[0] + di, current[1] + dj)
                if (neighbor[0] < 0 or neighbor[0] >= n_rows or
                    neighbor[1] < 0 or neighbor[1] >= n_cols):
                    continue
                # Skip obstacles.
                if cost_map[neighbor[0], neighbor[1]] == np.inf:
                    continue
                # Skip processed neighbor
                if neighbor in closed_set:
                    continue
                new_cost = cost_so_far[current] + cost_map[neighbor[0], neighbor[1]]
                
                # Neighbor with a lower cost.
                if new_cost < cost_so_far.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    cost_so_far[neighbor] = new_cost
                    heapq.heappush(open_set, (new_cost, neighbor))
                    
    return None  # No path found.


def plan_path(cost_map, goal_coord, x_vals, y_vals):

    start_coord = (0.0, 0.0)
    start_idx = coord_to_index(start_coord[0], start_coord[1], x_vals, y_vals)
    goal_idx = coord_to_index(goal_coord[0], goal_coord[1], x_vals, y_vals)
    path_indices = dijkstra(cost_map, start_idx, goal_idx, x_vals, y_vals)
    if path_indices is None:
        return None
    path_coords = [index_to_coord(i, j, x_vals, y_vals) for (i, j) in path_indices]
    return path_coords, path_indices


# ------------------------------------
# Trajectory Following :
# ------------------------------------


def create_waypoints(path_coords, path_indices, best_orientation_map):

    waypoints = []
    for (x, y), (i, j) in zip(path_coords, path_indices):
        theta = best_orientation_map[i, j]
        waypoints.append((x, y, theta))
    return waypoints

def trajectory_following(waypoints, current_position):
    
    MIN_DIST_THRESHOLD = 0.2            # m
    MIN_ORIENTATION_THRESHOLD = 0.3     # rad
    LINEAR_SPEED = 1.0                  # m/s
    ANGULAR_SPEED = 0.4                 # rad/s

    print('Waypoints :', waypoints)

    if not waypoints:
        return (0.0, 0.0), 0.0
    
    if len(waypoints) > 1 :
        waypoints.pop(0)

    x_r, y_r, theta_r = current_position
    x_goal, y_goal, theta_goal = waypoints[0]
    theta_goal = theta_goal - np.pi/2   # Positioning's convention is zero at right and follower's convention is zero at the top

    # Distance error
    distance = np.sqrt((x_goal - x_r) ** 2 + (y_goal - y_r) ** 2)

    # Movement direction
    theta_target = np.arctan2(y_goal - y_r, x_goal - x_r)

    # Orientation error
    orientation_error = theta_goal - theta_r
    orientation_error = np.arctan2(np.sin(orientation_error), np.cos(orientation_error))  # Normalize to [-pi, pi]
    print('Orientation Error : ', orientation_error)
    

    # Next waypoint check
    if distance < MIN_DIST_THRESHOLD and abs(orientation_error) < MIN_ORIENTATION_THRESHOLD:
        waypoints.pop(0)
        return trajectory_following(waypoints, current_position)

    # Linear velocity
    '''
    Follower's convention is x positiove to the right, y positive to the top. Robot convention is x positive to the top, y positive to the left
    '''
    vx = LINEAR_SPEED * np.sin(theta_target) if distance >= MIN_DIST_THRESHOLD else 0.0
    vy = - LINEAR_SPEED * np.cos(theta_target) if distance >= MIN_DIST_THRESHOLD else 0.0

    # Angular velocity
    angular_velocity = ANGULAR_SPEED if orientation_error > MIN_ORIENTATION_THRESHOLD else (
        -ANGULAR_SPEED if orientation_error < -MIN_ORIENTATION_THRESHOLD else 0.0
    )

    print('Velocity Commands : ', vx, vy, angular_velocity)
    return (vx, vy), angular_velocity
