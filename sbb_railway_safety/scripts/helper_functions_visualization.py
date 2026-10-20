import numpy as np
from matplotlib.path import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge 
from matplotlib.patches import Polygon
from matplotlib.ticker import MultipleLocator
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2
from scripts.helper_functions_navigation import coord_to_index



def visualize_detections(camera_img, bounding_boxes, human_distances):

    for box, distance in zip(bounding_boxes, human_distances):
        x1, y1, x2, y2 = map(int, box)
        label = f"{distance:.2f}m" if distance else "N/A"
        cv2.rectangle(camera_img, (x1, y1), (x2, y2), (220, 100, 100), 2)
        cv2.putText(camera_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 100, 100), 2)
    return camera_img



def visualize_positioning(cost_map, best_ori_map, best_pose, x_vals, y_vals, workers):
    
    best_x, best_y, best_ori = best_pose
    best_ori_deg = np.rad2deg(best_ori)
        
    fig, ax = plt.subplots(figsize=(8, 7))
    X_mesh, Y_mesh = np.meshgrid(x_vals, y_vals, indexing='ij')
    sc = ax.scatter(
        X_mesh.flatten(), Y_mesh.flatten(), 
        c=cost_map.flatten(),
        cmap='viridis', s=2, marker='s'
    )
    if len(workers)>0:
        ax.scatter(workers[:, 0], workers[:, 1], color='red', marker='s', s=50, label='Workers')
    
    ax.plot(best_x, best_y, 'ko', markersize=10, label='Best Position')
    arrow_length = 1.0
    ax.arrow(
        best_x, best_y,
        arrow_length * np.cos(best_ori),
        arrow_length * np.sin(best_ori),
        head_width=0.3, head_length=0.4, fc='k', ec='k'
    )
    fov_radius = 5.0
    theta1 = best_ori_deg - 90 / 2
    theta2 = best_ori_deg + 90 / 2
    wedge = Wedge(
        center=(best_x, best_y), r=fov_radius,
        theta1=theta1, theta2=theta2,
        facecolor='yellow', alpha=0.3
    )
    ax.add_patch(wedge)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend(loc='upper right')
 
    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.array(canvas.renderer.buffer_rgba())

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    
    return img_bgr



def visualize_path(cost_map, x_vals, y_vals, path_coords, goal_coord, workers, best_orientation_map):
    
    # start_coord = (0.0, 0.0)
    start_coord = (0.2631578947368416, 0.1315789473684208)
    path_coords = np.array(path_coords)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim([x_vals[0], x_vals[-1]])
    ax.set_ylim([y_vals[0], y_vals[-1]])
    sc = ax.imshow(cost_map.T, origin='lower', extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]], cmap='tab20b', alpha=0.7)
    # plt.colorbar(sc, label='Cost')
    ax.plot(path_coords[:, 0], path_coords[:, 1], 'r.-', linewidth=2, markersize=5, label='Path')
    ax.plot(start_coord[0], start_coord[1], 'go', markersize=10, label='Start')
    ax.plot(goal_coord[0], goal_coord[1], 'ko', markersize=10, label='Goal')
    ax.scatter(workers[:, 0], workers[:, 1], color='red', marker='s', s=50, label='Workers')

    arrow_length = 0.4
    for (x, y) in path_coords:
        i, j = coord_to_index(x, y, x_vals, y_vals)
        best_ori = best_orientation_map[i, j]
        ax.arrow(
            x, y,
            arrow_length * np.cos(best_ori),
            arrow_length * np.sin(best_ori),
            head_width=0.15, head_length=0.2, fc='k', ec='k'
        )

    # ax.legend()
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.grid(True, linewidth=1.0, color='black', alpha=1.0)

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    fig.tight_layout(pad=0)
    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.array(canvas.renderer.buffer_rgba())

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    
    return img_bgr


