import rosbag
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import cv2
import matplotlib.cm as cm
from ultralytics import YOLO


# -------------------------
# Parameters :
# -------------------------


model = YOLO('yolov8s.pt')

transformation_matrix = np.array([
    [0.9999353156286391, 0.006113157552193651, -0.00959134314868424, -0.0031391209194647106],
    [0.010503421799860262, -0.17276975498184988, 0.9849062340619058, 0.06680829559166526],
    [0.0043637929772131255, -0.9849432679440241, -0.1728227885511584, -0.0779811866869374],
    [0.0, 0.0, 0.0, 1.0]])

K = np.array([
    [700.1733239760443, 0, 755.8118827756588],
    [0, 700.0133240981131, 554.4259117189192],
    [0, 0, 1]])

D = np.array([-0.038599716351541, -0.005987974259361224, 0.005538347735041447, -0.002212063745000479])

safe_zone_coords = [
    (7.138659, 46.941301),
    (7.13848299, 46.94144631),
    (7.140534, 46.942412),
    (7.140700, 46.942279)]


# -------------------------
# Helper Functions :
# -------------------------


def undistort_image(image):
    height, width = image.shape[:2]
    K[0, 0] *= width / 1440.0
    K[1, 1] *= height / 1080.0
    K[0, 2] *= width / 1440.0
    K[1, 2] *= height / 1080.0
    undistorted_image = cv2.fisheye.undistortImage(image, K=K, D=D, Knew=K, new_size=(width, height))
    return undistorted_image

def preprocess_image(camera_msg):
    compressed_data = np.frombuffer(camera_msg.data, np.uint8)
    camera_img = cv2.imdecode(compressed_data, cv2.IMREAD_COLOR)
    camera_img = undistort_image(camera_img)
    for i in range(3):
        camera_img[:, :, i] = cv2.equalizeHist(camera_img[:, :, i])
    return camera_img

def transform_lidar_points(lidar_msg):
    point_list = []
    for point in pc2.read_points(lidar_msg, skip_nans=True):
        x, y, z = point[0], point[1], point[2]
        point_list.append([x, y, z])
    points = np.array(point_list, dtype=np.float64)
    ones = np.ones((points.shape[0], 1))
    points_homogeneous = np.hstack((points, ones))
    points_transformed = (transformation_matrix @ points_homogeneous.T).T[:, :3]
    return points_transformed

def project_to_image_with_depth(points, K, image_shape):
    points = points[points[:, 2] > 0]
    points_proj = (K @ points.T).T
    image_points = points_proj[:, :2] / points_proj[:, 2, None]
    depths = points[:, 2]
    valid_mask = (
        (0 <= image_points[:, 0]) & (image_points[:, 0] < image_shape[1]) &
        (0 <= image_points[:, 1]) & (image_points[:, 1] < image_shape[0])
    )
    image_points = image_points[valid_mask]
    depths = depths[valid_mask]
    # Flip the projected points
    image_points[:, 0] = image_shape[1] - image_points[:, 0]
    image_points[:, 1] = image_shape[0] - image_points[:, 1]
    return image_points, depths

def detect_humans(camera_img):
    results = model(camera_img)
    boxes = results[0].boxes.xyxy.cpu().numpy() 
    class_ids = results[0].boxes.cls.cpu().numpy()
    return boxes[class_ids == 0]

def calculate_human_distances(image_points, depths, bounding_boxes):
    distances = []
    for box in bounding_boxes:
        x1, y1, x2, y2 = map(int, box)
        mask = (
            (image_points[:, 0] >= x1) & (image_points[:, 0] <= x2) &
            (image_points[:, 1] >= y1) & (image_points[:, 1] <= y2)
        )
        box_depths = depths[mask]
        if len(box_depths) > 0:
            # distances.append(box_depths.min())
            
            # Find the bin with the highest count
            counts, bin_edges = np.histogram(box_depths, bins=30)
            dominant_index = np.argmax(counts)
            dominant_distance = (bin_edges[dominant_index] + bin_edges[dominant_index + 1]) / 2
            distances.append(dominant_distance)
        else:
            distances.append(None)
    return distances