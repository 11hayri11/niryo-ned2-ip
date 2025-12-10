import numpy as np
import cv2
import pyniryo as pyn
from scipy.spatial.transform import Rotation as R

# ---------------------- Parameters ----------------------
cube_size = 0.04  # meters
hover_height = 0.05  # meters above cube
view_poses = [
    [0.000, 0.6, -0.8, 0.000, -1.2, 0.000],
    [-0.350, 0.6, -0.7, 0.402, -1.2, 0.000],
    [0.350, 0.6, -0.7, -0.402, -1.2, 0.000],
]

# Load calibration
intrinsics = np.load("calibration_data/intrinsics_fisheye.npz")
K = intrinsics["K"].astype(np.float64)
D = intrinsics["D"].astype(np.float64)

handeye = np.load("calibration_data/handeye_TSAI.npz")
R_cam2gripper = handeye["R"]
t_cam2gripper = handeye["t"].reshape(3,1)

# ---------------------- Initialize robot ----------------------
robot_ip = "169.254.200.200"
robot = pyn.NiryoRobot(robot_ip)
robot.enable_tcp(True)

# ---------------------- Cube detection ----------------------
def detect_orange_cube(hsv, img=None, visualize=False):
    lower_orange = np.array([3, 150, 150])
    upper_orange = np.array([30, 255, 255])

    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    if img is not None:
        roi = cv2.bitwise_and(img, img, mask=mask)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3,3), 0)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        grad_mag = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, edges = cv2.threshold(grad_mag, 50, 255, cv2.THRESH_BINARY)
        edges = cv2.bitwise_and(edges, mask)
    else:
        edges = mask

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cubes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue
        hull = cv2.convexHull(cnt)
        if cv2.contourArea(hull)/area < 0.85:
            continue
        x, y, w, h = cv2.boundingRect(hull)
        aspect = w / float(h)
        if not 0.7 < aspect < 1.3:
            continue
        M = cv2.moments(hull)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"]/M["m00"])
        cy = int(M["m01"]/M["m00"])
        cubes.append({"contour": hull, "bbox": (x,y,w,h), "center": (cx,cy)})
        if visualize and img is not None:
            cv2.drawContours(img, [hull], -1, (0,255,0), 2)
            cv2.circle(img, (cx, cy), 5, (0,0,255), -1)

    if visualize and img is not None:
        cv2.imshow("Cube detection improved", img)
        cv2.waitKey(500)

    if cubes:
        return [max(cubes, key=lambda c: cv2.contourArea(c["contour"]))]
    else:
        return []

def extract_top_face(cube, img=None, visualize=True):
    hull = cv2.convexHull(cube["contour"])
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.02*peri, True)

    if approx.shape[0] != 4:
        pts = hull.reshape(-1,2)
        tl = pts[np.argmin(pts[:,0]+pts[:,1])]
        tr = pts[np.argmin(pts[:,0]-pts[:,1])]
        bl = pts[np.argmax(pts[:,0]-pts[:,1])]
        br = pts[np.argmax(pts[:,0]+pts[:,1])]
        approx = np.array([tl,tr,br,bl]).reshape(-1,1,2)

    # add center
    cx, cy = cube["center"]
    center = np.array([[cx, cy]], dtype=np.float32)
    points = np.vstack([approx.reshape(-1,2), center])

    if visualize and img is not None:
        img_vis = img.copy()
        cv2.polylines(img_vis, [approx.astype(np.int32)], True, (0, 255, 0), 2)
        for i, pt in enumerate(points):
            pt_int = tuple(pt.astype(int))
            cv2.circle(img_vis, pt_int, 5, (0,0,255), -1)
            cv2.putText(img_vis, str(i), pt_int, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
        cv2.imshow("Cube top face + center", img_vis)
        cv2.waitKey(500)

    return points  # 4 corners + center

# ---------------------- Main ----------------------
img_points_all = []
obj_points = []

# 3D cube top corners relative to cube center (X,Y axes rotated 45 deg)
s = cube_size/2
angle45 = np.deg2rad(45)
rot45 = np.array([[np.cos(angle45), -np.sin(angle45), 0],
                  [np.sin(angle45),  np.cos(angle45), 0],
                  [0, 0, 1]])
# corners relative to center in XY plane
cube_top_3D = np.array([[ s,  s, 0],
                        [ s, -s, 0],
                        [-s, -s, 0],
                        [-s,  s, 0]], dtype=np.float32)
cube_top_3D = (rot45[:3,:3] @ cube_top_3D.T).T
# add center at (0,0,0)
cube_top_3D = np.vstack([cube_top_3D, np.zeros((1,3), dtype=np.float32)])

I = np.eye(3)
D_zero = np.zeros((1,5))

for idx, pose in enumerate(view_poses, start=1):
    print(f"Moving to pose {idx}...")
    robot.move(pyn.JointsPosition(*pose))
    img_compressed = robot.get_img_compressed()
    img = cv2.imdecode(np.frombuffer(img_compressed, np.uint8), cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    cubes = detect_orange_cube(hsv)
    if not cubes:
        print(f"No cube detected in view {idx}")
        continue

    cube = max(cubes, key=lambda c: cv2.contourArea(c["contour"]))
    top_points_2D = extract_top_face(cube, img, visualize=True)
    img_points_all.append(top_points_2D)
    obj_points.append(cube_top_3D)

cv2.destroyAllWindows()

# ---------------------- Triangulate cube center ----------------------
# Only use the center point (last point in each view)
centers_2D = [pts[-1].reshape(1,1,2).astype(np.float32) for pts in img_points_all]
world_points = []

for i, (pose, center2D) in enumerate(zip(view_poses, centers_2D)):
    # Camera pose in base frame
    xg, yg, zg, rollg, pitchg, yawg = pose
    R_gripper2base = R.from_euler('xyz', [rollg, pitchg, yawg]).as_matrix()
    t_gripper2base = np.array([xg, yg, zg]).reshape(3,1)
    R_cam2base = R_gripper2base @ R_cam2gripper
    t_cam2base = R_gripper2base @ t_cam2gripper + t_gripper2base

    # Undistort 2D
    undistorted = cv2.fisheye.undistortPoints(center2D, K, D)
    # back-project ray in camera frame (z=1)
    uv = undistorted[0,0]
    ray_cam = np.array([uv[0], uv[1], 1.0]).reshape(3,1)
    ray_cam /= np.linalg.norm(ray_cam)
    # transform ray to base frame
    ray_base = R_cam2base @ ray_cam
    origin_base = t_cam2base.flatten()
    world_points.append((origin_base, ray_base.flatten()))

# Simple least-squares intersection of rays to get cube center
A = np.zeros((3,3))
b = np.zeros(3)
for o, d in world_points:
    d = d.reshape(3,1)
    A += np.eye(3) - d @ d.T
    b += (np.eye(3) - d @ d.T) @ o
cube_center_base = np.linalg.solve(A, b)
print("Triangulated cube center in base frame:", cube_center_base)