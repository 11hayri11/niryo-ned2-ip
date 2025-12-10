import numpy as np
import pyniryo as pyn
import cv2
from scipy.spatial.transform import Rotation as R

# ---------------------- Load calibration ----------------------
intrinsics = np.load("calibration_data/intrinsics_fisheye.npz")
K = intrinsics["K"].astype(np.float64)
D = intrinsics["D"].astype(np.float64)

calib_data = np.load("calibration_data/handeye_TSAI.npz")
R_cam2gripper = calib_data["R"]
t_cam2gripper = calib_data["t"].reshape(3,1)

# ---------------------- Initialize robot ----------------------
robot_ip = "169.254.200.200"
robot = pyn.NiryoRobot(robot_ip)
robot.enable_tcp(True)

# ---------------------- Cube detection ----------------------
def detect_orange_cube(hsv):
    # Orange range in HSV
    # You can tweak these based on lighting: print pixel HSV values to refine
    lower_orange = np.array([5, 120, 100])   # lower hue, more saturation
    upper_orange = np.array([25, 255, 255])  # upper hue limit for orange

    # Create mask
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cubes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / float(h)

        if 0.7 < aspect < 1.3:  # roughly square
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cubes.append({
                "contour": cnt,
                "bbox": (x, y, w, h),
                "center": (cx, cy)
            })

    return cubes

def extract_top_face(cube, img=None, visualize=True):
    hull = cv2.convexHull(cube["contour"])
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.02 * peri, True)

    # If not exactly 4 corners, approximate as square
    if approx.shape[0] != 4:
        pts = hull.reshape(-1, 2)
        tl = pts[np.argmin(pts[:, 0] + pts[:, 1])]
        tr = pts[np.argmin(pts[:, 0] - pts[:, 1])]
        bl = pts[np.argmax(pts[:, 0] - pts[:, 1])]
        br = pts[np.argmax(pts[:, 0] + pts[:, 1])]
        approx = np.array([tl, tr, br, bl]).reshape(-1, 1, 2)

    # Compute center of top face as intersection of diagonals
    corners = approx.reshape(4, 2).astype(np.float32)
    c1 = (corners[0] + corners[2]) / 2.0
    c2 = (corners[1] + corners[3]) / 2.0
    center = ((c1 + c2) / 2.0).astype(np.float32)

    if visualize and img is not None:
        img_vis = img.copy()
        # Draw square
        cv2.polylines(img_vis, [approx.astype(np.int32)], True, (0, 255, 0), 2)
        # Draw corners
        for pt in corners:
            cv2.circle(img_vis, tuple(pt.astype(int)), 6, (0, 0, 255), -1)
        # Draw diagonals
        cv2.line(img_vis, tuple(corners[0].astype(int)), tuple(corners[2].astype(int)), (255, 0, 0), 2)
        cv2.line(img_vis, tuple(corners[1].astype(int)), tuple(corners[3].astype(int)), (255, 0, 0), 2)
        # Draw center
        cv2.circle(img_vis, tuple(center.astype(int)), 6, (0, 255, 255), -1)
        cv2.putText(img_vis, "Top center", tuple(center.astype(int) + np.array([5, -5])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow("Top face + center", img_vis)
        cv2.waitKey(0)

    # Return both corners and center
    return corners, center

# ---------------------- Main ----------------------
pose = [0.000, 0.070, -0.700, 0.000, -0.858, 0.000]
cube_size = 0.04

robot.move(pyn.JointsPosition(*pose))
img_compressed = robot.get_img_compressed()
img = pyn.image_functions.uncompress_image(img_compressed)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cubes = detect_orange_cube(hsv)

if not cubes:
    print("No yellow cube detected.")
else:
    cube = max(cubes, key=lambda c: cv2.contourArea(c["contour"]))

    # Draw full cube contour
    img_cube = img.copy()
    cv2.drawContours(img_cube, [cube["contour"]], -1, (255,0,0), 2)
    cv2.putText(img_cube, "Detected cube", cube["center"], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
    cv2.imshow("Cube detection", img_cube)
    cv2.waitKey(0)

    # Extract top face and visualize
    top_corners_2D, center = extract_top_face(cube, img, visualize=True)

    # Undistort points for PnP
    corners = top_corners_2D.reshape(-1,1,2).astype(np.float32)
    undistorted = cv2.fisheye.undistortPoints(corners, K, D)

    # 3D cube top corners
    s = cube_size/2
    objp = np.array([[ s,  s, 0],
                     [ s, -s, 0],
                     [-s, -s, 0],
                     [-s,  s, 0]], dtype=np.float32)

    I = np.eye(3)
    D_zero = np.zeros((1,5))
    success, rvec, tvec = cv2.solvePnP(objp, undistorted, I, D_zero, flags=cv2.SOLVEPNP_ITERATIVE)

    if success:
        # Enforce positive Z (optional, for visualization)
        if tvec[2] < 0:
            rvec = -rvec
            tvec = -tvec

        # ---------------------- Project world axes ----------------------
        axis_len = 0.05  # 5 cm axes
        axis = np.array([[axis_len, 0, 0],  # X - red
                         [0, axis_len, 0],  # Y - green
                         [0, 0, axis_len]], dtype=np.float32)  # Z - blue
        axis_fisheye = axis.reshape(-1, 1, 3).astype(np.float64)

        rvec_f = rvec.reshape(3, 1).astype(np.float64)
        tvec_f = tvec.reshape(3, 1).astype(np.float64)

        imgpts, _ = cv2.fisheye.projectPoints(axis_fisheye, rvec_f, tvec_f, K, D)


        # ---------------------- Draw axes on image ----------------------
        def draw_axes(img, corners, imgpts):
            corner = tuple(corners[0].ravel().astype(int))
            imgpts = imgpts.reshape(-1, 2).astype(int)
            img = cv2.line(img, corner, tuple(imgpts[0]), (0, 0, 255), 3)  # X - red
            img = cv2.line(img, corner, tuple(imgpts[1]), (0, 255, 0), 3)  # Y - green
            img = cv2.line(img, corner, tuple(imgpts[2]), (255, 0, 0), 3)  # Z - blue
            return img


        img_with_axes = draw_axes(img.copy(), corners, imgpts)
        cv2.imshow("Cube with World Axes", img_with_axes)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print("Rotation vector (camera frame):", rvec.flatten())
        print("Translation vector (camera frame):", tvec.flatten())

        # ---------------- Cube pose in base frame ----------------
        # Get gripper pose in base frame
        gripper_pose = robot.forward_kinematics(pose)


        # Niryo forward_kinematics returns [x,y,z,roll,pitch,yaw]
        def rpy2mat(roll, pitch, yaw):
            Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
            Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
            Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
            return Rz @ Ry @ Rx


        xg, yg, zg, rollg, pitchg, yawg = gripper_pose
        R_gripper2base = rpy2mat(rollg, pitchg, yawg)
        t_gripper2base = np.array([xg, yg, zg]).reshape(3, 1)

        # Cube in gripper frame
        R_cube2cam, _ = cv2.Rodrigues(rvec)
        t_cube2cam = tvec

        # Cube in base: Rb = Rg*Rc*Rc2cam, tb = Rg*Rc*t_cube2cam + Rg*t_cam2gripper + t_gripper2base
        R_cube2gripper = R_cam2gripper @ R_cube2cam
        t_cube2gripper = R_cam2gripper @ t_cube2cam + t_cam2gripper

        R_cube2base = R_gripper2base @ R_cube2gripper
        t_cube2base = R_gripper2base @ t_cube2gripper + t_gripper2base

        print("Cube position in base frame:", t_cube2base.flatten())
        print("Cube orientation in base frame:\n", R_cube2base)

        # ---------------- Parameters ----------------
        hover_height = 0.05  # 5 cm above the cube
        gripper_open_orientation = True  # adjust if you want gripper open

        # t_cube2base and R_cube2base from previous code
        # ---------------- Compute target gripper pose ----------------

        # Compute the "hover" position above the cube
        # Move along the cube's local Z-axis (up)
        cube_z_axis = R_cube2base[:, 2]  # Z-axis of cube in base frame
        t_target = t_cube2base.flatten() + hover_height * np.array([0, 0, 1])

        # Use cube rotation directly for gripper orientation
        # Convert rotation matrix to RPY for Niryo
        r = R.from_matrix(R_cube2base)
        roll, pitch, yaw = r.as_euler('xyz', degrees=False)

        print("Target gripper position (base frame):", t_target)
        print("Target gripper orientation (RPY):", (roll, pitch, yaw))

        # ---------------- Move robot ----------------
        # robot.move(pyn.PoseObject(x, y, z, roll, pitch, yaw))
        # Create PoseObject with metadata
        # 3. Construct PoseObject **with metadata**
        target_pose = pyn.PoseObject(
            t_target[0],
            t_target[1],
            t_target[2],
            roll,
            pitch,
            yaw
        )
        print(target_pose)

        # 4. Move above the cube
        robot.move(target_pose, linear=True)

        print("Gripper moved above the cube.")
    else:
        print("PnP failed.")

cv2.destroyAllWindows()
