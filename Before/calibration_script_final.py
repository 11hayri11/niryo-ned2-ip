import cv2
import numpy as np
import pyniryo as pyn
import os
import pickle
import time
from pyniryo.vision import uncompress_image

def rpy_to_rot_matrix(rpy):
    """
    Convert RPY (roll, pitch, yaw) to rotation matrix.
    rpy: [roll, pitch, yaw] in radians
    Returns 3x3 rotation matrix.
    """
    roll, pitch, yaw = rpy

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])

    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])

    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])

    # Apply rotations in order: roll → pitch → yaw
    R = Rz @ Ry @ Rx
    return R

# -----------------------------
# 1. Connect to Niryo Ned2
# -----------------------------
robot_ip = "129.187.231.226"
robot = pyn.NiryoRobot(robot_ip)
robot.enable_tcp(True)

# -----------------------------
# 2. Chessboard settings
# -----------------------------
chessboard_size = (8, 6)
square_size = 0.025  # meters

# Prepare 3D world points for each chessboard view
objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
objp[:, :2] = np.mgrid[
    0:chessboard_size[0], 0:chessboard_size[1]
].T.reshape(-1, 2) * square_size

print(objp)
# -----------------------------
# 3. Storage for calibration
# -----------------------------
obj_points = []         # Chessboard corners in world coordinates
img_points = []         # Detected corners in image coordinates
eef_poses = []          # TCP poses from the robot
captured_images = []    # Store images

# -----------------------------
# 4. Define robot poses
# -----------------------------
robot_poses = [
    [0.000, 0.610, -0.163, 0.000, -1.866, 0.000],
    [-1.146, 0.338, -0.245, 0.714, -1.809, 0.000],
    [0.000, 0.338, -0.245, 0.051, -1.636, 0.000],
    [1.146, 0.338, -0.245, -0.714, -1.809, 0.000],
    [1.146, 0.447, -0.644, -0.714, -1.809, 0.000],
    [-1.146, 0.447, -0.644, 0.714, -1.809, 0.000],
    [-1.5, 0.6, -0.644, 0.714, -1.920, 0.000],
    [1.5, 0.6, -0.644, -0.714, -1.920, 0.000],
    [0.000, 0.447, -0.936, 0.714, -0.708, 0.000],
    [0.000, 0.447, -1.160, 0.000, -0.491, 0.000],
    [-0.350, 0.447, -0.848, 0.402, -0.941, 0.000],
    [0.350, 0.447, -0.848, -0.402, -0.941, 0.000],
    [0.767, -0.030, 0.091, -0.743, -1.897, 0.000],
    [0.000, -0.471, 0.205, 0.085, -1.899, 0.000],
    [0.506, -0.471, 0.205, -0.437, -1.897, 0.000],
    [-0.506, -0.471, 0.205, 0.437, -1.897, 0.000],
    [-0.725, -0.612, 0.193, 0.624, -1.670, 0.000],
    [-0.200, -0.612, 0.193, 0.194, -1.670, 0.000],
    [0.306, -0.612, 0.193, -0.346, -1.670, 0.000],
    [0.306, 0.610, 0.011, -0.254, -1.912, 0.000],
    [0.808, 0.610, -0.337, -0.806, -1.912, 0.000],
    [-0.510, 0.610, -0.085, 0.392, -1.825, 0.000],
]

# -----------------------------
# 5. Capture images and detect corners
# -----------------------------
for idx, pose in enumerate(robot_poses, start=1):
    robot.move(pyn.JointsPosition(*pose))
    time.sleep(0.5)  # wait 0.5 s for the robot to reach the pose

    # Capture image
    img_compressed = robot.get_img_compressed()
    img = uncompress_image(img_compressed)  # Already a BGR image
    print(f"Resolution: {img.shape[0]}x{img.shape[1]}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    print(f"Pose {idx}/{len(robot_poses)} - Chessboard found: {found}")

    if found:
        objp_reshaped = objp.reshape(-1, 1, 3).astype(np.float32)  # shape (N,1,3)
        corners_reshaped = corners.astype(np.float32)  # shape (N,1,2)

        obj_points.append(objp_reshaped)
        img_points.append(corners_reshaped)

        # Get TCP coordinates via forward kinematics
        eef_pose = robot.forward_kinematics(pose)
        eef_poses.append(eef_pose)

        cv2.drawChessboardCorners(img, chessboard_size, corners, found)
        captured_images.append(img)

        cv2.imshow("Chessboard", img)
        cv2.waitKey(1)

cv2.destroyAllWindows()

# -----------------------------
# 6. Intrinsic camera calibration (fisheye model)
# -----------------------------
print("\nStarting fisheye camera calibration...")

# Initialize parameters
N_OK = len(obj_points)
K = np.zeros((3,3))
D = np.zeros((4,1))
rvecs_target2cam = [np.zeros((1,1,3), dtype=np.float64) for _ in range(N_OK)]
tvecs_target2cam = [np.zeros((1,1,3), dtype=np.float64) for _ in range(N_OK)]

rms, K, D, rvecs_target2cam, tvecs_target2cam = cv2.fisheye.calibrate(
    obj_points,
    img_points,
    gray.shape[::-1],  # (width, height)
    K,
    D,
    rvecs_target2cam,
    tvecs_target2cam,
    flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC,
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
)

print(f"\nFisheye calibration complete:")
print(f"  -> RMS reprojection error: {rms:.6f}")
print(f"  -> Camera matrix (K):\n{K}")
print(f"  -> Distortion coefficients (D): {D.ravel()}")

# Compute average reprojection error
def compute_fisheye_reprojection_error(obj_points_f, img_points_f, rvecs, tvecs, K, D):
    total_error = 0
    for objp, imgp, rvec, tvec in zip(obj_points_f, img_points_f, rvecs, tvecs):
        projected, _ = cv2.fisheye.projectPoints(objp, rvec, tvec, K, D)
        projected = projected.reshape(-1, 2)
        err = np.linalg.norm(projected - imgp.reshape(-1, 2), axis=1).mean()
        total_error += err
    return total_error / len(obj_points_f)

mean_error = compute_fisheye_reprojection_error(
    obj_points, img_points, rvecs_target2cam, tvecs_target2cam, K, D
)
print(f"  -> Mean reprojection error per image: {mean_error:.4f} px")

# Optional: visualize one undistorted image
img = captured_images[0]
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, gray.shape[::-1], cv2.CV_16SC2)
undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
cv2.imshow("Undistorted (fisheye)", undistorted)
cv2.waitKey(0)
cv2.destroyAllWindows()

# -----------------------------
# 8. Convert Niryo poses to matrices
# -----------------------------
def niryo_pose_to_matrix(pose):
    x, y, z, roll, pitch, yaw = pose
    R = rpy_to_rot_matrix(np.array([roll, pitch, yaw]))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T

T_base_eef = [niryo_pose_to_matrix(pose) for pose in eef_poses]

R_gripper2base = [T[:3, :3] for T in T_base_eef]
t_gripper2base = [T[:3, 3].reshape(3, 1) for T in T_base_eef]

# -----------------------------
# 9. Hand-eye calibration (TSAI only)
# -----------------------------
print("\n=== Hand-Eye Calibration (TSAI) ===")

# Compute hand-eye using only TSAI
R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
    R_gripper2base, t_gripper2base,
    rvecs_target2cam, tvecs_target2cam,
    method=cv2.CALIB_HAND_EYE_TSAI
)

print("R_cam2gripper:\n", R_cam2gripper)
print("t_cam2gripper:\n", t_cam2gripper.T)

# -----------------------------
# 10. Save all relevant calibration info
# -----------------------------
save_dir = "calibration_data"
os.makedirs(save_dir, exist_ok=True)

# Save fisheye intrinsics
np.savez(os.path.join(save_dir, "intrinsics_fisheye.npz"), K=K, D=D)

# Save hand-eye calibration (TSAI)
np.savez(os.path.join(save_dir, "handeye_TSAI.npz"), R=R_cam2gripper, t=t_cam2gripper)

# Save captured images
for i, img in enumerate(captured_images):
    cv2.imwrite(os.path.join(save_dir, f"image_{i:02d}.png"), img)

# Save object/image points + robot poses
with open(os.path.join(save_dir, "points_and_poses.pkl"), "wb") as f:
    pickle.dump({
        "obj_points": obj_points,          # 3D chessboard points
        "img_points": img_points,          # 2D detected corners
        "eef_poses": eef_poses,            # TCP poses at each image
        "joint_poses": robot_poses,        # Robot joint positions used
        "rvecs_target2cam": rvecs_target2cam,      # PnP rotation matrices
        "tvecs_target2cam": tvecs_target2cam       # PnP translation vectors
    }, f)

print(f"Calibration data saved in '{save_dir}'")