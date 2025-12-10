import os
import numpy as np
import cv2
import pyniryo as pyn
import pickle

# -----------------------------
# Settings
# -----------------------------
save_dir = "calibration_data"
verbose = True

# -----------------------------
# 1. Connect to Niryo Ned2
# -----------------------------
robot_ip = "169.254.200.200"
robot = pyn.NiryoRobot(robot_ip)
robot.enable_tcp(True)

# -----------------------------
# Load fisheye intrinsics
# -----------------------------
intrinsics_file = os.path.join(save_dir, "intrinsics_fisheye.npz")
intrinsics = np.load(intrinsics_file)
K = intrinsics["K"]
D = intrinsics["D"]

print("\nFisheye intrinsics loaded:")
print("K =\n", np.array2string(K, precision=4, suppress_small=True))
print("D =\n", np.array2string(D.ravel(), precision=4, suppress_small=True))

# -----------------------------
# 2. Load hand-eye calibration
# -----------------------------
handeye_file = os.path.join(save_dir, "handeye.npz")
handeye = np.load(handeye_file)
R_cam2gripper = handeye["R"]
t_cam2gripper = handeye["t"]

print("Hand-eye calibration loaded:")
print("R_cam2gripper =\n", np.array2string(R_cam2gripper, precision=4, suppress_small=True))
print("t_cam2gripper =\n", np.array2string(t_cam2gripper.ravel(), precision=4, suppress_small=True))

# -----------------------------
# 3. Load corresponding TCP poses
# -----------------------------
poses_file = os.path.join(save_dir, "points_and_poses.pkl")
with open(poses_file, "rb") as f:
    data = pickle.load(f)

eef_poses = data["eef_poses"]


def pose_to_matrix(pose):
    x, y, z, roll, pitch, yaw = pose
    R = cv2.Rodrigues(np.array([roll, pitch, yaw]))[0]
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


T_base_eef = [pose_to_matrix(p) for p in eef_poses]

# -----------------------------
# 4. Debug each pose interactively
# -----------------------------
obj_points = data.get("obj_points", [])
img_points = data.get("img_points", [])

P_cam = np.array([[0.0], [0.0], [0.0]])  # origin in camera frame

for idx, (pose, T_tcp) in enumerate(zip(eef_poses, T_base_eef), start=1):
    print(f"\n--- Pose {idx} ---")
    print("TCP position (base frame) =", np.array2string(pose[:3], precision=4))
    print("TCP orientation (rpy) =", np.array2string(pose[3:], precision=4))

    # Move robot to this pose
    robot.move(pyn.JointsPosition(*pose))

    # Transform camera origin to gripper and base frame
    P_gripper = R_cam2gripper @ P_cam + t_cam2gripper
    P_base_h = T_tcp @ np.vstack((P_gripper, [1]))
    P_base = P_base_h[:3]

    print("P_gripper =", np.array2string(P_gripper.ravel(), precision=4))
    print("P_base =", np.array2string(P_base.ravel(), precision=4))

    # Optional: check PnP reprojection error
    if idx - 1 < len(obj_points) and idx - 1 < len(img_points):
        objp = obj_points[idx - 1].astype(np.float32)
        corners = img_points[idx - 1].astype(np.float32)
        if corners.ndim == 2:
            corners = corners.reshape(-1, 1, 2)

        success, rvec, tvec = cv2.solvePnP(objp, corners, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
        if success:
            projected, _ = cv2.projectPoints(objp, rvec, tvec, K, D)
            error = np.linalg.norm(corners.reshape(-1, 2) - projected.reshape(-1, 2), axis=1).mean()
            print("Mean reprojection error =", round(error, 4), "px")

            # Show image with projected points
            if "captured_images" in data:
                img = data["captured_images"][idx - 1]
                for pt in projected.reshape(-1, 2):
                    cv2.circle(img, tuple(pt.astype(int)), 3, (0, 0, 255), -1)
                cv2.imshow(f"Pose {idx}", img)
                cv2.waitKey(0)  # Wait until window closed
                cv2.destroyAllWindows()
        else:
            print("PnP failed")

    input("Press Enter to continue to the next pose...")  # Optional pause
