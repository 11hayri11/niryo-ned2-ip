# check_handeye_residuals.py
# ---------------------------------------------
# Sanity check for OpenCV calibrateHandEye result.
# We verify the AX = XB relation using the SAME convention
# as the OpenCV docs for eye-in-hand calibration.
#
# OpenCV equation (eye-in-hand):
#   A_ij X = X B_ij
#
# where for a pair of views (i, j):
#   A_ij = ( ^baseT_gripper(j) )^{-1}  ^baseT_gripper(i)
#   B_ij =   ^camT_target(j)          ( ^camT_target(i) )^{-1}
#
# and X = ^gripperT_camera  (output of calibrateHandEye).
#
# We compute:
#   ΔT_ij = X^{-1} A_ij X B_ij^{-1}
#   and measure how close ΔT_ij is to identity.
# ---------------------------------------------

import os
import pickle
import numpy as np
import cv2

# ------------------------------------------------------------------
# 1) CONFIG: choose which calibration run to inspect
# ------------------------------------------------------------------
# Change this path to any of your recent calibration runs:
# e.g. "calibration_data/run_20251128_104545"
run_dir = r"calibration_data\run_20251128_145707"

print(f"[INFO] Using run_dir = {run_dir}")

bundle_path = os.path.join(run_dir, "points_and_poses.pkl")
handeye_path = os.path.join(run_dir, "handeye_TSAI.npz")

# ------------------------------------------------------------------
# 2) Load bundle (poses + vision extrinsics)
# ------------------------------------------------------------------
print(f"[INFO] Loading bundle: {bundle_path}")
with open(bundle_path, "rb") as f:
    bundle = pickle.load(f)

eef_poses = bundle["eef_poses"]            # list of [x,y,z,roll,pitch,yaw]
rvecs_target2cam = bundle["rvecs_target2cam"]  # list of (1,1,3) or (3,) rvecs
tvecs_target2cam = bundle["tvecs_target2cam"]  # list of tvecs

num_views = len(eef_poses)
print(f"[INFO] Loaded {num_views} eef_poses and {len(rvecs_target2cam)} vision extrinsics from bundle.")

# ------------------------------------------------------------------
# 3) Load hand–eye result: ^gripperT_camera
# ------------------------------------------------------------------
print(f"[INFO] Loading hand-eye result: {handeye_path}")
he = np.load(handeye_path)
R_cam2gripper = he["R"]          # this is ^gripperR_camera
t_cam2gripper = he["t"].reshape(3, 1)  # ^grippert_camera  (3x1)

# Build ^gripperT_camera
T_gripper_camera = np.eye(4)
T_gripper_camera[:3, :3] = R_cam2gripper
T_gripper_camera[:3,  3] = t_cam2gripper.ravel()

norm_t = np.linalg.norm(t_cam2gripper)
print(f"[INFO] Hand-eye: ||^grippert_camera|| = {norm_t:.4f} m (~{norm_t*100:.1f} cm)")

# ------------------------------------------------------------------
# 4) Helper: Niryo pose -> ^baseT_gripper
# ------------------------------------------------------------------
def rpy_to_rot_matrix(rpy):
    """Same helper as in redoCalibrationV2.py"""
    roll, pitch, yaw = rpy
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll),  np.cos(roll)]])
    Ry = np.array([[ np.cos(pitch), 0, np.sin(pitch)],
                   [0,              1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[ np.cos(yaw), -np.sin(yaw), 0],
                   [ np.sin(yaw),  np.cos(yaw), 0],
                   [0,             0,           1]])
    return Rz @ Ry @ Rx

def niryo_pose_to_matrix(pose):
    """
    Convert Niryo TCP pose [x,y,z,roll,pitch,yaw] expressed in BASE frame
    into ^baseT_gripper (4x4).
    """
    x, y, z, roll, pitch, yaw = pose
    R = rpy_to_rot_matrix(np.array([roll, pitch, yaw]))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = [x, y, z]
    return T

# ------------------------------------------------------------------
# 5) Build ^baseT_gripper(i) and ^camT_target(i)
# ------------------------------------------------------------------
T_base_gripper = [niryo_pose_to_matrix(p) for p in eef_poses]
print(f"[INFO] Built {len(T_base_gripper)} transforms ^baseT_gripper from eef_poses.")
print("[DEBUG] Example ^baseT_gripper[0]:")
print(T_base_gripper[0])

T_cam_target = []
for rvec, tvec in zip(rvecs_target2cam, tvecs_target2cam):
    # Make sure they are 3-vectors
    r = np.array(rvec, dtype=np.float64).reshape(3,)
    t = np.array(tvec, dtype=np.float64).reshape(3, 1)

    # Rodrigues: rotation vector -> 3x3 rotation matrix
    R_ct, _ = cv2.Rodrigues(r)   # ^camR_target
    t_ct = t                     # ^camt_target (3x1)

    T = np.eye(4)
    T[:3, :3] = R_ct
    T[:3,  3] = t_ct.ravel()
    T_cam_target.append(T)

print(f"[INFO] Built {len(T_cam_target)} transforms ^camT_target from rvecs/tvecs.")

# ------------------------------------------------------------------
# 6) Compute AX = XB residuals with OpenCV’s convention
# ------------------------------------------------------------------
X = T_gripper_camera
X_inv = np.linalg.inv(X)

rot_residuals = []
trans_residuals = []
full_residuals = []

worst_idx = None
worst_trans = -1.0

num_pairs = 0
for i in range(num_views):
    for j in range(i + 1, num_views):
        # A_ij = ( ^baseT_gripper(j) )^{-1}  ^baseT_gripper(i)
        A_ij = np.linalg.inv(T_base_gripper[j]) @ T_base_gripper[i]

        # B_ij = ^camT_target(j) ( ^camT_target(i) )^{-1}
        B_ij = T_cam_target[j] @ np.linalg.inv(T_cam_target[i])

        # If AX = XB holds exactly, then:
        #   X^{-1} A X B^{-1} = I
        ΔT = X_inv @ A_ij @ X @ np.linalg.inv(B_ij)

        R_err = ΔT[:3, :3]
        t_err = ΔT[:3,  3]

        # Rotation residual: ||R_err - I||_F
        rot_res = np.linalg.norm(R_err - np.eye(3), ord="fro")

        # Translation residual: ||t_err||_2
        trans_res = np.linalg.norm(t_err)

        # Full 4x4 residual: ||ΔT - I||_F
        full_res = np.linalg.norm(ΔT - np.eye(4), ord="fro")

        rot_residuals.append(rot_res)
        trans_residuals.append(trans_res)
        full_residuals.append(full_res)

        if trans_res > worst_trans:
            worst_trans = trans_res
            worst_idx = (i, j)

        num_pairs += 1

print(f"[INFO] Number of pose pairs (i<j): {num_pairs}")

rot_residuals = np.array(rot_residuals)
trans_residuals = np.array(trans_residuals)
full_residuals = np.array(full_residuals)

print("\n[RESULT] Tsai AX = XB residuals over all pose pairs (OpenCV convention)")
print("  Rotation residual ‖R_err - I‖_F:")
print(f"    mean = {rot_residuals.mean():.3e}")
print(f"    max  = {rot_residuals.max():.3e}")

print("\n  Translation residual ‖t_err‖ (meters):")
print(f"    mean = {trans_residuals.mean():.3e} m")
print(f"    max  = {trans_residuals.max():.3e} m")

print("  Translation residual ‖t_err‖ (millimeters):")
print(f"    mean = {trans_residuals.mean()*1000:.3f} mm")
print(f"    max  = {trans_residuals.max()*1000:.3f} mm")

print("\n  Full 4x4 residual ‖ΔT - I‖_F:")
print(f"    mean = {full_residuals.mean():.3e}")
print(f"    max  = {full_residuals.max():.3e}")

if worst_idx is not None:
    i, j = worst_idx
    print(f"\n[DEBUG] Worst translation pair (i,j) = ({i}, {j}), "
          f"‖t_err‖ = {worst_trans*1000:.3f} mm")

print("\n[DONE] Hand–eye residual check finished.")
