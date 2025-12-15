# aruco_detect_estimate_motion.py
"""
I will build a solid pipeline for an marker detection and accurate pose estimation
--> Add a pick & place motion if accurate enough

Well structured and Step for Step:
    1. Implement pre-requisites (imports, configs, helper, connect robot, initialize camera)
    2. First: Good marker detection (try different position, and angles, and rotation)
    3. Than: Add a pose estimation (Test for accuracy)
    4. When all that works good -> add motion pipeline (pnp, stack, sort, ...)

Always test and check the steps before moving to the next one.

Marker side length: 0.026m (2,6cm)
orange cube side length: 0.036m (3,6cm)
using - aruco.DICT_4X4_50
ID 0 = orange cube, ID 1 = blue cube

We will use the run: run_20251211_153839 from calibration_data_charuco/

"""

import time
import json
from pathlib import Path
import datetime

import cv2
import numpy as np
import pyniryo as pyn

# ========================
# Step 1: Configs + Helper
# ========================
# ---Configs---
ROBOT_IP = "129.187.231.226"
RUN_DIR = Path("calibration_data_charuco/run_20251211_153839")

CAMERA_INDEX = 0
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
SHOW_PREVIEW = True

# Fixed observation pose
OBS_JOINTS = [0.2717, 0.4843, -0.9719, -0.0137, -0.9972, 0.2670]
Q6_REF = 0.0

ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50
TARGET_ID = 0
MARKER_LEN_M = 0.026

# ---helpers---
def load_intrinsics(run_dir: Path):
    data = np.load(run_dir / "intrinsics_charuco_pinhole_final.npz")
    K = data["K"]
    dist = data["dist"].reshape(-1, 1) if data["dist"].ndim == 2 else data["dist"]
    return K, dist

def load_handeye(run_dir: Path):
    data = np.load(run_dir / "handeye_charuco_hand_TSAI.npz")
    return data["T_hand_camera"]

def make_aruco_detector():
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, params)
    return detector

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

def niryo_pose_to_matrix_from_obj(pose):
    """
    Build ^baseT_tcp from a Niryo pose object (x,y,z,roll,pitch,yaw).
    """
    T = np.eye(4, dtype=float)
    R = rpy_to_rot_matrix([pose.roll, pose.pitch, pose.yaw])
    T[:3, :3] = R
    T[:3,  3] = [pose.x, pose.y, pose.z]
    return T

def fk_base_to_hand_virtual(robot, joints, q6_ref=0.0):
    """
    Compute a 'virtual hand' frame ^baseT_hand using the robot's own FK.

    Idea:
      - The camera is rigid to the wrist (joint 5) and does NOT move with q6.
      - We therefore build a frame that depends only on joints 1..5 by
        *artificially freezing* q6 to a fixed reference angle q6_ref
        before calling the robot's forward kinematics.

    Parameters
    ----------
    robot : pyn.NiryoRobot
        Connected robot instance, used to call forward_kinematics.
    joints : array-like of length 6
        Actual joint angles [q1..q6] in radians.
    q6_ref : float, optional
        Reference angle (rad) used for joint 6 when computing the FK.

    Returns
    -------
    T_hand : (4,4) ndarray
        Homogeneous transform ^baseT_hand for this configuration.
    """
    q = np.array(joints, dtype=float).reshape(6,)
    q[5] = float(q6_ref)          # overwrite actual q6 → freeze it

    # PyNiryo: forward_kinematics(joints: list[float]) → Pose object
    pose_fk = robot.forward_kinematics(q.tolist())

    # Re-use your existing helper to make a 4x4 from pose
    T_hand = niryo_pose_to_matrix_from_obj(pose_fk)
    return T_hand

def live_marker_detection(cap, detector, target_id=0):
    aruco = cv2.aruco
    print("\n[INFO] Step5: Live ArUco detection (raw frame)")
    print("       Keys: [q]=quit, [s]=snapshot print IDs")

    last_t = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Camera read failed")
            continue

        # FPS (simple smoothing)
        now = time.time()
        dt = now - last_t
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        last_t = now

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)

        ids_list = [] if ids is None else ids.ravel().tolist()
        rejected_n = 0 if rejected is None else len(rejected)

        vis = frame.copy()
        if ids is not None:
            aruco.drawDetectedMarkers(vis, corners, ids)

        # Small “health line”
        overlay = f"fps={fps:.1f}  ids={ids_list}  rejected={rejected_n}  target={target_id}"
        cv2.putText(vis, overlay, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, overlay, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Step5 - ArUco detect (raw)", vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("s"):
            print(f"[SNAPSHOT] ids={ids_list} | rejected={rejected_n}")
    cv2.destroyAllWindows()

# ================================================
# Step 2: Connect to Niryo Ned2 + Move to OBS_POSE
# ================================================
print("\n[INFO] Moving to observation pose and connecting to Niryo Ned2 robot at", ROBOT_IP, "...")

t0 = time.time()
robot = pyn.NiryoRobot(ROBOT_IP)
robot.enable_tcp(True) # Check if TCP is set to the right one!
robot.set_learning_mode(False)

# Optional (guarded) info:
try:
    print("[INFO] Robot joints (rad):", np.round(robot.get_joints(), 3))
except Exception:
    pass

# Move to a fixed observation pose
robot.move(pyn.JointsPosition(*OBS_JOINTS))
time.sleep(0.3)       # small settle time

print("[INFO] OBS_JOINTS =", np.round(OBS_JOINTS, 4).tolist())
print("[INFO] joints (actual)        =", np.round(robot.get_joints(), 4).tolist())

pose_live = robot.get_pose()
print(
    "[INFO] Robot moved to OBS pose: "
    f"x={pose_live.x:.4f}, y={pose_live.y:.4f}, z={pose_live.z:.4f}, "
    f"r={pose_live.roll:.4f}, p={pose_live.pitch:.4f}, yaw={pose_live.yaw:.4f} "
    f"(took {time.time() - t0:.2f}s)"
)

print(f"[INFO] Robot connection established successfully. (took {time.time()-t0:.2f}s)")

# ========================================
# Step 3: Initialize external (USB) camera
# ========================================
print(f"\n[INFO] Opening external camera (index {CAMERA_INDEX}) ...")
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# Test if the camera opened successfully
if not cap.isOpened():
    raise RuntimeError("[ERROR] Could not open camera. Check USB connection or index.")

# Grab one frame to confirm resolution
ret, frame = cap.read()
if not ret:
    raise RuntimeError("[ERROR] Failed to read test frame from camera.")

h, w = frame.shape[:2]
print(f"[INFO] Camera connected successfully. Current resolution: {w}×{h}")

# Optionally show one quick preview window (press any key to continue)
if SHOW_PREVIEW:
    cv2.imshow("Camera Test Frame", frame)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

print("\n[SETUP COMPLETE] Robot and camera are ready.\n")

# ========================================
# Step 4: Load calibration + sanity checks
# ========================================
print("\n[INFO] Step4: Load calibration artifacts (golden run)")

intr_path = RUN_DIR / "intrinsics_charuco_pinhole_final.npz"
he_path   = RUN_DIR / "handeye_charuco_hand_TSAI.npz"

intr = np.load(intr_path, allow_pickle=True)
K = intr["K"].astype(np.float64)
dist = intr["dist"].reshape(-1).astype(np.float64)

print("[INFO] Loaded intrinsics:", intr_path)
print("[INFO] K:\n", K)
print("[INFO] dist:", dist, "| shape:", dist.shape)

if "image_size" in intr:
    w_cal, h_cal = intr["image_size"].tolist()   # (width, height)
    print(f"[INFO] intrinsics image_size = {w_cal}x{h_cal}")
    if (w, h) != (w_cal, h_cal):
        print("[WARN] Camera resolution differs from calibration image_size!")
        print("       -> Set camera to calibrated resolution for reliable pose.")
else:
    print("[WARN] intrinsics file has no image_size key")

he = np.load(he_path, allow_pickle=True)
T_hand_camera = he["T_hand_camera"].astype(np.float64)
print("[INFO] Loaded hand-eye:", he_path)
print("[INFO] ||t_hand_camera|| =", float(np.linalg.norm(T_hand_camera[:3, 3])))

# ===================================
# Step 5: Live ArUco marker detection
# ===================================
detector = make_aruco_detector()
live_marker_detection(cap, detector, target_id=TARGET_ID)

cap.release()
robot.close_connection()