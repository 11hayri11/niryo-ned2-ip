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
OBS_JOINTS_1 = [0.2717, 0.4843, -0.9719, -0.0137, -0.9972, 0.2670]

# Extra viewpoints (try to keep q6 the same if possible; q6 doesn't change the camera much in your setup)
OBS_JOINTS_2 = [-0.2411, 0.4585, -0.9961, -0.0674, -1.0186, 0.2670]
OBS_JOINTS_3 = [-0.7723, 0.4115, -0.9627, -0.0720, -1.0140, 0.2670]

OBS_VIEWS = [OBS_JOINTS_1, OBS_JOINTS_2, OBS_JOINTS_3]
PRIMARY_OBS = OBS_JOINTS_1

Q6_REF = 0.0

ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50
TARGET_ID = 0
MARKER_LEN_M = 0.026

# log folder to not pollute the calibration run
LOG_DIR = Path("aruco_logs") / datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# --- motion check config ---
DO_MOVE_ON_SNAPSHOT = True          # set False for dry-run
Z_HOVER = 0.05                      # 5 cm above marker
Z_MIN   = 0.03                      # never go below 3 cm
Z_SAFE  = 0.12                      # safe travel height
# rough workspace bounds (adjust if you want)
X_MIN, X_MAX = 0.12, 0.45
Y_MIN, Y_MAX = -0.25, 0.25

RETURN_TO_OBS = True
HOVER_PAUSE_S = 0.2

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

def estimate_marker_once(frame_bgr, detector, K, dist, marker_len_m, target_id):
    aruco = cv2.aruco
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is None:
        return None

    ids_list = ids.ravel().tolist()
    if target_id not in ids_list:
        return None

    i = ids_list.index(target_id)
    img_pts = corners[i].reshape(4, 2).astype(np.float32)

    L = float(marker_len_m)
    obj_pts = np.array([
        [-L/2,  L/2, 0],
        [ L/2,  L/2, 0],
        [ L/2, -L/2, 0],
        [-L/2, -L/2, 0],
    ], dtype=np.float32)

    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None

    # pixel-area of marker (bigger = usually better)
    area = float(abs(cv2.contourArea(img_pts)))

    # reprojection error (smaller = better)
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    reproj_err = float(np.mean(np.linalg.norm(proj - img_pts, axis=1)))

    rejected_n = 0 if rejected is None else len(rejected)

    return dict(rvec=rvec, tvec=tvec, area=area, reproj_err=reproj_err,
                ids_list=ids_list, rejected_n=rejected_n)

def multiview_best_pose(robot, cap, detector, K, dist, marker_len_m,
                        T_hand_camera, q6_ref, target_id, log_dir, obs_views):
    best = None

    for k, q in enumerate(obs_views):
        robot.move_joints(q)          # ok (deprecated warning is fine for now)
        time.sleep(0.25)

        # flush a few frames after motion
        for _ in range(3):
            cap.grab()
        ok, frame = cap.read()
        if not ok:
            continue

        est = estimate_marker_once(frame, detector, K, dist, marker_len_m, target_id)
        if est is None:
            continue

        joints_now = robot.get_joints()
        T_base_hand = fk_base_to_hand_virtual(robot, joints_now, q6_ref=q6_ref)
        T_cam_marker = T_from_rvec_tvec(est["rvec"], est["tvec"])
        T_base_marker = T_base_hand @ T_hand_camera @ T_cam_marker

        # debug image with axes
        vis = frame.copy()
        cv2.drawFrameAxes(vis, K, dist, est["rvec"], est["tvec"], float(marker_len_m) * 1.5, 2)

        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        cv2.imwrite(str(log_dir / f"{stamp}_view{k}_raw.png"), frame)
        cv2.imwrite(str(log_dir / f"{stamp}_view{k}_vis.png"), vis)
        np.savez(str(log_dir / f"{stamp}_view{k}_T_base_marker.npz"),
                 T_base_marker=T_base_marker,
                 joints=joints_now,
                 area=est["area"],
                 reproj_err=est["reproj_err"])

        # choose best: max area, tie-break min reproj_err
        key = (-est["area"], est["reproj_err"])
        if (best is None) or (key < best["key"]):
            best = dict(key=key, est=est, T_base_marker=T_base_marker)

    return best  # None if no valid view

def make_aruco_detector():
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    params = aruco.DetectorParameters()

    # ✅ Subpixel corner refinement
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5          # try 5 or 7
    params.cornerRefinementMaxIterations = 30
    params.cornerRefinementMinAccuracy = 0.01

    return aruco.ArucoDetector(dictionary, params)


def in_bounds(x, y, z):
    return (X_MIN <= x <= X_MAX) and (Y_MIN <= y <= Y_MAX) and (z >= 0.0)

def move_hover_above(robot, x, y, z_hover, rpy_fixed):
    """
    Safe 3-step move: lift -> move XY at safe Z -> descend to hover Z.
    """
    # clamp hover height
    z_hover = max(float(z_hover), Z_MIN)
    z_travel = max(Z_SAFE, z_hover)

    if not in_bounds(x, y, z_hover):
        print(f"[MOVE][SKIP] target out of bounds: x={x:.3f}, y={y:.3f}, z={z_hover:.3f}")
        return

    roll, pitch, yaw = rpy_fixed

    # 1) lift where you are (same XY, up to z_travel)
    p = robot.get_pose()
    robot.move(pyn.PoseObject(p.x, p.y, z_travel, roll, pitch, yaw))

    # 2) travel in XY at z_travel
    robot.move(pyn.PoseObject(x, y, z_travel, roll, pitch, yaw))

    # 3) descend to hover
    robot.move(pyn.PoseObject(x, y, z_hover, roll, pitch, yaw))

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

def T_from_rvec_tvec(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T

def live_marker_detection(cap, detector, K, dist, marker_len_m,
                          robot, T_hand_camera, q6_ref, log_dir,
                          target_id=0, overlay_base_pose=False):
    aruco = cv2.aruco
    print("\n[INFO] Step5: Live ArUco detection (raw frame)")
    print("       Keys: [q]=quit, [s]=snapshot print IDs")

    last_t = time.time()
    fps = 0.0

    while True:
        ok_pnp = False
        rvec = None
        tvec = None
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

        # ==================================================
        # Step 6: Pose estimation (ID=target_id) + draw axes
        # ==================================================
        if ids is not None and target_id in ids_list:
            i = ids_list.index(target_id)  # take the first match
            img_pts = corners[i].reshape(4, 2).astype(np.float32)

            L = float(marker_len_m)
            obj_pts = np.array([
                [-L/2,  L/2, 0],
                [ L/2,  L/2, 0],
                [ L/2, -L/2, 0],
                [-L/2, -L/2, 0],
            ], dtype=np.float32)

            ok_pnp, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, K, dist,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )

            if ok_pnp:
                # Draw XYZ axes on the marker
                cv2.drawFrameAxes(vis, K, dist, rvec, tvec, L * 1.5, 2)

                # Optional: show tvec on screen (meters)
                tv = tvec.reshape(3)
                cv2.putText(vis, f"tvec=[{tv[0]:.3f},{tv[1]:.3f},{tv[2]:.3f}] m",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

        # Small “health line”
        overlay = f"fps={fps:.1f}  ids={ids_list}  rejected={rejected_n}  target={target_id}"
        cv2.putText(vis, overlay, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, overlay, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Step5+6 - ArUco detect (raw) + pose", vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("s"):
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

            # (optional) save what you currently see in the live window
            cv2.imwrite(str(log_dir / f"{stamp}_LIVE_raw.png"), frame)
            cv2.imwrite(str(log_dir / f"{stamp}_LIVE_vis.png"), vis)

            best = None
            try:
                best = multiview_best_pose(
                    robot, cap, detector, K, dist, marker_len_m,
                    T_hand_camera, q6_ref, target_id, log_dir, OBS_VIEWS
                )
            finally:
                # always go back
                if RETURN_TO_OBS:
                    robot.move_joints(PRIMARY_OBS)
                    time.sleep(0.2)

            if best is None:
                print("[SNAPSHOT] Multi-view: target not found in any view.")
                continue

            T_base_marker = best["T_base_marker"]
            p = T_base_marker[:3, 3]
            x_m, y_m, z_m = float(p[0]), float(p[1]), float(p[2])
            z_hover = z_m + Z_HOVER

            est = best["est"]
            print(f"[BEST] reproj_err={est['reproj_err']:.3f}px area={est['area']:.0f}px^2 "
                f"tvec_cam=[{est['tvec'][0][0]:.3f},{est['tvec'][1][0]:.3f},{est['tvec'][2][0]:.3f}]")

            print(f"[MOVE] base_xyz={x_m:.4f},{y_m:.4f},{z_m:.4f} -> hover_z={z_hover:.4f}")

            # optional: save the final chosen best transform too
            np.savez(str(log_dir / f"{stamp}_BEST_T_base_marker.npz"),
                     T_base_marker=T_base_marker,
                     reproj_err=est["reproj_err"],
                     area=est["area"])
            
            if DO_MOVE_ON_SNAPSHOT:
                try:
                    robot.open_gripper()
                except Exception:
                    pass

                move_hover_above(robot, x_m, y_m, z_hover, OBS_RPY)
                time.sleep(HOVER_PAUSE_S)

                if RETURN_TO_OBS:
                    robot.move_joints(PRIMARY_OBS)
                    time.sleep(0.2)

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
robot.move(pyn.JointsPosition(*OBS_JOINTS_1))
time.sleep(0.3)       # small settle time

print("[INFO] OBS_JOINTS =", np.round(OBS_JOINTS_1, 4).tolist())
print("[INFO] joints (actual)        =", np.round(robot.get_joints(), 4).tolist())

pose_live = robot.get_pose()
OBS_RPY = (pose_live.roll, pose_live.pitch, pose_live.yaw)
print("[INFO] OBS_RPY =", np.round(OBS_RPY, 4))
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
live_marker_detection(cap, detector, K, dist, MARKER_LEN_M,
                      robot, T_hand_camera, Q6_REF, LOG_DIR,
                      target_id=TARGET_ID, overlay_base_pose=False)

cap.release()
robot.close_connection()

# ==================================================
# Step 6: Pose estimation (ID=target_id) + draw axes
# ==================================================
# --> Implementation in the Step5 helper live loop!
# In def live_marker_detection(cap, detector, K, dist, marker_len_m, target_id=0)

# ==================================
# Step 7: Pose into robot base frame
# ==================================
# --> Changes in def live_marker_detection & added def T_from_rvec_tvec