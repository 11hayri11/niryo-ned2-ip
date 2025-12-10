# ==================
# redoCalibration.py
# ==================
import cv2
import numpy as np
import pyniryo as pyn
import os
import pickle
import time
from datetime import datetime

# ============================================================
# Step 1: RPY helper
# (We’ll need this later for building transformation matrices)
# ============================================================
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

# =============================
# Step 2: Connect to Niryo Ned2
# =============================
robot_ip = "129.187.231.226"
print("\n[INFO] Connecting to Niryo Ned2 robot at", robot_ip, "...")

t0 = time.time()
robot = pyn.NiryoRobot(robot_ip)
robot.enable_tcp(True) # Check if TCP is set to the right one!
robot.set_learning_mode(False)

# Optional (guarded) info:
try:
    print("[INFO] Robot joints (rad):", np.round(robot.get_joints(), 3))
except Exception:
    pass

print(f"[INFO] Robot connection established successfully. (took {time.time()-t0:.2f}s)")

# ========================================
# Step 3: Initialize external (USB) camera
# ========================================
CAMERA_INDEX = 0
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

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
cv2.imshow("Camera Test Frame", frame)
cv2.waitKey(1000)           # press any key to close
cv2.destroyAllWindows()

print("\n[SETUP COMPLETE] Robot and camera are ready.\n")

# ===========================
# Step 4: Chessboard settings
# ===========================
chessboard_size = (8, 6)
square_size = 0.022  # meters

# Detection flags (help with varying lighting/contrast)
CB_FLAGS = (cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)  # Tells OpenCV to normalize lighting and adapt thresholds for reliable detection

# Prepare 3D world points for each chessboard view
objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
objp[:, :2] = np.mgrid[
    0:chessboard_size[0], 0:chessboard_size[1]
].T.reshape(-1, 2) * square_size

print(f"[INFO] Chessboard: inner corners = {chessboard_size}, square_size = {square_size} m")
print("[INFO] Example of first 5 object points:\n", np.round(objp[:5], 4)) # Or add print(objp) to see all 48 points

# ===============================
# Step 5: Storage for calibration
# ===============================
# Each calibration run gets its own timestamped folder
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir   = os.path.join("calibration_data", f"run_{timestamp}")
os.makedirs(run_dir, exist_ok=True)

print(f"[INFO] New calibration run folder created: {run_dir}")

# Core data containers
obj_points = []         # Chessboard corners in world coordinates
img_points = []         # Detected corners in image coordinates
eef_poses = []          # TCP poses from the robot
captured_images = []    # Store images
fail_log = []           # Optional: record which poses failed to detect the board

# ==========================
# Step 6: Define robot poses (hardcoded baseline – 25 joint-space poses, in radians)
# ==========================
robot_poses = [
    [-0.0022,0.4994,-1.1869,-0.0091,-0.4188,-0.0259],    # pose-1
    [-0.0022,0.4994,-1.34,-0.0167,0.029,-0.0597],        # pose-2
    [-0.4801,0.4994,-1.1309,0.3268,-0.7164,-0.0551],     # pose-3
    [0.2336,0.5115,-1.1688,-0.3358,-0.471,0.4357],       # pose-4
   # [0.4725,0.4539,-1.34,-0.0351,-0.1458,0.632],         # pose-5
    [0.6232,0.4569,-1.2566,-0.6595,-0.7609,1.3806],      # pose-6
    [0.0601,0.4675,-1.037,0.2961,-0.6167,0.451],         # pose-7
   # [-0.6231,0.4675,-0.9233,0.4111,-0.9971,-0.4662],     # pose-8
    [-0.8773,0.4691,-0.7112,0.5738,-1.5018,0.1488],      # pose-9
    [0.3675,-0.1156,-0.7779,0.3314,-0.3283,0.6934],      # pose-10
   # [0.2884,-0.405,-0.2916,0.2608,-0.8775,0.4664],       # pose-11
    [-0.3416,-0.2459,-0.5007,0.02,-0.7394,-0.9003],      # pose-12
    [-0.0205,-0.1111,-0.5173,0.0951,-0.9327,-0.0459],    # pose-13
    [-0.0205,0.1858,-1.34,0.0369,0.3281,-0.052],         # pose-14
   # [-0.0205,0.4797,-0.7112,0.0307,-1.1014,-0.0489],     # pose-15
   # [-0.3127,0.4751,-0.7567,1.4573,-0.5584,1.5417],      # pose-16
    [0.8271,0.4009,-0.9552,-1.0138,-1.3561,0.0583],      # pose-17
    [-0.0646,0.5918,-0.5325,0.0138,-1.2871,-0.121],      # pose-18
    [-0.0661,0.61,-0.2462,-0.1256,-1.5954,-0.2545],      # pose-19
   # [-0.8286,0.61,-0.6779,0.3989,-1.5724,-0.1824],       # pose-20
    [1.0189,0.4978,-0.5507,-0.2928,-1.689,0.2593],       # pose-21
    [0.0342,-0.4792,0.2234,-0.0351,-1.5356,0.0691],      # pose-22
    [-0.0585,-0.3459,-0.2462,-0.098,-1.0677,-0.0259],    # pose-23
    [0.0342,0.463,-1.3233,-0.052,-0.0246,0.0967],        # pose-24
    [-0.0129,-0.311,-0.387,-0.1088,-0.8269,0.0246]       # pose-25
]

# Quick sanity checks/prints
print(f"[INFO] Loaded {len(robot_poses)} robot poses (joint space, radians).")
if robot_poses:
    print("[INFO] Sample pose [j1..j6] (rad):", np.round(robot_poses[0], 3))
    assert all(len(p) == 6 for p in robot_poses), "[ERROR] Each pose must have 6 joint values."

# =========================================
# Step 7: Capture images and detect corners
# =========================================
print("\n[STEP 7] Starting image capture & corner detection...\n")

for idx, pose in enumerate(robot_poses, start=1):
    print(f"[MOVE] Pose {idx}/{len(robot_poses)}")
    robot.move(pyn.JointsPosition(*pose)) # *pose - builds a "joint configuration"
    time.sleep(0.9) # wait for robot + camera to stabilize

    pose_live = robot.get_pose() # use pose_live.x, pose_live.y, pose_live.z, pose_live.roll, pose_live.pitch, pose_live.yaw

    # Optional debug: print the TCP for first and last pose
    if idx == 1 or idx == len(robot_poses):
        print(
            f"  [DEBUG] Live TCP after move: "
            f"x={pose_live.x:.4f}, y={pose_live.y:.4f}, z={pose_live.z:.4f}, "
            f"r={pose_live.roll:.4f}, p={pose_live.pitch:.4f}, yaw={pose_live.yaw:.4f}"
        )

    # Capture frame from external camera
    ret, frame = cap.read()
    if not ret:
        print(f"[WARN] Pose {idx}: Failed to grab frame — skipping.")
        fail_log.append(idx)
        continue

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # find the chess board corners with our FLAGS
    found, corners = cv2.findChessboardCorners(
        gray,
        chessboard_size,  # (8,6)
        CB_FLAGS          # ADAPTIVE_THRESH | NORMALIZE_IMAGE
    )

    if not found or corners is None:
        print(f"  ↳ Chessboard found: False")
        fail_log.append(idx)
        continue

    print(f"  ↳ Chessboard found: {found}")

    # Refine corner locations to subpixel accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria) # Now corners_refined is what I append as img_points
    
    # build TCP pose array (later for Step 9/10)
    tcp_vec = np.array([
        pose_live.x,
        pose_live.y,
        pose_live.z,
        pose_live.roll,
        pose_live.pitch,
        pose_live.yaw
    ], dtype=np.float64)

    # append to calibration container
    obj_points.append(objp.reshape(-1, 1, 3).astype(np.float32)) # 3D: (N, 1, 3)
    img_points.append(corners_refined.astype(np.float32))        # 2D: (N, 1, 2)
    eef_poses.append(tcp_vec)                                    # 6D: (x,y,z,roll,pitch,yaw)

    # Draw corners for visualization and store debug image
    dbg = frame.copy()
    cv2.drawChessboardCorners(dbg, chessboard_size, corners_refined, True)
    captured_images.append(dbg)
    cv2.imshow("Chessboard detection", dbg)
    cv2.waitKey(1)

cv2.destroyAllWindows()

print(f"\n[INFO] Image capture complete.")
print(f"[INFO] Successful detections: {len(img_points)}/{len(robot_poses)}")
if fail_log:
    print(f"[WARN] Failed poses: {fail_log}\n")

assert len(obj_points) == len(img_points) == len(eef_poses), \
    "[ERROR] Mismatched lengths: obj_points, img_points, eef_poses"

# ====================================
# Step 8: Intrinsic camera calibration (fisheye model)
# ====================================
print("\n[STEP 8] Starting fisheye camera calibration...")

# Sanity checks
num_sets = len(img_points)
if num_sets < 10:
    raise RuntimeError(f"[ERROR] Not enough detections for calibration: {num_sets} found, need >= 10.")

# Image size from the last captured image (or any captured frame)
if len(captured_images) == 0:
    raise RuntimeError("[ERROR] No captured images stored. Cannot determine image size.")
H, W = captured_images[0].shape[:2]
img_size = (W, H)
print(f"[INFO] Calibrating with {num_sets} images at resolution {W}×{H}")

# Ensure fisheye-preferred dtypes/shapes
obj_points_fe = [op.astype(np.float64) for op in obj_points]          # (N,1,3)
img_points_fe = [ip.astype(np.float64) for ip in img_points]          # (N,1,2)

# Initialize outputs
K  = np.zeros((3, 3), dtype=np.float64)
D  = np.zeros((4, 1), dtype=np.float64)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC  # good default

rms, K, D, rvecs_target2cam, tvecs_target2cam = cv2.fisheye.calibrate(
    objectPoints=obj_points_fe,
    imagePoints=img_points_fe,
    image_size=img_size,
    K=K,
    D=D,
    rvecs=None,
    tvecs=None,
    flags=flags,
    criteria=criteria
)

print("\n[RESULT] Fisheye calibration complete:")
print(f"  - RMS reprojection error: {rms:.6f}")
print(f"  - K (intrinsic matrix):\n{np.array2string(K, formatter={'float_kind':lambda x: f'{x: .6f}'})}")
print(f"  - D (distortion coeffs): {D.ravel()}")

# Compute mean reprojection error per image
def mean_fisheye_reproj_error(obj_pts, img_pts, rvecs, tvecs, K, D):
    total = 0.0
    for op, ip, rv, tv in zip(obj_pts, img_pts, rvecs, tvecs):
        proj, _ = cv2.fisheye.projectPoints(op, rv, tv, K, D)  # (N,1,2)
        err = np.linalg.norm(proj.reshape(-1,2) - ip.reshape(-1,2), axis=1).mean()
        total += err
    return total / len(obj_pts)

mean_err = mean_fisheye_reproj_error(obj_points_fe, img_points_fe, rvecs_target2cam, tvecs_target2cam, K, D)
print(f"  - Mean reprojection error per image: {mean_err:.4f} px")

# Optional - Quick undistortion preview
try:
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K, img_size, cv2.CV_16SC2
    )
    preview = cv2.remap(captured_images[0], map1, map2, interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Undistorted preview (fisheye)", preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except Exception as e:
    print(f"[WARN] Undistortion preview skipped: {e}")

# Save intrinsics now (to this run's folder)
np.savez(os.path.join(run_dir, "intrinsics_fisheye.npz"), K=K, D=D)
print(f"[INFO] Saved intrinsics to: {os.path.join(run_dir, 'intrinsics_fisheye.npz')}")

# ======================================================
# Step 9: Convert Niryo poses to transformation matrices
# ======================================================
print("\n[STEP 9] Converting TCP poses to homogeneous transformation matrices...")

def niryo_pose_to_matrix(pose):
    """
    Convert a Niryo TCP pose [x, y, z, roll, pitch, yaw] expressed in the BASE frame
    into the 4x4 homogeneous transform ^baseT_gripper (base <- gripper).
    """
    x, y, z, roll, pitch, yaw = pose
    R = rpy_to_rot_matrix(np.array([roll, pitch, yaw]))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = [x, y, z]
    return T

# Convert all TCP poses collected during calibration
# Each eef_pose is [x, y, z, roll, pitch, yaw] of the gripper in the *base* frame
T_base_gripper = [niryo_pose_to_matrix(pose) for pose in eef_poses]

# Split into rotation and translation parts for OpenCV's calibrateHandEye
# OpenCV calls these R_gripper2base / t_gripper2base, but they are actually ^baseR_gripper, ^baset_gripper
R_base_gripper = [T[:3, :3]              for T in T_base_gripper]
t_base_gripper = [T[:3, 3].reshape(3, 1) for T in T_base_gripper]

print(f"[INFO] Converted {len(T_base_gripper)} poses into transformation matrices.")
print("[INFO] Sample translation (m):", np.round(t_base_gripper[0].ravel(), 4))

# =============================
# Step 10: Hand–Eye calibration (TSAI)
# =============================
print("\n[STEP 10] Hand–Eye calibration (Tsai & Lenz)…")

# OpenCV expects lists of rotations/translations (robot & vision)
# - Robot side:  ^baseR_gripper, ^baset_gripper   (we already built these as R_gripper2base / t_gripper2base)
# - Vision side: ^camR_target,  ^camt_target      (rvecs_target2cam / tvecs_target2cam from fisheye.calibrate)

R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
    R_base_gripper,
    t_base_gripper,
    rvecs_target2cam,
    tvecs_target2cam,
    method=cv2.CALIB_HAND_EYE_TSAI
)

print("[RESULT] ^gripperR_camera:\n", np.array2string(R_cam2gripper, precision=6, suppress_small=True))
print("[RESULT] ^grippert_camera (m):", np.round(t_cam2gripper.ravel(), 6))
print("[RESULT] ^grippert_camera (cm):", np.round(t_cam2gripper.ravel() * 100.0, 2))

# Quick orthonormality sanity check on R
RtR = R_cam2gripper.T @ R_cam2gripper
orth_err = np.linalg.norm(RtR - np.eye(3))
det_R   = np.linalg.det(R_cam2gripper)
print(f"[CHECK] Orthonormality ||RᵀR - I|| = {orth_err:.2e}   det(R) = {det_R:.6f}")

t_norm = np.linalg.norm(t_cam2gripper)
print(f"[CHECK] ||^grippert_camera|| = {t_norm:.4f} m (~{t_norm*100:.1f} cm)")

# (Optional) Build the 4x4 ^gripperT_camera for convenience
T_gripper_camera = np.eye(4)
T_gripper_camera[:3, :3] = R_cam2gripper
T_gripper_camera[:3,  3] = t_cam2gripper.ravel()

# Compute inverse: camera → gripper
T_camera_gripper = np.linalg.inv(T_gripper_camera) # If I ever mount the camera elswhere but to the arm of the robot

# Save both directions
np.savez(os.path.join(run_dir, "handeye_TSAI.npz"),
         R=R_cam2gripper, t=t_cam2gripper,
         T_gripper_camera=T_gripper_camera,
         T_camera_gripper=T_camera_gripper)

print(f"[INFO] Saved both transforms to: {os.path.join(run_dir, 'handeye_TSAI.npz')}")

print("[INFO] Done — Hand–Eye calibration complete ✅")

# =================================================
# Step 11: Save all calibration artifacts + cleanup
# =================================================
print("\n[STEP 11] Saving artifacts and cleaning up…")

# A) Save annotated images for auditability
img_count = 0
for i, img in enumerate(captured_images):
    out_path = os.path.join(run_dir, f"image_{i:02d}.png")
    ok = cv2.imwrite(out_path, img)
    if ok:
        img_count += 1
print(f"[INFO] Saved {img_count} annotated images to {run_dir}")

# B) Save correspondences and robot poses (pickle bundle)
bundle_path = os.path.join(run_dir, "points_and_poses.pkl")
with open(bundle_path, "wb") as f:
    pickle.dump(
        {
            "chessboard_size": chessboard_size,
            "square_size_m": square_size,
            "obj_points": obj_points,             # list of (N,1,3)
            "img_points": img_points,             # list of (N,1,2)
            "eef_poses": eef_poses,               # TCP poses [x,y,z,r,p,y]
            "joint_poses": robot_poses,           # the 26 joint-space poses used
            "rvecs_target2cam": rvecs_target2cam, # per-image extrinsics (fisheye)
            "tvecs_target2cam": tvecs_target2cam,
            "fail_log": fail_log,                 # indices that failed detection (if any)
        },
        f,
    )
print(f"[INFO] Saved points & poses bundle: {bundle_path}")

# Quick summary line for your log
print(
    f"[SUMMARY] Sets: {len(img_points)}/{len(robot_poses)} | "
    f"Run folder: {run_dir}"
)

# --- Move robot to home pose (joint space) ---
try:
    HOME_JOINTS = [-0.0007, 0.4994, -1.2506, 0.0, 0.0014, 0.0062]
    print("[INFO] Moving robot to home pose...")
    robot.set_learning_mode(False)
    robot.move(pyn.JointsPosition(*HOME_JOINTS))
    print("[INFO] Reached home pose.")
except Exception as e:
    print(f"[WARN] Could not move to home pose: {e}")

# C) Cleanup resources
try:
    cap.release()
except Exception:
    pass

try:
    cv2.destroyAllWindows()
except Exception:
    pass

try:
    robot.close_connection()
except Exception:
    pass

print("[DONE] Calibration session finished. Resources released.\n")