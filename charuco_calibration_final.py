# ============================
# charuco_calibration_final.py
# ============================
import cv2
import numpy as np
import pyniryo as pyn
import json
import time
from datetime import datetime
from pathlib import Path


# =============================================
# Step 1: Helper - RPY helper + transforms + FK
# =============================================
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

# ----------------------------------------------------
# Helper (TEMP): FK base -> "hand" approximated by TCP
# ----------------------------------------------------
def fk_base_to_hand_temp(joints, T_base_tcp):
    """
    TEMPORARY placeholder FK.

    For now we approximate the 'hand' (wrist / joint5 frame) with the TCP frame.
    This lets us wire the hand–eye pipeline end-to-end without changing math
    elsewhere. Later, this will be replaced by a true FK up to joint 5.
    """
    # 'joints' is shape (6,), but we don't use it yet.
    return T_base_tcp.copy()

# ===============================
# Step 2: Global config + run dir
# ===============================
# Charuco board parameters:
ARUCO_DICT = cv2.aruco.DICT_6X6_250

CHARUCO_SQUARES_X = 7   # number of squares along X (columns)
CHARUCO_SQUARES_Y = 5   # number of squares along Y (rows)

SQUARE_LENGTH_M = 0.021   # [m]
MARKER_LENGTH_M = 0.011  # [m] marker is half of square size here

# Create dictionary and board (3D geometry in meters)
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
charuco_board = cv2.aruco.CharucoBoard(
    (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
    SQUARE_LENGTH_M,
    MARKER_LENGTH_M,
    aruco_dict,
)

# Detector parameters + CharucoDetector (new API)
aruco_params = cv2.aruco.DetectorParameters()
charuco_detector = cv2.aruco.CharucoDetector(
    charuco_board,
    detectorParams=aruco_params
)

print("[INFO] Charuco board configuration:")
print(f"       squares X x Y = {CHARUCO_SQUARES_X} x {CHARUCO_SQUARES_Y}")
print(f"       square_length = {SQUARE_LENGTH_M} m")
print(f"       marker_length = {MARKER_LENGTH_M} m")
print(f"       dictionary    = {ARUCO_DICT}")

# --- Run directory for this calibration session ---
BASE_CALIB_DIR = Path("calibration_data_charuco")
BASE_CALIB_DIR.mkdir(exist_ok=True)

RUN_ID = datetime.now().strftime("run_%Y%m%d_%H%M%S")
RUN_DIR = BASE_CALIB_DIR / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

print(f"[INFO] Charuco calibration run directory: {RUN_DIR}")

# Paths for all artifacts of this *final* pipeline
CHARUCO_CONFIG_PATH   = RUN_DIR / "charuco_config.json"
INTRINSICS_PATH       = RUN_DIR / "intrinsics_charuco_pinhole_final.npz"
T_CAM_BOARD_PATH      = RUN_DIR / "T_camera_board_all_final.npz"
T_BASE_TCP_PATH       = RUN_DIR / "T_base_tcp_all_final.npz"    # ^baseT_tcp
T_HAND_BASE_PATH      = RUN_DIR / "T_base_hand_all_final.npz"   # "hand" = wrist-like frame
JOINTS_PATH           = RUN_DIR / "joints_all_final.npz"
HE_HAND_EYE_PATH      = RUN_DIR / "handeye_charuco_hand_TSAI.npz"

# Save small JSON config (board + dict + run info)
config = {
    "aruco_dict": int(ARUCO_DICT),
    "charuco_squares_x": CHARUCO_SQUARES_X,
    "charuco_squares_y": CHARUCO_SQUARES_Y,
    "square_length_m": SQUARE_LENGTH_M,
    "marker_length_m": MARKER_LENGTH_M,
    "run_id": RUN_ID,
    "created_at": datetime.now().isoformat(timespec="seconds"),
}
with CHARUCO_CONFIG_PATH.open("w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

print("[INFO] Saved Charuco configuration to", CHARUCO_CONFIG_PATH)

# -------------------------
# Helper: Charuco detection
# -------------------------
def detect_charuco(frame, min_charuco=10):
    """
    Run Charuco detection on a BGR frame.
    Returns a dict with:
      ok: bool
      reason: "ok" | "no_markers" | "few_charuco"
      marker_corners, marker_ids
      charuco_corners, charuco_ids
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # One call: detect markers + interpolate Charuco corners
    charuco_corners, charuco_ids, marker_corners, marker_ids = \
        charuco_detector.detectBoard(gray)

    # 1) Check markers
    if marker_ids is None or len(marker_ids) == 0:
        return {
            "ok": False,
            "reason": "no_markers",
            "marker_corners": [],
            "marker_ids": None,
            "charuco_corners": None,
            "charuco_ids": None,
        }

    # 2) Check ChArUco corners
    if charuco_corners is None or charuco_ids is None or len(charuco_ids) < min_charuco:
        return {
            "ok": False,
            "reason": "few_charuco",
            "marker_corners": marker_corners,
            "marker_ids": marker_ids,
            "charuco_corners": charuco_corners,
            "charuco_ids": charuco_ids,
        }

    # 3) All good
    return {
        "ok": True,
        "reason": "ok",
        "marker_corners": marker_corners,
        "marker_ids": marker_ids,
        "charuco_corners": charuco_corners,
        "charuco_ids": charuco_ids,
    }

# ----------------------------------------
# Helper: draw Charuco detections on frame
# ----------------------------------------
def draw_charuco_detections(frame, det):
    """
    Return a copy of 'frame' with detected markers & Charuco corners drawn,
    plus an overlay text with counts and status.
    """
    vis = frame.copy()

    marker_corners = det.get("marker_corners", None)
    marker_ids     = det.get("marker_ids", None)
    charuco_corners = det.get("charuco_corners", None)
    charuco_ids     = det.get("charuco_ids", None)

    # Draw detected markers
    if marker_corners is not None and marker_ids is not None:
        cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)

    # Draw detected Charuco corners
    if charuco_corners is not None and charuco_ids is not None:
        cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)

    # Overlay text: counts + ok / reason
    num_m = 0 if marker_ids is None else len(marker_ids)
    num_c = 0 if charuco_ids is None else len(charuco_ids)
    status = det.get("reason", "unknown")

    text = f"markers={num_m}, corners={num_c}, status={status}"
    cv2.putText(vis, text, (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return vis

# =============================
# Step 3: Connect to Niryo Ned2
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
# Step 4: Initialize external (USB) camera
# ========================================
CAMERA_INDEX = 0
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
SHOW_PREVIEW = True

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

# ==============================
# Step 5: Live preview & capture
# ==============================
MIN_FRAMES = 20  # desired minimum number of good calibration views

T_base_tcp_list = []   # renamed: this is ^baseT_tcp (same as old "gripper")
image_paths = []

joints_all = []        # NEW: store q1..q6 per view
tcp_poses_all = []     # NEW: store [x,y,z,roll,pitch,yaw] per view

print("\n[CAPTURE] Live Charuco capture started.")
print("          Controls in the camera window:")
print("          - Press 'c' to CAPTURE current pose & frame")
print("          - Press 'q' to QUIT (after you have enough frames)\n")

while True:
    # 1) Read live frame
    ret, frame = cap.read()
    if not ret:
        print("[WARN] Failed to grab frame from camera. Stopping capture.")
        break

    # 2) Show live preview
    cv2.imshow("Charuco Live View", frame)
    key = cv2.waitKey(1) & 0xFF

    # --- Handle keys ---
    if key == ord('q'):
        # Try to quit
        if len(T_base_tcp_list) < MIN_FRAMES:
            print(f"[WARN] Only {len(T_base_tcp_list)} frames captured.")
            print(f"       Recommended at least {MIN_FRAMES}. Press 'q' again to force quit.")
            # Small trick: require a second 'q' to really quit if not enough frames
            key2 = cv2.waitKey(0) & 0xFF
            if key2 == ord('q'):
                print("[CAPTURE] Forcing quit.")
                break
            else:
                print("[CAPTURE] Continuing capture.")
                continue
        else:
            print(f"[CAPTURE] Quitting with {len(T_base_tcp_list)} frames.")
            break

    elif key == ord('c'):
        # --- Capture current pose + frame ---
        print("[CAPTURE] Capturing current pose & frame...")

        # 1) Get robot pose and build ^baseT_tcp
        pose = robot.get_pose()
        T_bt = niryo_pose_to_matrix_from_obj(pose)   # uses helper + rpy_to_rot_matrix

        # Also log joints and TCP pose values (NEW)
        q = np.array(robot.get_joints(), dtype=float)
        tcp_vec = np.array([pose.x, pose.y, pose.z,
                            pose.roll, pose.pitch, pose.yaw], dtype=float)

        # 2) Freeze the current frame as candidate
        candidate = frame.copy()

        # Run Charuco detection on candidate frame
        det = detect_charuco(candidate)
        candidate_vis = draw_charuco_detections(candidate, det)

        num_m = 0 if det["marker_ids"] is None else len(det["marker_ids"])
        num_c = 0 if det["charuco_ids"] is None else len(det["charuco_ids"])
        print(f"[CAPTURE] Charuco detection: markers={num_m}, corners={num_c}, "
              f"ok={det['ok']}, reason={det['reason']}")
        if not det["ok"]:
            print("[CAPTURE] [NOTE] Detection considered weak; usually better to discard this pose.")

        # Show the annotated (or raw) candidate
        cv2.imshow("Captured Candidate", candidate_vis)
        print("[CAPTURE] Check 'Captured Candidate' window. Press 'y' to keep, 'n' to discard.")

        while True:
            key_keep = cv2.waitKey(0) & 0xFF
            if key_keep == ord('y'):
                idx = len(T_base_tcp_list)
                img_name = f"charuco_{idx:02d}.png"
                img_path = RUN_DIR / img_name

                # Save image
                cv2.imwrite(str(img_path), candidate)

                # Store transforms & logs
                T_base_tcp_list.append(T_bt)
                image_paths.append(img_name)

                joints_all.append(q)
                tcp_poses_all.append(tcp_vec)

                print(f"[INFO] Saved frame #{idx} as {img_name}")
                break
            elif key_keep == ord('n'):
                print("[INFO] Discarded this frame/pose.")
                break
            else:
                print("[CAPTURE] Press 'y' to keep, 'n' to discard.")

        cv2.destroyWindow("Captured Candidate")

        if len(T_base_tcp_list) >= MIN_FRAMES:
            print(f"[INFO] You now have {len(T_base_tcp_list)} frames.")
            print("      You can quit with 'q' when you're satisfied.")

# After loop: save robot transforms + image paths + joint logs to disk
cv2.destroyWindow("Charuco Live View")
cap.release()
cv2.destroyAllWindows()

if len(T_base_tcp_list) > 0:
    # Stack transforms
    T_base_tcp_all = np.stack(T_base_tcp_list, axis=0)    # (N, 4, 4)
    image_names_arr = np.array(image_paths)

    # Save ^baseT_tcp + image names
    np.savez(
        T_BASE_TCP_PATH,
        T_base_tcp=T_base_tcp_all,
        image_names=image_names_arr,
    )
    print(f"[SAVE] Saved {len(T_base_tcp_list)} T_base_tcp matrices and image names "
          f"to {T_BASE_TCP_PATH}")

    # Also stack and save joints + TCP pose vectors (NEW)
    joints_all_arr = np.stack(joints_all, axis=0)         # (N, 6)
    tcp_poses_all_arr = np.stack(tcp_poses_all, axis=0)   # (N, 6)

    np.savez(
        JOINTS_PATH,
        joints_all=joints_all_arr,
        tcp_poses_all=tcp_poses_all_arr,
        image_names=image_names_arr,
    )
    print(f"[SAVE] Saved joints + TCP poses to {JOINTS_PATH}")
else:
    print("[WARN] No frames were captured. Nothing saved for this run.")

# ================================================
# Step 6: Build Charuco point sets for calibration
# ================================================
print("\n[OFFLINE] Building Charuco object/image point sets for calibration...")

if not T_BASE_TCP_PATH.exists():
    raise RuntimeError(f"[ERROR] Expected file not found: {T_BASE_TCP_PATH}")

data = np.load(T_BASE_TCP_PATH, allow_pickle=True)
image_names = data["image_names"]  # array of strings, e.g. 'charuco_00.png', ...

print(f"[OFFLINE] Found {len(image_names)} captured views in this run.")

MIN_CHARUCO_CALIB = 10  # min corners per image for intrinsics

all_obj_points = []      # list of (N_i, 1, 3) arrays
all_img_points = []      # list of (N_i, 1, 2) arrays
valid_image_names = []
image_size = None        # will be set from the first valid image

for img_name in image_names:
    img_path = RUN_DIR / str(img_name)
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[SKIP] Could not load image {img_path}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Re-run Charuco detection
    charuco_corners, charuco_ids, marker_corners, marker_ids = \
        charuco_detector.detectBoard(gray)

    num_c = 0 if charuco_ids is None else len(charuco_ids)

    if charuco_corners is None or charuco_ids is None or num_c < MIN_CHARUCO_CALIB:
        print(f"[SKIP] {img_name}: only {num_c} Charuco corners (need >= {MIN_CHARUCO_CALIB})")
        continue

    # Get corresponding 3D board points + 2D image points for these IDs
    obj_pts, img_pts = charuco_board.matchImagePoints(
        charuco_corners,
        charuco_ids
    )

    all_obj_points.append(obj_pts)
    all_img_points.append(img_pts)
    valid_image_names.append(str(img_name))

    # Remember image size for calibration (width, height)
    if image_size is None:
        h, w = gray.shape[:2]
        image_size = (w, h)

    print(f"[OK]   {img_name}: using {num_c} Charuco corners for calibration.")

print(f"\n[OFFLINE] Using {len(valid_image_names)} / {len(image_names)} views for calibration.")
if len(valid_image_names) < 5:
    print("[WARN] Very few good views detected; calibration might be unstable.")

# ============================================
# Step 7: Intrinsic calibration (pinhole model)
# ============================================
num_views = len(all_obj_points)
print("\n[CALIB] Starting intrinsic calibration (pinhole model) with Charuco points...")

if num_views < 5 or image_size is None:
    print(f"[CALIB][WARN] Only {num_views} valid views available or no image_size; "
          "skipping calibration.")
else:
    # 1) Prepare data for cv2.calibrateCamera
    obj_points_cv = [op.reshape(-1, 3).astype(np.float32) for op in all_obj_points]
    img_points_cv = [ip.reshape(-1, 2).astype(np.float32) for ip in all_img_points]

    image_size_cv = image_size  # (width, height) from Step 6

    # 2) Run calibration
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points_cv,
        img_points_cv,
        image_size_cv,
        None,
        None
    )

    print(f"[CALIB] Done. Using {num_views} views.")
    print(f"[CALIB] RMS reprojection error = {rms:.4f} pixels")
    print("[CALIB] Camera matrix K:\n", K)
    print("[CALIB] Distortion coefficients (k1,k2,p1,p2,k3,...):\n", dist.ravel())

    # 3) Save intrinsics to our final-path NPZ
    dist_row = dist.reshape(1, -1).astype(np.float64)

    np.savez(
        INTRINSICS_PATH,
        K=K.astype(np.float64),
        dist=dist_row,
        image_size=np.array(image_size_cv, dtype=np.int64),
        valid_image_names=np.array(valid_image_names),
        rms=float(rms)
    )
    print(f"[CALIB] Saved Charuco-based intrinsics to {INTRINSICS_PATH}")

    # 4) Quick undistortion sanity check on the first used image
    test_name = valid_image_names[0]
    test_img_path = RUN_DIR / test_name
    test_img = cv2.imread(str(test_img_path))
    if test_img is None:
        print(f"[CALIB][WARN] Could not load test image {test_img_path} for undistort preview.")
    else:
        h, w = test_img.shape[:2]
        newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))

        undist = cv2.undistort(test_img, K, dist, None, newK)

        cv2.imshow("Charuco - original (pinhole calib test)", test_img)
        cv2.imshow("Charuco - undistorted (pinhole calib)", undist)
        print(f"[CALIB] Showing original vs undistorted for {test_name}.")
        print("        Close the windows to continue...")
        cv2.waitKey(0)
        cv2.destroyWindow("Charuco - original (pinhole calib test)")
        cv2.destroyWindow("Charuco - undistorted (pinhole calib)")

    # ===================================================
    # Step 8: Build ^cameraT_board for each view and save
    # ===================================================
    print("\n[POSE] Building ^cameraT_board transforms for each calibration view...")

    T_camera_board_list = []

    for i, name in enumerate(valid_image_names):
        # rvecs[i], tvecs[i] describe board pose in the camera frame
        rvec = rvecs[i]
        tvec = tvecs[i].reshape(3)

        # Convert Rodrigues vector to 3x3 rotation
        R, _ = cv2.Rodrigues(rvec)

        # Build 4x4 homogeneous transform: ^cameraT_board
        T_cb = np.eye(4, dtype=np.float64)
        T_cb[:3, :3] = R
        T_cb[:3,  3] = tvec

        T_camera_board_list.append(T_cb)

        # Debug: print basic info
        dist_norm = np.linalg.norm(tvec)
        print(f"[POSE] view {i:02d}, image {name}: "
              f"|t| = {dist_norm:.3f} m, "
              f"t = [{tvec[0]: .3f}, {tvec[1]: .3f}, {tvec[2]: .3f}]")

    # Stack into a single (N,4,4) array
    T_camera_board_all = np.stack(T_camera_board_list, axis=0)  # shape (N, 4, 4)

    # Save to your new final path
    np.savez(
        T_CAM_BOARD_PATH,
        T_camera_board=T_camera_board_all,
        image_names=np.array(valid_image_names),
        K=K.astype(np.float64),
        dist=dist  # keep same shape as from calibrateCamera
    )

    print(f"[POSE] Saved ^cameraT_board for {len(valid_image_names)} views to {T_CAM_BOARD_PATH}")

    # ==============================================================
    # Step 9: Hand–eye calibration (Tsai) with TEMP hand≈tcp via FK
    # ==============================================================

    print("\n[HANDEYE] Starting hand–eye calibration (Tsai, TEMP hand≈tcp)...")

    # --- Load robot-side data: base->tcp + image_names ---
    if not T_BASE_TCP_PATH.exists():
        print(f"[HANDEYE][ERROR] Robot TCP file not found: {T_BASE_TCP_PATH}")
    else:
        tcp_npz = np.load(T_BASE_TCP_PATH, allow_pickle=True)
        print(f"[HANDEYE] {T_BASE_TCP_PATH.name} keys:", list(tcp_npz.keys()))

        if "T_base_tcp" not in tcp_npz or "image_names" not in tcp_npz:
            print("[HANDEYE][ERROR] Missing 'T_base_tcp' or 'image_names' in TCP file; aborting hand–eye.")
        elif not JOINTS_PATH.exists():
            print(f"[HANDEYE][ERROR] Joints file not found: {JOINTS_PATH}; aborting hand–eye.")
        elif not T_CAM_BOARD_PATH.exists():
            print(f"[HANDEYE][ERROR] Camera-board file not found: {T_CAM_BOARD_PATH}; aborting hand–eye.")
        else:
            # --- Load joints (robot FK input) ---
            joints_npz = np.load(JOINTS_PATH, allow_pickle=True)
            print(f"[HANDEYE] {JOINTS_PATH.name} keys:", list(joints_npz.keys()))

            if "joints_all" not in joints_npz or "image_names" not in joints_npz:
                print("[HANDEYE][ERROR] joints_all_final.npz must contain 'joints_all' and 'image_names'; aborting.")
            else:
                # Robot side: base -> tcp, joints, image names
                T_base_tcp_all = tcp_npz["T_base_tcp"]                  # (N_tcp, 4, 4)
                tcp_image_names = [str(n) for n in tcp_npz["image_names"]]

                joints_all = joints_npz["joints_all"]                  # (N_tcp, 6) expected
                joints_image_names = [str(n) for n in joints_npz["image_names"]]

                if len(tcp_image_names) != len(joints_image_names):
                    print("[HANDEYE][WARN] TCP and joints image name counts differ; will align via names.")

                # Map image name -> index on robot side
                name_to_idx_robot = {name: i for i, name in enumerate(tcp_image_names)}

                # --- TEMP: build T_base_hand_all using FK approx (hand ≈ tcp) ---
                T_base_hand_list = []
                for i, name in enumerate(tcp_image_names):
                    T_bt = T_base_tcp_all[i]
                    q_i = joints_all[i]         # (unused in TEMP FK, but kept for future real FK)
                    T_bh = fk_base_to_hand_temp(q_i, T_bt)
                    T_base_hand_list.append(T_bh)

                T_base_hand_all = np.stack(T_base_hand_list, axis=0)    # (N_tcp, 4, 4)
                np.savez(
                    T_HAND_BASE_PATH,
                    T_base_hand=T_base_hand_all,
                    image_names=np.array(tcp_image_names),
                )
                print(f"[HANDEYE] Saved TEMP T_base_hand (≈T_base_tcp) to {T_HAND_BASE_PATH}")

                # --- Load camera-side board poses: ^cameraT_board ---
                cb_npz = np.load(T_CAM_BOARD_PATH, allow_pickle=True)
                print(f"[HANDEYE] {T_CAM_BOARD_PATH.name} keys:", list(cb_npz.keys()))

                if "T_camera_board" not in cb_npz or "image_names" not in cb_npz:
                    print("[HANDEYE][ERROR] T_camera_board_all_final.npz must contain "
                          "'T_camera_board' and 'image_names'; aborting.")
                else:
                    T_camera_board_all = cb_npz["T_camera_board"]      # (N_cb, 4, 4)
                    board_image_names = [str(n) for n in cb_npz["image_names"]]

                    # ---- Pair up robot and vision data via image names ----
                    R_hand2base = []
                    t_hand2base = []
                    R_target2cam = []
                    t_target2cam = []
                    used_names = []

                    for k, img_name in enumerate(board_image_names):
                        if img_name not in name_to_idx_robot:
                            print(f"[HANDEYE][WARN] {img_name} missing on robot side; skipping.")
                            continue

                        idx_robot = name_to_idx_robot[img_name]

                        # Robot side: ^baseT_hand
                        T_bh = T_base_hand_all[idx_robot]
                        R_bh = T_bh[:3, :3]
                        t_bh = T_bh[:3, 3]

                        # Vision side: ^cameraT_board  (target = board)
                        T_cb = T_camera_board_all[k]
                        R_cb = T_cb[:3, :3]
                        t_cb = T_cb[:3, 3]

                        R_hand2base.append(R_bh)
                        t_hand2base.append(t_bh)
                        R_target2cam.append(R_cb)
                        t_target2cam.append(t_cb)
                        used_names.append(img_name)

                    num_pairs = len(used_names)
                    print(f"[HANDEYE] Using {num_pairs} pose pairs for hand–eye.")

                    if num_pairs < 3:
                        print("[HANDEYE][ERROR] Not enough pose pairs for hand–eye (need >= 3). Skipping.")
                    else:
                        R_hand2base = np.array(R_hand2base, dtype=np.float64)
                        t_hand2base = np.array(t_hand2base, dtype=np.float64)
                        R_target2cam = np.array(R_target2cam, dtype=np.float64)
                        t_target2cam = np.array(t_target2cam, dtype=np.float64)

                        # --- Call OpenCV hand–eye (Tsai) ---
                        R_cam2hand, t_cam2hand = cv2.calibrateHandEye(
                            R_hand2base, t_hand2base,
                            R_target2cam, t_target2cam,
                            method=cv2.CALIB_HAND_EYE_TSAI,
                        )

                        # According to OpenCV naming, this is cam->hand = ^handT_camera
                        T_hand_camera = np.eye(4, dtype=np.float64)
                        T_hand_camera[:3, :3] = R_cam2hand
                        T_hand_camera[:3, 3]  = t_cam2hand.reshape(3)

                        norm_t = np.linalg.norm(t_cam2hand)
                        print("[HANDEYE] ^handT_camera (Tsai, TEMP hand≈tcp):")
                        print(T_hand_camera)
                        print("[HANDEYE] Translation (m):", t_cam2hand.ravel())
                        print(f"[HANDEYE] ||t|| = {norm_t:.4f} m")

                        # Save for diagnostics
                        np.savez(
                            HE_HAND_EYE_PATH,
                            T_hand_camera=T_hand_camera,
                            R_hand_camera=T_hand_camera[:3, :3],
                            t_hand_camera=T_hand_camera[:3, 3],
                            used_image_names=np.array(used_names),
                            method="TSAI",
                            rms_intrinsics=float(rms),
                        )
                        print(f"[HANDEYE] Saved hand–eye result to {HE_HAND_EYE_PATH}")


