# ======================
# charuco_calibration.py
# ======================
import cv2
import numpy as np
import pyniryo as pyn
import os
import json
import time
from datetime import datetime
from pathlib import Path

# ============================================================
# Step 1: RPY helper + transforms + drawing helper
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

def draw_charuco_detections(frame, det):
    """
    Draw detected ArUco markers and ChArUco corners on a copy of frame.
    Also overlay counts of markers/corners.
    """
    vis = frame.copy()

    marker_corners = det["marker_corners"]
    marker_ids = det["marker_ids"]
    charuco_corners = det["charuco_corners"]
    charuco_ids = det["charuco_ids"]

    # Draw markers (if any)
    if marker_ids is not None and len(marker_ids) > 0:
        cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)

    # Draw Charuco corners (if any)
    if charuco_corners is not None and charuco_ids is not None and len(charuco_ids) > 0:
        # If OpenCV has a helper, use it:
        if hasattr(cv2.aruco, "drawDetectedCornersCharuco"):
            cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)
        else:
            # Fallback: draw small circles manually
            for pt in charuco_corners:
                u, v = int(pt[0][0]), int(pt[0][1])
                cv2.circle(vis, (u, v), 4, (0, 255, 0), -1)

    # Overlay text: number of markers and charuco corners
    num_m = 0 if marker_ids is None else len(marker_ids)
    num_c = 0 if charuco_ids is None else len(charuco_ids)
    text = f"M={num_m}  C={num_c}  ok={det['ok']}"

    cv2.putText(
        vis,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    print("[DEBUG] draw_charuco_detections: markers",
      0 if det["marker_ids"] is None else len(det["marker_ids"]),
      "corners",
      0 if det["charuco_ids"] is None else len(det["charuco_ids"]))

    return vis

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

# =============================
# Step 4: Charuco + run configs
# =============================
# Charuco board parameters:
ARUCO_DICT = cv2.aruco.DICT_6X6_250

CHARUCO_SQUARES_X = 7   # number of squares along X (columns)
CHARUCO_SQUARES_Y = 5   # number of squares along Y (rows)

SQUARE_LENGTH_M = 0.021   # [m]  <-- TODO: replace with measured value later
MARKER_LENGTH_M = 0.011  # [m] marker is half of square size here

# Create the Charuco dictionary and board object.
# This board object encodes the 3D geometry of the corners in *meters*.
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
charuco_board = cv2.aruco.CharucoBoard(
    (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
    SQUARE_LENGTH_M,
    MARKER_LENGTH_M,
    aruco_dict,
)

# Create an ArUco detector instance (new OpenCV API)
# --- ArUco / ChArUco detection parameters ---
aruco_params = cv2.aruco.DetectorParameters()
""" aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params) """

# NEW: CharucoDetector (does markers + charuco corners in one go)
charuco_detector = cv2.aruco.CharucoDetector(
    charuco_board,              # board geometry
    # you can leave charucoParams default
    detectorParams=aruco_params # reuse your marker detector params
)

print("[INFO] Charuco board configuration:")
print(f"       squares X x Y = {CHARUCO_SQUARES_X} x {CHARUCO_SQUARES_Y}")
print(f"       square_length = {SQUARE_LENGTH_M} m")
print(f"       marker_length = {MARKER_LENGTH_M} m")
print(f"       dictionary    = {ARUCO_DICT}")

# Run directory for this calibration session
BASE_CALIB_DIR = Path("calibration_data_charuco")
RUN_ID = datetime.now().strftime("run_%Y%m%d_%H%M%S")
RUN_DIR = BASE_CALIB_DIR / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

print(f"[INFO] Charuco calibration run directory: {RUN_DIR}")

# Save a small JSON config so we always know which board / params were used.
config = {
    "aruco_dict": int(ARUCO_DICT),
    "charuco_squares_x": CHARUCO_SQUARES_X,
    "charuco_squares_y": CHARUCO_SQUARES_Y,
    "square_length_m": SQUARE_LENGTH_M,
    "marker_length_m": MARKER_LENGTH_M,
    "run_id": RUN_ID,
    "created_at": datetime.now().isoformat(timespec="seconds"),
}

with open(RUN_DIR / "charuco_config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

print("[INFO] Saved Charuco configuration to", RUN_DIR / "charuco_config.json")

def detect_charuco(frame, min_charuco=10):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # One call: detect markers + interpolate charuco corners
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

# ==============================
# Step 5: Live preview & capture
# ==============================
MIN_FRAMES = 20  # desired minimum number of good calibration views

T_base_gripper_list = []
image_paths = []

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
        if len(T_base_gripper_list) < MIN_FRAMES:
            print(f"[WARN] Only {len(T_base_gripper_list)} frames captured.")
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
            print(f"[CAPTURE] Quitting with {len(T_base_gripper_list)} frames.")
            break

    elif key == ord('c'):
        # --- Capture current pose + frame ---
        print("[CAPTURE] Capturing current pose & frame...")

        # 1) Get robot pose and build T_base_gripper
        pose = robot.get_pose()
        pose_list = [pose.x, pose.y, pose.z, pose.roll, pose.pitch, pose.yaw]
        T_bg = niryo_pose_to_matrix(pose_list)

        # 2) Freeze the current frame as candidate
        candidate = frame.copy()

        # Run Charuco detection on candidate frame
        det = detect_charuco(candidate)
        candidate_vis = draw_charuco_detections(candidate, det)

        num_m = 0 if det["marker_ids"] is None else len(det["marker_ids"])
        num_c = 0 if det["charuco_ids"] is None else len(det["charuco_ids"])
        print(f"[CAPTURE] Charuco detection: markers={num_m}, corners={num_c}, ok={det['ok']}, reason={det['reason']}")
        if not det["ok"]:
            print("[CAPTURE] [NOTE] Detection considered weak; usually better to discard this pose.")
        # Show the annoted candidate
        cv2.imshow("Captured Candidate", candidate_vis)
        print("[CAPTURE] Check 'Captured Candidate' window. Press 'y' to keep, 'n' to discard.")

        while True:
            key_keep = cv2.waitKey(0) & 0xFF
            if key_keep == ord('y'):
                idx = len(T_base_gripper_list)
                img_name = f"charuco_{idx:02d}.png"
                img_path = RUN_DIR / img_name

                cv2.imwrite(str(img_path), candidate)
                T_base_gripper_list.append(T_bg)
                image_paths.append(img_name)

                print(f"[INFO] Saved frame #{idx} as {img_name}")
                break
            elif key_keep == ord('n'):
                print("[INFO] Discarded this frame/pose.")
                break
            else:
                print("[CAPTURE] Press 'y' to keep, 'n' to discard.")

        cv2.destroyWindow("Captured Candidate")

        if len(T_base_gripper_list) >= MIN_FRAMES:
            print(f"[INFO] You now have {len(T_base_gripper_list)} frames.")
            print("      You can quit with 'q' when you're satisfied.")

# After loop: save robot transforms + image paths to disk
cv2.destroyWindow("Charuco Live View")
cap.release()
cv2.destroyAllWindows()

if len(T_base_gripper_list) > 0:
    T_base_gripper_all = np.stack(T_base_gripper_list, axis=0)  # shape (N, 4, 4)
    np.savez(RUN_DIR / "T_base_gripper_all.npz",
             T_base_gripper=T_base_gripper_all,
             image_paths=np.array(image_paths))
    print(f"[SAVE] Saved {len(T_base_gripper_list)} T_base_gripper matrices and image names "
          f"to {RUN_DIR / 'T_base_gripper_all.npz'}")
else:
    print("[WARN] No frames were captured. Nothing saved for this run.")

# ================================================
# Step 6: Build Charuco point sets for calibration
# ================================================
print("\n[OFFLINE] Building Charuco object/image point sets for calibration...")

npz_path = RUN_DIR / "T_base_gripper_all.npz"
if not npz_path.exists():
    raise RuntimeError(f"[ERROR] Expected file not found: {npz_path}")

data = np.load(npz_path, allow_pickle=True)
image_paths = data["image_paths"] # array of strings (e.g. 'charuco_00.png', ...)

print(f"[OFFLINE] Found {len(image_paths)} captured views in this run.")

# Reusing charuco_board and charuco_detector from Step 4.
MIN_CHARUCO_CALIB = 10

all_obj_points = []   # list of (N_i, 1, 3) arrays
all_img_points = [] # list of (N_i, 1, 3) arrays
valid_image_names = []

for img_name in image_paths:
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

    obj_pts, img_pts = charuco_board.matchImagePoints(
        charuco_corners,
        charuco_ids
    )

    all_obj_points.append(obj_pts)
    all_img_points.append(img_pts)
    valid_image_names.append(str(img_name))

    print(f"[OK]   {img_name}: using {num_c} Charuco corners for calibration.")

print(f"\n[OFFLINE] Using {len(valid_image_names)} / {len(image_paths)} views for calibration.")

if len(valid_image_names) < 5:
    print("[WARN] Very few good views detected; calibration might be unstable.")


# =====================================================================
# Step 7: Intrinsic calibration (pinhole model) + quick undistort check
# =====================================================================
print("\n[CALIB] Starting intrinsic calibration (pinhole model) with Charuco points...")

num_views = len(all_obj_points)
if num_views < 5:
    print(f"[CALIB][WARN] Only {num_views} valid views available; skipping calibration.")
else:
    # 1) Prepare data for cv2.calibrateCamera
    obj_points_cv = [op.reshape(-1, 3) for op in all_obj_points]  # list of (Ni,3)
    img_points_cv = [ip.reshape(-1, 2) for ip in all_img_points]  # list of (Ni,2)

    image_size = (FRAME_WIDTH, FRAME_HEIGHT)  # (width, height)

    # 2) Run calibration
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points_cv,
        img_points_cv,
        image_size,
        None,
        None
    )

    print(f"[CALIB] Done. Using {num_views} views.")
    print(f"[CALIB] RMS reprojection error = {ret:.4f} pixels")
    print("[CALIB] Camera matrix K:\n", K)
    print("[CALIB] Distortion coefficients (k1,k2,p1,p2,k3,...):\n", dist.ravel())

    # 3) Save intrinsics
    intr_path = RUN_DIR / "intrinsics_charuco_pinhole.npz"
    np.savez(
        intr_path,
        K=K,
        dist=dist,
        image_size=np.array(image_size),
        valid_image_names=np.array(valid_image_names),
        rms=ret
    )
    print(f"[CALIB] Saved Charuco-based intrinsics to {intr_path}")

    # 4) Quick undistortion sanity check on the first used image
    test_name = valid_image_names[0]
    test_img_path = RUN_DIR / test_name
    test_img = cv2.imread(str(test_img_path))
    if test_img is None:
        print(f"[CALIB][WARN] Could not load test image {test_img_path} for undistort preview.")
    else:
        h, w = test_img.shape[:2]

        # Get an "optimal" new camera matrix (can adjust alpha if needed)
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
        T_cb = np.eye(4)
        T_cb[:3, :3] = R
        T_cb[:3, 3]  = tvec

        T_camera_board_list.append(T_cb)

        # Debug: print basic info
        dist_norm = np.linalg.norm(tvec)
        print(f"[POSE] view {i:02d}, image {name}: "
              f"|t| = {dist_norm:.3f} m, "
              f"t = [{tvec[0]: .3f}, {tvec[1]: .3f}, {tvec[2]: .3f}]")

    # Stack into a single (N,4,4) array
    T_camera_board_all = np.stack(T_camera_board_list, axis=0)  # shape (N, 4, 4)

    pose_path = RUN_DIR / "T_camera_board_all.npz"
    np.savez(
        pose_path,
        T_camera_board=T_camera_board_all,
        image_names=np.array(valid_image_names),
        K=K,
        dist=dist
    )

    print(f"[POSE] Saved ^cameraT_board for {len(valid_image_names)} views to {pose_path}")

    # =====================================================
    # Step 9: Hand–eye calibration (Tsai) with Charuco data
    # =====================================================
    print("\n[HANDEYE] Starting hand–eye calibration (Tsai) with Charuco data...")

    # Map image name -> index in T_base_gripper_all
    name_to_idx_bg = {str(name): i for i, name in enumerate(image_paths)}

    R_gripper2base = []
    t_gripper2base = []
    R_target2cam  = []
    t_target2cam  = []
    used_names    = []

    for k, img_name in enumerate(valid_image_names):
        img_name = str(img_name)
        if img_name not in name_to_idx_bg:
            print(f"[HANDEYE][WARN] {img_name} not found in T_base_gripper_all; skipping.")
            continue

        idx_bg = name_to_idx_bg[img_name]

        # --- Robot part: ^baseT_gripper (we already stored it this way)
        T_bg = T_base_gripper_all[idx_bg]      # ^baseT_gripper
        R_bg = T_bg[:3, :3]                    # ^baseR_gripper
        t_bg = T_bg[:3, 3]                     # ^baset_gripper

        # --- Vision part: board -> camera from calibrateCamera
        # rvecs/tvecs[k] map points in board frame to camera frame:
        # X_cam = R * X_board + t
        R_tc, _ = cv2.Rodrigues(rvecs[k])      # ^cameraR_board
        t_tc = tvecs[k].reshape(3)             # ^camerat_board

        R_gripper2base.append(R_bg)            # gripper -> base (as OpenCV names it)
        t_gripper2base.append(t_bg)
        R_target2cam.append(R_tc)              # target (board) -> cam
        t_target2cam.append(t_tc)
        used_names.append(img_name)

    R_gripper2base = np.array(R_gripper2base, dtype=np.float64)
    t_gripper2base = np.array(t_gripper2base, dtype=np.float64)
    R_target2cam   = np.array(R_target2cam,   dtype=np.float64)
    t_target2cam   = np.array(t_target2cam,   dtype=np.float64)

    print(f"[HANDEYE] Using {len(used_names)} pose pairs for hand–eye calibration.")

    # --- Call OpenCV hand–eye (Tsai)
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam,  t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI,
    )

    # According to OpenCV naming, this is "cam2gripper" -> transform from camera frame to gripper frame.
    # In your notation that is exactly ^gripperT_camera.
    T_gripper_camera = np.eye(4, dtype=np.float64)
    T_gripper_camera[:3, :3] = R_cam2gripper
    T_gripper_camera[:3, 3]  = t_cam2gripper.reshape(3)

    norm_t = np.linalg.norm(t_cam2gripper)
    print("[HANDEYE] ^gripperT_camera (Tsai):")
    print(T_gripper_camera)
    print("[HANDEYE] Translation (m):", t_cam2gripper.ravel())
    print(f"[HANDEYE] ||t|| = {norm_t:.4f} m")

    # Save for later use and diagnostics
    handeye_path = RUN_DIR / "handeye_charuco_TSAI.npz"
    np.savez(
        handeye_path,
        T_gripper_camera=T_gripper_camera,
        R_gripper_camera=T_gripper_camera[:3, :3],
        t_gripper_camera=T_gripper_camera[:3, 3],
        used_image_names=np.array(used_names),
        method="TSAI",
        rms_intrinsics=ret,
    )
    print(f"[HANDEYE] Saved hand–eye result to {handeye_path}")

    # ============================================================
    # Step 9: Hand–Eye calibration (Tsai) using Charuco extrinsics
    # ============================================================
    """ print("\n[HANDEYE] Starting hand–eye calibration (Tsai) with Charuco data...")

    handeye_ok = True
    # Load robot poses (base <- gripper) for this run
    try:
        robot_npz = np.load(RUN_DIR / "T_base_gripper_all.npz", allow_pickle=True)
    except Exception as e:
        print(f"[HANDEYE][ERR] Could not load T_base_gripper_all.npz: {e}")
        handeye_ok = False

    # Load camera <- board transforms we just saved
    try:
        board_npz = np.load(RUN_DIR / "T_camera_board_all.npz", allow_pickle=True)
    except Exception as e:
        print(f"[HANDEYE][ERR] Could not load T_camera_board_all.npz: {e}")
        handeye_ok = False

    if handeye_ok:
        T_base_gripper_all = robot_npz["T_base_gripper"]   # (N_capture, 4, 4)
        img_paths_bg       = robot_npz["image_paths"]      # (N_capture,)

        T_camera_board_all = board_npz["T_camera_board"]   # (N_valid, 4, 4)
        img_names_cb       = board_npz["image_names"]      # (N_valid,)

        # Map filename -> index in robot poses
        index_by_name = {str(name): idx for idx, name in enumerate(img_paths_bg)}

        R_gripper2base =  []
        t_gripper2base = []
        R_target2cam = []
        t_target2cam = []
        used_names = []

        for i, name in enumerate(img_names_cb):
            name_str = str(name)

            if name_str not in index_by_name:
                print(f"[HANDEYE][WARN] {name_str} not found in robot pose list; skipping.")
                continue

            idx_bg = index_by_name[name_str]

            T_bg = T_base_gripper_all[idx_bg]   # ^baseT_gripper
            T_cb = T_camera_board_all[i]        # ^cameraT_board

            # Robot side: gripper -> base  (we already store base <- gripper)
            R_g2b = T_bg[:3, :3]
            t_g2b = T_bg[:3, 3]

            # Target side: board -> camera
            R_t2c = T_cb[:3, :3]
            t_t2c = T_cb[:3, 3]

            R_gripper2base.append(R_g2b)
            t_gripper2base.append(t_g2b.reshape(3, 1))
            R_target2cam.append(R_t2c)
            t_target2cam.append(t_t2c.reshape(3, 1))
            used_names.append(name_str)

        n_pairs = len(R_gripper2base)
        if n_pairs < 4:
            print(f"[HANDEYE][WARN] Only {n_pairs} valid pose pairs, need at least 4. "
                  f"Skipping hand–eye.")
        else:
            print(f"[HANDEYE] Using {n_pairs} pose pairs for hand–eye calibration.")

            R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                R_gripper2base, t_gripper2base,
                R_target2cam,   t_target2cam,
                method=cv2.CALIB_HAND_EYE_TSAI
            )

            # Build ^gripperT_camera (g <- c) from the output
            T_gripper_camera = np.eye(4)
            T_gripper_camera[:3, :3] = R_cam2gripper
            T_gripper_camera[:3, 3]  = t_cam2gripper.reshape(3)

            t_gc = T_gripper_camera[:3, 3]
            print("[HANDEYE] ^gripperT_camera (Tsai):")
            print(T_gripper_camera)
            print(f"[HANDEYE] Translation (m): {t_gc}")
            print(f"[HANDEYE] ||t|| = {np.linalg.norm(t_gc):.4f} m")

            out_path = RUN_DIR / "handeye_charuco_TSAI.npz"
            np.savez(
                out_path,
                T_gripper_camera=T_gripper_camera,
                R_gripper_camera=R_cam2gripper,
                t_gripper_camera=t_cam2gripper,
                used_image_names=np.array(used_names),
                method="Tsai",
                rms_intrinsics=float(ret)
            )
            print(f"[HANDEYE] Saved hand–eye result to {out_path}") """