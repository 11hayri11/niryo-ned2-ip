# ----------------
# Step 0 - Imports
# ----------------
import os
import json
import numpy as np
import pyniryo as pyn
import cv2
from cv2 import aruco
import cv2
import time

# -----------------------
# Step 1 - helper, config
# -----------------------
# Helper: RPY -> R (ZYX / yaw→pitch→roll)
def rpy_to_rot_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    Rx = np.array([[1,0,0],
                   [0,np.cos(roll),-np.sin(roll)],
                   [0,np.sin(roll), np.cos(roll)]], dtype=float)
    Ry = np.array([[ np.cos(pitch),0,np.sin(pitch)],
                   [0,             1,            0],
                   [-np.sin(pitch),0,np.cos(pitch)]], dtype=float)
    Rz = np.array([[np.cos(yaw),-np.sin(yaw),0],
                   [np.sin(yaw), np.cos(yaw),0],
                   [0,           0,          1]], dtype=float)
    return Rz @ Ry @ Rx

def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """ Build 4x4 homogeneous transform from R(3x3), t(3,). """
    T = np.eye(4, dtype=float)
    T[:3,:3] = R
    T[:3, 3] = t.reshape(3,)
    return T

# Config / paths
run_dir = r"calibration_data_charuco\run_20251208_122545"

intrinsics_path = os.path.join(run_dir, "intrinsics_charuco_pinhole.npz")
handeye_path = os.path.join(run_dir, "handeye_charuco_TSAI.npz")

# Optinal: See which path it uses!
print(f"[CAL] Using run_dir: {run_dir}")
print(f"[CAL] intrinsics_path: {intrinsics_path}")
print(f"[CAL] handeye_path   : {handeye_path}")

# Table height in base frame
z_table_m = 0.0045 # 0.004 - 0.005 / With right TCP

# Camera index & resolution
cam_index = 0
cam_width = 1920
cam_height = 1080

# Which pose estimation method to use for this run
pose_method = "hsv_centroid"
# = Later: "aruco", "pnp", "learned", ... ====

# Small context object to share calibration & robot handles
class PoseContext:
    def __init__(self, robot, K, map1, map2, T_gripper_cam, z_table):
        self.robot = robot
        self.K = K
        self.map1 = map1
        self.map2 = map2
        self.T_gripper_cam = T_gripper_cam # 4x4, camera in gripper or vice versa
        self.z_table = z_table

# Fixed observation pose
OBS_POSE = pyn.PoseObject(
    0.1453,   # x
    0.0386,   # y
    0.154,   # z
    0.0831,  # roll
    1.4896,   # pitch
    0.0962,   # yaw
)

print("[DEBUG] OBS_POSE in code:", OBS_POSE)

# ------------------------------------
# Step 2 - Connect to Niryo & obs_pose
# ------------------------------------
robot_ip = "129.187.231.226"
print("[INFO] Step1: helper + configs")
print("\n[INFO] Step2: Connecting to Niryo Ned2 at", robot_ip, "...")

t0 = time.time()
robot = pyn.NiryoRobot(robot_ip)

if robot.need_calibration():
    print("[INFO] Robot needs calibration → calibrate_auto()")
    robot.calibrate_auto()

robot.enable_tcp(True)
robot.set_learning_mode(False)
robot.set_arm_max_velocity(50) # Safety Step

# Move to a fixed observation pose instead of home
robot.move(OBS_POSE)  # <--- use move() with PoseObject

pose_live = robot.get_pose()
print(
    "[INFO] Robot moved to OBS pose: "
    f"x={pose_live.x:.4f}, y={pose_live.y:.4f}, z={pose_live.z:.4f}, "
    f"r={pose_live.roll:.4f}, p={pose_live.pitch:.4f}, yaw={pose_live.yaw:.4f} "
    f"(took {time.time() - t0:.2f}s)"
)

# ------------------------------------------
# Step 3 - Load calibration (intrinsics, HE)
# ------------------------------------------
def load_intrinsics_fisheye(path: str):
    """
    Load fisheye intrinsics (K, D) from npz.
    Undistortion maps will be computed later in Step 4.
    """
    data = np.load(path)
    K = data["K"].astype(float)                 # 3x3
    D = data["D"].astype(float).reshape(-1)     # distortion coefficients
    print("[CAL] Loaded intrinsics from", path)
    print("[CAL] K:\n", K)
    print("[CAL] D:", D)
    return K, D

def load_handeye_T_gripper_cam(path: str):
    data = np.load(path)
    if "T_gripper_camera" in data:
        T_gc = data["T_gripper_camera"].astype(float)
        print("[CAL] Loaded T_gripper_camera from", path)
        return T_gc

    if "R_gc" in data and "t_gc" in data:
        R_gc = data["R_gc"]
        t_gc = data["t_gc"]
        T_gc = make_T(R_gc, t_gc)
        print("[CAL] Loaded T_gripper_cam (R_gc, t_gc) from", path)
        return T_gc

    if "R_tc" in data and "t_tc" in data:
        R_gc = data["R_tc"]
        t_gc = data["t_tc"]
        T_gc = make_T(R_gc, t_gc)
        print("[CAL] Loaded T_gripper_cam (R_tc, t_tc) from", path)
        return T_gc

    raise RuntimeError("Hand-eye npz does not contain expected keys.")
    
print("\n[INFO] Step3: Loading calibration...")
K, D = load_intrinsics_fisheye(intrinsics_path)
T_gripper_cam = load_handeye_T_gripper_cam(handeye_path)

# ----------------------------------------
# Step 4 - Undistort maps & camera capture
# ----------------------------------------
print("\n[INFO] Step4: Open camera & prepare undistortion maps")

# 4.1 Open camera
cap = cv2.VideoCapture(cam_index)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cam_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)

if not cap.isOpened():
    raise RuntimeError("[ERROR] Could not open camera. Check USB connection/index.")

# Let auto-exposure settle a bit
for _ in range(4):
    cap.grab()

ok, test_frame = cap.read()
if not ok:
    raise RuntimeError("[ERROR] Failed to read a test frame from camera.")

H_act, W_act = test_frame.shape[:2]
print(f"[INFO] Camera opened at {W_act}x{H_act} (requested {cam_width}x{cam_height})")

# 4.2 Build fisheye undistortion maps (same model as in pnp_clean.py)
D_4x1 = D.reshape(4, 1).astype(float)
map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    K, D_4x1,
    np.eye(3, dtype=float),  # no extra rectification rotation
    K,                       # keep same intrinsics for output
    (W_act, H_act),
    cv2.CV_16SC2,
)
print("[INFO] Undistortion maps ready.")

def grab_undistorted_frame():
    """
    Grab a frame from the USB camera and undistort it using precomputed maps.
    Returns:
        frame_raw: BGR image as captured
        frame_ud : undistorted BGR image
    """
    for _ in range(4):
        cap.grab()
    ok, frame_raw = cap.read()
    if not ok:
        raise RuntimeError("[ERROR] Camera read failed in grab_undistorted_frame()")
    frame_ud = cv2.remap(frame_raw, map1, map2, interpolation=cv2.INTER_LINEAR)
    return frame_raw, frame_ud

# 4.3 Build global context now that we know map1,map2
CTX = PoseContext(
    robot=robot,
    K=K,
    map1=map1,
    map2=map2,
    T_gripper_cam=T_gripper_cam,
    z_table=z_table_m,
)

print("[INFO] Step4 done. CTX ready for pose estimation methods.")

# -----------------------------------------
# Step 5 - Transform helpers (base <-> cam)
# -----------------------------------------
def compute_T_base_cam(ctx: PoseContext) -> np.ndarray:
    """
    Compute T_base<-camera for the CURRENT robot pose.

    Uses:
      - robot.get_pose()  -> PoseObject in base frame
      - ctx.T_gripper_cam -> fixed hand–eye transform (gripper<-camera)

    Returns:
      4x4 homogeneous matrix T_base_cam.
    """
    pose = ctx.robot.get_pose()  # PoseObject in base frame
    R_bg = rpy_to_rot_matrix(pose.roll, pose.pitch, pose.yaw)
    t_bg = np.array([pose.x, pose.y, pose.z], dtype=float)

    T_base_gripper = make_T(R_bg, t_bg)
    T_base_cam = T_base_gripper @ ctx.T_gripper_cam

    return T_base_cam

# -----------------------------------------
# Step 6 - Minimal test (no pose estimation)
# -----------------------------------------
def test_presteps(ctx: PoseContext):
    """
    Simple debug: grab one frame, undistort, compute T_base_cam 
    and show images. Useful if something feels off later.
    """
    print("\n[TEST] Grabbing one undistorted frame and computing T_base_cam…")

    frame_raw, frame_ud = grab_undistorted_frame()
    print(f"[TEST] frame_raw shape: {frame_raw.shape}")
    print(f"[TEST] frame_ud  shape: {frame_ud.shape}")

    T_base_cam = compute_T_base_cam(ctx)
    R_bc = T_base_cam[:3, :3]
    t_bc = T_base_cam[:3,  3]
    print("[DEBUG] T_base_cam:\n", T_base_cam)
    print(f"[DEBUG] Camera origin in base: x={t_bc[0]:.4f}, y={t_bc[1]:.4f}, z={t_bc[2]:.4f}")

    RtR = R_bc.T @ R_bc
    orth_err = np.linalg.norm(RtR - np.eye(3))
    detR = np.linalg.det(R_bc)
    print(f"[DEBUG] ||RᵀR - I|| = {orth_err:.2e}, det(R) = {detR:.6f}")

    SHOW_WINDOWS = True
    if SHOW_WINDOWS:
        cv2.imshow("raw", frame_raw)
        cv2.imshow("undistorted", frame_ud)
        print("[TEST] Press any key in the image window(s) to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("[TEST] Minimal pre-step test finished.")

# ------------------------------------
# Step 7 - ArUco based pose estimation
# ---------------------------------------------------------
# Step 7.1 - ArUco config (dictionary, detector, marker size)
# ---------------------------------------------------------
# Try to access the aruco module
try:
    aruco = cv2.aruco
except AttributeError:
    aruco = None
    print("[ARUCO ERROR] cv2.aruco module not available. Check your OpenCV installation.")

if aruco is not None:
    # Dictionary for our printed markers (4x4_50, ids 0..49)
    ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    # Try new API first (OpenCV 4.7+), fall back to old one
    _use_new_api = False
    _aruco_detector = None
    try:
        _aruco_params = aruco.DetectorParameters()
        _aruco_detector = aruco.ArucoDetector(ARUCO_DICT, _aruco_params)
        _use_new_api = True
        print("[ARUCO] Using ArUcoDetector (new API).")
    except Exception:
        _aruco_params = aruco.DetectorParameters_create()
        _aruco_detector = None
        _use_new_api = False
        print("[ARUCO] Using detectMarkers + DetectorParameters_create (old API).")

    # Measured marker side length in meters (your printed markers: 2.6 cm)
    ARUCO_MARKER_LEN_M = 0.026
else:
    ARUCO_DICT = None
    _aruco_params = None
    _aruco_detector = None
    _use_new_api = False
    ARUCO_MARKER_LEN_M = None

# ---------------------------------------
# Step 7.2 - estimate_pose_aruco function
# ---------------------------------------
ARUCO_DICT_NAME   = aruco.DICT_4X4_50
ARUCO_MARKER_LEN_M = 0.026  # your printed marker side (26 mm)

_aruco_dict   = aruco.getPredefinedDictionary(ARUCO_DICT_NAME)
_aruco_params = aruco.DetectorParameters()
_aruco_detector = aruco.ArucoDetector(_aruco_dict, _aruco_params)
print("[ARUCO] Using ArUcoDetector (new API).")


def estimate_pose_aruco(ctx: PoseContext, marker_id: int = 0, debug: bool = True):
    """
    Detect a single ArUco marker and compute its pose in the *base* frame.

    Returns:
        T_base_marker (4x4) or None if not found,
        info dict with some debug fields.
    """
    # 1) Grab undistorted frame
    frame_raw, frame_ud = grab_undistorted_frame()
    gray = cv2.cvtColor(frame_ud, cv2.COLOR_BGR2GRAY)

    # 2) Detect markers
    corners, ids, rejected = _aruco_detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        print("[ARUCO] No markers detected.")
        return None, {"status": "no_markers", "frame_ud": frame_ud}

    ids = ids.flatten()
    print(f"[ARUCO] Detected marker IDs: {ids}")

    # 3) Pick the requested marker_id
    matches = np.where(ids == marker_id)[0]
    if len(matches) == 0:
        print(f"[ARUCO] Marker id {marker_id} not found.")
        return None, {
            "status": "id_not_found",
            "ids": ids,
            "frame_ud": frame_ud,
        }

    # just take the first occurrence
    idx = int(matches[0])
    marker_corners = corners[idx].reshape(-1, 2).astype(np.float32)  # (4,2)

    # 4) Build 3D object points in marker frame (centered square on z=0)
    L = ARUCO_MARKER_LEN_M
    half = L / 2.0
    # OpenCV ArUco corner order is: top-left, top-right, bottom-right, bottom-left
    obj_pts = np.array([
        [-half,  half, 0.0],  # top-left
        [ half,  half, 0.0],  # top-right
        [ half, -half, 0.0],  # bottom-right
        [-half, -half, 0.0],  # bottom-left
    ], dtype=np.float32)

    # On the undistorted image we can assume zero distortion
    dist = None  # or np.zeros((4,1), dtype=np.float32)

    # 5) Solve PnP manually (replacement for estimatePoseSingleMarkers)
    if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE"):
        flag = cv2.SOLVEPNP_IPPE_SQUARE
    else:
        flag = cv2.SOLVEPNP_ITERATIVE

    ok, rvec, tvec = cv2.solvePnP(
        obj_pts,
        marker_corners,
        ctx.K,
        dist,
        flags=flag,
    )
    if not ok:
        print("[ARUCO] solvePnP failed.")
        return None, {"status": "pnp_failed", "frame_ud": frame_ud}
    
     # ---------- DEBUG: reprojection error ----------
    if debug:
        proj_pts, _ = cv2.projectPoints(
            obj_pts,
            rvec,
            tvec,
            ctx.K,
            dist,
        )
        proj_pts = proj_pts.reshape(-1, 2)
        img_pts_meas = marker_corners

        errs = np.linalg.norm(img_pts_meas - proj_pts, axis=1)
        print(f"[ARUCO CHECK] reprojection errors (px): {errs}")
        print(f"[ARUCO CHECK] mean={errs.mean():.2f} px, max={errs.max():.2f} px")

    # 6) Build T_cam_marker from rvec,tvec
    R_cm, _ = cv2.Rodrigues(rvec)
    t_cm = tvec.reshape(3,)
    T_cam_marker = make_T(R_cm, t_cm)

    # ---------- DEBUG: test two possible T_base_cam conventions ----------
    pose = ctx.robot.get_pose()
    R_bg = rpy_to_rot_matrix(pose.roll, pose.pitch, pose.yaw)
    t_bg = np.array([pose.x, pose.y, pose.z], dtype=float)
    T_base_gripper = make_T(R_bg, t_bg)

    T_gc = ctx.T_gripper_cam              # what we *think* is gripper <- camera
    T_cg = np.linalg.inv(T_gc)            # camera <- gripper (inverse)

    T_base_cam_A = T_base_gripper @ T_gc  # assumption A: saved as gripper <- camera
    T_base_cam_B = T_base_gripper @ T_cg  # assumption B: saved as camera <- gripper

    T_base_marker_A = T_base_cam_A @ T_cam_marker
    T_base_marker_B = T_base_cam_B @ T_cam_marker

    pA = T_base_marker_A[:3, 3]
    pB = T_base_marker_B[:3, 3]

    print(f"[DEBUG] A (using T_gripper_cam): x={pA[0]:.3f}, y={pA[1]:.3f}, z={pA[2]:.3f}")
    print(f"[DEBUG] B (using inv(T_gripper_cam)): x={pB[0]:.3f}, y={pB[1]:.3f}, z={pB[2]:.3f}")

    # 7) Compose T_base_marker (current code = A)
    T_base_cam = compute_T_base_cam(ctx)
    T_base_marker = T_base_cam @ T_cam_marker

    # ---------- DEBUG: rotation sanity check in base frame ----------
    if debug:
        R_bm = T_base_marker[:3, :3]
        RtR = R_bm.T @ R_bm
        orth_err = np.linalg.norm(RtR - np.eye(3))
        detR = np.linalg.det(R_bm)
        print(f"[ARUCO CHECK] ||RᵀR - I|| = {orth_err:.2e}, det(R) = {detR:.6f}")

    info = {
        "status": "ok",
        "rvec": rvec,
        "tvec": tvec,
        "T_cam_marker": T_cam_marker,
        "T_base_cam": T_base_cam,
        "corners": marker_corners,
        "frame_ud": frame_ud,
    }

    if debug:
        # simple overlay to see the detection
        draw = frame_ud.copy()
        aruco.drawDetectedMarkers(draw, [marker_corners.reshape(1, -1, 2)], np.array([[marker_id]]))
        # draw axis for visualization (use marker length for axis size)
        cv2.drawFrameAxes(draw, ctx.K, np.zeros((4,1)), rvec, tvec, L * 0.5)
        cv2.imshow("aruco_debug", draw)
        print("[ARUCO] Press any key to close debug window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # print pose summary
        t_bm = T_base_marker[:3, 3]
        print(f"[ARUCO] Marker {marker_id} in base frame: x={t_bm[0]:.3f}, y={t_bm[1]:.3f}, z={t_bm[2]:.3f}")

    return T_base_marker, info

if __name__ == "__main__":
    # Optional: run the old sanity test once in a while
    # test_presteps(CTX)

    print("\n[MAIN] Running single ArUco pose estimation for marker 0 (orange cube)…")
    T_base_marker, info = estimate_pose_aruco(CTX, marker_id=0, debug=True)

    if T_base_marker is None:
        print("[MAIN] No pose returned (see logs above).")
    else:
        R_bm = T_base_marker[:3, :3]
        t_bm = T_base_marker[:3, 3]
        print("[MAIN] T_base_marker:\n", T_base_marker)
        print(
            f"[MAIN] Marker position (base): "
            f"x={t_bm[0]:.3f}, y={t_bm[1]:.3f}, z={t_bm[2]:.3f}"
        )

    # Clean up
    try:
        robot.move_to_home_pose()
        print("[MAIN] Returned to home pose.")
    except Exception as e:
        print("[MAIN WARN] Could not move home:", e)

    cap.release()
    cv2.destroyAllWindows()
    print("[MAIN] Done.")