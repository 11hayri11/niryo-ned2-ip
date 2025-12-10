# --------------------------------
# Step 0 — Imports, config, helper
# --------------------------------
import os, time, json
import numpy as np
import cv2
import pyniryo as pyn

# === Calibration files from my latest run ====================
RUN_DIR = os.path.join("calibration_data_charuco", "run_20251208_122545")
K_PATH  = os.path.join(RUN_DIR, "intrinsics_charuco_for_pnp.npz")
HE_PATH = os.path.join(RUN_DIR, "handeye_charuco_TSAI.npz")

# === XY correction load/apply helpers =======================
CORR_PATH = os.path.join(RUN_DIR, "xy_correction_affine.json")
_CORR_MODE = None   # "affine" or "offset" or None
_A = None           # 2x3 affine matrix if present
_DX_DY = (0.0, 0.0) # offset fallback

def _load_xy_correction():
    global _CORR_MODE, _A, _DX_DY
    if not os.path.isfile(CORR_PATH):
        print("[CAL] No XY correction file yet.")
        _CORR_MODE, _A, _DX_DY = None, None, (0.0, 0.0)
        return
    try:
        with open(CORR_PATH, "r") as f:
            J = json.load(f)
        if "A" in J and J["A"]:
            _A = np.array(J["A"], dtype=float)      # shape (2,3)
            if _A.shape != (2,3):
                raise ValueError("Affine matrix must be 2x3")
            _CORR_MODE = "affine"
            print(f"[CAL] Loaded affine correction from {CORR_PATH}")
        elif "dx" in J and "dy" in J:
            _DX_DY = (float(J["dx"]), float(J["dy"]))
            _CORR_MODE = "offset"
            print(f"[CAL] Loaded offset correction from {CORR_PATH}")
        else:
            print("[CAL] Correction file present but empty/invalid; ignoring.")
            _CORR_MODE = None
    except Exception as e:
        print(f"[CAL] Failed to load XY correction: {e}")
        _CORR_MODE, _A, _DX_DY = None, None, (0.0, 0.0)
    finally:
        if _CORR_MODE == "affine":
            print(f"[CAL] Mode=affine, A=\n{_A}")
        elif _CORR_MODE == "offset":
            dx, dy = _DX_DY
            print(f"[CAL] Mode=offset, dx{dx:.4f}, dy={dy:.4f}")
        else:
            print("[CAL]Mode=none (No correction applied)")
def apply_xy_correction(x_vis, y_vis):
    """
    Apply the best-known correction to (x_vis,y_vis).
    - affine if A exists (>=3 calibration points)
    - else constant offset (1–2 points)
    - else passthrough
    """
    if _CORR_MODE == "affine" and _A is not None:
        v = np.array([x_vis, y_vis, 1.0], dtype=float)  # 3-vector
        xy = _A @ v                                     # (2,)
        return float(xy[0]), float(xy[1])
    elif _CORR_MODE == "offset":
        dx, dy = _DX_DY
        return x_vis + dx, y_vis + dy
    else:
        return x_vis, y_vis

# load on startup
_load_xy_correction()
# ===================

# === Table & cube ===
Z_TABLE_M   = 0.0045        # 0.004 - 0.005 m / With right TCP
CUBE_SIDE_M = 0.037
CUBE_HALF_M = CUBE_SIDE_M / 2.0

# === Camera ===
CAMERA_INDEX = 0
FRAME_W, FRAME_H = 1920, 1080

# === Helper: RPY -> R (ZYX / yaw→pitch→roll) ===
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

if __name__ == "__main__":
    print("[OK] Step 0 loaded.")
    print("RUN_DIR:", RUN_DIR)
    print("K_PATH :", K_PATH)
    print("HE_PATH:", HE_PATH)
    print("Z_TABLE:", Z_TABLE_M)

# --------------------------------
# Step 1 - Connect to Niryo & home
# --------------------------------
robot_ip = "129.187.231.226"
print("\n[INFO] Step1: Connecting to Niryo Ned2 at", robot_ip, "...")

t0 = time.time()
robot = pyn.NiryoRobot(robot_ip)

if robot.need_calibration():
    print("[INFO] Robot needs calibration → calibrate_auto()")
    robot.calibrate_auto()

robot.enable_tcp(True)
robot.set_learning_mode(False)
robot.set_arm_max_velocity(50) # Safety Step - slower while you're near the table
robot.move_to_home_pose() # Safety Step - Moves to home position
print(f"[INFO] Robot moved to home position and connection established successfully. (took {time.time()-t0:.2f}s)")

# ---------------------------------
# Step 2 - Move to observation pose
# ---------------------------------
print("\n[INFO] Step2: Move to observation pose (joints) & verify FK vs LIVE")

# Verified observation joints (rad)
obs_joints = pyn.JointsPosition(-0.0007, 0.5281, -0.2522, 0.0185, -1.7841, -0.0428)
robot.move(obs_joints)
time.sleep(0.5)  # Time to settle

# Compute FK from the same joints we just commanded
pose_fk   = robot.forward_kinematics(list(obs_joints))
pose_live = robot.get_pose()  # reported by robot

def _pp_pose(tag, p):
    print(f"[CHECK] {tag}: x={p.x:.4f}, y={p.y:.4f}, z={p.z:.4f} | "
          f"roll={p.roll:.3f}, pitch={p.pitch:.3f}, yaw={p.yaw:.3f}")

_pp_pose("FK   ",  pose_fk)
_pp_pose("LIVE ",  pose_live)

# Quick numeric deltas (to spot issues immediately)
d_pos  = np.array([pose_live.x - pose_fk.x, pose_live.y - pose_fk.y, pose_live.z - pose_fk.z])
d_rpy  = np.array([pose_live.roll - pose_fk.roll,
                   pose_live.pitch - pose_fk.pitch,
                   pose_live.yaw - pose_fk.yaw])

print(f"[DELTA] pos (m): {d_pos}  |  rpy (rad): {d_rpy}")

# Soft thresholds (tune if needed)
POS_TOL  = 0.010   # 10 mm
RPY_TOL  = 0.050   # ~3°
if np.linalg.norm(d_pos) > POS_TOL or np.max(np.abs(d_rpy)) > RPY_TOL:
    print("[WARN] FK vs LIVE mismatch exceeds tolerance. Keep an eye on this before hand–eye composition.")

# Keep FK pose for camera composition in Step 3
OBS_POSE_FK = pose_fk
# Ensure we keep both versions around
OBS_POSE_LIVE = robot.get_pose()

# ----------------------------------------------------
# Step 3 - Load K,D + hand–eye and compose T_base<-cam
# ----------------------------------------------------
print("\n[INFO] Step3: Load intrinsics & hand–eye, compose T_base<-camera")

# 3.1 Load intrinsics (K, D)
if not os.path.exists(K_PATH):
    raise FileNotFoundError(f"[ERROR] Intrinsics file not found: {K_PATH}")
intr = np.load(K_PATH)
K = intr["K"].astype(float)
D = intr["D"].astype(float).reshape(-1)
print("[INFO] K:\n", K)
print("[INFO] D:", D)

# 3.2 Load hand–eye (T_gripper_camera)
if not os.path.exists(HE_PATH):
    raise FileNotFoundError(f"[ERROR] Hand–eye file not found: {HE_PATH}")
he = np.load(HE_PATH, allow_pickle=True)
if "T_gripper_camera" in he:
    T_gripper_camera = he["T_gripper_camera"].astype(float)
else:
    R = he["R"].astype(float)
    t = he["t"].astype(float).reshape(3)
    T_gripper_camera = np.eye(4, dtype=float)
    T_gripper_camera[:3, :3] = R
    T_gripper_camera[:3,  3] = t
print("[INFO] T_gripper_camera:\n", T_gripper_camera)

# 3.3 Build T_base_gripper from the FK observation pose (keeps consistency)
R_bg = rpy_to_rot_matrix(OBS_POSE_LIVE.roll, OBS_POSE_LIVE.pitch, OBS_POSE_LIVE.yaw)
T_base_gripper = np.eye(4)
T_base_gripper[:3, :3] = R_bg
T_base_gripper[:3,  3] = [OBS_POSE_LIVE.x, OBS_POSE_LIVE.y, OBS_POSE_LIVE.z]
print("[INFO] T_base_gripper:\n", T_base_gripper)

# 3.4 Compose T_base_camera
T_base_cam = T_base_gripper @ T_gripper_camera
R_bc = T_base_cam[:3, :3]
t_bc = T_base_cam[:3,  3]
print("[INFO] T_base_camera:\n", T_base_cam)
print(f"[INFO] Camera origin in base (m): x={t_bc[0]:.4f}, y={t_bc[1]:.4f}, z={t_bc[2]:.4f}")

# 3.5 Quick orthonormality check on R_bc
RtR = R_bc.T @ R_bc
orth_err = np.linalg.norm(RtR - np.eye(3))
detR = np.linalg.det(R_bc)
print(f"[CHECK] ||RᵀR - I|| = {orth_err:.2e}   det(R) = {detR:.6f}")

# Keep for next steps
T_BASE_GRIPPER = T_base_gripper
T_BASE_CAM     = T_base_cam
K_mat, D_vec   = K, D

# ----------------------------------------------
# Step 4 - Open camera & build undistortion maps
# ----------------------------------------------
print("\n[INFO] Step4: Open camera & prepare undistortion maps")

# 4.1 Camera open (match setup)
CAMERA_INDEX = 0
REQ_W, REQ_H = 1920, 1080

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  REQ_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQ_H)

if not cap.isOpened():
    raise RuntimeError("[ERROR] Could not open camera. Check USB connection/index.")

# Let auto-exposure settle a bit
for _ in range(4):
    cap.grab()

ok, test_frame = cap.read()
if not ok:
    raise RuntimeError("[ERROR] Failed to read a test frame from camera.")

H_act, W_act = test_frame.shape[:2]
print(f"[INFO] Camera opened at {W_act}x{H_act} (requested {REQ_W}x{REQ_H})")

# 4.2 Build fisheye undistortion maps (rectify to same intrinsics)
# Ensure D has shape (4,1) for cv2.fisheye
D_4x1 = D_vec.reshape(4, 1).astype(float)
map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    K_mat, D_4x1, np.eye(3), K_mat, (W_act, H_act), cv2.CV_16SC2
)
print("[INFO] Undistortion maps ready.")

# Keep for next steps
CAP = cap
MAP1, MAP2 = map1, map2
FRAME_SIZE = (W_act, H_act)

# -------------------------------------------
# Step 5 - Take one snapshot and undistort it
# -------------------------------------------
print("\n[INFO] Step5: Snapshot (raw + undistorted)")

# Let exposure settle a touch more
for _ in range(4):
    CAP.grab()

ok, frame_raw = CAP.read()
if not ok:
    raise RuntimeError("[ERROR] Couldn’t grab a frame from the camera.")

# Undistort using the precomputed maps
frame_ud = cv2.remap(frame_raw, MAP1, MAP2, interpolation=cv2.INTER_LINEAR)

# Quick visual check (two windows)
cv2.imshow("Snapshot (raw)", frame_raw)
cv2.imshow("Snapshot (undistorted)", frame_ud)
print("  [UI] Press any key to close the preview windows…")
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optional: save to disk for debugging
import os, time
snap_dir = os.path.join(os.path.dirname(__file__), "debug_snaps")
os.makedirs(snap_dir, exist_ok=True)
stamp = time.strftime("%Y%m%d_%H%M%S")
raw_path = os.path.join(snap_dir, f"snap_raw_{stamp}.png")
ud_path  = os.path.join(snap_dir, f"snap_undist_{stamp}.png")
cv2.imwrite(raw_path, frame_raw)
cv2.imwrite(ud_path,  frame_ud)
print(f"[INFO] Saved snapshots:\n  raw  → {raw_path}\n  und  → {ud_path}")

# Keep for next steps
FRAME_RAW = frame_raw
FRAME_UD  = frame_ud

# ----------------------------------------------------------------
# Step 6 - HSV sliders on the undistorted snapshot + centroid pick
# ----------------------------------------------------------------
print("\n[INFO] Step6: HSV tuner (press 'q' to accept, 's' to save thresholds)")

# Work on a copy of the undistorted frame
frame_ud = FRAME_UD.copy()
hsv = cv2.cvtColor(frame_ud, cv2.COLOR_BGR2HSV)

# Create UI
win = "HSV Mask Tuner"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)

# Last good values (tweak as needed)
INIT = dict(H_low=7, H_high=13, S_low=113, V_low=141)

def put_trackbar(name, val, maxv):
    cv2.createTrackbar(name, win, 0, maxv, lambda v: None)
    cv2.setTrackbarPos(name, win, val)

put_trackbar("H_low",  INIT["H_low"],  179)
put_trackbar("H_high", INIT["H_high"], 179)
put_trackbar("S_low",  INIT["S_low"],  255)
put_trackbar("V_low",  INIT["V_low"],  255)

# Vars to export
U_det = V_det = None
AREA_det = 0

while True:
    # Read slider values (and ensure low <= high)
    Hl = cv2.getTrackbarPos("H_low",  win)
    Hh = cv2.getTrackbarPos("H_high", win)
    Sl = cv2.getTrackbarPos("S_low",  win)
    Vl = cv2.getTrackbarPos("V_low",  win)
    if Hl > Hh:
        Hl, Hh = Hh, Hl  # swap to keep consistency

    lower = np.array([Hl, Sl, Vl], dtype=np.uint8)
    upper = np.array([Hh, 255, 255], dtype=np.uint8)

    # Mask + clean-up
    mask = cv2.inRange(hsv, lower, upper)
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k1, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2, iterations=1)

    # Find largest plausible blob
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = frame_ud.copy()
    u_out = v_out = None
    area_out = 0

    if cnts:
        cnt = max(cnts, key=cv2.contourArea)
        area = float(cv2.contourArea(cnt))
        if area > 500:  # simple area floor to ignore specks
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                u = int(M["m10"] / M["m00"])
                v = int(M["m01"] / M["m00"])
                u_out, v_out, area_out = u, v, area
                # draw
                cv2.drawContours(overlay, [cnt], -1, (0,255,0), 2)
                cv2.circle(overlay, (u, v), 6, (0,0,255), -1)
                cv2.putText(overlay, f"(u,v)=({u},{v}), A={int(area)}",
                            (u+10, v-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    # Build 3-panel preview: original | mask | overlay
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    panel = np.hstack([frame_ud, mask_bgr, overlay])
    cv2.imshow(win, panel)

    key = cv2.waitKey(15) & 0xFF
    if key == ord('s'):
        # Save thresholds to file
        import json, os, time
        params = dict(H_low=Hl, H_high=Hh, S_low=Sl, V_low=Vl)
        outp = os.path.join(os.path.dirname(__file__), f"hsv_{time.strftime('%Y%m%d_%H%M%S')}.json")
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2)
        print(f"[INFO] Saved HSV params → {outp}")
    elif key == ord('q') or key == 27:  # q or ESC to finish
        # Export last good centroid (if any)
        if u_out is not None:
            U_det, V_det, AREA_det = u_out, v_out, area_out
        break

# After your Step 6 tuner loop ends and you have final thresholds:
HSV_LAST = (Hl, Hh, Sl, Vl)   # keep the final sliders for reuse in 7.5

cv2.destroyWindow(win)

if U_det is None:
    print("[WARN] No valid blob selected. Consider adjusting sliders or lighting.")
else:
    print(f"[RESULT] Centroid (u,v)=({U_det},{V_det}), area={AREA_det:.1f}")

# Keep for next steps
U_CENTROID = U_det
V_CENTROID = V_det
MASK_BIN   = mask

# -------------------------------------------
# Step 7 - Pixel (u,v) -> World (x,y,z_table)
# -------------------------------------------
def pixel_to_world_on_plane(u, v, K, T_base_cam, z_plane):
    """Back-project pixel to a 3D ray and intersect with plane z=z_plane (in base frame)."""
    K_inv = np.linalg.inv(K)
    p_cam = np.array([float(u), float(v), 1.0], dtype=float)
    r_cam = K_inv @ p_cam
    r_cam = r_cam / np.linalg.norm(r_cam)

    R_bc = T_base_cam[:3, :3]
    t_bc = T_base_cam[:3,  3]
    d_b  = R_bc @ r_cam           # ray direction in base frame
    O_b  = t_bc                   # camera origin in base frame

    eps = 1e-9
    if abs(d_b[2]) < eps:
        raise RuntimeError("Ray nearly parallel to the table plane; adjust observation pose.")

    t = (z_plane - O_b[2]) / d_b[2]
    if t <= 0:
        print(f"[WARN] Intersection behind the camera (t={t:.6f}). Check Z_TABLE_M / pose.")
    X_b = O_b + t * d_b
    return X_b  # (x,y,z)

# Use the centroid from Step 6
Xb = pixel_to_world_on_plane(U_CENTROID, V_CENTROID, K, T_base_cam, Z_TABLE_M)
x_vis, y_vis, z_on_plane = float(Xb[0]), float(Xb[1]), float(Xb[2])

# Apply learned XY correction (affine preferred, else offset)
x_pick_raw, y_pick_raw = x_vis, y_vis
x_pick, y_pick = apply_xy_correction(x_pick_raw, y_pick_raw)

# (Optional) clamp to a conservative reachable envelope (tune to the table)
def _clamp(v, lo, hi): return max(lo, min(hi, v))
X_MIN, X_MAX = 0.12, 0.42
Y_MIN, Y_MAX = -0.18, 0.18
x_pick = _clamp(x_pick, X_MIN, X_MAX)
y_pick = _clamp(y_pick, Y_MIN, Y_MAX)
print(f"[SAFE] XY clamped -> ({x_pick:.3f}, {y_pick:.3f}) within X[{X_MIN:.2f},{X_MAX:.2f}], Y[{Y_MIN:.2f},{Y_MAX:.2f}]")

# Suggest a grasp height (cube 3.7 cm -> half = 18.5 mm + 2 mm pad)
cube_half = 0.037 / 2.0
z_pick = Z_TABLE_M + cube_half + 0.002

print(f"[RESULT] Vision raw XY: ({x_pick_raw:.4f}, {y_pick_raw:.4f})")
print(f"[CORR]   Corrected XY : ({x_pick:.4f}, {y_pick:.4f})")
print(f"[INFO]   z_on_plane   : {z_on_plane:.4f}")
print(f"[INFO]   z_pick       : {z_pick:.4f}  (table {Z_TABLE_M:.4f} + {cube_half:.4f} + 0.002)")

# One sanity: center-pixel ray to table should land roughly forward of camera origin
K_inv = np.linalg.inv(K)
cx, cy = float(K[0,2]), float(K[1,2])
Xc = pixel_to_world_on_plane(cx, cy, K, T_base_cam, Z_TABLE_M)
r_cam_c = K_inv @ np.array([cx, cy, 1.0], float); r_cam_c /= np.linalg.norm(r_cam_c)
d_b_c = T_base_cam[:3,:3] @ r_cam_c
print(f"[CHECK] Center (cx,cy)=({cx:.2f},{cy:.2f}) -> world (x,y,z)=({Xc[0]:.4f},{Xc[1]:.4f},{Xc[2]:.4f})")
print(f"[CHECK] Camera origin base (x,y)=({T_base_cam[0,3]:.4f},{T_base_cam[1,3]:.4f})  dz_ray={d_b_c[2]:.3f}")

# --- Step 7.6: one-shot pendant check to validate corrected XY ---
try:
    _ = input("[GT] Jog tip to cube TOP, press Enter, then paste pendant 'x y' (m): ")
    gt_str = input("[GT] Pendant x y (e.g. 0.312 -0.0096): ").strip()
    xs, ys = gt_str.split()
    x_gt, y_gt = float(xs), float(ys)
    dx, dy = (x_pick - x_gt), (y_pick - y_gt)
    err = (dx**2 + dy**2) ** 0.5
    print(f"[GT] corrected=({x_pick:.4f},{y_pick:.4f}) vs pendant=({x_gt:.4f},{y_gt:.4f})  ->  Δ=({dx:.4f},{dy:.4f}) | ||Δ||={err*1000:.1f} mm")
    if err <= 0.01:
        print("[GT] ✅ Within 1 cm — safe to proceed to motion.")
    else:
        print("[GT] ⚠️ >1 cm. Add 1–2 samples with Step 7.5 near this region to refine the affine.")
except Exception as e:
    print("[GT] Skip check:", e)

# Make targets available to later steps
X_PICK_CORR = x_pick
Y_PICK_CORR = y_pick
Z_PICK      = z_pick

# -------------------------------------------------------
# Step 7.5 — XY calibration loop (collect several samples)
# -------------------------------------------------------
print("\n[STEP 7.5] XY calibration loop — collect ≥3 samples for an affine correction.")
print("Instructions:")
print("  • Leave the robot at the SAME observation pose.")
print("  • Move the cube to a new spot, then press Enter to capture.")
print("  • Jog tip to cube TOP and paste pendant 'x y' in meters (e.g. 0.312 -0.0096).")
print("  • Type 'done' to finish.\n")

# Warm-start from existing samples (optional)
try:
    if os.path.isfile(CORR_PATH):
        with open(CORR_PATH, "r") as f:
            J_prev = json.load(f)
        if isinstance(J_prev.get("samples"), list) and len(J_prev["samples"]) >= 1:
            samples = [tuple(map(float, s)) for s in J_prev["samples"]]
            print(f"[CAL] Warm-started with {len(samples)} existing samples.")
        else:
            samples = []
    else:
        samples = []
except Exception:
    samples = []

# --- utility: compute vision estimate for current frame using HSV_LAST ---
def _vision_estimate_xy_once():
    # 1) grab + undistort
    for _ in range(4): cap.grab()
    ok, frame_raw = cap.read()
    if not ok: raise RuntimeError("Camera read failed in 7.5")
    frame_ud = cv2.remap(frame_raw, map1, map2, interpolation=cv2.INTER_LINEAR)
    # 2) HSV threshold with last tuned values
    Hl, Hh, Sl, Vl = HSV_LAST
    hsv = cv2.cvtColor(frame_ud, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        (int(Hl), int(Sl), int(Vl)),
        (int(Hh), 255, 255)
    )
    # morph (light clean-up)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    # 3) centroid of biggest blob
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError("No blob found; adjust HSV or move cube.")
    c = max(cnts, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] < 1e-6: raise RuntimeError("Degenerate contour.")
    u = int(M["m10"] / M["m00"])
    v = int(M["m01"] / M["m00"])
    # 4) pixel -> world on table z = Z_TABLE_M
    K_inv = np.linalg.inv(K)
    p_cam = np.array([float(u), float(v), 1.0], dtype=float)
    r_cam = K_inv @ p_cam
    r_cam = r_cam / np.linalg.norm(r_cam)
    R_bc = T_base_cam[:3,:3]
    t_bc = T_base_cam[:3, 3]
    d_b  = R_bc @ r_cam
    O_b  = t_bc
    t_param = (Z_TABLE_M - O_b[2]) / d_b[2]
    X_b = O_b + t_param * d_b
    x_vis, y_vis = float(X_b[0]), float(X_b[1])
    return x_vis, y_vis, (u, v), frame_ud, mask

samples = []  # (x_vis,y_vis,x_gt,y_gt)
while True:
    ans = input("[CAL] Move cube to a NEW spot, press Enter to capture (or type 'done'): ").strip().lower()
    if ans == "done":
        break
    try:
        x_vis, y_vis, (u,v), img_ud, mask = _vision_estimate_xy_once()
        print(f"[CAL] Vision estimate: (x_vis,y_vis)=({x_vis:.4f}, {y_vis:.4f})  at pixel (u,v)=({u},{v})")
    except Exception as e:
        print("[CAL] Vision estimate FAILED ->", e)
        continue

    user = input("[CAL] Paste pendant 'x y' in meters (e.g. 0.312 -0.0096): ").strip()
    try:
        x_gt, y_gt = map(float, user.split())
    except Exception:
        print("[CAL] Invalid input; skipping this sample.")
        continue

    samples.append((x_vis, y_vis, x_gt, y_gt))
    N = len(samples)

    # ---- Update correction: constant offset for 1–2, affine for ≥3 ----
    to_save = {}
    if N >= 3:
        V = np.array([[sx, sy, 1.0] for (sx, sy, gx, gy) in samples], float)  # (N,3)
        Gx = np.array([gx for (_,_,gx,_) in samples], float)
        Gy = np.array([gy for (_,_,_,gy) in samples], float)
        ax, *_ = np.linalg.lstsq(V, Gx, rcond=None)
        ay, *_ = np.linalg.lstsq(V, Gy, rcond=None)
        A = np.vstack([ax, ay])  # (2,3)
        to_save = {"A": A.tolist(), "samples": samples}
        print(f"[CAL] Updated AFFINE A=\n{A}")
    else:
        dx = float(np.mean([gx - sx for (sx,sy,gx,gy) in samples]))
        dy = float(np.mean([gy - sy for (sx,sy,gx,gy) in samples]))
        to_save = {"dx": dx, "dy": dy, "samples": samples}
        print(f"[CAL] Updated OFFSET dx={dx:+.4f}  dy={dy:+.4f}")

    # persist & reload
    try:
        with open(CORR_PATH, "w") as f:
            json.dump(to_save, f, indent=2)
        print(f"[CAL] Saved correction to {CORR_PATH}")
    except Exception as e:
        print("[CAL] Save failed ->", e)

    _load_xy_correction()  # refresh in-memory correction
    # quick check
    xc, yc = apply_xy_correction(x_vis, y_vis)
    print(f"[CAL] Check apply → corrected: ({xc:.4f}, {yc:.4f})  vs GT: ({x_gt:.4f}, {y_gt:.4f})")

print(f"[STEP 7.5] Done. Total samples collected: {len(samples)}")

# -------------------------------------------
# Step 7.7 — Recompute pick XY with new correction
# -------------------------------------------
try:
    input("[7.7] Place the cube at the desired pick location, then press Enter to re-detect: ")
    x_vis2, y_vis2, (u2, v2), img_ud2, mask2 = _vision_estimate_xy_once()
    print(f"[7.7] New vision estimate: (x_vis2,y_vis2)=({x_vis2:.4f}, {y_vis2:.4f}) at pixel (u,v)=({u2},{v2})")

    # Apply updated correction
    x_pick_raw2, y_pick_raw2 = x_vis2, y_vis2
    x_pick2, y_pick2 = apply_xy_correction(x_pick_raw2, y_pick_raw2)

    # Clamp to safe envelope
    x_pick2 = _clamp(x_pick2, X_MIN, X_MAX)
    y_pick2 = _clamp(y_pick2, Y_MIN, Y_MAX)

    print(f"[7.7] Corrected XY after new affine: ({x_pick2:.4f}, {y_pick2:.4f})")

    # Update globals used by Steps 8–9
    X_PICK_CORR = x_pick2
    Y_PICK_CORR = y_pick2
    # Z_PICK stays as before (same table & cube), but we could recompute if you want

except Exception as e:
    print(f"[7.7 WARN] Failed to recompute pick target after calibration: {e}")
    print("[7.7 INFO] Falling back to old X_PICK_CORR / Y_PICK_CORR.")

# ============ Step 8: progressive glide to over-pick & hover ============
# Use the FINAL corrected pick target from Step 7.7
x_pick = X_PICK_CORR
y_pick = Y_PICK_CORR
z_pick = Z_PICK   # if you defined Z_PICK earlier from Step 6

Z_OVER   = OBS_POSE_FK.z          # lateral travel height (~0.326 m)
CLEAR_HOVER = 0.020               # 20 mm above cube top
Z_HOVER  = z_pick + CLEAR_HOVER   # safe hover

# pick-friendly vertical orientation (use your measured good one)
r_pick, p_pick, yaw_pick = (2.575, 1.544, 2.617)  # roll, pitch, yaw (rad)

def PO(x, y, z, r, p, yaw):
    return pyn.PoseObject(x, y, z, r, p, yaw)

# 8a) rotate-in-place at observation (keep X/Y/Z = obs, just set RPY)
pA = PO(OBS_POSE_FK.x, OBS_POSE_FK.y, Z_OVER, r_pick, p_pick, yaw_pick)
robot.move(pA)
live = robot.get_pose()
print(f"[8a] at obs: x={live.x:.3f}, y={live.y:.3f}, z={live.z:.3f} | r={live.roll:.3f}, p={live.pitch:.3f}, yaw={live.yaw:.3f}")

# 8b) progressive lateral glide at Z_OVER
print(f"[8b PLAN] over-pick @ Z_OVER: corrected XY=({x_pick:.3f},{y_pick:.3f}), Z_OVER={Z_OVER:.3f}")

def progressive_glide(x0, y0, x1, y1, n=12, z=Z_OVER):
    for i in range(1, n+1):
        a  = i / float(n)
        xi = (1 - a) * x0 + a * x1
        yi = (1 - a) * y0 + a * y1
        try:
            robot.move(PO(xi, yi, z, r_pick, p_pick, yaw_pick))
            l = robot.get_pose()
            print(f"[8b {i:02d}/{n}] x={l.x:.3f}, y={l.y:.3f}, z={l.z:.3f}")
        except Exception as e:
            print(f"[8b {i:02d}/{n}] IK fail @ (x={xi:.3f},y={yi:.3f},z={z:.3f}) -> {e}")
            # try a small Z bump, then come back
            try:
                robot.move(PO(xi, yi, z + 0.02, r_pick, p_pick, yaw_pick))
                robot.move(PO(xi, yi, z,        r_pick, p_pick, yaw_pick))
                l = robot.get_pose()
                print(f"[8b {i:02d}/{n}] recovered via +2cm: x={l.x:.3f}, y={l.y:.3f}, z={l.z:.3f}")
            except Exception as e2:
                print(f"[8b {i:02d}/{n}] still failing after Z-lift -> abort")
                return False
    return True

ok = progressive_glide(OBS_POSE_FK.x, OBS_POSE_FK.y, x_pick, y_pick, n=12, z=Z_OVER)
if not ok:
    raise RuntimeError("[ABORT] Could not reach over-pick at safe Z.")

# 8c) descend to hover at corrected XY, keeping pick-friendly RPY
p_hover = PO(x_pick, y_pick, Z_HOVER, r_pick, p_pick, yaw_pick)
print(f"[8c PLAN] hover target: x={p_hover.x:.3f}, y={p_hover.y:.3f}, z={p_hover.z:.3f}")

def try_move_pose(pose):
    try:
        robot.move(pose)
        return True
    except Exception:
        return False

if try_move_pose(p_hover):
    print("[8c] reached hover over cube.")
else:
    print("[8c WARN] IK failed at hover; trying small yaw sweeps…")
    for d in (-0.10, +0.10, -0.20, +0.20):
        if try_move_pose(PO(x_pick, y_pick, Z_HOVER, r_pick, p_pick, yaw_pick + d)):
            print(f"[8c] success with yaw offset {d:+.2f} rad")
            break
    else:
        print("[8c ABORT] Unable to descend to hover (yaw tweaks failed). Staying at Z_OVER.")

# ----------------------------------------------------
# Step 9 — cautious descend, grasp, lift (diagnostic)
# ----------------------------------------------------
HOVER_PAD  = 0.020   # 20 mm above cube top (hover)
PRE_PAD    = 0.012   # 12 mm above cube top (pre-grasp)
GRASP_PAD  = 0.000   # 6 mm above cube top (grasp attempt)

Z_HOVER = z_pick + HOVER_PAD
Z_PRE   = z_pick + PRE_PAD
Z_GRASP = z_pick + GRASP_PAD

# use the SAME RPY you used in Step 8
# ensure these exist: r_pick, p_pick, yaw_pick
XZ_YZ_STR = lambda: f"x={x_pick:.3f}, y={y_pick:.3f}"

def _pose_xyz(x, y, z):
    return pyn.PoseObject(x, y, z, r_pick, p_pick, yaw_pick)

def _move_Z(label, z_target, step=0.008, frame_name=None, x=None, y=None):
    """
    Move straight in Z with small increments at fixed (x, y, r_pick/p_pick/yaw_pick).
    - If x,y are None, default to (x_pick, y_pick) for backward compatibility.
    """
    if x is None: x = x_pick
    if y is None: y = y_pick

    z_now = robot.get_pose().z
    direction = 1.0 if z_target > z_now else -1.0
    n_full = int(abs(z_target - z_now) // step)
    remainder = abs(z_target - z_now) - n_full * step
    segments = [step] * n_full + ([remainder] if remainder > 1e-6 else [])

    for i, dz in enumerate(segments, 1):
        z_cmd = z_now + direction * dz
        moved = False
        err_msg = None

        # Try linear Cartesian if available
        try:
            if hasattr(robot, "move_linear"):
                if frame_name is None:
                    robot.move_linear(_pose_xyz(x, y, z_cmd))
                else:
                    robot.move_linear(_pose_xyz(x, y, z_cmd), frame_name)
                live = robot.get_pose()
                if abs(live.z - z_now) > 0.001:
                    print(f"[9Z] {label}/linear {i}/{len(segments)} -> z={live.z:.3f}")
                    z_now = live.z
                    moved = True
                else:
                    err_msg = "linear executed but no z change"
        except Exception as e:
            err_msg = f"linear: {e}"

        # Fallback: regular move
        if not moved:
            try:
                if frame_name is None:
                    robot.move(_pose_xyz(x, y, z_cmd))
                else:
                    robot.move(_pose_xyz(x, y, z_cmd), frame_name)
                live = robot.get_pose()
                if abs(live.z - z_now) > 0.001:
                    print(f"[9Z] {label}/move   {i}/{len(segments)} -> z={live.z:.3f}")
                    z_now = live.z
                    moved = True
                else:
                    err_msg = "move executed but no z change"
            except Exception as e:
                err_msg = f"move: {e}"

        # IK -> joints fallback
        if not moved:
            try:
                if hasattr(robot, "inverse_kinematics"):
                    q = robot.inverse_kinematics(_pose_xyz(x, y, z_cmd))
                    robot.move(pyn.JointsPosition(*q))
                    live = robot.get_pose()
                    if abs(live.z - z_now) > 0.001:
                        print(f"[9Z] {label}/IK     {i}/{len(segments)} -> z={live.z:.3f}")
                        z_now = live.z
                        moved = True
                    else:
                        err_msg = "IK executed but no z change"
                else:
                    err_msg = "no IK API on this PyNiryo version"
            except Exception as e:
                err_msg = f"IK: {e}"

        if not moved:
            print(f"[9Z ERR] {label} step {i}/{len(segments)} failed ({err_msg}).")
            return False

    # Snap to exact target
    try:
        if hasattr(robot, "move_linear"):
            if frame_name is None:
                robot.move_linear(_pose_xyz(x, y, z_target))
            else:
                robot.move_linear(_pose_xyz(x, y, z_target), frame_name)
        else:
            if frame_name is None:
                robot.move(_pose_xyz(x, y, z_target))
            else:
                robot.move(_pose_xyz(x, y, z_target), frame_name)
    except Exception:
        pass

    live = robot.get_pose()
    print(f"[9Z] {label} final -> z={live.z:.3f}")
    return True

""" def _move_Z(label, z_target, step=0.008, frame_name=None):
    
    Move straight in Z with small increments at fixed (x_pick, y_pick, r_pick/p_pick/yaw_pick).
   
    z_now = robot.get_pose().z
    direction = 1.0 if z_target > z_now else -1.0
    n_full = int(abs(z_target - z_now) // step)
    remainder = abs(z_target - z_now) - n_full * step
    segments = [step] * n_full + ([remainder] if remainder > 1e-6 else [])

    for i, dz in enumerate(segments, 1):
        z_cmd = z_now + direction * dz
        moved = False
        err_msg = None

        # Try linear Cartesian if available
        try:
            if hasattr(robot, "move_linear"):
                if frame_name is None:
                    robot.move_linear(_pose_xyz(x_pick, y_pick, z_cmd))
                else:
                    robot.move_linear(_pose_xyz(x_pick, y_pick, z_cmd), frame_name)
                live = robot.get_pose()
                if abs(live.z - z_now) > 0.001:
                    print(f"[9Z] {label}/linear {i}/{len(segments)} -> z={live.z:.3f}")
                    z_now = live.z
                    moved = True
                else:
                    err_msg = "linear executed but no z change"
        except Exception as e:
            err_msg = f"linear: {e}"

        # Fallback: regular move
        if not moved:
            try:
                if frame_name is None:
                    robot.move(_pose_xyz(x_pick, y_pick, z_cmd))
                else:
                    robot.move(_pose_xyz(x_pick, y_pick, z_cmd), frame_name)
                live = robot.get_pose()
                if abs(live.z - z_now) > 0.001:
                    print(f"[9Z] {label}/move   {i}/{len(segments)} -> z={live.z:.3f}")
                    z_now = live.z
                    moved = True
                else:
                    err_msg = "move executed but no z change"
            except Exception as e:
                err_msg = f"move: {e}"

        # IK -> joints fallback
        if not moved:
            try:
                if hasattr(robot, "inverse_kinematics"):
                    q = robot.inverse_kinematics(_pose_xyz(x_pick, y_pick, z_cmd))
                    robot.move(pyn.JointsPosition(*q))
                    live = robot.get_pose()
                    if abs(live.z - z_now) > 0.001:
                        print(f"[9Z] {label}/IK     {i}/{len(segments)} -> z={live.z:.3f}")
                        z_now = live.z
                        moved = True
                    else:
                        err_msg = "IK executed but no z change"
                else:
                    err_msg = "no IK API on this PyNiryo version"
            except Exception as e:
                err_msg = f"IK: {e}"

        if not moved:
            print(f"[9Z ERR] {label} step {i}/{len(segments)} failed ({err_msg}).")
            return False

    # Snap to exact target
    try:
        if hasattr(robot, "move_linear"):
            if frame_name is None:
                robot.move_linear(_pose_xyz(x_pick, y_pick, z_target))
            else:
                robot.move_linear(_pose_xyz(x_pick, y_pick, z_target), frame_name)
        else:
            if frame_name is None:
                robot.move(_pose_xyz(x_pick, y_pick, z_target))
            else:
                robot.move(_pose_xyz(x_pick, y_pick, z_target), frame_name)
    except Exception:
        pass

    live = robot.get_pose()
    print(f"[9Z] {label} final -> z={live.z:.3f}")
    return True """

print(f"[9 PLAN] {XZ_YZ_STR()} | Zs: OVER={Z_OVER:.3f}, HOVER={Z_HOVER:.3f}, PRE={Z_PRE:.3f}, GRASP={Z_GRASP:.3f}")

# Tool + safety
try:
    robot.update_tool()
except Exception as e:
    print("[9 WARN] update_tool:", e)
try:
    if robot.get_learning_mode():
        robot.set_learning_mode(False)
except Exception as e:
    print("[9 WARN] learning_mode:", e)

# Ensure over-pick at correct RPY
try:
    robot.move(_pose_xyz(x_pick, y_pick, Z_OVER))
    live = robot.get_pose()
    print(f"[9 START] at over-pick: z={live.z:.3f}")
except Exception as e:
    print("[9 ERR] move to over-pick:", e)

# Open gripper
try:
    robot.open_gripper(100)
    print("[9] gripper opened.")
except Exception as e:
    print("[9 WARN] open_gripper:", e)

# Descend ladder
if not _move_Z("to HOVER", Z_HOVER, step=0.010):
    print("[9 ABORT] Could not reach HOVER.")
if not _move_Z("to PRE",   Z_PRE,   step=0.008):
    print("[9 ABORT] Could not reach PRE.")
if not _move_Z("to GRASP", Z_GRASP, step=0.004):
    print("[9 ABORT] Could not reach GRASP.")

# Close + lift
try:
    print("[9] closing gripper…")
    robot.close_gripper(100)
    time.sleep(0.25)
except Exception as e:
    print("[9 WARN] close_gripper:", e)

if not _move_Z("lift to OVER", Z_OVER, step=0.010):
    print("[9 WARN] lift had an issue.")

print("[9 DONE] grasp sequence complete.")

# ---------------------------------------------
# Step 10 - Choose place target (XY + Z levels)
# ---------------------------------------------
print("\n[STEP 10] Define place target on the table.")

# Default place location
# TODO: measure a nice drop point with the pendant and update:
X_PLACE_DEFAULT = 0.193     # example, adjust!
Y_PLACE_DEFAULT = -0.100    # example, adjust!

# Allow user to override via pendant coordinates (optional)
user_place = input("[10] Place target 'x y' in meters (or press Enter to use defaults): ").strip()
if user_place:
    try:
        xp, yp = map(float, user_place.split())
        X_PLACE_RAW, Y_PLACE_RAW = xp, yp
        print(f"[10] Using user-defined place XY=({X_PLACE_RAW:.4f}, {Y_PLACE_RAW:.4f})")
    except Exception as e:
        print(f"[10 WARN] Could not parse input ({e}); falling back to defaults.")
        X_PLACE_RAW, Y_PLACE_RAW = X_PLACE_DEFAULT, Y_PLACE_DEFAULT
else:
    X_PLACE_RAW, Y_PLACE_RAW = X_PLACE_DEFAULT, Y_PLACE_DEFAULT
    print(f"[10] Using default place XY=({X_PLACE_RAW:.4f}, {Y_PLACE_RAW:.4f})")

# Clamp to same safe envelope as pick
X_PLACE = _clamp(X_PLACE_RAW, X_MIN, X_MAX)
Y_PLACE = _clamp(Y_PLACE_RAW, Y_MIN, Y_MAX)
if (X_PLACE != X_PLACE_RAW) or (Y_PLACE != Y_PLACE_RAW):
    print(f"[10 SAFE] Place XY clamped from ({X_PLACE_RAW:.3f}, {Y_PLACE_RAW:.3f}) "
          f"to ({X_PLACE:.3f}, {Y_PLACE:.3f}) within X[{X_MIN:.2f},{X_MAX:.2f}], Y[{Y_MIN:.2f},{Y_MAX:.2f}]")
else:
    print(f"[10 SAFE] Place XY inside envelope: ({X_PLACE:.3f}, {Y_PLACE:.3f})")

# Z levels for placing the cube
# Reuse cube_half (from Step 7) and table height
PLACE_HOVER_PAD = 0.020   # 20 mm above cube center
PLACE_PRE_PAD   = 0.012   # 12 mm above cube center
PLACE_PAD       = 0.006   # 6 mm above cube center (release height)

Z_PLACE_CENTER = Z_TABLE_M + cube_half
Z_PLACE_HOVER  = Z_PLACE_CENTER + PLACE_HOVER_PAD
Z_PLACE_PRE    = Z_PLACE_CENTER + PLACE_PRE_PAD
Z_PLACE        = Z_PLACE_CENTER + PLACE_PAD

print(f"[10 Z] Z_TABLE={Z_TABLE_M:.3f}, cube_half={cube_half:.3f}")
print(f"[10 Z] Z_PLACE_CENTER={Z_PLACE_CENTER:.3f}, "
      f"HOVER={Z_PLACE_HOVER:.3f}, PRE={Z_PLACE_PRE:.3f}, PLACE={Z_PLACE:.3f}")

# Globals for later steps (glide + Z-ladder for place)
X_PLACE_G   = X_PLACE
Y_PLACE_G   = Y_PLACE
Z_PLACE_HOV = Z_PLACE_HOVER
Z_PLACE_PRE = Z_PLACE_PRE
Z_PLACE_FIN = Z_PLACE

print("[STEP 10] Place target defined.")

# --------------------------------------------------------
# Step 11 - progressive glide from over-pick to over-place
# --------------------------------------------------------
print("\n[STEP 11] Glide from over-pick to over-place at Z_OVER.")

# (Optional) re-assert over-pick pose to be safe
try:
    robot.move(PO(x_pick, y_pick, Z_OVER, r_pick, p_pick, yaw_pick))
    live = robot.get_pose()
    print(f"[11 START] Reached over-pick: x={live.x:.3f}, y={live.y:.3f}, z={live.z:.3f}")
except Exception as e:
    print(f"[11 WARN] Could not re-move to over-pick: {e}")
    live = robot.get_pose()
    print(f"[11 INFO] Continuing from current pose x={live.x:.3f}, y={live.y:.3f}, z={live.z:.3f}")

print(f"[11 PLAN] over-place @ Z_OVER: XY=({X_PLACE_G:.3f},{Y_PLACE_G:.3f}), Z_OVER={Z_OVER:.3f}")

ok_place_glide = progressive_glide(
    live.x, live.y,          # start from current XY (ideally x_pick,y_pick)
    X_PLACE_G, Y_PLACE_G,    # target XY for place
    n=12,
    z=Z_OVER
)

if not ok_place_glide:
    raise RuntimeError("[11 ABORT] Could not reach over-place at safe Z.")
else:
    live2 = robot.get_pose()
    print(f"[11 DONE] Now at over-place: x={live2.x:.3f}, y={live2.y:.3f}, z={live2.z:.3f}")

# ------------------------------------------------------------
# Step 12 — descend at place XY (hover → pre → place-release)
# Step 13 — open gripper and lift back to Z_OVER
# ------------------------------------------------------------
print("\n[STEP 12/13] Descend to place height, release cube, lift.")

# (Optional) ensure we are really over-place at correct RPY
try:
    robot.move(PO(X_PLACE_G, Y_PLACE_G, Z_OVER, r_pick, p_pick, yaw_pick))
    live = robot.get_pose()
    print(f"[12 START] at over-place: x={live.x:.3f}, y={live.y:.3f}, z={live.z:.3f}")
except Exception as e:
    print(f"[12 WARN] Could not re-move to over-place: {e}")
    live = robot.get_pose()
    print(f"[12 INFO] Continuing from current pose x={live.x:.3f}, y={live.y:.3f}, z={live.z:.3f}")

print(f"[12 PLAN] Z ladder at place XY=({X_PLACE_G:.3f},{Y_PLACE_G:.3f}) "
      f"Zs: OVER={Z_OVER:.3f}, HOVER={Z_PLACE_HOV:.3f}, PRE={Z_PLACE_PRE:.3f}, PLACE={Z_PLACE_FIN:.3f}")

# Descend ladder at place XY
if not _move_Z("PLACE to HOVER", Z_PLACE_HOV, step=0.010, x=X_PLACE_G, y=Y_PLACE_G):
    print("[12 ABORT] Could not reach PLACE_HOVER.")
if not _move_Z("PLACE to PRE",   Z_PLACE_PRE, step=0.008, x=X_PLACE_G, y=Y_PLACE_G):
    print("[12 ABORT] Could not reach PLACE_PRE.")
if not _move_Z("to PLACE",       Z_PLACE_FIN, step=0.004, x=X_PLACE_G, y=Y_PLACE_G):
    print("[12 ABORT] Could not reach PLACE (release height).")

# Step 13 — Open gripper and lift back to Z_OVER
try:
    print("[13] opening gripper to release cube…")
    robot.open_gripper(100)
    time.sleep(0.25)
except Exception as e:
    print(f"[13 WARN] open_gripper: {e}")

if not _move_Z("lift from PLACE to OVER", Z_OVER, step=0.030, x=X_PLACE_G, y=Y_PLACE_G):
    print("[13 WARN] lift from PLACE had an issue.")

# Optional: glide back to observation pose
try:
    print("[13] gliding back to observation XY at Z_OVER…")
    ok_back = progressive_glide(
        X_PLACE_G, Y_PLACE_G,
        OBS_POSE_FK.x, OBS_POSE_FK.y,
        n=12,
        z=Z_OVER
    )
    if not ok_back:
        print("[13 WARN] Glide back to obs failed; check IK / workspace.")
except Exception as e:
    print(f"[13 WARN] Return glide failed: {e}")

print("[13 DONE] place sequence complete (cube should be on the table).")