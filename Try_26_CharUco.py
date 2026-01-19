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
RUN_DIR = Path("calibration_data_charuco") / "run_20260113_144602"

# camera init
CAMERA_INDEX = 0
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
SHOW_PREVIEW = True

SETTLE_CAM_S   = 0.3
FLUSH_FRAMES   = 6
# ------------------

# ArUco
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50
TARGET_ID     = 0
MARKER_LEN_M  = 0.026
# -----------------------------------

# Virtual hand reference (used later in pose estimation)
Q6_REF = 0.0

# Logging (separate from calibration run)
LOG_DIR = Path("aruco_logs") / datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Subpixel refinement (good to keep)
ENABLE_SUBPIX     = True
SUBPIX_WINSIZE    = 5
SUBPIX_MAX_ITERS  = 30
SUBPIX_MIN_ACC    = 0.01

# Multi-view (we will use later, but keep the switch here)
USE_MULTIVIEW_ON_SNAPSHOT = True

# Observation Views
# Convention: view0 = PRIMARY view. The rest are optional coverage views.
OBS_VIEWS = [
    [0.0692,0.4766,-0.8567,0.0108,-1.2426,-0.0581],    # view0 primary
    [-0.8453,-0.1323,-0.5704,0.4572,-1.5448,-0.1655],  # view1 from right
    [0.7237,-0.3019,-0.1719,-0.2836,-1.7657,0.2087],   # view2 from left
    [0.2762,-0.1156,-0.5825,-0.2806,-1.3499,0.155],    # view3 right coverage
    [-0.2153,-0.0671,-0.5719,0.7026,-1.4405,-0.1164],  # view4 left coverage
    [0.6339,0.163,-0.7112,0.0323,-1.0447,0.5569],      # view5 bad zone coverage
    [0.0692,-0.4156,-0.1386,-0.1379,-1.7196,0.2347],   # view6 right side edge zone
]

PRIMARY_OBS = OBS_VIEWS[0]

# Detect BOTH markers (orange + blue)
TARGET_IDS = [0, 1]           # 0=orange, 1=blue (adjust if needed)
REQUIRE_BOTH = True           # if True: we "want" both present
AREA_MIN_PX2 = 3500           # gate for “too small / too blurry”
# -----------------------------------------------------------------

# Step 6: Snapshot settings -------------------
SNAP_N                 = 12      # how many good samples to collect
SNAP_REPROJ_MAX_PX     = 0.40    # quality gate
SNAP_AREA_MIN_PX2      = 3500    # can be >= AREA_MIN_PX2 (often same is fine)
SNAP_TIMEOUT_S         = 6.0     # stop if we can't get enough good frames
SNAP_REQUIRE_ALL_IDS   = False   # True: need N samples for ALL target_ids
SNAP_SAVE_NPZ          = True
# ----------------------------

# Step 8: Motion configs ------------------------
ENABLE_MOTION = True
MOTION_KEY = "p"                 # press to execute pick+place-back once
MOTION_SELECT_MODE = "best"
ACTIVE_ID_FOR_MOTION = 0
MANUAL_ID_FOR_MOTION = 0

CUBE_HALF_M = 0.0185             # 37mm cube -> half
PICK_PAD_M  = 0.0020             # small pad above mid-height
Z_THRESH_MARKER_ON_CUBE = 0.020  # if z_marker > this -> marker likely on cube top

Z_PRE_OFFSET_M   = 0.020         # pre-grasp height above pick
Z_HOVER_OFFSET_M = 0.050         # hover height above pick
Z_OVER_MIN_M     = 0.18          # ensure travel is not too low

MOVE_SETTLE_S = 0.25
GRIPPER_WAIT_S = 0.35
# -------------------

# Multi-view functionality
USE_MULTIVIEW_ON_SNAPSHOT = True

MV_REPROJ_MAX_PX = 0.35          # candidate gate across views (can reuse SNAP_REPROJ_MAX_PX)
MV_MEAN_AREA_MIN = 4500          # candidate gate across views
MV_MAXDEV_MAX_MM = 2.5           # optional: reject “jittery” views
MV_SPREAD_MAX_MM = 15.0          # optional: if views disagree too much -> warn/abort for motion
MV_SPREAD_PRIORITIZE_REPROJ_MM = 6.0   # if cross-view disagreement > this, prefer reproj/area over consensus
MV_DEBUG_CHOICE = False               # set True to print ranking diagnostics

# helper ---------
def load_intrinsics(run_dir: Path):
    intr_path = run_dir / "intrinsics_charuco_pinhole_final.npz"
    data = np.load(intr_path)

    K = data["K"].astype(np.float64)
    dist = data["dist"].astype(np.float64).reshape(-1)   # keep 1D

    image_size = None
    if "image_size" in data.files:
        image_size = tuple(data["image_size"].tolist())  # (w, h)

    return intr_path, K, dist, image_size

def load_handeye(run_dir: Path):
    he_path = run_dir / "handeye_charuco_hand_TSAI.npz"
    data = np.load(he_path)

    T_hand_camera = data["T_hand_camera"].astype(np.float64)

    q6_ref = None
    if "q6_ref" in data.files:
        q6_ref = float(data["q6_ref"])

    used_image_names = None
    if "used_image_names" in data.files:
        used_image_names = data["used_image_names"]

    return he_path, T_hand_camera, q6_ref, used_image_names

def rotmat_to_rpy_zyx(R: np.ndarray) -> np.ndarray:
    """
    Inverse of R = Rz(yaw) @ Ry(pitch) @ Rx(roll) (same convention as rpy_to_rot_matrix).
    Returns [roll, pitch, yaw] in radians.
    """
    R = R.astype(float)
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-9

    if not singular:
        roll  = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw   = np.arctan2(R[1, 0], R[0, 0])
    else:
        # gimbal-lock fallback
        roll  = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw   = 0.0

    return np.array([roll, pitch, yaw], dtype=float)


def closest_to_median_index(xyz: np.ndarray) -> int:
    med = np.median(xyz, axis=0)
    d = np.linalg.norm(xyz - med, axis=1)
    return int(np.argmin(d))


def save_snapshot_npz(path: Path, snap: dict):
    """
    snap structure:
      snap[mid]["xyz"]  -> (N,3)
      snap[mid]["T"]    -> (N,4,4)
      snap[mid]["area"] -> (N,)
      snap[mid]["reproj"]->(N,)
    """
    out = {}
    for mid, d in snap.items():
        out[f"id{mid}_xyz"] = d["xyz"]
        out[f"id{mid}_T"] = d["T"]
        out[f"id{mid}_area"] = d["area"]
        out[f"id{mid}_reproj"] = d["reproj"]
    np.savez(path, **out)

def draw_axes_if_visible(frame, K, dist, rvec, tvec, axis_len, thickness=2):
    h, w = frame.shape[:2]

    # project origin + 3 axis endpoints
    obj = np.float32([
        [0, 0, 0],
        [axis_len, 0, 0],
        [0, axis_len, 0],
        [0, 0, axis_len],
    ])
    imgpts, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    imgpts = imgpts.reshape(-1, 2)

    # if any endpoint is outside -> skip drawing (prevents the OpenCV warning)
    if np.any(imgpts[:, 0] < 0) or np.any(imgpts[:, 0] >= w) or np.any(imgpts[:, 1] < 0) or np.any(imgpts[:, 1] >= h):
        return False

    cv2.drawFrameAxes(frame, K, dist, rvec, tvec, axis_len, thickness)
    return True

def snapshot_one_view(cap, detector, K, dist, marker_len_m,
                      T_base_hand, T_hand_camera,
                      target_ids, area_min_px2,
                      snap_n, reproj_max_px, timeout_s):
    """
    Collect snap_n valid pose samples per marker (for the CURRENT robot view).
    Returns: dict mid -> result dict (contains xyz/T/area/reproj arrays + summary fields)
    """
    t0 = time.time()
    snap = {}  # mid -> lists

    while True:
        if (time.time() - t0) > timeout_s:
            break

        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            continue

        ids_flat = ids.flatten().tolist()

        for i, mid in enumerate(ids_flat):
            if mid not in target_ids:
                continue

            c = corners[i].reshape(4, 2)
            area = marker_area_px2(c)
            if area < area_min_px2:
                continue

            out = solve_pnp_marker(c, K, dist, marker_len_m)
            if out[0] is None:
                continue
            (rvec, tvec), reproj = out
            if reproj > reproj_max_px:
                continue

            T_cam_marker = T_from_rvec_tvec(rvec, tvec)
            T_base_marker = T_base_hand @ T_hand_camera @ T_cam_marker
            xyz = T_base_marker[:3, 3].copy()

            if mid not in snap:
                snap[mid] = {"xyz": [], "T": [], "area": [], "reproj": []}
            snap[mid]["xyz"].append(xyz)
            snap[mid]["T"].append(T_base_marker.copy())
            snap[mid]["area"].append(float(area))
            snap[mid]["reproj"].append(float(reproj))

        # stop condition: at least one id has snap_n (or all ids if you prefer)
        if SNAP_REQUIRE_ALL_IDS:
            done = all(len(snap.get(mid, {}).get("xyz", [])) >= snap_n for mid in target_ids)
        else:
            done = any(len(d["xyz"]) >= snap_n for d in snap.values())

        if done:
            break

    # convert to arrays + compute summary
    results = {}
    for mid, d in snap.items():
        xyz = np.array(d["xyz"], dtype=float)[:snap_n]
        Tset = np.array(d["T"], dtype=float)[:snap_n]
        area = np.array(d["area"], dtype=float)[:snap_n]
        reproj = np.array(d["reproj"], dtype=float)[:snap_n]

        if xyz.shape[0] == 0:
            continue

        mean = xyz.mean(axis=0)
        std_mm = xyz.std(axis=0) * 1000.0
        med = np.median(xyz, axis=0)
        max_dev_mm = np.max(np.linalg.norm(xyz - med[None, :], axis=1)) * 1000.0
        mean_area = float(area.mean()) if area.size else float("nan")
        mean_reproj = float(reproj.mean()) if reproj.size else float("nan")

        rep_i = closest_to_median_index(xyz)
        T_rep = Tset[rep_i].copy()
        rpy_deg = np.degrees(rotmat_to_rpy_zyx(T_rep[:3, :3]))

        results[int(mid)] = {
            "n": int(xyz.shape[0]),
            "xyz": xyz, "T": Tset, "area": area, "reproj": reproj,
            "mean_xyz": mean, "std_mm": std_mm, "median_xyz": med,
            "max_dev_mm": float(max_dev_mm),
            "mean_area": float(mean_area), "mean_reproj": float(mean_reproj),
            "rep_T": T_rep, "rep_rpy_deg": rpy_deg
        }
    return results


def choose_multiview_candidate(cands):
    """
    cands: list of result dicts (each already has median_xyz, mean_reproj, mean_area, max_dev_mm, etc.)
    Returns: chosen dict, spread_mm
    """
    meds = np.array([c["median_xyz"] for c in cands], dtype=float)
    med_of_meds = np.median(meds, axis=0)

    dists_m = np.linalg.norm(meds - med_of_meds[None, :], axis=1)
    dists_mm = dists_m * 1000.0
    spread_mm = float(np.max(dists_mm)) if len(dists_mm) else float("inf")

    # Build a sortable key per candidate (smaller is better).
    # Switch strategy depending on how much the views disagree.
    quality_first = (spread_mm > MV_SPREAD_PRIORITIZE_REPROJ_MM)

    keys = []
    for i, c in enumerate(cands):
        reproj = float(c.get("mean_reproj", 1e9))
        area   = float(c.get("mean_area", 0.0))
        maxdev = float(c.get("max_dev_mm", 1e9))
        dist   = float(dists_mm[i])

        if quality_first:
            # Prefer best pose quality; consensus only as late tie-break.
            key = (reproj, -area, maxdev, dist)
        else:
            # Prefer consensus; still use quality as tie-break.
            key = (dist, reproj, -area, maxdev)

        keys.append(key)

    best_idx = min(range(len(cands)), key=lambda i: keys[i])

    if MV_DEBUG_CHOICE:
        mode = "QUALITY_FIRST" if quality_first else "CONSENSUS_FIRST"
        print(f"[CHOOSE] mode={mode} spread_mm={spread_mm:.1f}")
        order = sorted(range(len(cands)), key=lambda i: keys[i])
        for rank, i in enumerate(order[:min(5, len(order))]):
            c = cands[i]
            print(f"  rank={rank} view={c.get('view_index')} key={keys[i]} "
                  f"reproj={c.get('mean_reproj'):.3f}px area={c.get('mean_area'):.0f} "
                  f"maxdev={c.get('max_dev_mm'):.2f}mm dist={dists_mm[i]:.1f}mm")

    return cands[best_idx], spread_mm

def multiview_snapshot(robot, cap, detector, K, dist, marker_len_m,
                      T_hand_camera, q6_ref,
                      obs_views, target_ids,
                      snap_n, reproj_max_px, area_min_px2, timeout_s):
    """
    Moves through obs_views, collects per-view snapshot results, then selects best per ID.
    Returns: last_snapshot_result dict: mid -> chosen result dict (same format as before),
             plus prints per-view + final selection.
    """
    all_views = []  # list of dict mid->result

    for vi, q in enumerate(obs_views):
        print(f"\n[MULTIVIEW] View {vi}/{len(obs_views)-1}: moving...")
        move_to_joints(robot, q)
        time.sleep(MOVE_SETTLE_S)
        flush_camera(cap, n=8)

        joints = robot.get_joints()
        T_base_hand = fk_base_to_hand_virtual(robot, joints, q6_ref=q6_ref)

        view_res = snapshot_one_view(
            cap, detector, K, dist, marker_len_m,
            T_base_hand, T_hand_camera,
            target_ids=target_ids,
            area_min_px2=area_min_px2,
            snap_n=snap_n,
            reproj_max_px=reproj_max_px,
            timeout_s=timeout_s
        )
        all_views.append(view_res)

        # per-view print (same style)
        for mid, r in view_res.items():
            mean = r["mean_xyz"]; std_mm = r["std_mm"]; med = r["median_xyz"]
            print(f"[MULTIVIEW][V{vi}] id={mid} n={r['n']} "
                  f"mean_xyz=[{mean[0]:.4f},{mean[1]:.4f},{mean[2]:.4f}] "
                  f"std_mm=[{std_mm[0]:.2f},{std_mm[1]:.2f},{std_mm[2]:.2f}] "
                  f"median_xyz=[{med[0]:.4f},{med[1]:.4f},{med[2]:.4f}] "
                  f"max_dev_mm={r['max_dev_mm']:.2f} mean_area={r['mean_area']:.0f} "
                  f"mean_reproj={r['mean_reproj']:.3f}px")

    # go back to primary obs at end
    move_to_joints(robot, PRIMARY_OBS)
    time.sleep(MOVE_SETTLE_S)
    flush_camera(cap, n=8)

    # choose best per marker ID
    last_snapshot_result = {}
    for mid in target_ids:
        cands = []
        for vi, view_res in enumerate(all_views):
            if mid not in view_res:
                continue
            r = view_res[mid]
            # candidate gates
            if r["n"] < snap_n:
                continue
            if r["mean_reproj"] > MV_REPROJ_MAX_PX:
                continue
            if r["mean_area"] < MV_MEAN_AREA_MIN:
                continue
            if r["max_dev_mm"] > MV_MAXDEV_MAX_MM:
                continue

            rr = dict(r)  # copy
            rr["view_index"] = vi
            cands.append(rr)

        if not cands:
            print(f"[MULTIVIEW] id={mid}: no valid candidates after gating.")
            continue

        chosen, spread_mm = choose_multiview_candidate(cands)
        last_snapshot_result[int(mid)] = chosen

        print(f"[MULTIVIEW] id={mid}: chose view={chosen['view_index']} "
              f"median_xyz=[{chosen['median_xyz'][0]:.4f},{chosen['median_xyz'][1]:.4f},{chosen['median_xyz'][2]:.4f}] "
              f"mean_area={chosen['mean_area']:.0f} mean_reproj={chosen['mean_reproj']:.3f}px "
              f"spread_mm={spread_mm:.1f}")

        if spread_mm > MV_SPREAD_MAX_MM:
            print(f"[MULTIVIEW][WARN] id={mid}: view disagreement spread={spread_mm:.1f}mm > {MV_SPREAD_MAX_MM}mm")

    return last_snapshot_result

def choose_pick_id_for_motion(last_snapshot_result, mode, active_id, manual_id):
    """
    last_snapshot_result: dict mid -> {
        "mean_reproj", "mean_area", "max_dev_mm", "n", ...
    }
    Returns: (pick_id, reason_str)
    """
    if not last_snapshot_result:
        return None, "no snapshot stored"

    ids = sorted(last_snapshot_result.keys())
    if len(ids) == 1:
        return ids[0], "only one stored"

    mode = str(mode).lower().strip()

    if mode == "manual":
        if manual_id in last_snapshot_result:
            return manual_id, f"manual id={manual_id}"
        return None, f"manual id={manual_id} not stored"

    if mode == "active":
        if active_id in last_snapshot_result:
            return active_id, f"active id={active_id}"
        return None, f"active id={active_id} not stored"

    # default: "best"
    def score(mid):
        d = last_snapshot_result[mid]
        mean_reproj = float(d.get("mean_reproj", 1e9))
        mean_area   = float(d.get("mean_area", 0.0))
        max_dev_mm  = float(d.get("max_dev_mm", 1e9))
        n           = int(d.get("n", 0))
        # lower reproj better, higher area better, lower jitter better, higher n better
        return (mean_reproj, -mean_area, max_dev_mm, -n)

    best_id = min(ids, key=score)
    return best_id, "best-by-(reproj,area,jitter,n)"

# Step 8: motion
def move_to_joints(robot, q):
    """Move robot to joint configuration q (length 6). Forces float types."""
    q = [float(v) for v in q]
    robot.move(pyn.JointsPosition(*q))

def move_pose(robot, x, y, z, rpy, linear=False):
    """Move to cartesian pose using Niryo PoseObject."""
    pose = pyn.PoseObject(float(x), float(y), float(z),
                          float(rpy[0]), float(rpy[1]), float(rpy[2]))
    robot.move(pose, linear=bool(linear))

def flush_camera(cap, n=8):
    """Read and discard a few frames (useful after robot motion)."""
    for _ in range(int(n)):
        cap.read()

def compute_pick_z(z_marker, cube_half_m, pad_m, z_thresh_m=0.02):
    """
    If marker is on cube top: z_pick = z_marker - cube_half + pad.
    If marker is on table:    z_pick = z_marker + cube_half + pad.
    """
    z_marker = float(z_marker)
    if z_marker > float(z_thresh_m):
        return z_marker - float(cube_half_m) + float(pad_m)
    else:
        return z_marker + float(cube_half_m) + float(pad_m)

def move_hover_above(robot, x, y, z_hover, rpy, z_safe):
    """
    Safe 3-step: lift to z_safe (at current xy), travel xy at z_safe, descend to z_hover.
    """
    cur = robot.get_pose()
    z_safe = max(float(z_safe), float(cur.z), float(z_hover))
    move_pose(robot, cur.x, cur.y, z_safe, rpy, linear=False)
    move_pose(robot, x, y, z_safe, rpy, linear=False)
    move_pose(robot, x, y, z_hover, rpy, linear=True)

def descend_ladder(robot, x, y, z_hover, z_pick, rpy, pre_dz):
    """
    Hover -> Pre -> Pick (vertical-ish). Uses linear moves for safety.
    """
    z_hover = float(z_hover)
    z_pick  = float(z_pick)
    z_pre   = max(z_pick + float(pre_dz), z_pick)  # ensure above pick

    # If we're already below z_pre somehow, go back to hover first.
    cur = robot.get_pose()
    if cur.z < z_pre - 0.002:
        move_pose(robot, x, y, z_hover, rpy, linear=True)

    move_pose(robot, x, y, z_pre,  rpy, linear=True)
    move_pose(robot, x, y, z_pick, rpy, linear=True)

# For Step 5: Detection
def make_aruco_detector():
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    params = aruco.DetectorParameters()

    # Subpixel corner refinement (OpenCV will do it inside detectMarkers)
    if ENABLE_SUBPIX and hasattr(params, "cornerRefinementMethod"):
        params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = int(SUBPIX_WINSIZE)
        params.cornerRefinementMaxIterations = int(SUBPIX_MAX_ITERS)
        params.cornerRefinementMinAccuracy = float(SUBPIX_MIN_ACC)

    # Newer OpenCV
    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, params)
        return detector, dictionary, params

    # Fallback older OpenCV (no ArucoDetector)
    return None, dictionary, params

# For Step 6: Pose estimation
def rpy_to_rot_matrix(rpy):
    roll, pitch, yaw = rpy
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll),  np.cos(roll)]], dtype=float)
    Ry = np.array([[ np.cos(pitch), 0, np.sin(pitch)],
                   [0,              1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]], dtype=float)
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw),  np.cos(yaw), 0],
                   [0,            0,           1]], dtype=float)
    return Rz @ Ry @ Rx

def niryo_pose_to_matrix_from_obj(pose):
    T = np.eye(4, dtype=float)
    T[:3, :3] = rpy_to_rot_matrix([pose.roll, pose.pitch, pose.yaw])
    T[:3,  3] = [pose.x, pose.y, pose.z]
    return T

def fk_base_to_hand_virtual(robot, joints, q6_ref=0.0):
    q = np.array(joints, dtype=float).reshape(6,)
    q[5] = float(q6_ref)  # freeze q6
    pose_fk = robot.forward_kinematics(q.tolist())
    return niryo_pose_to_matrix_from_obj(pose_fk)

def T_from_rvec_tvec(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T

def solve_pnp_marker(corners_4x2, K, dist, marker_len_m):
    # OpenCV aruco corners order: TL, TR, BR, BL
    s = float(marker_len_m)
    objp = np.array([[-s/2,  s/2, 0],
                     [ s/2,  s/2, 0],
                     [ s/2, -s/2, 0],
                     [-s/2, -s/2, 0]], dtype=np.float64)

    imgp = corners_4x2.reshape(4, 2).astype(np.float64)

    # Prefer IPPE_SQUARE if available
    flag = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE)
    ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, dist, flags=flag)
    if not ok:
        return None, None

    # reprojection error (mean pixel error over 4 corners)
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    reproj = float(np.mean(np.linalg.norm(proj - imgp, axis=1)))
    return (rvec, tvec), reproj

def marker_area_px2(corners_1x4x2: np.ndarray) -> float:
    c = corners_1x4x2.reshape(-1, 2).astype(np.float32)
    return float(cv2.contourArea(c))

def live_marker_detection(cap, detector_tuple, K, dist, marker_len_m,
                          robot, T_hand_camera, q6_ref, log_dir,
                          target_ids=(0, 1), area_min_px2=3500, require_both=False):
    detector, dictionary, params = detector_tuple

    print("\n[INFO] Step5: Live ArUco detection + basic pose (raw frame)")
    print("       Keys: [q]=quit, [s]=print base pose now")
    print(f"       Target IDs: {list(target_ids)} | require_both={require_both} | area_min={area_min_px2}px^2")

    last_pose_dict = {}  # id -> dict with rvec,tvec,reproj,area,T_base_marker
    last_info_lines = []
    last_status_line = ""
    last_snapshot_result = {}  # mid -> dict with median_xyz, rep_T, etc.
    last_fail_print_t = 0.0
    consec_fail = 0

    # --- Step 6 snapshot state ---
    snap_active = False
    snap_t0 = 0.0
    snap = {}  # mid -> dict of lists

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            consec_fail += 1
            now = time.time()
            if now - last_fail_print_t > 1.0:  # print at most once per second
                print(f"[WARN] Camera frame read failed. (consecutive={consec_fail})")
                last_fail_print_t = now
            time.sleep(0.01)
            continue
        else:
            consec_fail = 0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        ok_ids = []
        info_lines = []
        last_pose_dict = {}

        # We'll compute FK once per frame (only if we actually need it)
        T_base_hand = None
        joints = None

        if ids is not None and len(ids) > 0:
            ids_flat = ids.flatten().tolist()

            for i, mid in enumerate(ids_flat):
                c = corners[i].reshape(4, 2)
                area = marker_area_px2(c)
                passed = (area >= area_min_px2)

                # overlay marker outline
                cv2.polylines(frame, [c.astype(np.int32)], True, (0, 255, 0), 2)

                # overlay text
                cx, cy = int(np.mean(c[:, 0])), int(np.mean(c[:, 1]))
                txt = f"id={mid} area={int(area)} {'OK' if passed else 'FAIL'}"
                cv2.putText(frame, txt, (cx - 60, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                info_lines.append(txt)

                if (mid in target_ids) and passed:
                    out = solve_pnp_marker(c, K, dist, marker_len_m)
                    if out[0] is None:
                        continue

                    (rvec, tvec), reproj = out
                    ok_ids.append(mid)

                    # FK once per frame (not per marker)
                    if T_base_hand is None:
                        joints = robot.get_joints()
                        T_base_hand = fk_base_to_hand_virtual(robot, joints, q6_ref=q6_ref)

                    T_cam_marker = T_from_rvec_tvec(rvec, tvec)
                    T_base_marker = T_base_hand @ T_hand_camera @ T_cam_marker

                    last_pose_dict[mid] = {
                        "area": area,
                        "reproj": reproj,
                        "rvec": rvec,
                        "tvec": tvec,
                        "T_base_marker": T_base_marker,
                    }

                    # --- Step 6: collect snapshot samples (only while active) ---
                    if snap_active:
                        if (reproj <= SNAP_REPROJ_MAX_PX) and (area >= SNAP_AREA_MIN_PX2):
                            if mid not in snap:
                                snap[mid] = {"xyz": [], "T": [], "area": [], "reproj": []}
                            snap[mid]["xyz"].append(T_base_marker[:3, 3].copy())
                            snap[mid]["T"].append(T_base_marker.copy())
                            snap[mid]["area"].append(float(area))
                            snap[mid]["reproj"].append(float(reproj))

                    # Draw axes every frame for every valid pose
                    draw_axes_if_visible(frame, K, dist, rvec, tvec, marker_len_m * 0.7, 2)

        ok_unique = sorted(set(ok_ids))
        missing = [m for m in target_ids if m not in ok_unique]
        status_ok = (len(missing) == 0) if require_both else (len(ok_unique) > 0)

        status_line = f"[STATUS] OK_IDS={ok_unique}  missing={missing}  STATUS={'OK' if status_ok else 'NO'}"
        cv2.putText(frame, status_line, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if SHOW_PREVIEW:
            cv2.imshow("ArUco Live", frame)

        key = cv2.waitKey(1) & 0xFF

        # Start snapshot on 's'
        if key == ord("s"):
            if USE_MULTIVIEW_ON_SNAPSHOT:
                print(f"[SNAPSHOT] MULTIVIEW start: views={len(OBS_VIEWS)} N={SNAP_N}, reproj<={SNAP_REPROJ_MAX_PX}, area>={SNAP_AREA_MIN_PX2}")
                last_snapshot_result = multiview_snapshot(
                    robot, cap, detector, K, dist, marker_len_m,
                    T_hand_camera, q6_ref,
                    obs_views=OBS_VIEWS,
                    target_ids=target_ids,
                    snap_n=SNAP_N,
                    reproj_max_px=SNAP_REPROJ_MAX_PX,
                    area_min_px2=SNAP_AREA_MIN_PX2,
                    timeout_s=SNAP_TIMEOUT_S
                )
                print("[SNAPSHOT] Stored for motion IDs:", sorted(last_snapshot_result.keys()))
            else:
                # allow cancel with 's' again if you want
                snap_active = False
                print("[SNAPSHOT] Canceled.")

        # quick manual selection: press '0' or '1'
        if key in (ord("0"), ord("1")):
            MANUAL_ID_FOR_MOTION = int(chr(key))
            MOTION_SELECT_MODE = "manual"
            print(f"[MOTION] Manual selection: MANUAL_ID_FOR_MOTION={MANUAL_ID_FOR_MOTION} (mode=manual)")

        if key == ord(MOTION_KEY):
            preferred_id = int(ACTIVE_ID_FOR_MOTION)

            pick_id, reason = choose_pick_id_for_motion(
                last_snapshot_result,
                mode=MOTION_SELECT_MODE,
                active_id=preferred_id,
                manual_id=MANUAL_ID_FOR_MOTION
            )

            if pick_id is None:
                # fallback: if exactly one snapshot exists, use it
                if len(last_snapshot_result) == 1:
                    pick_id = next(iter(last_snapshot_result.keys()))
                    print(f"[MOTION] {reason}; fallback to only stored id={pick_id}.")
                else:
                    print(f"[MOTION] {reason}. Press 's' first (and ensure IDs are stored).")
                    continue

            # Optional: print why we picked it + its metrics
            md = last_snapshot_result[pick_id]
            print(
                f"[MOTION] Pick selection: mode={MOTION_SELECT_MODE} -> id={pick_id} ({reason}) | "
                f"mean_reproj={md.get('mean_reproj', float('nan')):.3f}px "
                f"mean_area={md.get('mean_area', float('nan')):.0f} "
                f"max_dev_mm={md.get('max_dev_mm', float('nan')):.2f} "
                f"n={md.get('n', -1)}"
            )

            xyz = last_snapshot_result[pick_id]["median_xyz"]
            x_m, y_m, z_m = [float(v) for v in xyz]

            z_pick  = compute_pick_z(z_m, CUBE_HALF_M, PICK_PAD_M, z_thresh_m=Z_THRESH_MARKER_ON_CUBE)
            z_hover = z_pick + Z_HOVER_OFFSET_M

            print(f"[MOTION] Using id={pick_id} median_xyz={[x_m,y_m,z_m]}  z_pick={z_pick:.4f}  z_hover={z_hover:.4f}")

            # --- Pick ---
            robot.open_gripper()
            move_hover_above(robot, x_m, y_m, z_hover, OBS_RPY, Z_OVER_MIN_M)
            descend_ladder(robot, x_m, y_m, z_hover, z_pick, OBS_RPY, Z_PRE_OFFSET_M)
            robot.close_gripper()
            move_pose(robot, x_m, y_m, z_hover, OBS_RPY, linear=True)

            # go OBS before placing (your requested behavior)
            move_to_joints(robot, PRIMARY_OBS)
            flush_camera(cap, n=8)

            # --- Place back at same XY ---
            move_hover_above(robot, x_m, y_m, z_hover, OBS_RPY, Z_OVER_MIN_M)
            descend_ladder(robot, x_m, y_m, z_hover, z_pick, OBS_RPY, Z_PRE_OFFSET_M)
            robot.open_gripper()
            move_pose(robot, x_m, y_m, z_hover, OBS_RPY, linear=True)

            # after placing: lift first, then return OBS (your requested tweak)
            move_to_joints(robot, PRIMARY_OBS)
            flush_camera(cap, n=8)

            print("[MOTION] Done.")

            # ---- Close gripper after motion ----
            time.sleep(GRIPPER_WAIT_S)
            robot.close_gripper()
            time.sleep(GRIPPER_WAIT_S)


        # If active: check completion / timeout and print summary once
        if snap_active:
            # How many samples per id?
            counts = {mid: len(d["xyz"]) for mid, d in snap.items()}
            # decide if "done"
            if SNAP_REQUIRE_ALL_IDS:
                done = all(counts.get(mid, 0) >= SNAP_N for mid in target_ids)
            else:
                done = any(c >= SNAP_N for c in counts.values())

            timed_out = (time.time() - snap_t0) > SNAP_TIMEOUT_S

            if done or timed_out:
                snap_active = False

                if len(snap) == 0:
                    print("[SNAPSHOT] No valid samples collected.")
                else:
                    # Print summary per ID (only those with >=1 sample)
                    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    for mid, d in snap.items():
                        xyz   = np.array(d["xyz"], dtype=float)
                        Tset  = np.array(d["T"], dtype=float)
                        area  = np.array(d["area"], dtype=float)
                        reproj = np.array(d["reproj"], dtype=float)

                        # If we collected more than SNAP_N, keep first SNAP_N for consistency
                        if xyz.shape[0] > SNAP_N:
                            xyz    = xyz[:SNAP_N]
                            Tset   = Tset[:SNAP_N]
                            area   = area[:SNAP_N]
                            reproj = reproj[:SNAP_N]

                        mean = xyz.mean(axis=0)
                        std_mm = xyz.std(axis=0) * 1000.0
                        med = np.median(xyz, axis=0)

                        # NEW: max deviation from median (mm)
                        max_dev_mm = np.max(np.linalg.norm(xyz - med[None, :], axis=1)) * 1000.0

                        # NEW: mean area + mean reproj
                        mean_area = float(area.mean()) if area.size else float("nan")
                        mean_reproj = float(reproj.mean()) if reproj.size else float("nan")

                        rep_i = closest_to_median_index(xyz)
                        Rrep = Tset[rep_i][:3, :3]
                        rpy_deg = np.degrees(rotmat_to_rpy_zyx(Rrep))

                        print(f"[SNAPSHOT] id={mid}  n={xyz.shape[0]}  "
                            f"mean_xyz=[{mean[0]:.4f},{mean[1]:.4f},{mean[2]:.4f}]  "
                            f"std_mm=[{std_mm[0]:.2f},{std_mm[1]:.2f},{std_mm[2]:.2f}]  "
                            f"median_xyz=[{med[0]:.4f},{med[1]:.4f},{med[2]:.4f}]  "
                            f"max_dev_mm={max_dev_mm:.2f}  "
                            f"mean_area={mean_area:.0f}  "
                            f"mean_reproj={mean_reproj:.3f}px  "
                            f"rep_rpy_deg=[{rpy_deg[0]:.2f},{rpy_deg[1]:.2f},{rpy_deg[2]:.2f}]")

                    if SNAP_SAVE_NPZ:
                        # pack and save
                        packed = {}
                        # reset + store latest snapshot results for motion
                        last_snapshot_result = {}
                        for mid, d in snap.items():
                            mid_i = int(mid)

                            xyz   = np.array(d["xyz"], dtype=float)
                            Tset  = np.array(d["T"], dtype=float)
                            area  = np.array(d["area"], dtype=float)
                            reproj = np.array(d["reproj"], dtype=float)

                            if xyz.shape[0] > SNAP_N:
                                xyz    = xyz[:SNAP_N]
                                Tset   = Tset[:SNAP_N]
                                area   = area[:SNAP_N]
                                reproj = reproj[:SNAP_N]

                            mean = xyz.mean(axis=0)
                            std_mm = xyz.std(axis=0) * 1000.0
                            med = np.median(xyz, axis=0)

                            max_dev_mm = np.max(np.linalg.norm(xyz - med[None, :], axis=1)) * 1000.0
                            mean_area = float(area.mean()) if area.size else float("nan")
                            mean_reproj = float(reproj.mean()) if reproj.size else float("nan")

                            rep_i = closest_to_median_index(xyz)
                            T_rep = Tset[rep_i].copy()
                            Rrep = T_rep[:3, :3]
                            rpy_deg = np.degrees(rotmat_to_rpy_zyx(Rrep))

                            # ✅ THIS is what motion needs later
                            last_snapshot_result[mid_i] = {
                                "median_xyz": med.copy(),
                                "rep_T": T_rep,
                                "mean_xyz": mean.copy(),
                                "std_mm": std_mm.copy(),
                                "max_dev_mm": float(max_dev_mm),
                                "mean_area": float(mean_area),
                                "mean_reproj": float(mean_reproj),
                                "rep_rpy_deg": rpy_deg.copy(),
                                "n": int(xyz.shape[0]),
                            }

                            # (your existing print stays the same)
                            print(f"[SNAPSHOT] id={mid_i}  n={xyz.shape[0]}  ...")

                            print("[SNAPSHOT] Stored for motion IDs:", sorted(last_snapshot_result.keys()))

        if key == ord("q"):
            break

        # store last state for printing
        last_info_lines = info_lines
        last_status_line = status_line

        if key == ord("s"):
            print(last_status_line)
            for line in last_info_lines:
                print(" ", line)

            if len(last_pose_dict) == 0:
                print("[POSE] No valid pose (no target IDs passed area/solvePnP).")
            else:
                for mid, d in last_pose_dict.items():
                    tvec = d["tvec"].reshape(3)
                    base_xyz = d["T_base_marker"][:3, 3]
                    print(f"[POSE] id={mid} area={int(d['area'])} reproj={d['reproj']:.3f}px "
                          f"tvec_cam=[{tvec[0]:.3f},{tvec[1]:.3f},{tvec[2]:.3f}] "
                          f"base_xyz=[{base_xyz[0]:.4f},{base_xyz[1]:.4f},{base_xyz[2]:.4f}]")

    cv2.destroyAllWindows()
# -----------------------------------------------------------------------------

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

# Choose which observation view to move to
OBS_IDX = 0
obs_q = OBS_VIEWS[OBS_IDX]          # 6 floats
print("[INFO] OBS_IDX =", OBS_IDX)
print("[INFO] obs_q (target) =", np.round(obs_q, 4).tolist())

# Move to observation pose
robot.move(pyn.JointsPosition(*obs_q))
time.sleep(0.3)  # small settle time

# Print actual state
print("[INFO] joints (actual) =", np.round(robot.get_joints(), 4).tolist())

pose_live = robot.get_pose()
OBS_RPY = np.array([pose_live.roll, pose_live.pitch, pose_live.yaw], dtype=float)
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

# Let camera settle exposure / focus a bit
time.sleep(SETTLE_CAM_S)

# Flush a few frames so we don't start with old buffered images
for _ in range(FLUSH_FRAMES):
    cap.grab()

# Grab one frame to confirm resolution
ret, frame = cap.read()
if not ret or frame is None:
    raise RuntimeError("[ERROR] Failed to read test frame from camera.")

h, w = frame.shape[:2]
print(f"[INFO] Camera connected successfully. Current resolution: {w}×{h}")

# Optionally show one quick preview window (press any key to continue)
if SHOW_PREVIEW:
    cv2.imshow("Camera Test Frame", frame)
    cv2.waitKey(800)
    cv2.destroyAllWindows()

print("\n[SETUP COMPLETE] Robot and camera are ready.\n")

# ========================================
# Step 4: Load calibration + sanity checks
# ========================================
print("\n[INFO] Step4: Load calibration artifacts")

# --- Intrinsics ---
intr_path, K, dist, image_size = load_intrinsics(RUN_DIR)

print("[INFO] Loaded intrinsics:", intr_path)
print("[INFO] K:\n", K)
print("[INFO] dist:", dist, "| shape:", dist.shape)

if image_size is not None:
    w_cal, h_cal = image_size
    print(f"[INFO] intrinsics image_size = {w_cal}x{h_cal}")
    if (w, h) != (w_cal, h_cal):
        print("[WARN] Camera resolution differs from calibration image_size!")
        print("       -> Set camera to calibrated resolution for reliable pose.")
else:
    print("[WARN] intrinsics file has no image_size key")

# --- Hand-eye ---
he_path, T_hand_camera, q6_ref_file, used_names = load_handeye(RUN_DIR)

print("[INFO] Loaded hand-eye:", he_path)
print("[INFO] ||t_hand_camera|| =", float(np.linalg.norm(T_hand_camera[:3, 3])))

if q6_ref_file is not None:
    print(f"[INFO] hand-eye q6_ref = {q6_ref_file:.4f} rad (script Q6_REF={Q6_REF:.4f})")
    if abs(q6_ref_file - float(Q6_REF)) > 1e-6:
        print("[WARN] q6_ref mismatch! Your runtime Q6_REF should match the calibration q6_ref.")
else:
    print("[WARN] hand-eye file has no q6_ref key")

if used_names is not None:
    print(f"[INFO] hand-eye used_image_names: {len(used_names)} views")

# ===================================
# Step 5: Live ArUco marker detection 
# ==================================+
# Step 6: pose estimation
# ==================================+
# Step 7: Snapshot settings
# ==================================+
# Step 8: motion
# ==================================
detector_tuple = make_aruco_detector()

live_marker_detection(
    cap, detector_tuple,
    K, dist, MARKER_LEN_M,
    robot, T_hand_camera, Q6_REF, LOG_DIR,
    target_ids=(0, 1),
    area_min_px2=3500,
    require_both=False  # set True if you want "only OK when both are good"
)

cap.release()
robot.close_connection()