"""
aruco_pose_only_jog_check.py

Pose-only ArUco estimation for manual jogging comparison in Niryo Studio.

Workflow:
1) Run script (robot goes to OBS pose and stays there).
2) Place marker (on table or on cube).
3) Press 's' to snapshot (takes N good samples, prints mean/std in base frame).
4) In Niryo Studio: jog TCP tip to the marker center and note the pose.
5) Compare Niryo Studio base pose vs printed estimate -> you get your bias vector.

Keys:
  q : quit
  s : snapshot (collect N accepted samples -> print mean/std)
"""

import time
import datetime
from pathlib import Path

import cv2
import numpy as np
import pyniryo as pyn


# =========================
# Config
# =========================
ROBOT_IP = "129.187.231.226"

# >>> USE YOUR NEW BEST RUN HERE <<<
RUN_DIR = Path(r"calibration_data_charuco\run_20260113_144602")

CAMERA_INDEX = 0
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

# Observation pose (safe + stable)
OBS_JOINTS_1 = [0.2717, 0.4843, -0.9719, -0.0137, -0.9972, 0.2670]
OBS_RPY_FIXED = None  # will be filled from live get_pose at OBS

# Optional extra views: set to [OBS_JOINTS_1] for strict single-view testing
OBS_JOINTS_2 = [-0.2411, 0.4585, -0.9961, -0.0674, -1.0186, 0.2670]
OBS_JOINTS_3 = [-0.7723, 0.4115, -0.9627, -0.0720, -1.0140, 0.2670]
OBS_VIEWS = [OBS_JOINTS_1]  # start pose-only testing with single view

Q6_REF = 0.0  # virtual hand q6 freeze

ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50
TARGET_ID = 0
MARKER_LEN_M = 0.026

# Snapshot sampling + gates
N_GOOD_SAMPLES = 12          # how many accepted samples to collect per snapshot
MAX_TRIES = 60               # max frames to look at to reach N_GOOD_SAMPLES
REPROJ_MAX_PX = 0.40         # reject bad PnP
AREA_MIN_PX2 = 4500          # reject tiny marker (far/blurry)

# Logging
LOG_DIR = Path("aruco_logs") / datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
LOG_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Helpers
# =========================
def rpy_to_rot_matrix(rpy):
    roll, pitch, yaw = rpy
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll),  np.cos(roll)]], dtype=float)
    Ry = np.array([[ np.cos(pitch), 0, np.sin(pitch)],
                   [0,             1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]], dtype=float)
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw),  np.cos(yaw), 0],
                   [0,            0,           1]], dtype=float)
    return Rz @ Ry @ Rx

def rot_matrix_to_rpy(R):
    # ZYX (yaw-pitch-roll) consistent with R = Rz*Ry*Rx
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-9
    if not singular:
        roll  = np.arctan2(R[2,1], R[2,2])
        pitch = np.arctan2(-R[2,0], sy)
        yaw   = np.arctan2(R[1,0], R[0,0])
    else:
        roll  = np.arctan2(-R[1,2], R[1,1])
        pitch = np.arctan2(-R[2,0], sy)
        yaw   = 0.0
    return np.array([roll, pitch, yaw], dtype=float)

def niryo_pose_to_matrix(pose):
    T = np.eye(4, dtype=float)
    T[:3, :3] = rpy_to_rot_matrix([pose.roll, pose.pitch, pose.yaw])
    T[:3,  3] = [pose.x, pose.y, pose.z]
    return T

def fk_base_to_hand_virtual(robot, joints, q6_ref=0.0):
    q = np.array(joints, dtype=float).reshape(6,)
    q[5] = float(q6_ref)
    pose_fk = robot.forward_kinematics(q.tolist())
    return niryo_pose_to_matrix(pose_fk)

def T_from_rvec_tvec(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T

def load_intrinsics(run_dir: Path):
    data = np.load(run_dir / "intrinsics_charuco_pinhole_final.npz", allow_pickle=True)
    K = data["K"].astype(np.float64)
    dist = data["dist"].reshape(-1).astype(np.float64)
    img_size = None
    if "image_size" in data.files:
        img_size = tuple(data["image_size"].tolist())  # (w,h)
    return K, dist, img_size

def load_handeye(run_dir: Path):
    data = np.load(run_dir / "handeye_charuco_hand_TSAI.npz", allow_pickle=True)
    T_hand_camera = data["T_hand_camera"].astype(np.float64)
    return T_hand_camera

def make_aruco_detector():
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    params = aruco.DetectorParameters()

    # subpixel refinement helps a lot
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 30
    params.cornerRefinementMinAccuracy = 0.01

    return aruco.ArucoDetector(dictionary, params)

def estimate_marker_pose(frame_bgr, detector, K, dist, marker_len_m, target_id):
    aruco = cv2.aruco
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is None:
        return None, (corners, ids, rejected)

    ids_list = ids.ravel().tolist()
    if target_id not in ids_list:
        return None, (corners, ids, rejected)

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
        return None, (corners, ids, rejected)

    # area + reprojection error
    area = float(abs(cv2.contourArea(img_pts)))
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    reproj = float(np.mean(np.linalg.norm(proj - img_pts, axis=1)))

    return {"rvec": rvec, "tvec": tvec, "area": area, "reproj": reproj, "img_pts": img_pts}, (corners, ids, rejected)

def flush_camera(cap, n=5):
    for _ in range(n):
        cap.grab()

def snapshot_mean_pose(robot, cap, detector, K, dist, T_hand_camera, q6_ref,
                       marker_len_m, target_id, obs_views):
    """
    Move through obs_views, but for pose-only validation it's recommended to keep obs_views=[OBS_JOINTS_1].
    Collect N_GOOD_SAMPLES accepted samples from the *current* view and return mean/std.
    """
    best_view_idx = None
    best_stats = None

    for view_idx, q in enumerate(obs_views):
        robot.move_joints(q)
        time.sleep(0.25)
        flush_camera(cap, 6)

        samples_T = []
        tries = 0

        while len(samples_T) < N_GOOD_SAMPLES and tries < MAX_TRIES:
            tries += 1
            ok, frame = cap.read()
            if not ok:
                continue

            est, detpack = estimate_marker_pose(frame, detector, K, dist, marker_len_m, target_id)
            if est is None:
                continue

            if est["reproj"] > REPROJ_MAX_PX or est["area"] < AREA_MIN_PX2:
                continue

            joints_now = robot.get_joints()
            T_base_hand = fk_base_to_hand_virtual(robot, joints_now, q6_ref=q6_ref)
            T_cam_marker = T_from_rvec_tvec(est["rvec"], est["tvec"])
            T_base_marker = T_base_hand @ T_hand_camera @ T_cam_marker
            samples_T.append((T_base_marker, est, frame))

        if len(samples_T) < max(4, N_GOOD_SAMPLES // 2):
            # not enough usable samples in this view
            continue

        # compute translation stats
        P = np.array([T[:3, 3] for (T, _, _) in samples_T], dtype=float)
        mean = P.mean(axis=0)
        std = P.std(axis=0)

        # pick a representative (closest to mean) for logging/visual
        d = np.linalg.norm(P - mean.reshape(1,3), axis=1)
        k_best = int(np.argmin(d))
        T_rep, est_rep, frame_rep = samples_T[k_best]

        # score: prefer larger marker and lower reproj on representative sample
        score = (-est_rep["area"], est_rep["reproj"])

        stats = {
            "view_idx": view_idx,
            "mean": mean,
            "std": std,
            "T_rep": T_rep,
            "est_rep": est_rep,
            "frame_rep": frame_rep,
            "score": score,
            "n": len(samples_T),
        }

        if best_stats is None or score < best_stats["score"]:
            best_stats = stats
            best_view_idx = view_idx

    return best_stats


# =========================
# Main
# =========================
print(f"\n[INFO] Connecting to robot {ROBOT_IP} and moving to OBS...")
robot = pyn.NiryoRobot(ROBOT_IP)
robot.enable_tcp(True)
robot.set_learning_mode(False)

robot.move(pyn.JointsPosition(*OBS_JOINTS_1))
time.sleep(0.3)

pose_obs = robot.get_pose()
OBS_RPY_FIXED = (pose_obs.roll, pose_obs.pitch, pose_obs.yaw)
print("[INFO] OBS pose:", f"x={pose_obs.x:.4f} y={pose_obs.y:.4f} z={pose_obs.z:.4f} "
      f"r={pose_obs.roll:.4f} p={pose_obs.pitch:.4f} yaw={pose_obs.yaw:.4f}")

print(f"\n[INFO] Opening camera index {CAMERA_INDEX} ...")
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
if not cap.isOpened():
    raise RuntimeError("Could not open camera.")

ok, frame = cap.read()
if not ok:
    raise RuntimeError("Could not read camera frame.")

h, w = frame.shape[:2]
print(f"[INFO] Camera resolution: {w}x{h}")

print(f"\n[INFO] Loading calibration from: {RUN_DIR}")
K, dist, img_size = load_intrinsics(RUN_DIR)
T_hand_camera = load_handeye(RUN_DIR)
print("[INFO] ||t_hand_camera|| =", float(np.linalg.norm(T_hand_camera[:3, 3])))

if img_size is not None:
    if (w, h) != img_size:
        print(f"[WARN] Camera resolution {w}x{h} != intrinsics image_size {img_size}. "
              f"Try to match for best accuracy.")

detector = make_aruco_detector()
aruco = cv2.aruco

print("\n[INFO] Live window running.")
print("      Keys: [s]=snapshot mean pose, [q]=quit")

last_est = None

while True:
    ok, frame = cap.read()
    if not ok:
        continue

    est, (corners, ids, rejected) = estimate_marker_pose(frame, detector, K, dist, MARKER_LEN_M, TARGET_ID)

    vis = frame.copy()
    ids_list = [] if ids is None else ids.ravel().tolist()
    rej_n = 0 if rejected is None else len(rejected)

    if ids is not None:
        aruco.drawDetectedMarkers(vis, corners, ids)

    if est is not None:
        cv2.drawFrameAxes(vis, K, dist, est["rvec"], est["tvec"], MARKER_LEN_M * 1.5, 2)
        tv = est["tvec"].reshape(3)
        cv2.putText(vis, f"tvec_cam=[{tv[0]:.3f},{tv[1]:.3f},{tv[2]:.3f}]m  reproj={est['reproj']:.3f}px  area={est['area']:.0f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)

    overlay = f"ids={ids_list} rejected={rej_n} target={TARGET_ID}  gate: reproj<{REPROJ_MAX_PX}, area>{AREA_MIN_PX2}"
    cv2.putText(vis, overlay, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(vis, overlay, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 1, cv2.LINE_AA)

    cv2.imshow("ArUco Pose-Only (Jog Check)", vis)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord("s"):
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        print(f"\n[SNAPSHOT] {stamp} collecting samples... (N={N_GOOD_SAMPLES}, reproj<{REPROJ_MAX_PX}, area>{AREA_MIN_PX2})")

        stats = snapshot_mean_pose(
            robot, cap, detector, K, dist, T_hand_camera, Q6_REF,
            MARKER_LEN_M, TARGET_ID, OBS_VIEWS
        )

        if stats is None:
            print("[SNAPSHOT] No valid pose (not enough good samples).")
            continue

        mean = stats["mean"]
        std = stats["std"]
        T_rep = stats["T_rep"]
        R_rep = T_rep[:3, :3]
        rpy_rep = rot_matrix_to_rpy(R_rep)

        print(f"[RESULT] view={stats['view_idx']}  n_good={stats['n']}")
        print(f"  base_xyz mean (m): [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
        print(f"  base_xyz std  (mm): [{std[0]*1000:.2f}, {std[1]*1000:.2f}, {std[2]*1000:.2f}]")
        print(f"  marker rpy (deg) from representative: [{np.degrees(rpy_rep[0]):.2f}, {np.degrees(rpy_rep[1]):.2f}, {np.degrees(rpy_rep[2]):.2f}]")
        print("  -> Now jog in Niryo Studio to the marker center and compare.")

        # Save representative images + transform
        est_rep = stats["est_rep"]
        frame_rep = stats["frame_rep"]
        vis_rep = frame_rep.copy()
        cv2.drawFrameAxes(vis_rep, K, dist, est_rep["rvec"], est_rep["tvec"], MARKER_LEN_M * 1.5, 2)

        cv2.imwrite(str(LOG_DIR / f"{stamp}_rep_raw.png"), frame_rep)
        cv2.imwrite(str(LOG_DIR / f"{stamp}_rep_vis.png"), vis_rep)
        np.savez(str(LOG_DIR / f"{stamp}_T_base_marker_mean.npz"),
                 base_xyz_mean=mean,
                 base_xyz_std=std,
                 T_base_marker_rep=T_rep,
                 view_idx=stats["view_idx"],
                 n_good=stats["n"],
                 reproj=est_rep["reproj"],
                 area=est_rep["area"])

        last_est = mean.copy()

cv2.destroyAllWindows()
cap.release()
robot.close_connection()
print("[DONE]")
