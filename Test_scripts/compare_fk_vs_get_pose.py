# compare_fk_vs_get_pose.py
import time
import numpy as np
import pyniryo as pyn

# ---------- helper: RPY -> rotation matrix (same as redoCalibrationV2) ----------
def rpy_to_rot_matrix(rpy):
    """
    Convert RPY (roll, pitch, yaw) to rotation matrix.
    rpy: [roll, pitch, yaw] in radians
    Returns 3x3 rotation matrix.
    """
    roll, pitch, yaw = rpy

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll),  np.cos(roll)]])

    Ry = np.array([[ np.cos(pitch), 0, np.sin(pitch)],
                   [0,              1,             0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])

    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw),  np.cos(yaw), 0],
                   [0,                      0, 1]])

    # Apply rotations in order: roll → pitch → yaw
    R = Rz @ Ry @ Rx
    return R

def pose_to_vec(pose_obj):
    """Convert Niryo PoseObject -> [x, y, z, roll, pitch, yaw] np.array."""
    return np.array([
        pose_obj.x,
        pose_obj.y,
        pose_obj.z,
        pose_obj.roll,
        pose_obj.pitch,
        pose_obj.yaw,
    ], dtype=np.float64)

def vec_to_T(vec6):
    """[x,y,z,roll,pitch,yaw] -> 4x4 homogeneous transform (base <- gripper)."""
    x, y, z, roll, pitch, yaw = vec6
    R = rpy_to_rot_matrix([roll, pitch, yaw])
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3]  = [x, y, z]
    return T

# ---------- the 26 joint-space poses (copy from calibration scripts) ----------
robot_poses = [
    [-0.0022,0.4994,-1.1869,-0.0091,-0.4188,-0.0259],    # pose-1
    [-0.0022,0.4994,-1.34,-0.0167,0.029,-0.0597],        # pose-2
    [-0.4801,0.4994,-1.1309,0.3268,-0.7164,-0.0551],     # pose-3
    [0.2336,0.5115,-1.1688,-0.3358,-0.471,0.4357],       # pose-4
    [0.4725,0.4539,-1.34,-0.0351,-0.1458,0.632],         # pose-5
    [0.6232,0.4569,-1.2566,-0.6595,-0.7609,1.3806],      # pose-6
    [0.0601,0.4675,-1.037,0.2961,-0.6167,0.451],         # pose-7
    [-0.6231,0.4675,-0.9233,0.4111,-0.9971,-0.4662],     # pose-8
    [-0.8773,0.4691,-0.7112,0.5738,-1.5018,0.1488],      # pose-9
    [0.3675,-0.1156,-0.7779,0.3314,-0.3283,0.6934],      # pose-10
    [0.2884,-0.405,-0.2916,0.2608,-0.8775,0.4664],       # pose-11
    [-0.3416,-0.2459,-0.5007,0.02,-0.7394,-0.9003],      # pose-12
    [-0.0205,-0.1111,-0.5173,0.0951,-0.9327,-0.0459],    # pose-13
    [-0.0205,0.1858,-1.34,0.0369,0.3281,-0.052],         # pose-14
    [-0.0205,0.4797,-0.7112,0.0307,-1.1014,-0.0489],     # pose-15
    [-0.3127,0.4751,-0.7567,1.4573,-0.5584,1.5417],      # pose-16
    [0.8271,0.4009,-0.9552,-1.0138,-1.3561,0.0583],      # pose-17
    [-0.0646,0.5918,-0.5325,0.0138,-1.2871,-0.121],      # pose-18
    [-0.0661,0.61,-0.2462,-0.1256,-1.5954,-0.2545],      # pose-19
    [-0.8286,0.61,-0.6779,0.3989,-1.5724,-0.1824],       # pose-20
    [1.0189,0.4978,-0.5507,-0.2928,-1.689,0.2593],       # pose-21
    [0.0342,0.61,0.0598,0.1488,-1.9206,0.6765],          # pose-22
    [0.0342,-0.4792,0.2234,-0.0351,-1.5356,0.0691],      # pose-23
    [-0.0585,-0.3459,-0.2462,-0.098,-1.0677,-0.0259],    # pose-24
    [0.0342,0.463,-1.3233,-0.052,-0.0246,0.0967],        # pose-25
    [-0.0129,-0.311,-0.387,-0.1088,-0.8269,0.0246]       # pose-26
]

# ---------- connect to robot ----------
robot_ip = "129.187.231.226"

print(f"[INFO] Connecting to Niryo Ned2 at {robot_ip} ...")
robot = pyn.NiryoRobot(robot_ip)
robot.set_learning_mode(False)
robot.enable_tcp(True)  # make sure TCP is active

try:
    print("[INFO] Current joints (rad):", np.round(robot.get_joints(), 4))
except Exception:
    pass

print("[INFO] Connected.\n")

trans_errs = []
angle_errs = []

for idx, joints in enumerate(robot_poses, start=1):
    joints_arr = np.array(joints, dtype=np.float64)

    print(f"\n=== Pose {idx:02d} ===")
    print("Joints (rad):", np.round(joints_arr, 4))

    # ----- A) Forward kinematics (NO motion, pure kinematics) -----
    # In old redoCalibration, you did: eef_pose = robot.forward_kinematics(pose)
    pose_fk = robot.forward_kinematics(joints_arr.tolist())
    vec_fk  = pose_to_vec(pose_fk)
    T_fk    = vec_to_T(vec_fk)

    print("FK pose [x y z r p y]:", np.round(vec_fk, 5))

    # ----- B) Move and then get_pose (actual TCP in current state) -----
    robot.move(pyn.JointsPosition(*joints))
    time.sleep(0.8)  # let it settle
    pose_live = robot.get_pose()
    vec_live  = pose_to_vec(pose_live)
    T_live    = vec_to_T(vec_live)

    print("get_pose [x y z r p y]:", np.round(vec_live, 5))

    # ----- C) Differences in 6D -----
    d_xyz = vec_live[:3] - vec_fk[:3]
    d_rpy = vec_live[3:] - vec_fk[3:]

    print("Δxyz (live - FK) [m]:   ", np.round(d_xyz, 5))
    print("Δrpy (live - FK) [rad]:", np.round(d_rpy, 5))

    # ----- D) Full transform error (T_fk^-1 * T_live) -----
    T_err = np.linalg.inv(T_fk) @ T_live
    t_err = T_err[:3, 3]
    R_err = T_err[:3, :3]

    # translation error norm
    trans_norm = np.linalg.norm(t_err)
    trans_errs.append(trans_norm)

    # rotation error angle from rotation matrix
    # clamp trace to valid range for acos to avoid nan from tiny numeric noise
    trace_R = np.trace(R_err)
    cos_theta = max(min((trace_R - 1.0) / 2.0, 1.0), -1.0)
    theta = np.arccos(cos_theta)  # in radians
    angle_errs.append(theta)

    print(f"Transform error: |Δt| = {trans_norm*1000:.1f} mm, "
          f"angle ≈ {theta*180/np.pi:.2f} deg")

# ---------- summary ----------
trans_errs = np.array(trans_errs)
angle_errs = np.array(angle_errs)

print("\n[SUMMARY] FK vs get_pose over all 26 calibration poses")
print(f"  Translation |Δt| (mm): mean = {trans_errs.mean()*1000:.2f}, "
      f"min = {trans_errs.min()*1000:.2f}, max = {trans_errs.max()*1000:.2f}")
print(f"  Rotation angle error (deg): mean = {angle_errs.mean()*180/np.pi:.3f}, "
      f"min = {angle_errs.min()*180/np.pi:.3f}, max = {angle_errs.max()*180/np.pi:.3f}")

# optional: go back home
try:
    HOME_JOINTS = [-0.0007, 0.4994, -1.2506, 0.0, 0.0014, 0.0062]
    print("\n[INFO] Moving robot back to home pose...")
    robot.move(pyn.JointsPosition(*HOME_JOINTS))
    print("[INFO] Reached home pose.")
except Exception as e:
    print(f"[WARN] Could not move to home: {e}")

try:
    robot.close_connection()
except Exception:
    pass

print("\n[DONE] FK vs get_pose comparison finished.")
