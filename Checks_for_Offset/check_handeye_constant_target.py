import os
import pickle
import numpy as np
import cv2

# =====================================================================
# CONFIG – change this to the run you want to analyze
# =====================================================================
RUN_DIR = r"calibration_data\run_20251128_145707"  # <-- update as needed


# =====================================================================
# Helpers
# =====================================================================
def rpy_to_rot_matrix(rpy):
    """
    Convert [roll, pitch, yaw] (rad) to a 3x3 rotation matrix.
    Convention must match redoCalibrationV2.py (Rz * Ry * Rx).
    """
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([[1, 0, 0],
                   [0, cr, -sr],
                   [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp],
                   [0, 1, 0],
                   [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [0,   0,  1]], dtype=np.float64)

    return Rz @ Ry @ Rx


def niryo_pose_to_matrix(pose):
    """
    Niryo TCP pose [x, y, z, roll, pitch, yaw] in BASE frame
    -> 4x4 homogeneous transform ^baseT_gripper.
    """
    x, y, z, roll, pitch, yaw = pose
    R = rpy_to_rot_matrix(np.array([roll, pitch, yaw], dtype=np.float64))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


def rvec_tvec_to_matrix(rvec, tvec):
    """
    OpenCV rvec/tvec (target wrt camera) -> 4x4 ^camT_target.
    """
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T


# =====================================================================
# Main
# =====================================================================
def main():
    run_dir = RUN_DIR
    print(f"[INFO] Using run_dir = {run_dir}")

    bundle_path = os.path.join(run_dir, "points_and_poses.pkl")
    handeye_path = os.path.join(run_dir, "handeye_TSAI.npz")
    print(f"[INFO] Bundle path   = {bundle_path}")
    print(f"[INFO] Hand-eye path = {handeye_path}")

    # --- Load bundle (poses + vision extrinsics) ---
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)

    eef_poses          = np.array(bundle["eef_poses"], dtype=np.float64)
    rvecs_target2cam   = bundle["rvecs_target2cam"]
    tvecs_target2cam   = bundle["tvecs_target2cam"]

    N = len(eef_poses)
    print(f"[INFO] Loaded {N} eef_poses and {N} vision extrinsics.")

    # --- Load hand–eye result (as saved by redoCalibrationV2.py) ---
    data = np.load(handeye_path)
    R_gc = data["R"]  # this is what we called ^gripperR_camera
    t_gc = data["t"].reshape(3, 1)  # ^grippert_camera

    print("[INFO] Hand-eye transform (as saved in redoCalibrationV2):")
    print(R_gc)
    print("t (m):", t_gc.ravel())
    norm_t = np.linalg.norm(t_gc)
    print(f"[INFO] ||t|| = {norm_t:.4f} m")

    # Build ^gripperT_camera and its inverse ^cameraT_gripper
    T_gripper_camera = np.eye(4, dtype=np.float64)
    T_gripper_camera[:3, :3] = R_gc
    T_gripper_camera[:3, 3] = t_gc.ravel()

    T_camera_gripper = np.linalg.inv(T_gripper_camera)

    # --- Build ^baseT_gripper and ^camT_target for each view ---
    print("[INFO] Building transforms ^baseT_gripper and ^camT_target...")
    T_base_gripper_list = []
    T_cam_target_list   = []

    for pose, rvec, tvec in zip(eef_poses, rvecs_target2cam, tvecs_target2cam):
        T_bg = niryo_pose_to_matrix(pose)
        T_ct = rvec_tvec_to_matrix(rvec, tvec)
        T_base_gripper_list.append(T_bg)
        T_cam_target_list.append(T_ct)

    T_base_gripper = np.stack(T_base_gripper_list, axis=0)  # (N,4,4)
    T_cam_target   = np.stack(T_cam_target_list,   axis=0)  # (N,4,4)

    print("[DEBUG] Example ^baseT_gripper[0]:")
    print(np.array2string(T_base_gripper[0], formatter={'float_kind': lambda x: f"{x: .6f}"}))
    print("[DEBUG] Example ^camT_target[0]:")
    print(np.array2string(T_cam_target[0],   formatter={'float_kind': lambda x: f"{x: .6f}"}))

    # =================================================================
    # Variant A: interpret (R,t) as ^gripperT_camera
    # =================================================================
    # ^baseT_target_A(i) = ^baseT_gripper(i) * ^gripperT_camera * ^camT_target(i)
    T_base_target_A = T_base_gripper @ T_gripper_camera @ T_cam_target

    t_base_target_A = T_base_target_A[:, :3, 3]  # (N,3)
    mean_pos_A = t_base_target_A.mean(axis=0)

    deltas_A = t_base_target_A - mean_pos_A[None, :]
    norms_A = np.linalg.norm(deltas_A, axis=1)  # (N,)

    mean_dev_A = norms_A.mean()
    std_dev_A  = norms_A.std()
    min_dev_A  = norms_A.min()
    max_dev_A  = norms_A.max()

    print("\n[RESULT] Constant-target check – Variant A (R,t = ^gripperT_camera)")
    print(f"  Mean chessboard position ^baset_target_mean (m): {mean_pos_A}")
    print("  Per-image deviation from mean (‖Δt‖):")
    print(f"    mean = {mean_dev_A: .6f} m  ({mean_dev_A*1000: .1f} mm)")
    print(f"    std  = {std_dev_A: .6f} m  ({std_dev_A*1000: .1f} mm)")
    print(f"    min  = {min_dev_A: .6f} m  ({min_dev_A*1000: .1f} mm)")
    print(f"    max  = {max_dev_A: .6f} m  ({max_dev_A*1000: .1f} mm)")

    # Worst 5 (as before)
    worst_indices_A = np.argsort(-norms_A)[:5]
    print("\n  Worst 5 views by chessboard-position scatter (Variant A (R,t = ^gripperT_camera)):")
    for idx in worst_indices_A:
        print(f"    idx={idx:2d}, ‖Δt‖ = {norms_A[idx]*1000: .2f} mm, pos = {t_base_target_A[idx]}")

    # --- NEW: detailed per-view table for Variant A ---
    print("\n[DETAIL] Per-view deviations (Variant A, sorted by ‖Δt‖ descending)")
    print("  idx   |   ‖Δt‖ [mm]")
    print("  ------+-------------")
    sorted_indices_A = np.argsort(-norms_A)  # descending
    for idx in sorted_indices_A:
        print(f"  {idx:4d} | {norms_A[idx]*1000:8.2f}")

    # =================================================================
    # Variant B: interpret (R,t) as ^cameraT_gripper (sanity alternative)
    # =================================================================
    # ^baseT_target_B(i) = ^baseT_gripper(i) * ^cameraT_gripper * ^camT_target(i)
    T_base_target_B = T_base_gripper @ T_camera_gripper @ T_cam_target

    t_base_target_B = T_base_target_B[:, :3, 3]
    mean_pos_B = t_base_target_B.mean(axis=0)

    deltas_B = t_base_target_B - mean_pos_B[None, :]
    norms_B = np.linalg.norm(deltas_B, axis=1)

    mean_dev_B = norms_B.mean()
    std_dev_B  = norms_B.std()
    min_dev_B  = norms_B.min()
    max_dev_B  = norms_B.max()

    print("\n[RESULT] Constant-target check – Variant B (R,t = ^cameraT_gripper)")
    print(f"  Mean chessboard position ^baset_target_mean (m): {mean_pos_B}")
    print("  Per-image deviation from mean (‖Δt‖):")
    print(f"    mean = {mean_dev_B: .6f} m  ({mean_dev_B*1000: .1f} mm)")
    print(f"    std  = {std_dev_B: .6f} m  ({std_dev_B*1000: .1f} mm)")
    print(f"    min  = {min_dev_B: .6f} m  ({min_dev_B*1000: .1f} mm)")
    print(f"    max  = {max_dev_B: .6f} m  ({max_dev_B*1000: .1f} mm)")

    worst_indices_B = np.argsort(-norms_B)[:5]
    print("\n  Worst 5 views by chessboard-position scatter (Variant B (R,t = ^cameraT_gripper)):")
    for idx in worst_indices_B:
        print(f"    idx={idx:2d}, ‖Δt‖ = {norms_B[idx]*1000: .2f} mm, pos = {t_base_target_B[idx]}")

    print("\n[SUMMARY] Use the per-view table above (Variant A) to spot outliers.")
    print("  Large ‖Δt‖ (e.g. > 180–200 mm) are good candidates to inspect or drop.")
    print("[DONE] Constant-target check finished.")


if __name__ == "__main__":
    main()
