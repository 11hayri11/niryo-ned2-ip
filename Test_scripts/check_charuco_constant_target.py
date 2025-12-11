import os
import numpy as np

# =====================================================================
# CONFIG – Charuco run to analyze
# =====================================================================
RUN_DIR = r"calibration_data_charuco\run_20251208_122545"


def main():
    print(f"[INFO] Using Charuco run_dir = {RUN_DIR}")

    # -----------------------------------------------------------------
    # 1) Load ^baseT_gripper and image names from T_base_gripper_all.npz
    # -----------------------------------------------------------------
    bg_path = os.path.join(RUN_DIR, "T_base_gripper_all.npz")
    if not os.path.isfile(bg_path):
        raise FileNotFoundError(f"Could not find {bg_path}")

    bg_data = np.load(bg_path, allow_pickle=True)
    T_base_gripper_all = bg_data["T_base_gripper"]   # shape (N,4,4)
    image_paths_all = bg_data["image_paths"]         # shape (N,)

    # Normalize to simple filenames like "charuco_00.png"
    image_paths_all = np.array(
        [os.path.basename(str(name)) for name in image_paths_all]
    )

    N_bg = T_base_gripper_all.shape[0]
    print(f"[INFO] Loaded {N_bg} ^baseT_gripper entries from {bg_path}")

    # -----------------------------------------------------------------
    # 2) Load ^cameraT_board and image names from T_camera_board_all.npz
    # -----------------------------------------------------------------
    cb_path = os.path.join(RUN_DIR, "T_camera_board_all.npz")
    if not os.path.isfile(cb_path):
        raise FileNotFoundError(f"Could not find {cb_path}")

    cb_data = np.load(cb_path, allow_pickle=True)
    T_camera_board_all = cb_data["T_camera_board"]   # shape (M,4,4)
    image_names_all = cb_data["image_names"]         # shape (M,)

    image_names_all = np.array(
        [os.path.basename(str(name)) for name in image_names_all]
    )

    M_cb = T_camera_board_all.shape[0]
    print(f"[INFO] Loaded {M_cb} ^cameraT_board entries from {cb_path}")

    # -----------------------------------------------------------------
    # 3) Match views by filename (Charuco uses the same charuco_XX.png names)
    # -----------------------------------------------------------------
    name_to_idx_bg = {name: i for i, name in enumerate(image_paths_all)}

    idx_bg = []
    idx_cb = []
    used_names = []

    for j, name in enumerate(image_names_all):
        if name not in name_to_idx_bg:
            print(f"[WARN] image '{name}' in T_camera_board_all has no matching pose in T_base_gripper_all; skipping")
            continue
        idx_bg.append(name_to_idx_bg[name])
        idx_cb.append(j)
        used_names.append(name)

    idx_bg = np.array(idx_bg, dtype=int)
    idx_cb = np.array(idx_cb, dtype=int)
    used_names = np.array(used_names)

    print(f"[INFO] Matched {len(used_names)} views between robot poses and camera-board extrinsics.")
    if len(used_names) == 0:
        print("[ERROR] No matched views — cannot run constant-target check.")
        return

    print("       First few matched images:", used_names[:min(5, len(used_names))])

    T_base_gripper = T_base_gripper_all[idx_bg]      # shape (K,4,4)
    T_camera_board = T_camera_board_all[idx_cb]      # shape (K,4,4)
    K = T_base_gripper.shape[0]

    # -----------------------------------------------------------------
    # 4) Load hand–eye result ^gripperT_camera from handeye_charuco_TSAI.npz
    # -----------------------------------------------------------------
    he_path = os.path.join(RUN_DIR, "handeye_charuco_TSAI.npz")
    if not os.path.isfile(he_path):
        raise FileNotFoundError(f"Could not find {he_path}")

    he_data = np.load(he_path)
    print(f"[INFO] Hand-eye file keys: {he_data.files}")

    # Cases:
    # 1) Full 4x4 transform stored directly
    if "T_gripper_camera" in he_data:
        T_gripper_camera = he_data["T_gripper_camera"]
        if T_gripper_camera.shape != (4, 4):
            raise ValueError(f"T_gripper_camera has unexpected shape {T_gripper_camera.shape}")
        R_gc = T_gripper_camera[:3, :3]
        t_gc = T_gripper_camera[:3, 3]

    # 2) Separate R and t with Charuco naming
    elif "R_gripper_camera" in he_data and "t_gripper_camera" in he_data:
        R_gc = he_data["R_gripper_camera"]
        t_gc = he_data["t_gripper_camera"].reshape(3)
        T_gripper_camera = np.eye(4, dtype=np.float64)
        T_gripper_camera[:3, :3] = R_gc
        T_gripper_camera[:3, 3] = t_gc

    # 3) Fallbacks to older naming, if ever present
    elif "R" in he_data and "t" in he_data:
        R_gc = he_data["R"]
        t_gc = he_data["t"].reshape(3)
        T_gripper_camera = np.eye(4, dtype=np.float64)
        T_gripper_camera[:3, :3] = R_gc
        T_gripper_camera[:3, 3] = t_gc
    elif "R_gc" in he_data and "t_gc" in he_data:
        R_gc = he_data["R_gc"]
        t_gc = he_data["t_gc"].reshape(3)
        T_gripper_camera = np.eye(4, dtype=np.float64)
        T_gripper_camera[:3, :3] = R_gc
        T_gripper_camera[:3, 3] = t_gc
    else:
        raise KeyError("Could not find a valid (R,t) or T_gripper_camera in hand-eye file")

    print("\n[INFO] ^gripperT_camera (from hand–eye):")
    print(T_gripper_camera)
    print(f"[INFO] t_gc (m): {t_gc}")
    print(f"[INFO] ||t_gc|| = {np.linalg.norm(t_gc):.4f} m")

    # Also build inverse just in case (for Variant B)
    T_camera_gripper = np.linalg.inv(T_gripper_camera)

    # -----------------------------------------------------------------
    # 5) Variant A: interpret hand–eye as ^gripperT_camera (this is our intended convention)
    #    ^baseT_board(i) = ^baseT_gripper(i) * ^gripperT_camera * ^cameraT_board(i)
    # -----------------------------------------------------------------
    T_base_board_A = []
    for i in range(K):
        T_bb = T_base_gripper[i] @ T_gripper_camera @ T_camera_board[i]
        T_base_board_A.append(T_bb)
    T_base_board_A = np.stack(T_base_board_A, axis=0)

    t_base_board_A = T_base_board_A[:, :3, 3]   # shape (K,3)
    mean_pos_A = t_base_board_A.mean(axis=0)

    deltas_A = t_base_board_A - mean_pos_A[None, :]
    norms_A = np.linalg.norm(deltas_A, axis=1)

    mean_dev_A = norms_A.mean()
    std_dev_A  = norms_A.std()
    min_dev_A  = norms_A.min()
    max_dev_A  = norms_A.max()

    print("\n[RESULT] Constant-target check – Variant A (X = ^gripperT_camera)")
    print(f"  Mean board position ^baseT_board_mean (m): {mean_pos_A}")
    print("  Per-view deviation from mean (‖Δt‖):")
    print(f"    mean = {mean_dev_A: .6f} m  ({mean_dev_A*1000: .1f} mm)")
    print(f"    std  = {std_dev_A: .6f} m  ({std_dev_A*1000: .1f} mm)")
    print(f"    min  = {min_dev_A: .6f} m  ({min_dev_A*1000: .1f} mm)")
    print(f"    max  = {max_dev_A: .6f} m  ({max_dev_A*1000: .1f} mm)")

    # Show worst few views
    worst_indices_A = np.argsort(-norms_A)[:5]
    print("\n  Worst 5 views by board-position scatter (Variant A):")
    for idx in worst_indices_A:
        print(f"    idx={idx:2d}, img={used_names[idx]}, "
              f"‖Δt‖ = {norms_A[idx]*1000: .2f} mm, "
              f"pos = {t_base_board_A[idx]}")

    # Optional: detailed per-view mm table
    print("\n[DETAIL] Per-view deviations (Variant A, sorted by ‖Δt‖ descending)")
    print("  idx   |   ‖Δt‖ [mm]   | image")
    print("  ------+---------------+----------------")
    sorted_indices_A = np.argsort(-norms_A)
    for idx in sorted_indices_A:
        print(f"  {idx:4d} | {norms_A[idx]*1000:8.2f} | {used_names[idx]}")

    # -----------------------------------------------------------------
    # 6) Variant B: sanity alternative (pretend X = ^cameraT_gripper)
    #    ^baseT_board(i) = ^baseT_gripper(i) * ^cameraT_gripper * ^cameraT_board(i)
    #    (this *should* look worse if our convention is correct)
    # -----------------------------------------------------------------
    T_base_board_B = []
    for i in range(K):
        T_bb = T_base_gripper[i] @ T_camera_gripper @ T_camera_board[i]
        T_base_board_B.append(T_bb)
    T_base_board_B = np.stack(T_base_board_B, axis=0)

    t_base_board_B = T_base_board_B[:, :3, 3]
    mean_pos_B = t_base_board_B.mean(axis=0)

    deltas_B = t_base_board_B - mean_pos_B[None, :]
    norms_B = np.linalg.norm(deltas_B, axis=1)

    mean_dev_B = norms_B.mean()
    std_dev_B  = norms_B.std()
    min_dev_B  = norms_B.min()
    max_dev_B  = norms_B.max()

    print("\n[RESULT] Constant-target check – Variant B (X = ^cameraT_gripper)")
    print(f"  Mean board position ^baseT_board_mean (m): {mean_pos_B}")
    print("  Per-view deviation from mean (‖Δt‖):")
    print(f"    mean = {mean_dev_B: .6f} m  ({mean_dev_B*1000: .1f} mm)")
    print(f"    std  = {std_dev_B: .6f} m  ({std_dev_B*1000: .1f} mm)")
    print(f"    min  = {min_dev_B: .6f} m  ({min_dev_B*1000: .1f} mm)")
    print(f"    max  = {max_dev_B: .6f} m  ({max_dev_B*1000: .1f} mm)")

    worst_indices_B = np.argsort(-norms_B)[:5]
    print("\n  Worst 5 views by board-position scatter (Variant B):")
    for idx in worst_indices_B:
        print(f"    idx={idx:2d}, img={used_names[idx]}, "
              f"‖Δt‖ = {norms_B[idx]*1000: .2f} mm, "
              f"pos = {t_base_board_B[idx]}")

    print("\n[SUMMARY]")
    print("  Variant A should be the one that makes physical sense if X is really ^gripperT_camera.")
    print("  If Variant A has large scatter (e.g. > 50–80 mm), something in poses/hand-eye is off.")
    print("  Variant B is just a sanity cross-check: it will usually look much worse.")
    print("[DONE] Constant-target check finished.")


if __name__ == "__main__":
    main()
