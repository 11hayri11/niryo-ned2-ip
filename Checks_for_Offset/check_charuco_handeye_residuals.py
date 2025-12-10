import os
import numpy as np

# =====================================================================
# CONFIG – pick the Charuco run to analyze
# =====================================================================
RUN_DIR =  r"calibration_data_charuco\run_20251208_122545"  # <-- update if needed


def relative_transform(T1, T2):
    """
    Compute relative transform T_rel = T1^-1 * T2
    (i.e. transform that takes frame 1 into frame 2).
    """
    return np.linalg.inv(T1) @ T2


def main():
    print(f"[INFO] Using Charuco run_dir = {RUN_DIR}")

    # -----------------------------------------------------------------
    # 1) Load ^baseT_gripper for all captured views
    # -----------------------------------------------------------------
    path_bg = os.path.join(RUN_DIR, "T_base_gripper_all.npz")
    if not os.path.isfile(path_bg):
        raise FileNotFoundError(f"Could not find {path_bg}")

    data_bg = np.load(path_bg, allow_pickle=True)
    T_base_gripper_all = data_bg["T_base_gripper"]       # (N_bg, 4,4)
    names_bg = data_bg["image_paths"]                    # (N_bg,)
    names_bg = np.array([str(n) for n in names_bg])
    print(f"[INFO] Loaded {len(T_base_gripper_all)} ^baseT_gripper entries from {path_bg}")

    # -----------------------------------------------------------------
    # 2) Load ^cameraT_board for all views used in calibration
    # -----------------------------------------------------------------
    path_cb = os.path.join(RUN_DIR, "T_camera_board_all.npz")
    if not os.path.isfile(path_cb):
        raise FileNotFoundError(f"Could not find {path_cb}")

    data_cb = np.load(path_cb, allow_pickle=True)
    T_camera_board_all = data_cb["T_camera_board"]       # (N_cb, 4,4)
    names_cb = data_cb["image_names"]                    # (N_cb,)
    names_cb = np.array([str(n) for n in names_cb])
    print(f"[INFO] Loaded {len(T_camera_board_all)} ^cameraT_board entries from {path_cb}")

    # -----------------------------------------------------------------
    # 3) Match views by image name (should be 1:1 in a healthy run)
    # -----------------------------------------------------------------
    name2idx_bg = {n: i for i, n in enumerate(names_bg)}
    idx_bg = []
    idx_cb = []
    matched_names = []

    for j, name in enumerate(names_cb):
        if name in name2idx_bg:
            idx_bg.append(name2idx_bg[name])
            idx_cb.append(j)
            matched_names.append(name)
        else:
            print(f"[WARN] image {name} in T_camera_board_all has no match in T_base_gripper_all")

    idx_bg = np.array(idx_bg, dtype=int)
    idx_cb = np.array(idx_cb, dtype=int)
    matched_names = np.array(matched_names)

    N = len(matched_names)
    print(f"[INFO] Matched {N} views.")
    print("       First few matched images:", matched_names[:5])

    if N < 2:
        raise RuntimeError("Not enough matched views to compute residuals.")

    T_bg = T_base_gripper_all[idx_bg]  # (N,4,4)
    T_cb = T_camera_board_all[idx_cb]  # (N,4,4)

    # -----------------------------------------------------------------
    # 4) Load hand–eye result ^gripperT_camera
    # -----------------------------------------------------------------
    he_path = os.path.join(RUN_DIR, "handeye_charuco_TSAI.npz")
    if not os.path.isfile(he_path):
        raise FileNotFoundError(f"Could not find {he_path}")

    he_data = np.load(he_path)
    print(f"[INFO] Hand-eye file keys: {he_data.files}")

    if "T_gripper_camera" in he_data:
        T_gripper_camera = he_data["T_gripper_camera"]
        if T_gripper_camera.shape != (4, 4):
            raise ValueError(f"T_gripper_camera has unexpected shape {T_gripper_camera.shape}")
    elif "R_gripper_camera" in he_data and "t_gripper_camera" in he_data:
        R_gc = he_data["R_gripper_camera"]
        t_gc = he_data["t_gripper_camera"].reshape(3)
        T_gripper_camera = np.eye(4, dtype=np.float64)
        T_gripper_camera[:3, :3] = R_gc
        T_gripper_camera[:3, 3] = t_gc
    else:
        raise KeyError("Could not find 'T_gripper_camera' or (R_gripper_camera,t_gripper_camera)")

    print("\n[INFO] ^gripperT_camera (X) from hand–eye:")
    print(T_gripper_camera)

    # -----------------------------------------------------------------
    # 5) Build pairwise motions A_ij (gripper) and B_ij (camera)
    # -----------------------------------------------------------------
    # A_ij = ^gripper_i T_gripper_j  (from robot)
    # B_ij = ^camera_i T_camera_j    (from Charuco extrinsics)
    #
    # where ^camera_i T_camera_j = ^camera_i T_board * ^board T_camera_j
    #                            = T_cb[i] * inv(T_cb[j])

    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            pairs.append((i, j))
    print(f"[INFO] Number of pose pairs = {len(pairs)}")

    ang_errors_deg = []
    trans_errors_m = []

    for (i, j) in pairs:
        # Relative gripper motion
        A_ij = relative_transform(T_bg[i], T_bg[j])  # ^gripper_iT_gripper_j

        # Relative camera motion
        T_ciTb = T_cb[i]  # ^camera_iT_board
        T_cjTb = T_cb[j]  # ^camera_jT_board
        B_ij = T_ciTb @ np.linalg.inv(T_cjTb)  # ^camera_iT_camera_j

        # Hand-eye equation: A_ij * X  ≈  X * B_ij
        left  = A_ij @ T_gripper_camera
        right = T_gripper_camera @ B_ij

        # Residual transform E_ij = left * right^-1 (should be identity if perfect)
        E_ij = left @ np.linalg.inv(right)

        R_err = E_ij[:3, :3]
        t_err = E_ij[:3, 3]

        # Rotation error angle
        cos_angle = (np.trace(R_err) - 1.0) / 2.0
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_angle))

        ang_errors_deg.append(angle_deg)
        trans_errors_m.append(np.linalg.norm(t_err))

    ang_errors_deg = np.array(ang_errors_deg)
    trans_errors_m = np.array(trans_errors_m)

    # -----------------------------------------------------------------
    # 6) Summarize residuals
    # -----------------------------------------------------------------
    print("\n[RES] Hand–eye residuals over all pose pairs (A_ij X vs X B_ij):")
    print(f"  Number of pairs: {len(pairs)}")

    print("  Rotation error (deg):")
    print(f"    mean = {ang_errors_deg.mean(): .4f}")
    print(f"    std  = {ang_errors_deg.std(): .4f}")
    print(f"    min  = {ang_errors_deg.min(): .4f}")
    print(f"    max  = {ang_errors_deg.max(): .4f}")

    print("  Translation residual (meters):")
    print(f"    mean = {trans_errors_m.mean(): .4f}  ({trans_errors_m.mean()*1000: .1f} mm)")
    print(f"    std  = {trans_errors_m.std(): .4f}  ({trans_errors_m.std()*1000: .1f} mm)")
    print(f"    min  = {trans_errors_m.min(): .4f}  ({trans_errors_m.min()*1000: .1f} mm)")
    print(f"    max  = {trans_errors_m.max(): .4f}  ({trans_errors_m.max()*1000: .1f} mm)")

    # (Optional) print the worst K pairs
    K = 10
    worst_idx = np.argsort(-trans_errors_m)[:K]
    print(f"\n[RES] Worst {K} pairs by translation residual:")
    for idx in worst_idx:
        i, j = pairs[idx]
        print(f"    pair (i={i:2d}, j={j:2d}) -> "
              f"‖Δt‖ = {trans_errors_m[idx]*1000: .2f} mm, "
              f"ΔR = {ang_errors_deg[idx]: .2f} deg "
              f"({matched_names[i]} vs {matched_names[j]})")


if __name__ == "__main__":
    main()
