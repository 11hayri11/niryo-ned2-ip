import os
import numpy as np
import cv2

# ===============================================================
# CONFIG
# ===============================================================

# Charuco run you want to analyze
RUN_DIR = r"calibration_data_charuco\run_20251205_124800"

# Indices of "good" views in the *aligned* list (same indexing as in
# check_charuco_constant_target.py output for this run).
# These came from your last constant-target table (‖Δt‖ < ~90 mm):
GOOD_IDXS = [2, 12, 15, 5, 20, 13, 3, 1, 16, 19]
# Feel free to tweak this list after you inspect the images.


# ===============================================================
# Helpers
# ===============================================================

def load_and_align_poses(run_dir):
    """
    Load ^baseT_gripper and ^cameraT_board for a Charuco run.
    Assumes both NPZ files are saved in the same order of valid_image_names.
    Uses the image_names stored in T_camera_board_all.npz as canonical.
    """
    path_bg = os.path.join(run_dir, "T_base_gripper_all.npz")
    path_cb = os.path.join(run_dir, "T_camera_board_all.npz")

    # Load base->gripper transforms
    data_bg  = np.load(path_bg)
    T_bg_all = data_bg["T_base_gripper"]      # shape (N,4,4)

    # Load camera->board transforms and their image names
    data_cb  = np.load(path_cb)
    T_cb_all = data_cb["T_camera_board"]      # shape (N,4,4)
    names    = data_cb["image_names"]         # shape (N,)

    # Sanity check: all lengths must match
    N_bg = T_bg_all.shape[0]
    N_cb = T_cb_all.shape[0]
    N_nm = len(names)
    if not (N_bg == N_cb == N_nm):
        raise ValueError(
            f"Size mismatch: T_base_gripper={N_bg}, "
            f"T_camera_board={N_cb}, names={N_nm}"
        )

    print(f"[INFO] Loaded {N_bg} pose pairs (assumed in identical order).")
    print(f"[INFO] First few image names: {names[:5]}")

    # No need to re-align by name: they share the same ordering
    return T_bg_all, T_cb_all, names



def run_handeye_on_subset(T_bg, T_cb, names, good_idxs):
    """
    Given aligned T_bg, T_cb, and a list of subset indices, run
    cv2.calibrateHandEye on that subset and return X = ^gripperT_camera.
    """
    N = len(names)
    good_idxs = [i for i in good_idxs if 0 <= i < N]
    if not good_idxs:
        raise ValueError("GOOD_IDXS is empty or out of range after filtering.")

    print(f"[INFO] Using {len(good_idxs)} subset views for hand–eye:")
    for i in good_idxs:
        print(f"  idx={i:2d}, img={names[i]}")

    # Build the motion lists for OpenCV: R_gripper2base, t_gripper2base,
    # R_target2cam, t_target2cam

    R_gripper2base = []
    t_gripper2base = []
    R_target2cam   = []
    t_target2cam   = []

    for i in good_idxs:
        Tbg = T_bg[i]
        Tcb = T_cb[i]

        R_g2b = Tbg[:3, :3]          # ^baseR_gripper
        t_g2b = Tbg[:3, 3]           # ^baset_gripper

        R_t2c = Tcb[:3, :3]          # ^cameraR_board  (target->cam)
        t_t2c = Tcb[:3, 3]           # ^camerat_board

        R_gripper2base.append(R_g2b.astype(np.float64))
        t_gripper2base.append(t_g2b.astype(np.float64))
        R_target2cam.append(R_t2c.astype(np.float64))
        t_target2cam.append(t_t2c.astype(np.float64))

    # Run Tsai hand–eye
    print("\n[HANDEYE] Running cv2.calibrateHandEye on subset...")
    R_gc, t_gc = cv2.calibrateHandEye(
        R_gripper2base,
        t_gripper2base,
        R_target2cam,
        t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI,
    )

    T_gc = np.eye(4, dtype=np.float64)
    T_gc[:3, :3] = R_gc
    T_gc[:3, 3]  = t_gc.reshape(3)

    print("[HANDEYE] ^gripperT_camera (subset Tsai):")
    print(T_gc)
    print("[HANDEYE] t_gc (m):", t_gc.ravel())
    print(f"[HANDEYE] ||t_gc|| = {np.linalg.norm(t_gc):.4f} m")

    # Also return subset arrays and names for further checks
    T_bg_sub = T_bg[good_idxs]
    T_cb_sub = T_cb[good_idxs]
    names_sub = names[good_idxs]

    return T_gc, R_gc, t_gc, T_bg_sub, T_cb_sub, names_sub


def constant_target_on_subset(T_gc, T_bg_sub, T_cb_sub, names_sub):
    """
    Compute the constant-target scatter only on the subset:
    ^baseT_board = ^baseT_gripper * ^gripperT_camera * ^cameraT_board
    """
    # ^baseT_board for each subset view
    T_base_board = T_bg_sub @ T_gc @ T_cb_sub
    t_bb = T_base_board[:, :3, 3]  # (Ns,3)

    mean_pos = t_bb.mean(axis=0)
    deltas = t_bb - mean_pos[None, :]
    norms = np.linalg.norm(deltas, axis=1)

    mean_dev = norms.mean()
    std_dev  = norms.std()
    min_dev  = norms.min()
    max_dev  = norms.max()

    print("\n[CONST-TARGET][SUBSET] Constant-target scatter on subset:")
    print("  Mean board position ^baseT_board_mean (m):", mean_pos)
    print("  Per-view deviation from mean (‖Δt‖):")
    print(f"    mean = {mean_dev: .6f} m  ({mean_dev*1000: .1f} mm)")
    print(f"    std  = {std_dev: .6f} m  ({std_dev*1000: .1f} mm)")
    print(f"    min  = {min_dev: .6f} m  ({min_dev*1000: .1f} mm)")
    print(f"    max  = {max_dev: .6f} m  ({max_dev*1000: .1f} mm)")

    # Detailed table sorted by deviation
    print("\n  Per-view deviations (subset, sorted by ‖Δt‖ descending):")
    print("  idx_sub | ‖Δt‖ [mm] | image")
    print("  -------+-----------+----------------")
    order = np.argsort(-norms)  # descending
    for k in order:
        print(f"  {k:7d} | {norms[k]*1000:9.2f} | {names_sub[k]}")


def save_handeye_subset(run_dir, T_gc, R_gc, t_gc, names_sub):
    """
    Save the subset hand–eye to a new npz file in the run directory.
    """
    out_path = os.path.join(run_dir, "handeye_charuco_TSAI_subset.npz")
    np.savez(
        out_path,
        T_gripper_camera=T_gc,
        R_gripper_camera=R_gc,
        t_gripper_camera=t_gc.reshape(3),
        used_image_names=names_sub,
        method="Tsai_subset",
        rms_intrinsics=np.nan,  # unknown here, but included for compatibility
    )
    print(f"\n[SAVE] Wrote subset hand–eye result to {out_path}")


# ===============================================================
# Main
# ===============================================================

def main():
    print(f"[INFO] Using Charuco run_dir = {RUN_DIR}")
    T_bg, T_cb, names = load_and_align_poses(RUN_DIR)

    T_gc, R_gc, t_gc, T_bg_sub, T_cb_sub, names_sub = run_handeye_on_subset(
        T_bg, T_cb, names, GOOD_IDXS
    )

    constant_target_on_subset(T_gc, T_bg_sub, T_cb_sub, names_sub)
    save_handeye_subset(RUN_DIR, T_gc, R_gc, t_gc, names_sub)


if __name__ == "__main__":
    main()
