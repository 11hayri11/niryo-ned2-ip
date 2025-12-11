import numpy as np
from pathlib import Path

# ====== CONFIG: set this to the run you want to analyze ======
RUN_DIR = Path("calibration_data_charuco/run_20251211_153839")
# =============================================================

T_HAND_PATH   = RUN_DIR / "T_base_hand_all_final.npz"
T_CAM_BOARD_PATH = RUN_DIR / "T_camera_board_all_final.npz"
HE_PATH       = RUN_DIR / "handeye_charuco_hand_TSAI.npz"


def load_data():
    if not T_HAND_PATH.exists():
        raise FileNotFoundError(f"Missing {T_HAND_PATH}")
    if not T_CAM_BOARD_PATH.exists():
        raise FileNotFoundError(f"Missing {T_CAM_BOARD_PATH}")
    if not HE_PATH.exists():
        raise FileNotFoundError(f"Missing {HE_PATH}")

    hand_npz = np.load(T_HAND_PATH, allow_pickle=True)
    cam_board_npz = np.load(T_CAM_BOARD_PATH, allow_pickle=True)
    he_npz = np.load(HE_PATH, allow_pickle=True)

    T_base_hand_all = hand_npz["T_base_hand"]          # (N,4,4)
    names_hand = [str(n) for n in hand_npz["image_names"]]

    T_cam_board_all = cam_board_npz["T_camera_board"]  # (M,4,4)
    names_cam = [str(n) for n in cam_board_npz["image_names"]]

    T_hand_cam = he_npz["T_hand_camera"]               # (4,4)
    if "used_image_names" in he_npz.files:
        used_names = [str(n) for n in he_npz["used_image_names"]]
    else:
        # Fallback: assume all names_cam were used
        used_names = names_cam

    return T_base_hand_all, names_hand, T_cam_board_all, names_cam, T_hand_cam, used_names


def main():
    (T_base_hand_all, names_hand,
     T_cam_board_all, names_cam,
     T_hand_cam, used_names) = load_data()

    name_to_idx_hand = {n: i for i, n in enumerate(names_hand)}
    name_to_idx_cam  = {n: i for i, n in enumerate(names_cam)}

    T_base_board_list = []
    used_ok = []

    for nm in used_names:
        if nm not in name_to_idx_hand or nm not in name_to_idx_cam:
            print(f"[WARN] Image name {nm} not found in hand or cam-board arrays, skipping.")
            continue

        i_h = name_to_idx_hand[nm]
        i_c = name_to_idx_cam[nm]

        T_bh = T_base_hand_all[i_h]        # ^baseT_hand
        T_cb = T_cam_board_all[i_c]        # ^camT_board

        # Compose: base -> board
        T_bc = T_bh @ T_hand_cam           # ^baseT_cam
        T_bb = T_bc @ T_cb                 # ^baseT_board

        T_base_board_list.append(T_bb)
        used_ok.append(nm)

    if len(T_base_board_list) < 2:
        print("[ERROR] Not enough matched views for constant-target check.")
        return

    T_base_board_all = np.stack(T_base_board_list, axis=0)  # (K,4,4)
    K = T_base_board_all.shape[0]
    print(f"[INFO] Using {K} views for constant-target scatter check.")

    # ---- Position scatter ----
    pos = T_base_board_all[:, :3, 3]  # (K,3)
    pos_mean = pos.mean(axis=0)
    pos_centered = pos - pos_mean
    pos_err = np.linalg.norm(pos_centered, axis=1)

    print("\n[CONST-TARGET] Position of board in base frame (per view):")
    for nm, p, e in zip(used_ok, pos, pos_err):
        print(f"  {nm}: p = [{p[0]: .4f}, {p[1]: .4f}, {p[2]: .4f}], "
              f"dev_from_mean = {e*1000:.1f} mm")

    print("\n[CONST-TARGET] Position scatter stats:")
    print(f"  mean position [m]   = [{pos_mean[0]: .4f}, {pos_mean[1]: .4f}, {pos_mean[2]: .4f}]")
    print(f"  mean dev from mean  = {pos_err.mean()*1000:.1f} mm")
    print(f"  max  dev from mean  = {pos_err.max()*1000:.1f} mm")

    # ---- Orientation scatter (angle to mean) ----
    R_all = T_base_board_all[:, :3, :3]

    # Compute a crude "mean rotation" via averaging and re-orthonormalizing
    R_avg = R_all.mean(axis=0)
    # Orthonormalize via SVD
    U, _, Vt = np.linalg.svd(R_avg)
    R_mean = U @ Vt

    def rot_angle(R):
        # angle of rotation from R_mean^T R
        R_err = R_mean.T @ R
        tr = np.trace(R_err)
        tr = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
        return np.degrees(np.arccos(tr))

    ang_errs = np.array([rot_angle(R) for R in R_all])

    print("\n[CONST-TARGET] Orientation scatter:")
    for nm, ang in zip(used_ok, ang_errs):
        print(f"  {nm}: angle to mean = {ang:.3f} deg")

    print("\n[CONST-TARGET] Orientation scatter stats:")
    print(f"  mean angle dev  = {ang_errs.mean():.3f} deg")
    print(f"  max  angle dev  = {ang_errs.max():.3f} deg")


if __name__ == "__main__":
    main()
