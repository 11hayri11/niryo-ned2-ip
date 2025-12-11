import numpy as np
from pathlib import Path

# ====== CONFIG: set this to the run you want to analyze ======
RUN_DIR = Path("calibration_data_charuco/run_20251211_153839")
# =============================================================

T_HAND_PATH      = RUN_DIR / "T_base_hand_all_final.npz"
T_CAM_BOARD_PATH = RUN_DIR / "T_camera_board_all_final.npz"
HE_PATH          = RUN_DIR / "handeye_charuco_hand_TSAI.npz"


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

    T_base_hand_all = hand_npz["T_base_hand"]               # (N,4,4)
    names_hand = [str(n) for n in hand_npz["image_names"]]

    T_cam_board_all = cam_board_npz["T_camera_board"]       # (M,4,4)
    names_cam = [str(n) for n in cam_board_npz["image_names"]]

    T_hand_cam = he_npz["T_hand_camera"]                    # (4,4)

    if "used_image_names" in he_npz.files:
        used_names = [str(n) for n in he_npz["used_image_names"]]
    else:
        # Fallback: assume calibrateHandEye used all views in T_camera_board_all
        used_names = names_cam

    return T_base_hand_all, names_hand, T_cam_board_all, names_cam, T_hand_cam, used_names


def rotation_angle_deg(R):
    # Angle of rotation from 3x3 rotation matrix
    tr = np.trace(R)
    # numerical safety
    x = (tr - 1.0) / 2.0
    x = np.clip(x, -1.0, 1.0)
    return np.degrees(np.arccos(x))


def main():
    (T_base_hand_all, names_hand,
     T_cam_board_all, names_cam,
     T_hand_cam, used_names) = load_data()

    name_to_idx_hand = {n: i for i, n in enumerate(names_hand)}
    name_to_idx_cam  = {n: i for i, n in enumerate(names_cam)}

    # Build matched lists of ^baseT_hand and ^camT_board with the same ordering
    T_bh_list = []
    T_cb_list = []
    used = []

    for nm in used_names:
        if nm not in name_to_idx_hand or nm not in name_to_idx_cam:
            print(f"[WARN] Image name {nm} not found in hand or cam-board arrays, skipping.")
            continue
        i_h = name_to_idx_hand[nm]
        i_c = name_to_idx_cam[nm]

        T_bh_list.append(T_base_hand_all[i_h])
        T_cb_list.append(T_cam_board_all[i_c])
        used.append(nm)

    K = len(T_bh_list)
    if K < 2:
        print("[ERROR] Not enough matched views for AX=XB residuals.")
        return

    T_bh = np.stack(T_bh_list, axis=0)  # (K,4,4)
    T_cb = np.stack(T_cb_list, axis=0)  # (K,4,4)

    print(f"[INFO] Using {K} views for AX=XB residuals.")

    # Collect residuals over all pairs (i < j)
    pos_errs = []
    ang_errs = []
    pair_labels = []

    X = T_hand_cam

    for i in range(K):
        for j in range(i + 1, K):
            # A_ij = baseTh_i^-1 * baseTh_j   (hand motion)
            A = np.linalg.inv(T_bh[i]) @ T_bh[j]

            # B_ij = camTboard_i * (camTboard_j)^-1  (camera motion)
            B = T_cb[i] @ np.linalg.inv(T_cb[j])

            # We want A X ≈ X B  (Tsai AX=XB equation)
            # Residual transform: T_res = A X B^{-1} X^{-1} ≈ I
            T_res = A @ X @ np.linalg.inv(B) @ np.linalg.inv(X)

            R_res = T_res[:3, :3]
            t_res = T_res[:3, 3]

            pos_err = np.linalg.norm(t_res)          # [m]
            ang_err = rotation_angle_deg(R_res)      # [deg]

            pos_errs.append(pos_err)
            ang_errs.append(ang_err)
            pair_labels.append((used[i], used[j]))

    pos_errs = np.array(pos_errs)
    ang_errs = np.array(ang_errs)

    print("\n[AX=XB] Residual stats over all pairs (i<j):")
    print(f"  Num pairs         : {len(pos_errs)}")
    print(f"  mean pos error    : {pos_errs.mean()*1000:.2f} mm")
    print(f"  median pos error  : {np.median(pos_errs)*1000:.2f} mm")
    print(f"  max pos error     : {pos_errs.max()*1000:.2f} mm")
    print(f"  mean angle error  : {ang_errs.mean():.3f} deg")
    print(f"  median angle error: {np.median(ang_errs):.3f} deg")
    print(f"  max angle error   : {ang_errs.max():.3f} deg")

    # Optionally print top few worst pairs
    worst_idx = np.argsort(-pos_errs)[:5]
    print("\n[AX=XB] Worst 5 pairs by position error:")
    for k in worst_idx:
        i_name, j_name = pair_labels[k]
        print(f"  ({i_name}, {j_name}): "
              f"pos_err = {pos_errs[k]*1000:.2f} mm, "
              f"ang_err = {ang_errs[k]:.3f} deg")


if __name__ == "__main__":
    main()
