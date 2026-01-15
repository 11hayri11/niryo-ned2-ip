import numpy as np
import cv2
from pathlib import Path

# --------- USER SETTINGS ----------
RUN_DIR = Path(r"calibration_data_charuco\run_20260113_144602")
DROP_NAMES = {"charuco_08.png"}          # add more if needed
Q6_REF_DEFAULT = 0.0                     # just saved into output for info
OUT_NAME = "handeye_charuco_hand_TSAI_filtered.npz"  # new file name
# ----------------------------------


def _load_npz(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return np.load(str(path), allow_pickle=True)


def _to_name_list(arr):
    # image_names sometimes are numpy strings; normalize to python str
    return [str(x) for x in arr]


def _make_map(T_all, names):
    # Map image_name -> 4x4 transform
    return {n: T_all[i] for i, n in enumerate(names)}


def main():
    base_hand_path = RUN_DIR / "T_base_hand_all_final.npz"
    cam_board_path = RUN_DIR / "T_camera_board_all_final.npz"
    old_he_path    = RUN_DIR / "handeye_charuco_hand_TSAI.npz"

    bg = _load_npz(base_hand_path)
    cb = _load_npz(cam_board_path)

    # keys in your run (confirmed): ['T_base_hand','image_names','q6_ref'] and ['T_camera_board','image_names','K','dist']
    T_base_hand_all = bg["T_base_hand"]
    names_bg = _to_name_list(bg["image_names"])

    T_cam_board_all = cb["T_camera_board"]
    names_cb = _to_name_list(cb["image_names"])

    map_bg = _make_map(T_base_hand_all, names_bg)
    map_cb = _make_map(T_cam_board_all, names_cb)

    # intersection, then drop requested names
    common = sorted(set(map_bg.keys()) & set(map_cb.keys()))
    kept = [n for n in common if n not in DROP_NAMES]

    print(f"[INFO] Run dir: {RUN_DIR}")
    print(f"[INFO] Common views: {len(common)}")
    print(f"[INFO] Dropping: {sorted(DROP_NAMES)}")
    print(f"[INFO] Kept views: {len(kept)}")

    if len(kept) < 6:
        raise RuntimeError("Too few views left after filtering (need at least ~6-8).")

    # Build OpenCV calibrateHandEye inputs
    R_gripper2base, t_gripper2base = [], []
    R_target2cam,  t_target2cam  = [], []

    for n in kept:
        T_b_h = map_bg[n]  # baseT_hand  (hand->base) => gripper2base
        T_c_b = map_cb[n]  # cameraT_board (board->camera) => target2cam

        R_gripper2base.append(T_b_h[:3, :3].astype(np.float64))
        t_gripper2base.append(T_b_h[:3, 3].astype(np.float64).reshape(3, 1))

        R_target2cam.append(T_c_b[:3, :3].astype(np.float64))
        t_target2cam.append(T_c_b[:3, 3].astype(np.float64).reshape(3, 1))

    # Tsai hand-eye
    R_cam2hand, t_cam2hand = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam,  t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI,
    )

    T_hand_camera = np.eye(4, dtype=np.float64)
    T_hand_camera[:3, :3] = R_cam2hand
    T_hand_camera[:3, 3] = t_cam2hand.reshape(3)

    print("[RESULT] ^handT_camera:")
    print(T_hand_camera)
    print(f"[RESULT] ||t|| = {np.linalg.norm(T_hand_camera[:3, 3]):.4f} m")

    # Preserve some metadata if old file exists
    method = "TSAI"
    rms_intr = None
    q6_ref = Q6_REF_DEFAULT

    if old_he_path.exists():
        old = _load_npz(old_he_path)
        method = str(old.get("method", method))
        rms_intr = float(old["rms_intrinsics"]) if "rms_intrinsics" in old.files else None
        q6_ref = float(old["q6_ref"]) if "q6_ref" in old.files else q6_ref

    out_path = RUN_DIR / OUT_NAME
    np.savez(
        str(out_path),
        T_hand_camera=T_hand_camera,
        R_hand_camera=T_hand_camera[:3, :3],
        t_hand_camera=T_hand_camera[:3, 3],
        used_image_names=np.array(kept),
        dropped_image_names=np.array(sorted(DROP_NAMES)),
        method=method,
        rms_intrinsics=(-1.0 if rms_intr is None else rms_intr),
        q6_ref=float(q6_ref),
    )

    print(f"[SAVE] Wrote filtered hand-eye to: {out_path}")
    print("[NEXT] Now rerun:")
    print(f"  - Test_scripts/check_charuco_constant_target.py  (run_dir = {RUN_DIR})")
    print(f"  - Test_scripts/check_charuco_hand_AXXB_residuals.py (same run_dir)")
    print("  And temporarily point them to OUT_NAME (or rename OUT_NAME -> handeye_charuco_hand_TSAI.npz).")


if __name__ == "__main__":
    main()
