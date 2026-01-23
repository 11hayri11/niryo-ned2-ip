import os
import argparse
import numpy as np


# -----------------------------
# Helpers
# -----------------------------
def basename_array(arr):
    """Convert np array of strings/paths into plain basenames."""
    out = []
    for x in arr:
        s = str(x)
        out.append(os.path.basename(s))
    return np.array(out)


def load_npz(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Could not find: {path}")
    return np.load(path, allow_pickle=True)


def quat_from_R(R):
    """
    Convert rotation matrix to quaternion (w,x,y,z).
    Robust-ish for numeric noise.
    """
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    else:
        # pick largest diagonal
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S

    q = np.array([w, x, y, z], dtype=np.float64)
    # normalize + enforce consistent sign
    q /= (np.linalg.norm(q) + 1e-12)
    if q[0] < 0:
        q = -q
    return q


def mean_quat(quats):
    """
    Quaternion averaging via eigenvector of sum(q q^T).
    quats: (N,4) in (w,x,y,z)
    """
    Q = np.zeros((4, 4), dtype=np.float64)
    for q in quats:
        q = q.reshape(4, 1)
        Q += q @ q.T
    Q /= max(len(quats), 1)

    eigvals, eigvecs = np.linalg.eigh(Q)
    q_mean = eigvecs[:, np.argmax(eigvals)]
    q_mean = q_mean / (np.linalg.norm(q_mean) + 1e-12)
    if q_mean[0] < 0:
        q_mean = -q_mean
    return q_mean


def quat_angle_deg(q1, q2):
    """Angle between orientations represented by quaternions."""
    q1 = q1 / (np.linalg.norm(q1) + 1e-12)
    q2 = q2 / (np.linalg.norm(q2) + 1e-12)
    dot = float(np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0))
    return float(2.0 * np.degrees(np.arccos(dot)))


def print_scatter(name, T_list, used_names):
    """
    Print translation scatter and rotation scatter of a list of base->board transforms.
    """
    T_list = np.asarray(T_list, dtype=np.float64)
    t = T_list[:, :3, 3]
    R = T_list[:, :3, :3]

    t_mean = np.mean(t, axis=0)
    t_std = np.std(t, axis=0)
    t_med = np.median(t, axis=0)
    d = np.linalg.norm(t - t_med[None, :], axis=1)
    d_max = float(np.max(d)) if len(d) else 0.0

    quats = np.stack([quat_from_R(Ri) for Ri in R], axis=0)
    q_mean = mean_quat(quats)
    ang = np.array([quat_angle_deg(q_mean, qi) for qi in quats], dtype=np.float64)

    print(f"\n[{name}] Constant-target scatter over {len(T_list)} views")
    print(f"  Translation mean (m): [{t_mean[0]:.4f}, {t_mean[1]:.4f}, {t_mean[2]:.4f}]")
    print(f"  Translation std  (mm): [{1000*t_std[0]:.2f}, {1000*t_std[1]:.2f}, {1000*t_std[2]:.2f}]")
    print(f"  Max distance from median (mm): {1000*d_max:.2f}")

    print(f"  Rotation scatter vs mean: mean={np.mean(ang):.3f} deg, max={np.max(ang):.3f} deg")

    # worst 5 by translation distance
    worst = np.argsort(-d)[:min(5, len(d))]
    print("  Worst views by translation distance:")
    for idx in worst:
        print(f"    idx={idx:2d} img={used_names[idx]}  |Δt|={1000*d[idx]:.2f} mm  t=[{t[idx,0]:.4f},{t[idx,1]:.4f},{t[idx,2]:.4f}]")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, default=r"calibration_data_charuco/run_20260113_144602",
                    help="Charuco run directory, e.g. calibration_data_charuco\\run_YYYYMMDD_HHMMSS")
    args = ap.parse_args()
    run_dir = args.run_dir

    print(f"[INFO] Using Charuco run_dir = {run_dir}")

    # ---- Load base->hand list (virtual wrist/hand frame)
    base_hand_path = os.path.join(run_dir, "T_base_hand_all_final.npz")
    base_hand = load_npz(base_hand_path)
    if "T_base_hand" not in base_hand.files:
        raise KeyError(f"{base_hand_path} missing key 'T_base_hand'. Keys={base_hand.files}")
    if "image_names" not in base_hand.files:
        raise KeyError(f"{base_hand_path} missing key 'image_names'. Keys={base_hand.files}")

    T_base_hand_all = base_hand["T_base_hand"]
    names_hand = basename_array(base_hand["image_names"])
    print(f"[INFO] Loaded {len(T_base_hand_all)} T_base_hand from {base_hand_path}")

    # ---- Load camera->board list
    cam_board_path = os.path.join(run_dir, "T_camera_board_all_final.npz")
    cam_board = load_npz(cam_board_path)
    if "T_camera_board" not in cam_board.files:
        raise KeyError(f"{cam_board_path} missing key 'T_camera_board'. Keys={cam_board.files}")
    if "image_names" not in cam_board.files:
        raise KeyError(f"{cam_board_path} missing key 'image_names'. Keys={cam_board.files}")

    T_camera_board_all = cam_board["T_camera_board"]
    names_cb = basename_array(cam_board["image_names"])
    print(f"[INFO] Loaded {len(T_camera_board_all)} T_camera_board from {cam_board_path}")

    # ---- Load hand-eye (hand->camera)
    # New naming:
    he_path = os.path.join(run_dir, "handeye_charuco_hand_TSAI.npz")
    # Fallback older naming if needed:
    if not os.path.isfile(he_path):
        he_path_alt = os.path.join(run_dir, "handeye_charuco_TSAI.npz")
        if os.path.isfile(he_path_alt):
            he_path = he_path_alt
    he = load_npz(he_path)
    print(f"[INFO] Hand-eye file: {he_path}")
    print(f"[INFO] Hand-eye keys: {he.files}")

    # Determine T_hand_camera
    if "T_hand_camera" in he.files:
        T_hand_camera = he["T_hand_camera"]
    elif "T_gripper_camera" in he.files:
        # older convention name, still same meaning if you used gripper=tcp in that run
        T_hand_camera = he["T_gripper_camera"]
    else:
        # try R/t
        if "R_hand_camera" in he.files and "t_hand_camera" in he.files:
            R = he["R_hand_camera"]
            t = he["t_hand_camera"].reshape(3)
        elif "R_gripper_camera" in he.files and "t_gripper_camera" in he.files:
            R = he["R_gripper_camera"]
            t = he["t_gripper_camera"].reshape(3)
        else:
            raise KeyError("Could not find T_hand_camera / T_gripper_camera or matching R/t in hand-eye npz.")

        T_hand_camera = np.eye(4, dtype=np.float64)
        T_hand_camera[:3, :3] = R
        T_hand_camera[:3, 3] = t

    if T_hand_camera.shape != (4, 4):
        raise ValueError(f"T_hand_camera has shape {T_hand_camera.shape}, expected (4,4)")

    print("[INFO] ^handT_camera:")
    print(T_hand_camera)
    print(f"[INFO] ||t_hand_camera|| = {np.linalg.norm(T_hand_camera[:3,3]):.4f} m")

    # ---- Choose which image names to evaluate
    # Prefer hand-eye subset if available
    if "used_image_names" in he.files:
        used_names = basename_array(he["used_image_names"])
        print(f"[INFO] Using {len(used_names)} 'used_image_names' from hand-eye file.")
    else:
        used_names = np.intersect1d(names_hand, names_cb)
        print(f"[INFO] Using intersection of hand/cam names (N={len(used_names)}).")

    # Build lookup tables
    idx_hand = {n: i for i, n in enumerate(names_hand)}
    idx_cb_map = {n: i for i, n in enumerate(names_cb)}

    keep = []
    for n in used_names:
        if (n in idx_hand) and (n in idx_cb_map):
            keep.append(n)
    used_names = np.array(keep)

    if len(used_names) < 5:
        raise RuntimeError(f"Too few matched views ({len(used_names)}). Something is inconsistent with filenames.")

    # Gather matched transforms
    T_base_hand = np.stack([T_base_hand_all[idx_hand[n]] for n in used_names], axis=0)
    T_camera_board = np.stack([T_camera_board_all[idx_cb_map[n]] for n in used_names], axis=0)

    print(f"[INFO] Matched {len(used_names)} views between base-hand and camera-board.")
    print("       First few:", used_names[:min(5, len(used_names))])

    # ---- Variant A (expected): base->board = base->hand * hand->cam * cam->board
    T_base_board_A = []
    for i in range(len(used_names)):
        T_base_board_A.append(T_base_hand[i] @ T_hand_camera @ T_camera_board[i])
    T_base_board_A = np.stack(T_base_board_A, axis=0)

    # ---- Variant B sanity: invert hand-eye (as if file were cam->hand by mistake)
    T_camera_hand = np.linalg.inv(T_hand_camera)
    T_base_board_B = []
    for i in range(len(used_names)):
        T_base_board_B.append(T_base_hand[i] @ T_camera_hand @ T_camera_board[i])
    T_base_board_B = np.stack(T_base_board_B, axis=0)

    # ---- Print results
    print_scatter("Variant A (expected)", T_base_board_A, used_names)
    print_scatter("Variant B (inverted hand-eye sanity check)", T_base_board_B, used_names)

    print("\n[SUMMARY]")
    print("  Variant A should have LOW scatter (mm-level, ~1 deg-ish) if the run is good.")
    print("  Variant B should usually be much worse. If Variant B looks better, frame convention is wrong.")
    print("[DONE]")


if __name__ == "__main__":
    main()
