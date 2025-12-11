import os
import numpy as np

# ============================================================
# CONFIG
# ============================================================
RUN_DIR = r"calibration_data_charuco\run_20251205_124800"

# If you want to use all views, set this to None.
# If you want only a subset, list the indices here:
SUBSET_INDICES = [2, 12, 15, 5, 20, 13, 3, 1, 16, 19]
# SUBSET_INDICES = None   # <-- use this line instead if you want all views


# ============================================================
# Quaternion helpers (vector part first: [x,y,z,w])
# ============================================================
def rot_to_quat(R):
    """
    Rotation matrix -> unit quaternion [x, y, z, w].
    """
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
            qw = (R[2, 1] - R[1, 2]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 - R[0, 0] + R[1, 1] - R[2, 2]) * 2.0
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
            qw = (R[0, 2] - R[2, 0]) / S
        else:
            S = np.sqrt(1.0 - R[0, 0] - R[1, 1] + R[2, 2]) * 2.0
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
            qw = (R[1, 0] - R[0, 1]) / S

    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= np.linalg.norm(q)
    return q


def quat_to_rot(q):
    """
    Unit quaternion [x, y, z, w] -> 3x3 rotation matrix.
    """
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = np.array([
        [1 - 2 * (yy + zz),     2 * (xy - wz),         2 * (xz + wy)],
        [2 * (xy + wz),         1 - 2 * (xx + zz),     2 * (yz - wx)],
        [2 * (xz - wy),         2 * (yz + wx),         1 - 2 * (xx + yy)]
    ], dtype=np.float64)
    return R


def skew(v):
    """
    [v]_x for v shape (3,)
    """
    x, y, z = v
    return np.array([[0, -z, y],
                     [z, 0, -x],
                     [-y, x, 0]], dtype=np.float64)


def left_mult_mat(q):
    """
    4x4 left multiplication matrix L(q) so that:
      q ⊗ r = L(q) @ r   (both as [x,y,z,w])
    """
    v = q[:3]
    s = q[3]
    vx = skew(v)
    upper_left = s * np.eye(3) + vx
    upper_right = v.reshape(3, 1)
    lower_left = -v.reshape(1, 3)
    lower_right = np.array([[s]])
    return np.block([[upper_left, upper_right],
                     [lower_left, lower_right]])


def right_mult_mat(q):
    """
    4x4 right multiplication matrix R(q) so that:
      r ⊗ q = R(q) @ r
    """
    v = q[:3]
    s = q[3]
    vx = skew(v)
    upper_left = s * np.eye(3) - vx
    upper_right = v.reshape(3, 1)
    lower_left = -v.reshape(1, 3)
    lower_right = np.array([[s]])
    return np.block([[upper_left, upper_right],
                     [lower_left, lower_right]])


# ============================================================
# Load poses
# ============================================================
def load_poses(run_dir):
    path_bg = os.path.join(run_dir, "T_base_gripper_all.npz")
    path_cb = os.path.join(run_dir, "T_camera_board_all.npz")

    data_bg = np.load(path_bg)
    data_cb = np.load(path_cb)

    # transforms
    T_bg = data_bg["T_base_gripper"]   # (N,4,4)
    T_cb = data_cb["T_camera_board"]   # (N,4,4)

    # names: handle different key names or absence
    if "image_names" in data_bg.files:
        names = data_bg["image_names"]
    elif "names" in data_bg.files:
        names = data_bg["names"]
    else:
        N = T_bg.shape[0]
        names = np.array([f"view_{i:02d}.png" for i in range(N)], dtype=object)

    assert T_bg.shape == T_cb.shape
    N = T_bg.shape[0]
    print(f"[INFO] Loaded {N} views from {run_dir}")
    print(f"[INFO] First few images: {names[:5]}")
    return T_bg, T_cb, names


# ============================================================
# Solve AX = X B
# ============================================================
def solve_AX_XB(T_bg, T_cb):
    """
    T_bg[i] = ^baseT_gripper_i
    T_cb[i] = ^cameraT_board_i
    Want X = ^gripperT_camera.

    For each (i<j):
      A_ij = T_g_i^{-1} T_g_j
      B_ij = T_c_i^{-1} T_c_j
      A_ij X = X B_ij
    """
    N = T_bg.shape[0]
    As_R, Bs_R = [], []
    As_t, Bs_t = [], []

    for i in range(N):
        for j in range(i + 1, N):
            Tg_i = T_bg[i]
            Tg_j = T_bg[j]
            Tc_i = T_cb[i]
            Tc_j = T_cb[j]

            A = np.linalg.inv(Tg_i) @ Tg_j
            B = np.linalg.inv(Tc_i) @ Tc_j

            As_R.append(A[:3, :3])
            Bs_R.append(B[:3, :3])
            As_t.append(A[:3, 3])
            Bs_t.append(B[:3, 3])

    As_R = np.stack(As_R, axis=0)
    Bs_R = np.stack(Bs_R, axis=0)
    As_t = np.stack(As_t, axis=0)
    Bs_t = np.stack(Bs_t, axis=0)

    M_rows = []

    for Ra, Rb in zip(As_R, Bs_R):
        qa = rot_to_quat(Ra)
        qb = rot_to_quat(Rb)
        La = left_mult_mat(qa)
        Rb_mat = right_mult_mat(qb)
        M_rows.append(La - Rb_mat)

    M = np.concatenate(M_rows, axis=0)  # (4 * num_pairs, 4)

    # Solve M qx = 0 via SVD
    U, S, Vt = np.linalg.svd(M)
    qx = Vt[-1, :]
    qx /= np.linalg.norm(qx)
    # Make scalar part positive for consistency
    if qx[3] < 0:
        qx = -qx

    R_x = quat_to_rot(qx)

    # Solve translation: (Ra - I) t_x = R_x t_b - t_a
    A_big = []
    b_big = []
    I3 = np.eye(3)
    for Ra, ta, tb in zip(As_R, As_t, Bs_t):
        A_big.append(Ra - I3)
        b_big.append(R_x @ tb - ta)

    A_big = np.vstack(A_big)        # (3*num_pairs, 3)
    b_big = np.concatenate(b_big)   # (3*num_pairs,)

    t_x, *_ = np.linalg.lstsq(A_big, b_big, rcond=None)

    # Build full X
    X = np.eye(4, dtype=np.float64)
    X[:3, :3] = R_x
    X[:3, 3] = t_x
    return X


# ============================================================
# Diagnostics: AX=XB residuals & constant target scatter
# ============================================================
def diagnostics(T_bg, T_cb, X, names):
    # AX = X B residuals (same construction as above)
    N = T_bg.shape[0]
    rot_err_deg = []
    trans_err = []

    for i in range(N):
        for j in range(i + 1, N):
            Tg_i = T_bg[i]
            Tg_j = T_bg[j]
            Tc_i = T_cb[i]
            Tc_j = T_cb[j]

            A = np.linalg.inv(Tg_i) @ Tg_j
            B = np.linalg.inv(Tc_i) @ Tc_j

            left  = A @ X
            right = X @ B

            R_l, t_l = left[:3, :3], left[:3, 3]
            R_r, t_r = right[:3, :3], right[:3, 3]

            R_err = R_l @ R_r.T
            angle = np.arccos(
                np.clip((np.trace(R_err) - 1) / 2.0, -1.0, 1.0)
            )
            rot_err_deg.append(np.degrees(angle))
            trans_err.append(np.linalg.norm(t_l - t_r))

    rot_err_deg = np.array(rot_err_deg)
    trans_err = np.array(trans_err)

    print("\n[RES] AX=XB residuals (NumPy solver):")
    print(f"  Rotation error (deg): mean={rot_err_deg.mean():.2f}, "
          f"std={rot_err_deg.std():.2f}, "
          f"min={rot_err_deg.min():.2f}, max={rot_err_deg.max():.2f}")
    print(f"  Translation error (m): mean={trans_err.mean():.4f} "
          f"({trans_err.mean()*1000:.1f} mm), "
          f"std={trans_err.std():.4f} "
          f"({trans_err.std()*1000:.1f} mm), "
          f"min={trans_err.min():.4f} "
          f"({trans_err.min()*1000:.1f} mm), "
          f"max={trans_err.max():.4f} "
          f"({trans_err.max()*1000:.1f} mm)")

    # Constant-target check: ^baseT_board = ^baseT_gripper * X * ^cameraT_board
    T_g_c = X
    T_b_g = T_bg
    T_c_b = T_cb

    T_b_board = T_b_g @ T_g_c @ T_c_b
    t_b_board = T_b_board[:, :3, 3]
    mean_pos = t_b_board.mean(axis=0)
    deltas = t_b_board - mean_pos[None, :]
    norms = np.linalg.norm(deltas, axis=1)

    print("\n[CONST] Constant-target scatter (NumPy solver):")
    print(f"  Mean board position ^baseT_board_mean: {mean_pos}")
    print(f"  mean ‖Δt‖ = {norms.mean():.4f} m ({norms.mean()*1000:.1f} mm)")
    print(f"  std  ‖Δt‖ = {norms.std():.4f} m ({norms.std()*1000:.1f} mm)")
    print(f"  min  ‖Δt‖ = {norms.min():.4f} m ({norms.min()*1000:.1f} mm)")
    print(f"  max  ‖Δt‖ = {norms.max():.4f} m ({norms.max()*1000:.1f} mm)")

    worst_idx = np.argsort(-norms)[:5]
    print("\n  Worst 5 views by scatter:")
    for idx in worst_idx:
        print(f"    idx={idx:2d}, ‖Δt‖={norms[idx]*1000:7.2f} mm, "
              f"img={names[idx]}, pos={t_b_board[idx]}")


def main():
    T_bg, T_cb, names = load_poses(RUN_DIR)
    N = len(names)

    # --------------------------------------------------------------
    # Optional sub-selection of views
    # --------------------------------------------------------------
    if SUBSET_INDICES is not None:
        idx = np.array(SUBSET_INDICES, dtype=int)
        T_bg = T_bg[idx]
        T_cb = T_cb[idx]
        names = names[idx]
        print(f"[INFO] Using subset of {len(idx)} views out of {N}:")
        for i, nm in zip(idx, names):
            print(f"  idx={i:2d}, img={nm}")
    else:
        print(f"[INFO] Using all {N} views.")

    X = solve_AX_XB(T_bg, T_cb)
    print("\n[RESULT] NumPy AX=XB hand–eye solution (X = ^gripperT_camera):")
    print(X)
    t = X[:3, 3]
    print(f"[RESULT] t_gc = {t},  ||t_gc|| = {np.linalg.norm(t):.4f} m")

    diagnostics(T_bg, T_cb, X, names)


if __name__ == "__main__":
    main()
