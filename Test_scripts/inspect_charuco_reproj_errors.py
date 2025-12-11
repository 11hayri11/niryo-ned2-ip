import os
import glob
import json
import numpy as np
import cv2

# =====================================================================
# CONFIG
# =====================================================================
RUN_DIR = r"calibration_data_charuco\run_20251205_124800"
MIN_CHARUCO_CORNERS = 10  # skip very weak detections


def load_intrinsics(run_dir: str):
    """
    Load K, dist from intrinsics_charuco_pinhole.npz
    """
    path = os.path.join(run_dir, "intrinsics_charuco_pinhole.npz")
    data = np.load(path)

    if "K" in data:
        K = data["K"]
    elif "camera_matrix" in data:
        K = data["camera_matrix"]
    else:
        raise KeyError("Could not find 'K' or 'camera_matrix' in intrinsics file")

    if "dist" in data:
        dist = data["dist"]
    elif "dist_coeffs" in data:
        dist = data["dist_coeffs"]
    else:
        raise KeyError("Could not find 'dist' or 'dist_coeffs' in intrinsics file")

    return K, dist


def load_charuco_board(run_dir: str):
    """
    Load Charuco config (from charuco_config.json) and build:
      - board  (CharucoBoard)
      - detector (CharucoDetector)
    """
    cfg_path = os.path.join(run_dir, "charuco_config.json")
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    squaresX = int(
        cfg.get("squaresX",
        cfg.get("squares_x",
        cfg.get("squares_x_count", 7)))
    )
    squaresY = int(
        cfg.get("squaresY",
        cfg.get("squares_y",
        cfg.get("squares_y_count", 5)))
    )

    square_length = float(
        cfg.get("square_length",
        cfg.get("square_length_m", 0.021))
    )
    marker_length = float(
        cfg.get("marker_length",
        cfg.get("marker_length_m", 0.011))
    )

    dict_id = int(
        cfg.get("dictionary",
        cfg.get("aruco_dict_id", 10))
    )

    print("[INFO] Charuco config from JSON / defaults:")
    print(f"       squaresX x squaresY = {squaresX} x {squaresY}")
    print(f"       square_length       = {square_length} m")
    print(f"       marker_length       = {marker_length} m")
    print(f"       dictionary id       = {dict_id}")

    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)

    board = cv2.aruco.CharucoBoard(
        (squaresX, squaresY),
        square_length,
        marker_length,
        dictionary
    )

    # New-style detector: this is what your main calibration already uses
    detector = cv2.aruco.CharucoDetector(board)

    return board, detector


def inspect_run(run_dir: str):
    K, dist = load_intrinsics(run_dir)
    board, detector = load_charuco_board(run_dir)

    img_paths = sorted(glob.glob(os.path.join(run_dir, "charuco_*.png")))
    if not img_paths:
        raise RuntimeError(f"No charuco_*.png found in {run_dir}")

    results = []

    for path in img_paths:
        img_name = os.path.basename(path)
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] Could not read {img_name}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # CharucoDetector does marker detection + Charuco interpolation
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)

        if charuco_ids is None or len(charuco_ids) < MIN_CHARUCO_CORNERS:
            num_ch = 0 if charuco_ids is None else len(charuco_ids)
            print(f"[SKIP] {img_name}: only {num_ch} Charuco corners")
            continue

        num_ch = len(charuco_ids)

        # 3D object points for *just* the detected Charuco corners
        # Use getChessboardCorners() since .chessboardCorners does NOT exist in this build.
        all_corners_3d = board.getChessboardCorners()  # shape (Nx*Ny, 3)
        obj_points = all_corners_3d[charuco_ids.flatten(), :]  # (N,3)

        # 2D image points
        img_points = charuco_corners.reshape(-1, 2)  # (N,2)

        # Pose from solvePnP (same idea as in your charuco_calibration Step 8)
        ok, rvec, tvec = cv2.solvePnP(
            obj_points,
            img_points,
            K,
            dist,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            print(f"[SKIP] {img_name}: solvePnP failed")
            continue

        # Reproject object points and measure error
        proj, _ = cv2.projectPoints(obj_points, rvec, tvec, K, dist)
        proj = proj.reshape(-1, 2)

        err = np.linalg.norm(img_points - proj, axis=1)  # per-corner error in pixels
        mean_err = float(err.mean())
        max_err = float(err.max())

        results.append((img_name, int(num_ch), mean_err, max_err))

    return results


def main():
    print(f"[INFO] Inspecting Charuco reprojection errors for run_dir = {RUN_DIR}")
    results = inspect_run(RUN_DIR)
    if not results:
        print("[ERROR] No valid views to report.")
        return

    mean_errs = np.array([r[2] for r in results])
    max_errs  = np.array([r[3] for r in results])

    print("\n[SUMMARY] Reprojection error over all valid images:")
    print(f"  Num images        : {len(results)}")
    print(f"  mean(mean_err)    : {mean_errs.mean():.4f} px")
    print(f"  std(mean_err)     : {mean_errs.std():.4f} px")
    print(f"  min(mean_err)     : {mean_errs.min():.4f} px")
    print(f"  max(mean_err)     : {mean_errs.max():.4f} px")
    print(f"  mean(max_err)     : {max_errs.mean():.4f} px")
    print(f"  max(max_err)      : {max_errs.max():.4f} px")

    # Detailed per-image table, worst first
    results_sorted = sorted(results, key=lambda r: r[2], reverse=True)
    print("\n[DETAIL] Per-image reprojection errors (sorted by mean_err descending):")
    print("  image          | #corners | mean_err [px] | max_err [px]")
    print("  --------------+----------+--------------+------------")
    for img_name, num_ch, mean_err, max_err in results_sorted:
        print(f"  {img_name:12s} | {num_ch:8d} | {mean_err:12.4f} | {max_err:10.4f}")


if __name__ == "__main__":
    main()
