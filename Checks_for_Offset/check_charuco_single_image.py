import cv2
import numpy as np
import json
from pathlib import Path

# --------- 1) Select the run and load metadata ---------
RUN_DIR = Path(r"C:\Users\hayri\Downloads\Ned2\Ned2\calibration_data_charuco\run_20251202_140247")

npz_path = RUN_DIR / "T_base_gripper_all.npz"
cfg_path = RUN_DIR / "charuco_config.json"

data = np.load(npz_path, allow_pickle=True)
T_all = data["T_base_gripper"]   # shape (N, 4, 4)
paths = data["image_paths"]      # shape (N,)

print("Number of views in this run:", T_all.shape[0])
print("First few image filenames:", paths[:5])

# Pick first view for now
idx = 15
img_name = str(paths[idx])
img_path = RUN_DIR / img_name
print(f"\n[INFO] Using view {idx}, image {img_name}")

# --------- 2) Rebuild the Charuco board & detector from config ---------
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

aruco_dict_id = int(cfg["aruco_dict"])
squares_x = int(cfg["charuco_squares_x"])
squares_y = int(cfg["charuco_squares_y"])
square_len = float(cfg["square_length_m"])
marker_len = float(cfg["marker_length_m"])

print("\n[INFO] Board from config:")
print(f"  squares X x Y = {squares_x} x {squares_y}")
print(f"  square_length = {square_len} m")
print(f"  marker_length = {marker_len} m")
print(f"  aruco_dict    = {aruco_dict_id}")

aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
charuco_board = cv2.aruco.CharucoBoard(
    (squares_x, squares_y),
    square_len,
    marker_len,
    aruco_dict,
)

detector_params = cv2.aruco.DetectorParameters()
charuco_detector = cv2.aruco.CharucoDetector(
    charuco_board,
    detectorParams=detector_params,
)

# --------- 3) Load the image and run detection ---------
img = cv2.imread(str(img_path))
if img is None:
    raise RuntimeError(f"[ERROR] Could not load image: {img_path}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

charuco_corners, charuco_ids, marker_corners, marker_ids = \
    charuco_detector.detectBoard(gray)

num_m = 0 if marker_ids is None else len(marker_ids)
num_c = 0 if charuco_ids is None else len(charuco_ids)

print(f"\n[DETECTION] markers={num_m}, charuco corners={num_c}")

# Optional: show the annotated image
vis = img.copy()
if marker_ids is not None and len(marker_ids) > 0:
    cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)
if charuco_corners is not None and charuco_ids is not None and len(charuco_ids) > 0:
    if hasattr(cv2.aruco, "drawDetectedCornersCharuco"):
        cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)
    else:
        for pt in charuco_corners:
            u, v = int(pt[0][0]), int(pt[0][1])
            cv2.circle(vis, (u, v), 4, (0, 255, 0), -1)

cv2.imshow("Single Charuco Detection", vis)
cv2.waitKey(5000)  # show briefly
cv2.destroyAllWindows()

# --------- 4) Convert to object/image points for calibration ---------
if charuco_corners is None or charuco_ids is None or num_c < 4:
    print("[WARN] Not enough Charuco corners to build calibration points.")
else:
    obj_pts, img_pts = charuco_board.matchImagePoints(
        charuco_corners,
        charuco_ids
    )

    print("\n[POINTS] matchImagePoints output:")
    print("  obj_pts shape:", obj_pts.shape)   # (N, 3)
    print("  img_pts shape:", img_pts.shape)   # (N, 2)

    print("\n  First 5 object points (board frame, meters):")
    print(np.round(obj_pts[:5], 4))

    print("\n  First 5 image points (pixels):")
    print(np.round(img_pts[:5], 2))
