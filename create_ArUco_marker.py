# ============================================================================
# For now we will create 2 markers -> ID 0 = orange cube, ID 1 = blue cube
# markerLength = 0.03m (a bit smaller than cube side_length)
# Dictionary we will use for now -> DICT_4x4_50 (totally enough for our usage)
# ============================================================================
import os
import cv2
import numpy as np

# -----------------------
# Config
# -----------------------
# Dictionary: DICT_4X4_50 (IDs 0..49)
ARUCO_DICT_NAME = cv2.aruco.DICT_4X4_50

# IDs we want:
#   0 -> orange cube
#   1 -> blue cube
MARKERS = [
    (0, "orange_cube"),
    (1, "blue_cube"),
]

# Marker image size in pixels (square)
SIDE_PX = 600

# Output folder
OUT_DIR = "aruco_markers"


def create_aruco_marker(dictionary, marker_id: int, side_px: int, out_path: str):
    """
    Generate a single ArUco marker image and save it as a PNG.

    Uses cv2.aruco.generateImageMarker, which returns the marker image.
    """
    try:
        # Newer Python API: returns the image directly
        img = cv2.aruco.generateImageMarker(dictionary, marker_id, side_px)
    except AttributeError:
        # Fallback in case generateImageMarker is not found
        img = np.zeros((side_px, side_px), dtype=np.uint8)
        cv2.aruco.drawMarker(dictionary, marker_id, side_px, img, borderBits=1)

    cv2.imwrite(out_path, img)
    print(f"[GEN] Saved marker id={marker_id} to {out_path}")


if __name__ == "__main__":
    print("[INFO] Creating ArUco markers...")

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_NAME)
    os.makedirs(OUT_DIR, exist_ok=True)

    for marker_id, label in MARKERS:
        filename = f"aruco_4x4_50_id{marker_id}_{SIDE_PX}px_{label}.png"
        out_path = os.path.join(OUT_DIR, filename)
        create_aruco_marker(dictionary, marker_id, SIDE_PX, out_path)

    print("[INFO] Done. Check the 'aruco_markers' folder for the PNG files.")