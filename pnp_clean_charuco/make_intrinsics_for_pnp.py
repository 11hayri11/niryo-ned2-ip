import numpy as np

# 1) Load Charuco intrinsics
src_path = r"calibration_data_charuco\run_20251208_122545\intrinsics_charuco_pinhole.npz"
data = np.load(src_path)

K = data["K"]
dist = data["dist"].reshape(-1)  # flatten (1,5) -> (5,)

print("K:\n", K)
print("dist:", dist)

# 2) Save in the format pnp_clean.py expects: keys 'K' and 'D'
dst_path = r"calibration_data_charuco\run_20251208_122545\intrinsics_charuco_for_pnp.npz"
np.savez(dst_path, K=K, D=dist)

print("Saved compat intrinsics to:", dst_path)
