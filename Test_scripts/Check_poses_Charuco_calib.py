import numpy as np

data = np.load(
    r"C:\Users\hayri\Downloads\Ned2\Ned2\calibration_data_charuco\run_20251202_140247\T_base_gripper_all.npz",
    allow_pickle=True
)
T_all = data["T_base_gripper"]    # shape (N, 4, 4)
paths = data["image_paths"]       # filenames

print("T_all shape:", T_all.shape)
print("paths shape:", paths.shape)
print("First filenames:", paths[:5])

print("First transform:\n", T_all[0])
