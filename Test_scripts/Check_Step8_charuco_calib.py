import numpy as np

path = r"C:\Users\hayri\Downloads\Ned2\Ned2\calibration_data_charuco\run_20251203_140405\T_camera_board_all.npz"
data = np.load(path, allow_pickle=True)

T_all = data["T_camera_board"]      # (N, 4, 4)
names = data["image_names"]         # (N,)

print("T_all shape:", T_all.shape)
print("names shape:", names.shape)
print("First 3 names:", names[:3])

for i in range(3):
    T = T_all[i]
    t = T[:3, 3]
    print(f"\nView {i}, {names[i]}:")
    print("T =\n", np.round(T, 4))
    print("t =", np.round(t, 4), "|t| =", np.linalg.norm(t))
