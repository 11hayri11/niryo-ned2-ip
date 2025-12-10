import time
import math
import pyniryo as pyn
import numpy as np

# -----------------------
# Step 1 - Config / setup
# -----------------------
print("[INFO] Step1: helper + configs")

# Robot IP
robot_ip = "129.187.231.226" # Change in lecture

# Drawing plane & orientation (Hardcoded for now)
# in meters
draw_center_x = 0.2611
draw_center_y = 0.0155
z_draw        = 0.2579

# Safety height above drawing plane - Not sure if needed
z_safe = z_draw + 0.05 # 5cm above drawing plane

# Orientation of gripper - "pen-like"
r_draw   = 2.142
p_draw   = 1.531
yaw_draw = 2.246

# Optional: triangle size
tri_side = 0.14 # 14cm side length

# Optional: Rectangle size
rect_width  = 0.18 # 18cm
rect_height = 0.10 # 10cm

# Hexagon size - distance from center to each vertex
hex_radius = 0.11 # 11cm

# --------------------------------
# Step 2 - Connect to Niryo & home
# --------------------------------
print(f"\n[INFO] Step2: Connecting to Niryo Ned2 at {robot_ip} ...")

t0 = time.time()
robot = pyn.NiryoRobot(robot_ip)

# Calibrate if needed
if robot.need_calibration():
    print("[INFO] Robot needs calibration -> calibrate_auto()")
    robot.calibrate_auto()

# Basic safety settings
robot.enable_tcp(True)
robot.set_learning_mode(False)
robot.set_arm_max_velocity(50)
robot.move_to_home_pose() # Safe starting point

print(f"[INFO] Robot moved to home position and connection established successfully. (took {time.time() - t0:.2f}s)")


# ---------------------------
# Step 3 - Pose helper (draw)
# ---------------------------
def PO_xy(x: float, y: float, z: float):
    """
    Create a PoseObject at (x, y, z) with the fixed drawing orientation.
    All drawing motions will use this helper.
    """
    return pyn.PoseObject(x, y, z, r_draw, p_draw, yaw_draw)

# ===================== Triangle ===========================
def generate_triangle_vertices(center_xy, side, theta0=0.0):
    """
    Generate vertices of an equilateral triangle in the XY plane.

    center_xy: (cx, cy) in meters  -> triangle centroid in base frame
    side     : side length in meters
    theta0   : in-plane rotation (radians), 0 means one vertex along +X from center

    Returns:
        vertices_xy: list of (x, y) vertices, with the first vertex repeated at the end
                     so the path is closed.
    """
    cx, cy = center_xy
    R = side / math.sqrt(3.0)  # distance from centroid to each vertex (equilateral triangle)

    vertices = []
    for k in range(3):
        angle = theta0 + k * 2.0 * math.pi / 3.0
        x = cx + R * math.cos(angle)
        y = cy + R * math.sin(angle)
        vertices.append((x, y))

    # Close the loop by repeating the first vertex
    vertices.append(vertices[0])
    return vertices

def classify_triangle_vertices(vertices_xy):
    """
    vertices_xy: list of (x,y) with 4 entries: [v0, v1, v2, v0]
                 We only use the first 3 unique vertices.

    Returns:
        left, right, tip : each a (x,y) tuple
    """
    base_vertices = vertices_xy[:3]  # three unique corners

    # tip = vertex with max y
    ys = [v[1] for v in base_vertices]
    tip_idx = int(np.argmax(ys))
    tip = base_vertices[tip_idx]

    # remaining two -> left/right by x
    others = [v for i, v in enumerate(base_vertices) if i != tip_idx]
    if others[0][0] < others[1][0]:
        left, right = others[0], others[1]
    else:
        left, right = others[1], others[0]

    return left, right, tip


def build_triangle_path(center_xy, vertices_xy):
    """
    center_xy : (cx, cy) - still passed in but we no longer use it in the path
    vertices_xy : list from generate_triangle_vertices(...)

    Returns:
        path_xy: list of (x,y) in the order:
                 first_corner -> tip -> second_corner -> first_corner (closed loop)
    """
    left, right, tip = classify_triangle_vertices(vertices_xy)

    # wähle z.B. 'left' als erste Ecke
    first_corner  = left
    second_corner = right

    path_xy = [
        first_corner,  # erste Ecke
        tip,           # Spitze
        second_corner, # zweite Ecke
        first_corner,  # wieder erste Ecke
    ]
    return path_xy

# =========================== Rectangle ==============================
def generate_rectangle_vertices(center_xy, width, height, theta0=0.0):
    """
    Generate a rectangle centered at center_xy, axis-aligned (no rotation for now).

    Order (before path/story shaping):
        bl -> tl -> tr -> br -> bl  (closed loop)
    """
    cx, cy = center_xy
    w2 = width / 2.0
    h2 = height / 2.0

    # axis-aligned corners
    bl = (cx - w2, cy - h2)  # bottom-left
    tl = (cx - w2, cy + h2)  # top-left
    tr = (cx + w2, cy + h2)  # top-right
    br = (cx + w2, cy - h2)  # bottom-right

    vertices = [bl, tl, tr, br, bl]

    # Optional rotation: skip for now (theta0=0.0), you can add later if needed.
    return vertices


def build_rectangle_path(center_xy, vertices_xy):
    """
    Define a nice drawing order for the rectangle demo.

    We'll tell this story:
        bottom-left -> top-left -> top-right -> bottom-right -> bottom-left
    """

    # take the first 4 unique corners in given order
    bl, tl, tr, br = vertices_xy[:4]

    path_xy = [
        bl,
        tl,
        tr,
        br,
        bl,
    ]
    return path_xy

# ======================== Hexagon ==========================
def generate_hexagon_vertices(center_xy, radius, theta0=0.0):
    """
    Regular hexagon of given radius around center_xy.

    Returns 7 points: v0..v5, v0 to close the loop.
    """
    cx, cy = center_xy
    vertices = []
    for k in range(6):
        angle = theta0 + k * (math.pi / 3.0)  # 60° steps
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        vertices.append((x, y))

    vertices.append(vertices[0])  # close loop
    return vertices


def build_hexagon_path(center_xy, vertices_xy):
    """
    Drawing story for hexagon:

        v0 -> v1 -> v2 -> v3 -> v4 -> v5 -> v0
    """

    base_vertices = vertices_xy[:6]  # v0..v5
    v0, v1, v2, v3, v4, v5 = base_vertices

    path_xy = [
        v0,
        v1,
        v2,
        v3,
        v4,
        v5,
        v0,
    ]
    return path_xy

# ========== Shape drawing ===========
def draw_shape_xy(robot, vertices_xy):
    if not vertices_xy:
        print("[DRAW] No vertices given, nothing to draw.")
        return
    
    # 1) Go directly to first vertex at z_draw
    x0, y0 = vertices_xy[0]
    print(f"[DRAW] Move to first vertex at ({x0:.3f}, {y0:.3f}, {z_draw:.3f})")
    robot.move(PO_xy(x0, y0, z_draw))

    # Pause 1 sec at first vertex
    print("[DRAW] Pause 1s at first vertex.")
    time.sleep(1.0)

    # 2) Draw along all remaining vertices at z_draw
    for i, (x, y) in enumerate(vertices_xy[1:], start=1):
        print(f"[DRAW] Edge step {i}: to ({x:.3f}, {y:.3f}, {z_draw:.3f})")
        robot.move(PO_xy(x, y, z_draw))

    # 3) Stop at last vertex.
    x_last, y_last = vertices_xy[-1]
    print(f"[DRAW] Finished drawing at last vertex ({x_last:.3f}, {y_last:.3f}, {z_draw:.3f})")

    print("[DRAW] Shape drawing complete.")

def choose_shape_from_input():
    """
    Ask user which shape to draw and return one of:
      "triangle", "rectangle", "hexagon"
    """
    print("\n[MENU] Choose shape to draw:")
    print("  t = triangle")
    print("  r = rectangle")
    print("  h = hexagon")

    while True:
        ans = input("[MENU] Your choice [t/r/h]: ").strip().lower()
        if ans == "t":
            return "triangle"
        elif ans == "r":
            return "rectangle"
        elif ans == "h":
            return "hexagon"
        else:
            print("[MENU] Invalid choice. Please type 't', 'r', or 'h'.")

if __name__ == "__main__":
    center = (draw_center_x, draw_center_y)

    # Ask user which shape to draw
    shape = choose_shape_from_input()
    print(f"[MAIN] Selected shape: {shape}")

    if shape == "triangle":
        side = tri_side
        print(f"[MAIN] Drawing TRIANGLE at center=({center[0]:.3f}, {center[1]:.3f}), side={side:.3f}")
        tri_vertices = generate_triangle_vertices(center, side)
        print("[MAIN] Raw triangle vertices (XY):")
        for v in tri_vertices:
            print("   ", v)
        path_xy = build_triangle_path(center, tri_vertices)

    elif shape == "rectangle":
        print(f"[MAIN] Drawing RECTANGLE at center=({center[0]:.3f}, {center[1]:.3f}), "
              f"width={rect_width:.3f}, height={rect_height:.3f}")
        rect_vertices = generate_rectangle_vertices(center, rect_width, rect_height)
        print("[MAIN] Raw rectangle vertices (XY):")
        for v in rect_vertices:
            print("   ", v)
        path_xy = build_rectangle_path(center, rect_vertices)

    elif shape == "hexagon":
        print(f"[MAIN] Drawing HEXAGON at center=({center[0]:.3f}, {center[1]:.3f}), radius={hex_radius:.3f}")
        hex_vertices = generate_hexagon_vertices(center, hex_radius)
        print("[MAIN] Raw hexagon vertices (XY):")
        for v in hex_vertices:
            print("   ", v)
        path_xy = build_hexagon_path(center, hex_vertices)

    else:
        raise ValueError(f"Unknown shape: {shape}")

    print("[MAIN] Drawing path (XY):")
    for v in path_xy:
        print("   ", v)

    draw_shape_xy(robot, path_xy)

    try:
        robot.move_to_home_pose()
        print("[MAIN] Returned to home pose.")
    except Exception as e:
        print("[MAIN WARN] Could not move to home pose:", e)