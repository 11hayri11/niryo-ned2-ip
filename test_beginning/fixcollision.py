from pyniryo import NiryoRobot, PoseObject

IP = "129.187.231.226"
r = NiryoRobot(IP)
try:
    # 1) Enable learning mode to free the joints, move arm by hand to a safe spot
    r.set_learning_mode(True)
    print("Learning mode: ON — gently move arm to a high, obstacle-free pose.")
    input("Press Enter when positioned safely...")

    # 2) Disable learning mode (motors on)
    r.set_learning_mode(False)
    print("Learning mode: OFF")

    # 3) Clear the collision flag
    r.clear_collision_detected()
    print("Collision flag cleared.")

    # 4) Calibrate (good practice after a collision/backdrive)
    r.calibrate_auto()
    print("Calibrated.")

    # 5) Slow motion and try a very safe, high pose
    r.update_tool()
    r.set_arm_max_velocity(20)
    safe = PoseObject(0.20, 0.00, 0.25, -3.1416, 0.0, 0.0)
    print("Moving to safe high pose…")
    r.move(safe)
    print("Success.")
finally:
    r.close_connection()

