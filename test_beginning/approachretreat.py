""" from pyniryo import NiryoRobot, PoseObject

ROBOT_IP = "129.187.231.226"

def pick_and_place(robot, src_xy, dst_xy, z_table, clear=0.08):
    roll, pitch, yaw = -3.1416, 0.0, 0.0
    x_s, y_s = src_xy
    x_d, y_d = dst_xy
    z_approach = z_table + clear
    z_pick = z_table + 0.015
    z_place = z_table + 0.020

    # Pick
    robot.move(PoseObject(x_s, y_s, z_approach, roll, pitch, yaw))
    robot.move(PoseObject(x_s, y_s, z_pick,     roll, pitch, yaw))
    robot.grasp_with_tool()
    robot.move(PoseObject(x_s, y_s, z_approach, roll, pitch, yaw))

    # Place
    robot.move(PoseObject(x_d, y_d, z_approach, roll, pitch, yaw))
    robot.move(PoseObject(x_d, y_d, z_place,    roll, pitch, yaw))
    robot.release_with_tool()
    robot.move(PoseObject(x_d, y_d, z_approach, roll, pitch, yaw))

if __name__ == "__main__":
    robot = NiryoRobot(ROBOT_IP)
    try:
        robot.update_tool()
        robot.set_arm_max_velocity(20)
        z_table_guess = 0.17
        pick_and_place(robot, (0.19, -0.12), (0.20, 0.09), z_table_guess)
    finally:
        robot.close_connection() """