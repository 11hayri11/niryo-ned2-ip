import pyniryo as pyn

robot = pyn.NiryoRobot('129.187.231.226')
#robot.enable_tcp(True) // enables the TCP server interface on the robot; Not needed now but also doesnt hurt

robot.calibrate_auto()

robot.move(pyn.JointsPosition(0.0, 0.0, -0.9, 0.24, 0.0, 0.0))
robot.close_connection()