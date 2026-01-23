# niryo_ned2_kinematics.py
"""
Forward kinematics helpers for the Niryo Ned2.

Goal here:
    Given joint angles q = [j1..j6], compute ^baseT_hand,
    where 'hand' is the wrist / joint-5 frame (where the camera is mounted).

We will use the Product of Exponentials (PoE) formulation from Modern Robotics.
Fill in:
    - S_LIST (6 screw axes in SPACE frame)
    - M_HAND (home configuration of the hand frame)
"""

import numpy as np
# import modern_robotics as mr   # if you decide to use the MR library


# ----------------------------------------------------------
# Screw axes (SPACE frame) for the 6 joints of the Ned2 arm
# ----------------------------------------------------------
# S_LIST has shape (6, 6), each column is:
#   [wx, wy, wz, vx, vy, vz]^T   for that joint.
#
# TODO: fill with the correct values from the Ned2 URDF / DH model.
S_LIST = np.array([
    # w1x, w1y, w1z,  v1x, v1y, v1z
    [0.0, 0.0, 1.0,  0.0, 0.0, 0.0],   # <-- placeholder
    [0.0, 1.0, 0.0,  0.0, 0.0, 0.0],   # <-- placeholder
    [0.0, 1.0, 0.0,  0.0, 0.0, 0.0],   # ...
    [1.0, 0.0, 0.0,  0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0,  0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0,  0.0, 0.0, 0.0],
], dtype=float).T   # final shape (6, 6)


# --------------------------------------------------------------------
# Home configuration of the "hand" frame in base frame, q = [0..0]
# --------------------------------------------------------------------
# This is ^baseT_hand when all joints are at their zero position (rad = 0).
# You can get this from the URDF / kinematic description, or by querying
# FK in some library that already knows the geometry.
M_HAND = np.eye(4, dtype=float)
# e.g.:
# M_HAND[:3, 3] = [0.15, 0.0, 0.20]   # <-- placeholder translation


def fk_base_to_hand(q):
    """
    Compute ^baseT_hand for given Ned2 joint angles q = [j1..j6].

    Currently intended as:
        - 'hand' = joint-5 frame (camera mount frame),
          so only joints 1..5 affect its pose.

    Parameters
    ----------
    q : array-like, shape (6,)
        Joint angles [j1..j6] in radians.

    Returns
    -------
    T_bh : np.ndarray, shape (4, 4)
        Homogeneous transform ^baseT_hand.
    """
    q = np.asarray(q, dtype=float)
    assert q.shape == (6,)

    # We only need the first 5 joints for the hand frame
    q_hand = q[:5]

    # If you use ModernRobotics:
    #
    #   T_bh = mr.FKinSpace(M_HAND, S_LIST[:, :5], q_hand)
    #
    # For now we keep a placeholder identity until S_LIST & M_HAND are filled:
    T_bh = np.eye(4, dtype=float)

    return T_bh
