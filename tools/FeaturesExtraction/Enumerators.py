from enum import Enum


class BodyStraight(Enum):
   Neck            = 0   # 1  => 0
   Upper_body      = 1   # 1  => 8
   Right_arm       = 2   # 2  => 3
   Right_forearm   = 3   # 3  => 4
   Left_arm        = 4   # 5  => 6
   Left_forearm    = 5   # 6  => 7
   Right_thigh     = 6   # 9  => 10
   Right_leg       = 7   # 10 => 1
   Left_thigh      = 8   # 12 => 13
   Left_leg        = 9   # 13 => 1
   Right_hip       = 10  # 8  => 9
   Left_hip        = 11  # 8  => 12
   Right_shoulder  = 12  # 1  => 2
   Left_shoulder   = 13  # 1  => 5



