from threading import Lock
from copy import deepcopy
from math import atan2,asin

class SharedState:
    def __init__(self):
        self.landmarks = {}
        self.last_circle_id = -1
        self.yolo_busy = False
        self.ransac_busy = False
        self.wall = None
        self.wall_label = None
        self.attitude_quat = (0.0, 0.0, 0.0, 1.0) # x, y, z, w
        self.lock = Lock()
        self.ransac_lock = Lock()


    def update_attitude(self, x, y, z, w):
        with self.ransac_lock:
            self.attitude_quat = (x, y, z, w)

    def get_attitude_angles(self):
        """Returns (pitch_rad, roll_rad, yaw_rad)"""
        with self.ransac_lock:
            x, y, z, w = self.attitude_quat
            
        # Standard Quaternion to Euler conversion
        roll = atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
        yaw = atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        
        return pitch, roll, yaw

    def set_busy(self, busy=True):
        with self.lock:
            self.yolo_busy = busy

    def set_ransac_busy(self, busy=True):
        with self.ransac_lock:
            self.ransac_busy = busy

    def is_busy(self):
        with self.lock:
            return self.yolo_busy
        
    def is_ransac_busy(self):
        with self.ransac_lock:
            return self.ransac_busy
    
    def update_landmarks(self, new_dict, circle_id):
        with self.lock:
            self.landmarks.clear()
            self.landmarks.update(new_dict)
            self.last_circle_id = circle_id

    def clear_landmarks(self):
        with self.lock:
            self.landmarks.clear()
            self.last_circle_id = -1

    def update_wall(self, wall, wall_label):
        with self.ransac_lock:
            self.wall = wall
            self.wall_label = wall_label

    def clear_wall(self):
        with self.ransac_lock:
            self.wall = None

    def snapshot(self):
        with self.lock:
            with self.ransac_lock:
                landmarks = deepcopy(self.landmarks)

                return {
                    "landmarks" : landmarks,
                    "last_circle_id" : self.last_circle_id,
                    "yolo_busy" : self.yolo_busy,
                    "ransac_busy" : self.ransac_busy,
                    "wall" : self.wall
                }

