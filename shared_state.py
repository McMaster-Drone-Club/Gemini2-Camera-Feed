from threading import Lock
from copy import deepcopy

class SharedState:
    def __init__(self):
        self.landmarks = {}
        self.last_circle_id = -1
        self.yolo_busy = False
        self.ransac_busy = False
        self.wall = None
        self.lock = Lock()
        self.ransac_lock = Lock()

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

    def update_wall(self, wall):
        with self.ransac_lock:
            self.wall = wall

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

