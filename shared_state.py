from threading import Lock
from copy import deepcopy

class SharedState:
    def __init__(self):
        self.landmarks = {}
        self.last_circle_id = -1
        self.yolo_busy = False
        self.lock = Lock()

    def set_busy(self, busy=True):
        with self.lock:
            self.yolo_busy = busy

    def is_busy(self):
        with self.lock:
            return self.yolo_busy
    
    def update_landmarks(self, new_dict, circle_id):
        with self.lock:
            self.landmarks.clear()
            self.landmarks.update(new_dict)
            self.last_circle_id = circle_id

    def clear_landmarks(self):
        with self.lock:
            self.landmarks.clear()
            self.last_circle_id = -1

    def snapshot(self):
        with self.lock:
            landmarks = deepcopy(self.landmarks)

            return {
                "landmarks" : landmarks,
                "last_circle_id" : self.last_circle_id,
                "yolo_busy" : self.yolo_busy
            }

