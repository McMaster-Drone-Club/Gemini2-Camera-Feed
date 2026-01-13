from ultralytics import YOLO
from threading import Thread
from pyorbbecsdk import *
from math import sqrt


class YoloJob:
    def __init__(self, frame_bundle, circle, calibration, image_array):
        self.image_array = image_array
        self.depth_matrix = frame_bundle.depth_u16
        self.circle = circle
        self.depth_intrinsics = calibration.depth_intrinsics
        self.extrinsic = calibration.extrinsic


class YoloWorker:
    def __init__(self, shared_state, config):
        self.shared_state = shared_state
        self.config = config
        self.model_path = self.config.model_path
        self.model = YOLO(self.model_path) 

    def submit_job(self, job):
        if self.shared_state.is_busy():
            return False
        
        self.shared_state.set_busy()
        Thread(target=self.run_job, args=(job,), daemon=True).start()
        return True

    @staticmethod
    def compute_distance(point1, point2):
        x1, y1, z1 = point1.x, point1.y, point1.z
        x2, y2, z2 = point2.x, point2.y, point2.z

        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1

        return round(sqrt(dx * dx + dy * dy + dz * dz), 3)


    def run_job(self, job):
        local_landmarks = {}

        try:
            results = self.model(job.image_array)
            result = results[0]
            noBoxes = result.boxes is None or len(result.boxes) == 0

            if not noBoxes:
                for box in result.boxes: # each box is an actual objects 
                    pos = box.xyxy[0].cpu().numpy()
                    box_x_avg = int((pos[0] + pos[2]) / 2)
                    box_y_avg = int((pos[1] + pos[3]) / 2)

                    target_x_pixel, target_y_pixel = job.circle.x, job.circle.y

                    h, w = job.depth_matrix.shape
                    if not (0 <= target_x_pixel < w and 0 <= target_y_pixel < h):
                        continue
                    if not (0 <= box_x_avg < w and 0 <= box_y_avg < h):
                        continue

                    depth_target = job.depth_matrix[target_y_pixel, target_x_pixel]
                    depth_landmark = job.depth_matrix[box_y_avg, box_x_avg]


                    if depth_target == 0 or depth_landmark == 0:
                        continue

                    target_realworld = transformation2dto3d(OBPoint2f(target_x_pixel, target_y_pixel), depth_target, job.depth_intrinsics, job.extrinsic)
                    landmark_realworld = transformation2dto3d(OBPoint2f(box_x_avg, box_y_avg), depth_landmark, job.depth_intrinsics, job.extrinsic)

                    name = self.model.names[int(box.cls)]

                    distance = self.compute_distance(target_realworld, landmark_realworld)

                    local_landmarks[str(name)] = round(distance / 1000, 3)
                    
                if local_landmarks:
                    self.shared_state.update_landmarks(local_landmarks, job.circle.id)
                else:
                    self.shared_state.clear_landmarks()
            else:
                self.shared_state.clear_landmarks()
                return

        except Exception as e:
            print("Exception from yolo's run_job" + repr(e))

        finally:
            self.shared_state.set_busy(False)
