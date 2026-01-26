from yolo_worker import YoloJob
from ransac_worker import RansacJob
import cv2 as cv


class VisionApp:
    def __init__(self, camera, segmenter, yolo_worker, ransac_worker, state, renderer, app_config):
        self.camera =  camera
        self.segmenter = segmenter
        self.yolo_worker = yolo_worker
        self.ransac_worker = ransac_worker
        self.state = state
        self.renderer = renderer
        self.app_config = app_config
        self.calibration = camera.calibration
        self.frame_count = 0
        self.no_circle_count = 0


    def tick(self):
        try:
            self.frame_count += 1
            self.frame_bundle = self.camera.read()

            if not self.frame_bundle:
                return

            image = self.frame_bundle.color_image
            mask_code = self.app_config.mask_code

            self.top_circles = self.segmenter.segment(image, mask_code)

            none_count = 0
            for circle in self.top_circles:
                if circle is None:
                    none_count += 1
            
            if none_count == len(self.top_circles):
                self.no_circle_count += 1
                snapshot = self.state.snapshot()
                self.renderer.render(image, snapshot, False, False)
            else:
                self.no_circle_count = 0
                
                filtered_circles = [c for c in self.top_circles if c is not None]
                sorted_circles = sorted(filtered_circles, key= lambda c: c.roundness, reverse=True)
                best_circle = max(filtered_circles, key=lambda c: c.roundness)

                # only run yolo on best circle in top_circles list
                if self.frame_count % self.app_config.yolo_interval == 0:
                    self.submit_ransac(self.frame_bundle, self.calibration, image)
                    self.submit_yolo(best_circle, self.frame_bundle, self.calibration)

                for circle in sorted_circles:
                    if circle != best_circle:
                        cv.circle(image, (circle.x, circle.y), circle.r, (0, 0, 0), 3)         
                    else:
                        cv.circle(image, (circle.x, circle.y), circle.r, (180, 105, 255), 3)    

                snapshot = self.state.snapshot()
                
                if snapshot["wall"] is not None and len(snapshot["wall"]) > 0:
                    self.renderer.render(image, snapshot, True, True)
                else:
                    self.renderer.render(image, snapshot, True, False)
        except Exception as e:
            print("Exception from tick " + repr(e))


    def submit_yolo(self, circle, frame_bundle, calibration):
        try:
            if not self.state.is_busy():
                # crop out circle
                x, y, r = circle.x, circle.y, circle.r

                image_copy = frame_bundle.color_image.copy()
                cv.circle(image_copy, (x, y), r, (0, 0, 0), -1)
                job = YoloJob(frame_bundle, circle, calibration, image_copy)

                self.yolo_worker.submit_job(job)
        except Exception as e:
            print("Exception from submit_yolo " + repr(e))

    def submit_ransac(self, frame_bundle, calibration, image_array):
        try:
            if not self.state.is_ransac_busy():
                job = RansacJob(frame_bundle, calibration, image_array)
                self.ransac_worker.submit_job(job)
        except Exception as e:
            print("Exception from submit_ransac " + repr(e))


    def shutdown(self):
        self.camera.stop()
        print("Pipeline stopped and all windows closed.")