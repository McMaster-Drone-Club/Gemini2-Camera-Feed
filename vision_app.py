from yolo_worker import YoloJob
import cv2 as cv

class VisionApp:
    def __init__(self, camera, segmenter, yolo_worker, state, renderer, app_config):
        self.camera =  camera
        self.segmenter = segmenter
        self.yolo_worker = yolo_worker
        self.state = state
        self.renderer = renderer
        self.app_config = app_config
        self.calibration = camera.calibration
        self.frame_count = 0
        self.no_circle_count = 0


    def tick(self):
        self.frame_count += 1
        self.frame_bundle = self.camera.read()

        if not self.frame_bundle:
            return

        image = self.frame_bundle.color_image
        mask_code = self.app_config.mask_code

        self.circle = self.segmenter.segment(image, mask_code)
    
        if self.circle:
            cv.circle(image, (self.circle.x, self.circle.y), self.circle.r, (0, 0, 0), 3)

            if self.frame_count % self.app_config.yolo_interval == 0:
                self.submit_yolo(self.circle, self.frame_bundle, self.calibration)

            snapshot = self.state.snapshot()
            self.renderer.render(image, snapshot, True)
                    
        else:
            self.no_circle_count += 1
            snapshot = self.state.snapshot()
            self.renderer.render(image, snapshot, False)

        

    def submit_yolo(self, circle, frame_bundle, calibration):
        if not self.state.is_busy():
            # crop out circle
            x, y, r = circle.x, circle.y, circle.r

            image_copy = frame_bundle.color_image.copy()
            cv.circle(image_copy, (x, y), r, (0, 0, 0), -1)
            job = YoloJob(frame_bundle, circle, calibration, image_copy)

            self.yolo_worker.submit_job(job)


    def shutdown(self):
        self.camera.stop()
        print("Pipeline stopped and all windows closed.")