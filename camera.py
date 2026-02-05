import numpy as np
import cv2 as cv
from pyorbbecsdk import *

class Calibration:
    def __init__(self, camera):
        pipeline = camera.pipeline
        self.color_profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        self.depth_profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        self.color_profile = self.color_profile_list.get_default_video_stream_profile()
        self.depth_profile = self.depth_profile_list.get_default_video_stream_profile()
        self.depth_intrinsics = self.depth_profile.as_video_stream_profile().get_intrinsic()
        self.extrinsic = self.depth_profile.get_extrinsic_to(self.color_profile)


class FrameBundle:
    def __init__(self, camera):
        self.camera = camera

    def capture_frames(self):
        for _ in range(50):
            frames = self.camera.pipeline.wait_for_frames(500)
            if frames is None:
                continue

            aligned = self.camera.align_filter.process(frames)
            if aligned is None:
                continue

            frameset  = aligned.as_frame_set()
            if frameset is None:
                continue

            # Get color and depth frames
            self.color_frame = frameset.get_color_frame()
            self.depth_frame = frameset.get_depth_frame()

            if self.color_frame is None or self.depth_frame is None:
                continue
            
            self.color_image = self.camera.frame_to_bgr_image(self.color_frame)
            
            if self.depth_frame.get_format() != OBFormat.Y16:
                raise Exception("Depth format is not Y16")
            
            depth_u16 = np.frombuffer(self.depth_frame.get_data(), dtype=np.uint16).reshape(self.get_frame_dims())
            depth_u16 = np.where((depth_u16 > self.camera.app_config.min_depth) & (depth_u16 < self.camera.app_config.max_depth), depth_u16, 0).astype(np.uint16)
            self.depth_u16 = depth_u16.copy()

            return
            
        raise Exception("Could not capture frames")

    def get_frame_dims(self): # H, W
        return self.depth_frame.get_height(), self.depth_frame.get_width()
    

class OrbbecCamera:

    def __init__(self, app_config):
        self.pipeline = Pipeline()
        self.config = Config()
        self.app_config = app_config
        self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
        self.calibration = Calibration(self)


    # must call before using camera to align color and depth capture
    def start_and_sync(self):   
        try:
            self.config.enable_stream(self.calibration.color_profile)
            self.config.enable_stream(self.calibration.depth_profile)
            self.pipeline.enable_frame_sync()
            self.frame_bundle = FrameBundle(self)
            self.pipeline.start(self.config)
        except Exception as e:
            print(e)
            return False
        
        return True
        
    def read(self):
        try:
            self.frame_bundle.capture_frames()
        except Exception as e:
            print(e)
            return None
            
        return self.frame_bundle
    
    def stop(self):
        self.pipeline.stop()
        
    
    def frame_to_bgr_image(self, frame):
        width = frame.get_width()
        height = frame.get_height()
        color_format = frame.get_format()
        data = np.frombuffer(frame.get_data(), dtype=np.uint8)
        
        if color_format == OBFormat.RGB:
            image = data.reshape((height, width, 3))
            return cv.cvtColor(image, cv.COLOR_RGB2BGR)
        
        elif color_format == OBFormat.BGR:
            return data.reshape((height, width, 3))
    
        elif color_format == OBFormat.YUYV:
            image = data.reshape((height, width, 2))
            return cv.cvtColor(image, cv.COLOR_YUV2BGR_YUYV)
        
        elif color_format == OBFormat.MJPG:
            return cv.imdecode(data, cv.IMREAD_COLOR)
            
        elif color_format == OBFormat.I420:
            return i420_to_bgr(data, width, height)
            
        elif color_format == OBFormat.NV12:
            return nv12_to_bgr(data, width, height)
            
        elif color_format == OBFormat.NV21:
            return nv21_to_bgr(data, width, height)
            
        elif color_format == OBFormat.UYVY:
            image = data.reshape((height, width, 2))
            return cv.cvtColor(image, cv.COLOR_YUV2BGR_UYVY)

        else:
            print("Unsupported color format: {}".format(color_format))
            return None