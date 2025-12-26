from pyorbbecsdk import *
from config import AppConfig
from camera import OrbbecCamera
from segmentation import ColorSegmenter
from vision_app import VisionApp
from yolo_worker import YoloWorker
from shared_state import SharedState
from renderer import Renderer
from sys import exit

def run():
    app_config = AppConfig()
    orbec_camera = OrbbecCamera(app_config)
    segmenter = ColorSegmenter(app_config)
    state = SharedState()
    yolo = YoloWorker(state, app_config)
    rend = Renderer(app_config)

    app = VisionApp(orbec_camera, segmenter, yolo, state, rend, app_config)
    orbec_camera.start_and_sync()

    try:
        while not rend.should_quit():
            app.tick()
    finally:
        rend.close()
        app.shutdown()
        exit()

if __name__ == '__main__':
    run()