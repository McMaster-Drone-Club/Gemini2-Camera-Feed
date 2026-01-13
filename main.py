from pyorbbecsdk import *
from config import AppConfig
from camera import OrbbecCamera
from segmentation import ColorSegmenter
from vision_app import VisionApp
from yolo_worker import YoloWorker
from ransac_worker import RansacWorker
from shared_state import SharedState
from renderer import Renderer
from sys import exit

def run():
    app_config = AppConfig("yolov10n.pt", "rp")
    orbec_camera = OrbbecCamera(app_config)
    segmenter = ColorSegmenter(app_config)
    state = SharedState()
    yolo = YoloWorker(state, app_config)
    ransac = RansacWorker(state)
    rend = Renderer(app_config)

    app = VisionApp(orbec_camera, segmenter, yolo, ransac, state, rend, app_config)

    
    if not orbec_camera.start_and_sync():
        return

    try:
        while not rend.should_quit():
            app.tick()
    except Exception as e:
        print(repr(e))
        raise
    finally:
        rend.close()
        app.shutdown()
        exit()

if __name__ == '__main__':
    run()