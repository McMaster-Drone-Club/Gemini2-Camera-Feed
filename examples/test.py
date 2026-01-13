import cv2 as cv
import numpy as np
from pyorbbecsdk import *
from utils import frame_to_bgr_image
from math import pi, sqrt
from sys import exit
from ultralytics import YOLO
import threading

ESC_KEY = 27
MIN_DEPTH = 20  # 20mm
MAX_DEPTH = 10000  # 10000mm = 10 m
PRINT_INTERVAL = 1
AVG_INTERVAL = 30
YOLO_INTERVAL = 20

model = YOLO("yolov10m.pt") 

light_blue = (95, 80, 50)
dark_blue  = (130, 255, 255)

light_purple = (125, 40, 50)
dark_purple = (155, 255, 255)

light_green = (40, 50, 50)
dark_green = (75, 255, 255)

light_red1 = (0, 80, 50)
dark_red1  = (10, 255, 255)

light_red2 = (170, 80, 50)
dark_red2  = (179, 255, 255)

light_orange = (15, 80, 50)
dark_orange = (25, 255, 255)

light_yellow = (20, 40, 60)
dark_yellow = (35, 255, 255)

landmarks = {}
lock = threading.Lock()
yolo_busy = False


# computes circularity of shape; returns True if its >thresh (1 is perfect circle)
def isCircle(r, area, perim, thresh=0.7, min_area=500, min_radius=20):
    if r < min_radius:
        return False
    
    if area < min_area:
        return False
    
    roundness = 4 * pi * area / (perim ** 2)
    return roundness >= thresh



def segmentation(mask_code, image_stream=None, image_path='', bounding_box_color=(0, 0, 255)):
    img = None
    circle = None

    if image_path != '':
        img = cv.imread(image_path)
    elif image_stream is not None:
        img = image_stream
    else:
        raise Exception("Must supply either image stream (Matlike) or image file path (str)")

    # hsv makes defining masks easier because it ignores shadows
    
    blurred_img= cv.GaussianBlur(img,(5,5),0)
    hsv_img = cv.cvtColor(blurred_img, cv.COLOR_BGR2HSV)
     

    masks = []

    if 'r' in mask_code:
        masks.append(cv.inRange(hsv_img, light_red1, dark_red1))
        masks.append(cv.inRange(hsv_img, light_red2, dark_red2))
    if 'b' in mask_code:
        masks.append(cv.inRange(hsv_img, light_blue, dark_blue))
    if 'g' in mask_code:
        masks.append(cv.inRange(hsv_img, light_green, dark_green))
    if 'y' in mask_code:
        masks.append(cv.inRange(hsv_img, light_yellow, dark_yellow))
    if 'p' in mask_code:
        masks.append(cv.inRange(hsv_img, light_purple, dark_purple))
    if 'o' in mask_code:
        masks.append(cv.inRange(hsv_img, light_orange, dark_orange))

    size = len(masks)
    
    combined_mask = masks[0]
    if size > 1:
        for next_mask in masks[1:]:
            combined_mask = cv.bitwise_or(combined_mask, next_mask)


    kernel = np.ones((5, 5), np.uint8)

    # filtering out noise by erosion _+ dilation
    combined_mask_processed = cv.morphologyEx(combined_mask, cv.MORPH_OPEN, kernel)
    combined_mask_processed = cv.morphologyEx(combined_mask, cv.MORPH_CLOSE, kernel)

    contours, hierarchy = cv.findContours(combined_mask_processed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
     

    # check if each contour is a circle or not
    for contour in contours:
        contour_area = cv.contourArea(contour)
        contour_perim = cv.arcLength(contour, True)
        (x, y),r = cv.minEnclosingCircle(contour)
        center = (int(x),int(y))
        r = int(r)

        if isCircle(r, contour_area, contour_perim): # draw bounding box around circles only
            cv.circle(img,center,r,bounding_box_color,3)
            circle = int(x), int(y), int(r)

    return img, circle


def main():
    global landmarks, yolo_busy

    pipeline = Pipeline()
    config = Config()
    
    print("Pipeline started successfully. Press 'q' or ESC to exit.")

    profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = profile_list.get_default_video_stream_profile()
    config.enable_stream(color_profile)
    profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = profile_list.get_default_video_stream_profile()
    config.enable_stream(depth_profile)

    depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsic()
    extrinsic = depth_profile.get_extrinsic_to(color_profile)

    try:
        pipeline.enable_frame_sync()
    except Exception as e:
        print(e)

    try:
        pipeline.start(config)
    except Exception as e:
        print(e)
        return

    align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)


    # Set window size
    window_width = 1280
    window_height = 720
    cv.namedWindow("QuickStart Viewer", cv.WINDOW_NORMAL)
    cv.resizeWindow("QuickStart Viewer", window_width, window_height)

    frameCount =  0
    noCircleCount = 0

    while True:
        try:
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue

            frames = align_filter.process(frames)
            if not frames:
                continue

            frames  = frames.as_frame_set()

            frameCount += 1
            noCircleCount += 1

            # Get color frame
            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue
            color_image = frame_to_bgr_image(color_frame)
            raw = color_image.copy()

            # Get depth frame
            depth_frame = frames.get_depth_frame()
            if depth_frame is None:
                continue
            if depth_frame.get_format() != OBFormat.Y16:
                print("Depth format is not Y16")
                continue
            
            color_image, circle = segmentation('r', image_stream=color_image, bounding_box_color=(0, 0, 0))

            if circle is None and noCircleCount % 10 == 0:
                with lock:
                    noCircleCount = 0
                    landmarks.clear()
        
            # Process depth data
            width = depth_frame.get_width()
            height = depth_frame.get_height()
            scale = depth_frame.get_depth_scale() 

            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((height, width))
            depth_data = np.where((depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH), depth_data, 0).astype(np.uint16)
            

            if circle and frameCount % YOLO_INTERVAL == 0:
                # crop out circle
                x, y, r = circle

                raw_copy = raw.copy()
                cv.circle(raw_copy, (x, y), r, (0, 0, 0), -1)

                start = False

                with lock:
                    if not yolo_busy:
                        yolo_busy = True
                        start = True
                
                if start:
                    threading.Thread(target=yolo_inference, args=(raw_copy.copy(), circle, depth_data.copy(), depth_intrinsics, extrinsic), daemon=True).start()

                frameCount = 0


            # Resize and combine images
            color_image_resized = cv.resize(color_image, (window_width // 2, window_height))

            if (depth_frame.get_width() != color_frame.get_width() or depth_frame.get_height() != color_frame.get_height()):
                raise Exception("WARNING: depth and color not same size after alignment!")

            landmark_format = "No landmarks detected"
            
            with lock:
                lm = landmarks.copy()

            if lm and len(lm) > 0:
                landmark_format = ""                
                for key in lm.keys():
                    landmark_format += str(key) + " : " + str(lm[key]) + " meters\n"

                landmark_format = landmark_format.split("\n")
                i = 0

                for line in landmark_format:
                    cv.putText(color_image_resized, line, (30, 30 + 30 * i), cv.FONT_HERSHEY_PLAIN, 2, color=(255, 0, 0), thickness=3)
                    i += 1
            else:
                cv.putText(color_image_resized, landmark_format, (30, 30), cv.FONT_HERSHEY_PLAIN, 2, color=(255, 0, 0), thickness=3)
            
            cv.imshow("Live drone feed", color_image_resized)

            if cv.waitKey(1) in [ord('q'), ESC_KEY]:
                break
        except KeyboardInterrupt:
            break

    cv.destroyAllWindows()
    pipeline.stop()
    print("Pipeline stopped and all windows closed.")
    exit()


def get_distance(point1, point2):
    x1, y1, z1 = point1.x, point1.y, point1.z
    x2, y2, z2 = point2.x, point2.y, point2.z

    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1

    return round(sqrt(dx * dx + dy * dy + dz * dz), 3)


def yolo_inference(image_array, circle, depth_matrix, depth_intrinsics, extrinsic):    
    global landmarks, yolo_busy, lock

    try:
        results = model(image_array)
        result = results[0]
        noBoxes = result.boxes is None or len(result.boxes) == 0

        if not noBoxes:
            for box in result.boxes: # each box is an actual objects 
                pos = box.xyxy[0].cpu().numpy()
                box_x_avg = int((pos[0] + pos[2]) / 2)
                box_y_avg = int((pos[1] + pos[3]) / 2)

                target_x_pixel, target_y_pixel, _ = circle

                depth_target = depth_matrix[target_y_pixel, target_x_pixel]
                depth_landmark = depth_matrix[box_y_avg, box_x_avg]

                if depth_target == 0 or depth_landmark == 0:
                    continue

                target_realworld = transformation2dto3d(OBPoint2f(target_x_pixel, target_y_pixel), depth_target, depth_intrinsics, extrinsic)
                landmark_realworld = transformation2dto3d(OBPoint2f(box_x_avg, box_y_avg), depth_landmark, depth_intrinsics, extrinsic)

                name = model.names[int(box.cls)]

                distance = get_distance(target_realworld, landmark_realworld)

                with lock:
                    landmarks[str(name)] = round(distance / 1000, 3)

    except Exception as e:
        print(e)

    finally:
        with lock:
            yolo_busy = False



if __name__ == "__test__":
    main()