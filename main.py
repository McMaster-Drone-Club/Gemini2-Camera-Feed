import cv2 as cv
import numpy as np
from pyorbbecsdk import *
from utils import frame_to_bgr_image
from math import pi, radians, isfinite, tan
from sys import exit
from ultralytics import YOLO

ESC_KEY = 27
MIN_DEPTH = 20  # 20mm
MAX_DEPTH = 10000  # 10000mm = 10 m
PRINT_INTERVAL = 1
AVG_INTERVAL = 30
YOLO_INTERVAL = 20# Depth FOV
HFOV_DEPTH_DEG = 91.0
VFOV_DEPTH_DEG = 66.0
# RGB/Color FOV
HFOV_COLOR_DEG = 86.0
VFOV_COLOR_DEG = 55.0

# Precompute radians
HFOV_DEPTH_RAD = radians(HFOV_DEPTH_DEG)
VFOV_DEPTH_RAD = radians(VFOV_DEPTH_DEG)
HFOV_COLOR_RAD = radians(HFOV_COLOR_DEG)
VFOV_COLOR_RAD = radians(VFOV_COLOR_DEG)

model = YOLO("yolov10m.pt") 

light_blue = (95, 80, 50)
dark_blue  = (130, 255, 255)

light_purple = (130, 80, 50)
dark_purple = (143, 255, 255)

light_green = (50, 80, 50)
dark_green = (70, 255, 255)

light_red1 = (0, 80, 50)
dark_red1  = (10, 255, 255)

light_red2 = (170, 80, 50)
dark_red2  = (179, 255, 255)

light_yellow = (25, 90, 60)
dark_yellow = (32, 255, 255)

circles = [] # (x, y, r)


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
    pipeline = Pipeline()

    pipeline.start()
    print("Pipeline started successfully. Press 'q' or ESC to exit.")

    # Set window size
    window_width = 1280
    window_height = 720
    cv.namedWindow("QuickStart Viewer", cv.WINDOW_NORMAL)
    cv.resizeWindow("QuickStart Viewer", window_width, window_height)

    i = 0
    center_distance_avg = 0
    distances = [0] * AVG_INTERVAL

    frameCount =  0

    while True:
        try:
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue

            frameCount += 1

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
            
            color_image, circle = segmentation('ry', image_stream=color_image, bounding_box_color=(0, 0, 0))

            if circle and frameCount % YOLO_INTERVAL == 0:
                # crop out circle
                x, y, r = circle
                raw_copy = cv.circle(raw, (x, y), r, (0, 0, 0), -1)
                yolo_inference(raw_copy, circle)
                frameCount = 0

            # Process depth data
            width = depth_frame.get_width()
            height = depth_frame.get_height()
            scale = depth_frame.get_depth_scale() 

            
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((height, width))
            depth_data = depth_data.astype(np.float32) * scale
            depth_data = np.where((depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH), depth_data, 0).astype(np.uint16)

            # Create depth visualization
            depth_image = cv.normalize(depth_data, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
            depth_image = cv.applyColorMap(depth_image, cv.COLORMAP_JET)

            center_y = int(height / 2)
            center_x = int(width / 2)
            center_distance = depth_data[center_y, center_x]
            distances[i % AVG_INTERVAL] = center_distance
            i += 1

            valid = [d for d in distances if d > 0]

            if valid:
                center_distance_avg = sum(valid) / len(valid)

            # Resize and combine images
            color_image_resized = cv.resize(color_image, (window_width // 2, window_height))
            depth_image_resized = cv.resize(depth_image, (window_width // 2, window_height))
            combined_image = np.hstack((color_image_resized, depth_image_resized))

            cv.putText(combined_image, f"Center distance: {round(center_distance_avg / 10, 2)} cm", (30, 30), cv.FONT_HERSHEY_PLAIN, 2, color=(255, 0, 0), thickness=3)
            
            cv.imshow("QuickStart Viewer", combined_image)

            if cv.waitKey(1) in [ord('q'), ESC_KEY]:
                break
        except KeyboardInterrupt:
            break

    cv.destroyAllWindows()
    pipeline.stop()
    print("Pipeline stopped and all windows closed.")
    exit()


def pixel_to_camera_xyz(dx, dy, depth_m, d_w, d_h):
    """
    Map depth pixel (dx, dy, depth_m) to camera coordinates (X, Y, Z),
    using the DEPTH FOV.
    """
    if depth_m is None or depth_m <= 0 or not isfinite(depth_m):
        return None

    cx_d = d_w / 2.0
    cy_d = d_h / 2.0

    nx = (dx - cx_d) / (d_w / 2.0)
    ny = (dy - cy_d) / (d_h / 2.0)

    theta_x = nx * (HFOV_DEPTH_RAD / 2.0)
    theta_y = ny * (VFOV_DEPTH_RAD / 2.0)

    X = depth_m * tan(theta_x)
    Y = depth_m * tan(theta_y)
    Z = depth_m

    return np.array([X, Y, Z], dtype=np.float32)


def yolo_inference(image_array, circle):    
    results = model(image_array)
    
    # get x and y coordinate of landmark
    # get distance to circle
    
    result = results[0]
    noBoxes = result.boxes is None or len(result.boxes) == 0

    if not noBoxes:
        for box in result.boxes: # each box is an actual objects 
            pos = box.xyxy[0].cpu().numpy()
            box_x_avg = int((pos[0] + pos[2]) / 2)
            box_y_avg = int((pos[1] + pos[3]) / 2)

            print(f"Object position: {box_x_avg.item(), box_y_avg.item()} Circle location: {circle[0], circle[1]} ")

        annotated_frame = result.plot()
        cv.imshow("YOLO11 Detection", annotated_frame)



if __name__ == "__main__":
    main()
