from math import pi, isfinite
import cv2 as cv
import numpy as np

class Circle:
    def __init__(self, dims, roundness, circle_id):
        self.x = int(dims[0])
        self.y = int(dims[1])
        self.r = int(dims[2])
         
        self.roundness = roundness
        self.min_eq = 0.0003
        self.id = circle_id

    # computes circularity of shape (1 is perfect circle)
    @staticmethod
    def computeRoundness(area, perim):
        if perim ** 2 < 1e-6:
            return 0.0

        roundness = 4 * pi * area / (perim ** 2)
        
        if not isfinite(roundness):
            return 0.0
        
        return roundness

    # returns True if its >thresh 
    @staticmethod
    def isCircle(r, area, perim, thresh=0.7, min_area=500, min_radius=20):
        if r < min_radius:
            return False
        
        if area < min_area:
            return False

        roundness = Circle.computeRoundness(area, perim)        
        return roundness >= thresh
    
    def __gt__(self, other):
        return self.roundness - other.roundness > self.min_eq
    
    def __lt__(self, other):
        return other.roundness - self.roundness > self.min_eq
    
    def __eq__(self, other):
        return abs(self.roundness - other.roundness) < self.min_eq


class ColorSegmenter:
    def __init__(self, config):
        self.config = config

    def segment(self, image, mask_code):
        masks = []
        best_circle = None

        blurred_img= cv.GaussianBlur(image,(5,5),0)
        hsv_img = cv.cvtColor(blurred_img, cv.COLOR_BGR2HSV)

        if 'r' in mask_code:
            masks.append(cv.inRange(hsv_img, self.config.light_red1, self.config.dark_red1))
            masks.append(cv.inRange(hsv_img, self.config.light_red2, self.config.dark_red2))
        if 'b' in mask_code:
            masks.append(cv.inRange(hsv_img, self.config.light_blue, self.config.dark_blue))
        if 'g' in mask_code:
            masks.append(cv.inRange(hsv_img, self.config.light_green, self.config.dark_green))
        if 'y' in mask_code:
            masks.append(cv.inRange(hsv_img, self.config.light_yellow, self.config.dark_yellow))
        if 'p' in mask_code:
            masks.append(cv.inRange(hsv_img, self.config.light_purple, self.config.dark_purple))
        if 'o' in mask_code:
            masks.append(cv.inRange(hsv_img, self.config.light_orange, self.config.dark_orange))

        for mask in masks:
            kernel = np.ones((5, 5), np.uint8)

            # filtering out noise by erosion _+ dilation
            mask_processed = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
            mask_processed = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

            contours, _ = cv.findContours(mask_processed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            # check if each contour is a circle or not
            for contour in contours:
                contour_area = cv.contourArea(contour)
                contour_perim = cv.arcLength(contour, True)
                (x, y),r = cv.minEnclosingCircle(contour)
                r = int(r)
                
                if Circle.isCircle(r, contour_area, contour_perim):
                    roundness = Circle.computeRoundness(contour_area, contour_perim)
                    circle_id = f"{r}:{int(y)}:{int(x)}"
                    circle = Circle((x, y, r), roundness, circle_id)

                    if not best_circle or circle > best_circle:
                        best_circle = circle

        return best_circle

