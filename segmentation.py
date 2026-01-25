from math import pi, isfinite
import cv2 as cv
import numpy as np

class Circle:
    def __init__(self, dims, roundness, circle_id, color):
        self.x = int(dims[0])
        self.y = int(dims[1])
        self.r = int(dims[2])
         
        self.roundness = roundness
        self.min_eq = 0.003 # circularity threshold
        self.id = circle_id
        self.color = color

    # computes circularity of shape (1 is perfect circle)
    @staticmethod
    def computeRoundness(area, perim):
        if perim ** 2 < 1e-6:
            return 0.0

        roundness = 4 * pi * area / (perim ** 2)
        
        if not isfinite(roundness):
            return 0.0
        
        return roundness
    
    @staticmethod
    def computeSolidity(contour, area):
        hull = cv.convexHull(contour)
        hull_area = cv.contourArea(hull)
        
        if hull_area > 1e-6:
            return area / hull_area
        
        return 0.0


    # returns True if its >thresh 
    @staticmethod
    def isCircle(r, area, perim, contour, thresh1=0.7, thresh2=(0.6, 1.1), thresh3=(0.85, 1.0), min_area=300, min_radius=15):   
        if r < min_radius:
            return False
        
        if area < min_area:
            return False
        
        computed_area = pi * r ** 2

        if computed_area < 1e-6:
            return False

        fill = area / computed_area
        if fill < thresh2[0] or fill > thresh2[1]:
            return False
        
        solidity = Circle.computeSolidity(contour, area)
        if solidity < thresh3[0] or solidity > thresh3[1]:
            return False

        roundness = Circle.computeRoundness(area, perim)        
        return roundness >= thresh1
    
    # comparing circles via roundness. NOTE: None object is treated as smallest possible circle
    def __gt__(self, other):
        if other is None:
            return True
        
        return self.roundness - other.roundness > self.min_eq
    
    def __lt__(self, other):
        if other is None:
            return False
        
        return other.roundness - self.roundness > self.min_eq
    
    def __eq__(self, other):
        if other is None:
            return False

        return abs(self.roundness - other.roundness) < self.min_eq


class ColorSegmenter:
    def __init__(self, config):
        self.config = config

    def segment(self, image, mask_code, k=3):
        masks = []
        top_circles = [None] * k 

        blurred_img = cv.GaussianBlur(image,(5,5),0) # 5 by 5 kernel
        hsv_img = cv.cvtColor(blurred_img, cv.COLOR_BGR2HSV)

        if 'r' in mask_code:
            masks.append(('r', cv.inRange(hsv_img, self.config.light_red1, self.config.dark_red1)))
            masks.append(('r', cv.inRange(hsv_img, self.config.light_red2, self.config.dark_red2)))
        if 'b' in mask_code:
            masks.append(('b', cv.inRange(hsv_img, self.config.light_blue, self.config.dark_blue)))
        if 'g' in mask_code:
            masks.append(('g', cv.inRange(hsv_img, self.config.light_green, self.config.dark_green)))
        if 'y' in mask_code:
            masks.append(('y', cv.inRange(hsv_img, self.config.light_yellow, self.config.dark_yellow)))
        if 'p' in mask_code:
            masks.append(('p', cv.inRange(hsv_img, self.config.light_purple, self.config.dark_purple)))
        if 'o' in mask_code:
            masks.append(('o', cv.inRange(hsv_img, self.config.light_orange, self.config.dark_orange)))

        for mask in masks:
            color, hsv_range = mask

            kernel1 = np.ones((7, 7), np.uint8)
            kernel2 = np.ones((3, 3), np.uint8)
            # filtering out noise by erosion + dilation
            mask_processed = cv.morphologyEx(hsv_range, cv.MORPH_CLOSE, kernel1)
            mask_processed = cv.morphologyEx(mask_processed, cv.MORPH_OPEN, kernel2)

            contours, _ = cv.findContours(mask_processed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            # check if each contour is a circle or not
            for contour in contours:
                contour_area = cv.contourArea(contour)
                contour_perim = cv.arcLength(contour, True)
                (x, y),r = cv.minEnclosingCircle(contour)
                r = int(r)

                roundness = Circle.computeRoundness(contour_area, contour_perim)

                if Circle.isCircle(r, contour_area, contour_perim, contour):                 
                    roundness = Circle.computeRoundness(contour_area, contour_perim)
                    circle_id = f"{r}:{int(y)}:{int(x)}"
                    circle = Circle((x, y, r), roundness, circle_id, color)

                    for i in range(len(top_circles)):
                        if circle > top_circles[i]:
                            top_circles.insert(i, circle)
                            top_circles = top_circles[:k]
                            break

        return top_circles

