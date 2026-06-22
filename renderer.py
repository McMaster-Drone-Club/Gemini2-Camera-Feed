import cv2 as cv

class Renderer:
    ESC_KEY = 27

    def __init__(self, app_config):
        self.app_config = app_config


    def render(self, image, snapshot, circle, plane):
        try:
            if circle:
                lm = snapshot["landmarks"]

                if lm and len(lm) > 0:
                    landmark_format = ""                
                    for key in lm.keys():
                        landmark_format += str(key) + " : " + str(lm[key]) + " meters\n"

                    landmark_format = landmark_format.split("\n")
                    i = 0

                    for line in landmark_format:
                        cv.putText(image, line, (30, 30 + 30 * i), cv.FONT_HERSHEY_PLAIN, 2, color=(255, 0, 0), thickness=3)
                        i += 1
                else:
                    cv.putText(image, "No landmarks detected", (30, 30), cv.FONT_HERSHEY_PLAIN, 2, color=(255, 0, 0), thickness=3)
                
            else:
                cv.putText(image, "Searching for targets ... ", (30, 30), cv.FONT_HERSHEY_PLAIN, 2, color=(255, 0, 0), thickness=3)
            
            if plane:
                wall = snapshot["wall"]
                if wall is not None and wall.hull is not None:
                    cv.drawContours(image, [hull], -1, (0, 255, 0), 2)
            
            cv.imshow("Live drone feed", image)
        except Exception as e:
            print("Exception from renderer " + repr(e))

    def should_quit(self):
        return cv.waitKey(1) in [ord('q'), Renderer.ESC_KEY]
    
    def close(self):
        cv.destroyAllWindows()
       

