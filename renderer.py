import cv2 as cv

class Renderer:
    ESC_KEY = 27

    def __init__(self, app_config):
        self.app_config = app_config


    def render(self, image, snapshot, circle, plane):
        try:
            lm = snapshot["landmarks"]
            detections = snapshot.get("detections", [])

            for det in detections:
                x1 = det["x1"]
                y1 = det["y1"]
                x2 = det["x2"]
                y2 = det["y2"]
                label = det["label"]

                cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                (text_w, text_h), baseline = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                text_x = x1
                text_y = max(y1 - 8, text_h + 8)
                cv.rectangle(image, (text_x, text_y - text_h - baseline - 4), (text_x + text_w + 6, text_y + 2), (0, 0, 0), -1)
                cv.putText(image, label, (text_x + 3, text_y - 2), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv.LINE_AA)

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
                # Display landmarks even without circle detection, or show searching message
                if lm and len(lm) > 0:
                    landmark_format = ""
                    for key in lm.keys():
                        landmark_format += str(key) + " : " + str(lm[key]) + " meters\n"

                    landmark_format = landmark_format.split("\n")
                    i = 0

                    for line in landmark_format:
                        cv.putText(image, line, (30, 30 + 30 * i), cv.FONT_HERSHEY_PLAIN, 2, color=(0, 255, 255), thickness=3)
                        i += 1
            if plane:
                hull = snapshot["wall"]
                cv.drawContours(image, [hull], -1, (0, 255, 0), 3)
                wall_text = "Wall plane"
                cv.putText(image, wall_text, (30, image.shape[0] - 20), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
            
            cv.imshow("Live drone feed", image)
        except Exception as e:
            print("Exception from renderer " + repr(e))

    def should_quit(self):
        return cv.waitKey(1) in [ord('q'), Renderer.ESC_KEY]
    
    def close(self):
        cv.destroyAllWindows()
       

